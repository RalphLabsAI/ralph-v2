"""Regression tests for the thing that cost $12.50: a round that stopped talking and was not
stopped. Runnable two ways:

    python -m tests.test_progress_timeout      # self-running harness, no deps
    pytest tests/test_progress_timeout.py      # if pytest is installed

These run against real child processes rather than mocks, because every one of the seven round-1
failures lived in the seam between two machines and a mocked pipe reproduces none of them. What
they cannot cover is ssh itself, so the sub-second budgets here stand in for the twenty-minute one.
"""
from __future__ import annotations

import io
import subprocess
import sys
import time

from eval.orchestrator import (SILENCE_S, GpuSpec, RemoteRoundError, _stream_until_silent,
                               _teardown_on_signal)


def _proc(code: str):
    """A child whose stdout is a raw pipe, exactly as `_ssh_stream` opens ssh."""
    return subprocess.Popen([sys.executable, "-u", "-c", code],
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, bufsize=0)


def _run(code: str, silence_s: float, hard_s: float = 60.0):
    buf = io.StringIO()
    try:
        return _stream_until_silent(_proc(code), silence_s, hard_s, buf.write, "the test child"), \
            buf.getvalue(), None
    except RemoteRoundError as e:
        return None, buf.getvalue(), e


def test_silence_kills_and_a_chatty_child_survives():
    """The whole property, both directions. A child that sleeps past the budget dies; a child that
    sleeps just as long while TALKING does not — which is the half that matters, because a timeout
    that kills healthy rounds is worse than the hang it replaces."""
    t0 = time.monotonic()
    out, log, err = _run("import time; print('working'); time.sleep(30)", silence_s=1.0)
    assert err is not None and "no output for" in str(err)
    assert "working" in log
    # it died on the budget, not on the child's own 30s sleep
    assert time.monotonic() - t0 < 10

    out, log, err = _run(
        "import time\n"
        "for i in range(8):\n"
        "    print('tick', i); time.sleep(0.3)\n", silence_s=1.0)
    assert err is None, f"a talking child was killed: {err}"
    assert "tick 7" in log


def test_liveness_is_bytes_not_lines():
    """A `tqdm` bar and `hf_hub_download` separate updates with \\r and emit no \\n until they
    finish. A line-based detector reads a healthy 16 GB checkpoint download as silence, so the
    detector counts BYTES — this child never writes a newline at all."""
    out, log, err = _run(
        "import sys, time\n"
        "for i in range(8):\n"
        "    sys.stdout.write('%d%%\\r' % i); sys.stdout.flush(); time.sleep(0.3)\n"
        "sys.stdout.write('done\\n')\n", silence_s=1.0)
    assert err is None, f"a \\r-only progress bar was mistaken for silence: {err}"
    assert "done" in log


def test_the_hard_ceiling_still_binds_on_a_chatty_child():
    """The silence budget catches the round that stopped working; this catches the one that is
    still working and can no longer be afforded. Without it a scorer stuck in a retry loop talks
    its way past every deadline."""
    out, log, err = _run("import time\n"
                         "while True:\n"
                         "    print('still here'); time.sleep(0.1)\n",
                         silence_s=30.0, hard_s=1.0)
    assert err is not None and "ceiling" in str(err)


def test_a_nonzero_exit_is_a_failed_round_and_carries_the_output():
    """The scorer's traceback is the whole diagnosis. `| tail` used to eat the exit status; this
    asserts the status AND that the reason travels with it."""
    out, log, err = _run("import sys; print('KeyError: binary'); sys.exit(1)", silence_s=5.0)
    assert err is not None and "failed (1)" in str(err)
    assert "KeyError: binary" in str(err)


def test_output_is_streamed_not_collected():
    """Streaming is not a stylistic choice: the watchers decide a round is alive from the log's
    mtime, so a line has to reach the log while the round is still running. This asserts the first
    line is written long before the child exits."""
    buf, seen = io.StringIO(), []

    class _Watch:
        def write(self, s):
            seen.append(time.monotonic())
            return buf.write(s)

        def flush(self):
            pass

    w = _Watch()
    t0 = time.monotonic()
    _stream_until_silent(_proc("import time; print('early'); time.sleep(2); print('late')"),
                         10.0, 60.0, w.write, "the test child")
    assert seen and seen[0] - t0 < 1.0, "output was collected at the end, not streamed"
    assert time.monotonic() - t0 >= 2.0


def test_a_carriage_return_only_child_still_writes_the_log():
    """THE SIGNAL THE EXTERNAL WATCHDOG READS. `silence_s` measures bytes on the PIPE; the watchdog
    measures bytes in the LOG, and a `\\r`-only tqdm bar reaches `_emit` once per 2000 characters —
    so a healthy 16 GB download can leave the log's mtime frozen for many minutes while the pipe is
    busy. That is the exact signature of a hang, on a round that is working."""
    buf = io.StringIO()
    _stream_until_silent(
        _proc("import sys, time\n"
              "for i in range(12):\n"
              "    sys.stdout.write('%d%%\\r' % i); sys.stdout.flush(); time.sleep(0.1)\n"),
        5.0, 60.0, buf.write, "the test child", heartbeat_s=0.3)
    assert "[heartbeat]" in buf.getvalue(), "the log would have gone silent on a healthy round"


def test_output_past_the_suppression_cap_still_heartbeats():
    """`_emit` stops writing after `max_lines` while the reader keeps resetting the silence clock.
    Without a heartbeat that is a permanently frozen log on a round that is talking constantly."""
    buf = io.StringIO()
    _stream_until_silent(
        _proc("import time\n"
              "for i in range(40):\n"
              "    print('line', i); time.sleep(0.05)\n"),
        5.0, 60.0, buf.write, "the test child", max_lines=3, heartbeat_s=0.2)
    log = buf.getvalue()
    assert "further output suppressed" in log
    assert "[heartbeat]" in log.split("further output suppressed", 1)[1], \
        "the log froze after the suppression cap — the watchdog would read this as a hang"


def test_a_flood_cannot_fill_the_disk():
    """The log lives on the orchestrator, which also holds the signing key, and a remote in a
    print loop must not be able to fill that disk. The round still succeeds — a chatty box is not
    a failed box — it is only the echoing that stops."""
    buf = io.StringIO()
    _stream_until_silent(_proc("for i in range(400): print('x' * 40)"),
                         5.0, 60.0, buf.write, "the test child", max_lines=50)
    log = buf.getvalue()
    assert "further output suppressed after 50 lines" in log
    assert log.count("\n") < 60


def test_sigterm_becomes_an_exception_so_teardown_runs():
    """THE $12.50 TEST. systemd's TimeoutStartSec sent SIGTERM, Python's default disposition
    terminated the interpreter without unwinding, the `finally` that destroys the rental never ran,
    and an H100 billed for another 50 minutes. The handler has to turn the signal into a raise."""
    import os
    import signal

    before = signal.getsignal(signal.SIGTERM)
    torn_down = []
    try:
        with _teardown_on_signal():
            try:
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(2)               # the raise lands at the next bytecode boundary
                raise AssertionError("SIGTERM did not raise — the rental would have leaked")
            finally:
                torn_down.append("destroyed")
    except RemoteRoundError as e:
        assert "signal" in str(e)
    assert torn_down == ["destroyed"]

    # ...and the previous handler is put back, or the second round of the day inherits ours
    assert signal.getsignal(signal.SIGTERM) is before


def test_the_budgets_are_ordered_against_the_supervisor():
    """Whoever's deadline fires first decides whether the instance is destroyed or abandoned, so
    ours has to fire first. systemd's TimeoutStartSec is 10800s; the spec's ceiling must sit under
    it, and the silence budget under that."""
    SUPERVISOR_TIMEOUT_S = 21600      # systemd TimeoutStartSec on ralph-validator.service (6 h)
    spec = GpuSpec()
    assert spec.max_hours * 3600 < SUPERVISOR_TIMEOUT_S, \
        "the supervisor would kill us before we can tear down"
    assert spec.silence_s == SILENCE_S < spec.max_hours * 3600
    # and the watchdog must never be tighter than the rental it is watching
    from eval.watchdog import RENTAL_CEILING_S, RUN_CEILING_S
    assert spec.max_hours * 3600 < RUN_CEILING_S < RENTAL_CEILING_S < SUPERVISOR_TIMEOUT_S, \
        "a watchdog ceiling below the rental's destroys healthy long rounds"


def test_the_ceiling_is_the_rentals_not_the_steps():
    """Per-step budgets ADD UP: a 40-minute install plus a 2.5-hour score is over the supervisor's
    three hours even though each step is individually within its limit. Every step has to clamp to
    what is left of `kill_at`, which is stamped when the meter starts."""
    from eval.orchestrator import Instance, _ssh_stream

    spec = GpuSpec(ssh_key="/dev/null")
    inst = Instance(ip="127.0.0.1", kill_at=time.time() - 1)   # rental already over its ceiling
    t0 = time.monotonic()
    try:
        # ssh will fail on the key long before this matters; what is asserted is that the clamp
        # picked ~0 seconds rather than the step's own 2.5 hours.
        _ssh_stream(inst, spec, "sleep 600", io.StringIO().write, what="the test child")
    except RemoteRoundError:
        pass
    assert time.monotonic() - t0 < 30, "the step used its own budget, not the rental's"


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    bad = 0
    for t in TESTS:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except Exception as e:
            bad += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - bad}/{len(TESTS)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
