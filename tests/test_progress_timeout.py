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
    SUPERVISOR_TIMEOUT_S = 36000      # systemd TimeoutStartSec on ralph-validator.service (10 h)
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


def test_a_venv_without_pip_fails_immediately_and_says_why():
    """`python3 -m venv .venv 2>/dev/null;` — error hidden, `;` carrying on regardless. On an image
    with no ensurepip (massedcompute/desmoines, 2026-08-07) venv "succeeded", produced no pip, and
    the round died three commands later as a bare `127` with the real message discarded. Provider
    images vary and we do not pick them, so the install must repair what it can and fail LOUDLY at
    four minutes rather than silently at forty."""
    import subprocess
    import tempfile

    from eval.orchestrator import _VENV_CMD

    # the suppression that hid it must not come back
    assert "venv .venv 2>/dev/null" not in _VENV_CMD,         "the venv error is being discarded again — that is the whole bug"
    # it must be able to REPAIR the common case...
    assert "python3-venv" in _VENV_CMD and "get-pip" in _VENV_CMD
    # ...and REFUSE rather than continue when it cannot
    assert "exit 3" in _VENV_CMD, "a venv with no pip must stop the round, not carry on"
    assert _VENV_CMD.rstrip().endswith("&&"),         "the install must chain on success, so a failure short-circuits the rest"

    # and it has to be valid shell — this string is only ever executed on a box that bills
    with tempfile.NamedTemporaryFile("w", suffix=".sh") as fh:
        fh.write("set -e\n" + _VENV_CMD + " true\n")
        fh.flush()
        r = subprocess.run(["bash", "-n", fh.name], capture_output=True, text=True)
    assert r.returncode == 0, f"generated install shell does not parse: {r.stderr}"


def test_a_cpu_llama_build_is_refused_before_it_can_bill():
    """The CPU wheel accepts `n_gpu_layers` and IGNORES it, so a failed CUDA build yields a round
    that rents an H100, scores every submission on CPU at ~6x, and says so in one log line nobody
    reads. Observed on massedcompute/desmoines: the wheel came out 20 MB instead of 283 MB.

    Catching it at install costs five minutes. Catching it in score_job costs the parent and
    observer downloads first. Not catching it cost $12 of a round that had to be killed."""
    import inspect
    import os
    import re

    from eval import orchestrator as O

    src = inspect.getsource(O)
    assert "eval.gpu_check" in src, "nothing verifies the wheel that was actually built"
    assert "RALPH_ALLOW_CPU_STUDENTS" in src, "no deliberate escape hatch"
    m = re.search(r"verify_cmd = \(\n(.*?)\n    \) if gpu_students else \"\"", src, re.S)
    assert m, "verify_cmd not found in the shape the install builds"
    assert "exit 9" in m.group(1), "a CPU-only build must stop the install, not warn"


def test_the_cuda_probe_does_not_fail_closed_on_a_missing_symbol():
    """IT REFUSED A GOOD ROUND. The guard asked for `llama_supports_gpu_offload`; recent bindings
    no longer export it, the call raised, the shell's `||` read the non-zero exit as "no GPU", and
    a correctly built 283 MB CUDA wheel was rejected — $0.61 and an hour, and it would have blocked
    every round after it.

    An absent capability FUNCTION and an absent capability look identical through `getattr`, so one
    probe returning false is not evidence. The two errors are asymmetric: believing a CPU build is
    CUDA wastes a round's money; believing a CUDA build is CPU refuses a round that would have
    worked — and that is the one that happened."""
    import os
    import sys
    import tempfile
    import types

    from eval.gpu_check import probe

    real = sys.modules.get("llama_cpp")
    real_inner = sys.modules.get("llama_cpp.llama_cpp")
    try:
        # a binding with NO capability function, whose package ships a CUDA backend
        with tempfile.TemporaryDirectory() as d:
            os_path = os.path.join(d, "lib")
            os.makedirs(os_path)
            open(os.path.join(os_path, "libggml-cuda.so"), "w").close()
            pkg = types.ModuleType("llama_cpp")
            pkg.__file__ = os.path.join(d, "__init__.py")
            inner = types.ModuleType("llama_cpp.llama_cpp")   # deliberately no symbol
            sys.modules["llama_cpp"] = pkg
            sys.modules["llama_cpp.llama_cpp"] = inner
            ok, why = probe()
            assert ok, f"a CUDA build was called CPU because one symbol was missing: {why}"
            assert "cuda" in why.lower()

        # and a genuine CPU wheel is still caught
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "lib"))
            open(os.path.join(d, "lib", "libggml-cpu.so"), "w").close()
            pkg = types.ModuleType("llama_cpp")
            pkg.__file__ = os.path.join(d, "__init__.py")
            inner = types.ModuleType("llama_cpp.llama_cpp")
            inner.llama_supports_gpu_offload = lambda: False
            sys.modules["llama_cpp"] = pkg
            sys.modules["llama_cpp.llama_cpp"] = inner
            ok, why = probe()
            assert not ok, f"a CPU wheel passed as CUDA: {why}"
    finally:
        for k, v in (("llama_cpp", real), ("llama_cpp.llama_cpp", real_inner)):
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    # the probe must never raise, whatever it finds
    ok, why = probe()
    assert isinstance(ok, bool) and isinstance(why, str)


def test_a_region_whose_image_cannot_build_is_not_rented_again():
    """`rent`'s own `exclude` lives for one round's retries, so the cheapest broken candidate is
    picked again on the very next start and the subnet wedges on one bad image. This is about the
    IMAGE, never the GPU — `require_gpu` still binds the device name, and a different region is
    not a different measurement."""
    import os

    from eval.orchestrator import GpuSpec
    from eval.run_orchestrated import Config

    assert GpuSpec().exclude_regions == ()
    old = os.environ.get("RALPH_GPU_EXCLUDE")
    try:
        os.environ["RALPH_GPU_EXCLUDE"] = "massedcompute/desmoines-usa-1, foo/bar"
        cfg = Config.from_env()
        assert cfg.exclude_regions == ("massedcompute/desmoines-usa-1", "foo/bar"), \
            cfg.exclude_regions
    finally:
        if old is None:
            os.environ.pop("RALPH_GPU_EXCLUDE", None)
        else:
            os.environ["RALPH_GPU_EXCLUDE"] = old

    # and rent() must actually filter on it
    class _P(__import__("eval.orchestrator", fromlist=["x"]).ShadeformProvider):
        def __init__(self):
            self.ssh_key_id = ""

        def _key(self):
            return "k"

        def _api(self, method, path, body=None):
            return {"instance_types": [
                {"cloud": "massedcompute", "shade_instance_type": "H100", "hourly_price": 199,
                 "availability": [{"region": "desmoines-usa-1", "available": True}]}]}

    spec = GpuSpec(exclude_regions=("massedcompute/desmoines-usa-1",))
    try:
        _P().rent(spec, "ralph-round-9-1")
        raise AssertionError("rented an excluded region")
    except RuntimeError as e:
        assert "excluded" in str(e), e


def test_the_supervisor_deadline_must_be_outside_our_own():
    """THE LADDER'S OUTERMOST RING LIVES IN A UNIT FILE, so the ladder test — which asserts the
    rings this code owns — could not see it. On 2026-08-07 `kill_at` was raised 4.5 h -> 8 h and
    `TimeoutStartSec` was left at 6 h, leaving it INVERTED: systemd would have SIGTERMed a healthy
    6.5 h round 30 minutes from the end. Whoever kills the process decides whether the rental dies
    or leaks, which is the shape of the 226-minute hang."""
    from eval.orchestrator import GpuSpec
    from eval.run_orchestrated import _supervisor_deadline_problems

    spec = GpuSpec()
    outer_h = spec.max_hours + spec.provider_deadline_slack_h

    class _Run:
        def __init__(self, value):
            self.stdout, self.returncode = value, 0

    import subprocess
    real = subprocess.run
    try:
        # inside our deadlines -> a finding, naming both numbers
        subprocess.run = lambda *a, **k: _Run("6h")
        bad = _supervisor_deadline_problems(spec)
        assert bad and "INSIDE" in bad[0], bad
        assert "6h" in bad[0] and f"{outer_h:.2f}" in bad[0], bad[0]

        # comfortably outside -> silent
        subprocess.run = lambda *a, **k: _Run("10h")
        assert _supervisor_deadline_problems(spec) == []

        # compound and bare-seconds forms both parse
        subprocess.run = lambda *a, **k: _Run("1h 30min")
        assert _supervisor_deadline_problems(spec), "1h30 is inside an 8.5h ring"
        subprocess.run = lambda *a, **k: _Run("36000s")
        assert _supervisor_deadline_problems(spec) == []

        # no supervisor at all says NOTHING: an unsupervised round has no outer ring to invert
        subprocess.run = lambda *a, **k: _Run("infinity")
        assert _supervisor_deadline_problems(spec) == []
        subprocess.run = lambda *a, **k: _Run("")
        assert _supervisor_deadline_problems(spec) == []

        def _boom(*a, **k):
            raise OSError("no systemctl here")

        subprocess.run = _boom
        assert _supervisor_deadline_problems(spec) == [], "a missing systemctl must not block a round"
    finally:
        subprocess.run = real


def test_every_remote_call_carries_the_ssh_identity():
    """A RENTED BOX HAS NO KEY IN YOUR AGENT. `_SSH_OPTS` is options only — the identity and the
    port live beside it, and `eval.orchestrator._ssh` has always passed all three. bench/run_remote
    copied the options and not the rest, so every command came back `Permission denied (publickey)`
    and an H100 was rented, failed and destroyed inside three minutes for nothing.

    Cheap to lose, trivially avoidable, and exactly the class of thing a test should hold."""
    import inspect

    from bench import run_remote as B
    from eval.orchestrator import GpuSpec, Instance

    inst = Instance(id="i", ip="1.2.3.4", ssh_user="shadeform", ssh_port=2222)
    argv = B._ssh_argv(inst, GpuSpec())
    assert argv[0] == "ssh"
    assert "-i" in argv and GpuSpec().ssh_key in argv, argv
    assert "-p" in argv and "2222" in argv, argv
    for opt in ("StrictHostKeyChecking=no", "ConnectTimeout=15"):
        assert opt in argv, (opt, argv)

    # and no remote call may be built any other way, ANYWHERE IN THE MODULE. This used to scan only
    # main()'s body; moving the scp into a helper so a failed run's results are fetched before
    # teardown took it out of the window, and the check silently stopped covering it.
    src = inspect.getsource(B)
    for bad in ('["ssh", *_SSH_OPTS', '"ssh " + " ".join(_SSH_OPTS)'):
        assert bad not in src, f"a remote call bypasses _ssh_argv: {bad}"
    # scp takes -P (capital) for the port, not -p — a silent difference from ssh
    assert '"scp"' in src, "no scp call found — results are never retrieved"
    for at in [i for i in range(len(src)) if src.startswith('"scp"', i)]:
        scp = src[at:at + 200]
        assert '"-i"' in scp and '"-P"' in scp, scp


def test_a_rental_with_no_gpu_is_refused_before_the_install():
    """`wait_ready` reports the PROVIDER's view — created, running, reachable — and a box can
    satisfy all three with no card attached. Seen 2026-08-10 on a latitude H100: six nvidia kernel
    modules loaded, `nvidia-smi` answering "No devices were found", zero NVIDIA devices on the PCI
    bus. The driver was fine; the GPU simply was not there.

    `_assert_gpu` in score_job would eventually catch it — after torch and a llama.cpp source
    build, fifteen minutes in. One ssh catches it in seconds."""
    import eval.orchestrator as O

    inst, spec = O.Instance(id="i", ip="1.2.3.4"), O.GpuSpec()
    real = O._ssh
    try:
        O._ssh = lambda *a, **k: "GPU 0: NVIDIA H100 PCIe (UUID: GPU-abc)\n"
        assert O.gpu_devices(inst, spec) == ["GPU 0: NVIDIA H100 PCIe (UUID: GPU-abc)"]
        assert O.assert_gpu_present(inst, spec)

        # the exact output the broken box gave
        O._ssh = lambda *a, **k: "No devices were found\n"
        assert O.gpu_devices(inst, spec) == []
        try:
            O.assert_gpu_present(inst, spec)
            raise AssertionError("a GPU-less rental was accepted")
        except O.RemoteRoundError as e:
            assert "NO GPU" in str(e), e

        # an ssh that fails is NOT evidence of a GPU — fail closed
        def _boom(*a, **k):
            raise RuntimeError("ssh died")

        O._ssh = _boom
        assert O.gpu_devices(inst, spec) == []
    finally:
        O._ssh = real

    # and the round asks BEFORE installing
    import inspect
    src = inspect.getsource(O.run_remote_round)
    i_ready, i_gpu = src.index("wait_ready"), src.index("assert_gpu_present")
    assert i_gpu > i_ready, "the GPU check must follow wait_ready"
    # ...and precede the delegation to `_run_on`, which is what installs and scores. If the check
    # ran after that, it would save nothing — the fifteen minutes are already spent.
    assert i_gpu < src.index("_run_on"), \
        "the GPU check must precede the install, or it saves nothing"
    # a GPU-less box goes through the rent-retry loop rather than failing the round outright:
    # the region is excluded and another is tried, exactly as for a box that never booted
    assert "exclude=tuple(tried)" in src


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
