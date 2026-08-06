"""Regression tests for `eval/watchdog.py` — the answer to "is it making progress, or only
existing?". Runnable two ways:

    python -m tests.test_watchdog      # self-running harness, no deps
    pytest tests/test_watchdog.py      # if pytest is installed

THE CLOCK AND THE THRESHOLDS ARE INJECTED, THE FILES ARE REAL. `classify()` is pure over frozen
observations, so a 2.75-hour ceiling is testable in a millisecond. But the log is a real file with
real append semantics, real byte offsets and a real `stat`, because every one of the seven round-1
failures lived in a seam between two machines and a mocked file reproduces none of them.

The test that matters most is not the one that catches the hang. It is
`test_the_idle_box_is_not_an_alarm`: a watchdog that cries wolf on a quiet box gets disabled inside
a week, and then the real hang goes unwatched.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import time

from eval.orchestrator import GpuSpec, INSTALL_SILENCE_S, SILENCE_S
from eval.watchdog import (GRAMMAR, MINT_PREFIX, POLL_S, PROTECTED, RENTAL_CEILING_S,
                           RUN_BANNER, RUN_CEILING_S, STALE_S, LogView, UnitView,
                           WatchdogConfig, act, classify, read_log)

NOW = 1_756_000_000.0
RATE = 330                      # hourly_price is in cents, as the API returns it


def _cfg(**kw):
    d = dict(log_path="/dev/null", work_dir="/tmp", act=True, hung_act=True)
    d.update(kw)
    return WatchdogConfig(**d)


def _unit(running=True, pid=4242, start=NOW - 7200):
    """`start` defaults to well back, because `classify` clamps silence to the RUN's age — a round
    that began 60 s ago cannot have been silent for an hour, whatever the append-only log says.
    Tests that exercise the young-run case pass `start` explicitly."""
    return UnitView(load_state="loaded", active_state="activating" if running else "failed",
                    sub_state="start" if running else "failed", pid=pid if running else 0,
                    start_t=start, invocation="abc123", pid_alive=running)


def _owner(pid=4242, cmd="python -m eval.run_orchestrated", started=NOW - 600):
    """(pid, cmdline, start_time). The start time is read by `read_owners`, never by `classify` —
    classify is pure over frozen observations, and a /proc stat inside it meant a replayed pass
    silently lost every ZOMBIE_OWNER finding."""
    return (pid, cmd, started)


def _log(mtime, milestone="", line="", live=(), dead=(), stale=(), pid=4242, trusted=True):
    return LogView(mtime=mtime, inode=1, banner_pid=pid, milestone=milestone,
                   line=line or (milestone + "…" if milestone else ""),
                   claims_live=frozenset(live), claims_dead=frozenset(dead),
                   claims_stale=frozenset(stale), trusted=trusted)


def _inst(name, iid, created_s, price=RATE, status="active"):
    return {"name": name, "id": iid, "status": status, "hourly_price": price,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime(created_s))}


# --- the rung that keeps this turned on -------------------------------------------------------

def test_the_idle_box_is_not_an_alarm():
    """Unit failed, log 13 hours stale, nothing rented. That is the CORRECT resting state — the
    log is append-only and nobody has run a round since yesterday. A watchdog that alarms here is
    a watchdog that gets `systemctl disable`d, and then nothing watches the real hang."""
    v = classify(NOW, _unit(running=False), [], _log(NOW - 13 * 3600), [], _cfg(), {})
    assert v.state == "IDLE", v.state
    assert not v.alarms and not v.acting and not v.destroy


def test_a_fresh_run_does_not_inherit_the_previous_runs_tail():
    """The append-only trap. The file's last line is the PREVIOUS round's `scoring…` — the tightest
    budget there is — and the current run started 60 s ago and has not written its banner yet."""
    log = _log(NOW - 4000, milestone="scoring", pid=999)      # banner pid != unit pid
    v = classify(NOW, _unit(pid=4242, start=NOW - 60), [_owner(cmd="python -m eval.run_orchestrated")],
                 log, [_inst("ralph-round-1-1", "i-1", NOW - 60)], _cfg(), {})
    assert v.state != "HUNG", f"a 60-second-old run inherited a dead round's milestone: {v}"


def test_the_banner_scopes_the_milestone_to_this_run():
    """Two runs in one file: the milestone must come from the slice after the LAST banner."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "log")
        with open(p, "w") as fh:
            fh.write(f"{RUN_BANNER}pid=111 t=1 ===\n  rented aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
                     "  scoring (this is the expensive part)…\n")
            fh.write(f"{RUN_BANNER}pid=4242 t=2 ===\n  installing…\n")
        lv = read_log(p, _unit(pid=4242))
        assert lv.milestone == "installing", lv.milestone
        assert lv.banner_pid == 4242
        # the first run's rental is claimed by a DEAD slice, not by this one
        assert "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" in lv.claims_dead
        assert not lv.claims_live


def test_attempt_seven_at_the_moment_the_watchdog_would_have_acted():
    """The incident, replayed. A claimed rental is up, the unit is running, and the log stopped
    moving 33 minutes ago in `scoring`. Detection in tens of minutes against the 226 it took."""
    # BOTH observations must exceed the budget, and the first is taken a poll earlier — so the
    # silence has to be past `budget + POLL_S`, not merely past `budget`. Confirmation costs one
    # poll of exposure and buys immunity to a clock step; that trade is the point.
    log = _log(NOW - 2000, milestone="scoring", live=["i-real"])
    r = [_inst("ralph-round-1-69408", "i-real", NOW - 2400)]
    latches = {}
    first = classify(NOW - POLL_S, _unit(), [_owner()], log, r, _cfg(), latches)
    assert not first.acting, "one observation must never be enough"
    v = classify(NOW, _unit(), [_owner()], log, r, _cfg(), latches)
    assert v.state == "HUNG", v.state
    assert v.signal_pid == 4242
    assert [d.id for d in v.destroy] == ["i-real"]
    assert v.destroy[0].cost > 1.0


# --- false positives: the half that matters ---------------------------------------------------

def test_a_healthy_install_survives_and_a_dead_one_does_not():
    """The largest false-positive risk in the system: pip reports per package, so a 2.5 GB torch
    wheel is legitimately quiet for a long time."""
    r = [_inst("ralph-round-1-1", "i-1", NOW - 3000)]
    ok = classify(NOW, _unit(), [_owner()], _log(NOW - 1700, "installing", live=["i-1"]),
                  r, _cfg(), {})
    assert ok.state == "RUNNING", ok.state
    latches = {}
    classify(NOW - POLL_S, _unit(), [_owner()], _log(NOW - 2600, "installing", live=["i-1"]),
             r, _cfg(), latches)
    bad = classify(NOW, _unit(), [_owner()], _log(NOW - 2600, "installing", live=["i-1"]),
                   r, _cfg(), latches)
    assert bad.state == "HUNG", bad.state


def test_wait_ready_silence_is_not_a_hang():
    """`rented` → the provider can legitimately take 900 s to hand over a box, and that is the
    most expensive place in the round to be wrong: the meter has started and nothing exists yet."""
    v = classify(NOW, _unit(), [_owner()], _log(NOW - 890, "rented", live=["i-1"]),
                 [_inst("ralph-round-1-1", "i-1", NOW - 900)], _cfg(), {})
    assert v.state == "RUNNING", v.state


def test_scoring_budget_never_beats_the_in_process_killer():
    """`_stream_until_silent` owns the scoring window: when IT fires it destroys the rental itself
    and writes the accounting line. The watchdog is the backstop for a WEDGED detector, so its
    budget must be strictly the looser of the two."""
    v = classify(NOW, _unit(), [_owner()], _log(NOW - (SILENCE_S + 100), "scoring",
                                                   live=["i-1"]),
                 [_inst("ralph-round-1-1", "i-1", NOW - 3000)], _cfg(), {})
    assert v.state == "RUNNING", v.state


def test_an_unrecognised_milestone_gets_slower_never_faster():
    """These needles are prose in another module and WILL be reworded. A rename must fall back to
    the conservative budget, never to a shorter one."""
    r = [_inst("ralph-round-1-1", "i-1", NOW - 5000)]
    mild = classify(NOW, _unit(), [_owner()], _log(NOW - 2000, "", "  frobnicating", ["i-1"]),
                    r, _cfg(), {})
    assert mild.state == "RUNNING", mild.state
    assert max(b for _n, b, _w in GRAMMAR) <= STALE_S


def test_the_growing_head_is_out_of_jurisdiction_even_with_a_corpse_present():
    """The head of a round — chain reads, a lineage replay that grows with the trail — is silent
    and free, and no budget applies to it. The trap this closes: keying "is a rental in flight" on
    the name prefix rather than on THIS run's claim, so any leaked corpse re-arms the log rung."""
    corpse = _inst("ralph-round-9-9", "i-corpse", NOW - 5 * 3600)
    v = classify(NOW, _unit(), [_owner()],
                 _log(NOW - 5000, "", "  v2 commitments: 3", live=[]), [corpse],
                 _cfg(), {})
    assert v.signal_pid == 0, "a healthy head was signalled because a corpse was lying around"
    assert v.state != "HUNG"


def test_a_corpse_does_not_arm_a_kill_against_the_live_round():
    """One 5-hour orphan plus one healthy claimed rental: the healthy one must never be touched."""
    rentals = [_inst("ralph-round-9-9", "i-corpse", NOW - 5 * 3600),
               _inst("ralph-round-10-1", "i-live", NOW - 300)]
    latches = {}
    classify(NOW - POLL_S, _unit(), [_owner()], _log(NOW - 10, "scoring", live=["i-live"]),
             rentals, _cfg(), latches)
    v = classify(NOW, _unit(), [_owner()], _log(NOW - 10, "scoring", live=["i-live"]),
                 rentals, _cfg(), latches)
    ids = [d.id for d in v.destroy]
    assert "i-live" not in ids, ids
    assert "i-corpse" in ids, ids


# --- money: the rung that survives the box being unable to run code ---------------------------

def test_the_orphan_is_killed_with_no_reference_to_the_log():
    """The incident's last 52 minutes. mtime is maximally FRESH here — the point is that the
    orphan rung never consults it."""
    v = classify(NOW, _unit(running=False), [], _log(NOW, dead=["i-dead"]),
                 [_inst("ralph-round-1-1", "i-dead", NOW - 3600)], _cfg(),
                 {"orphan": {"key": "i-dead", "since": NOW - 600, "obs": 1}})
    assert v.state == "ORPHAN", v.state
    assert v.signal_pid == 0, "there is no process to signal — that is the whole point"
    assert [d.id for d in v.destroy] == ["i-dead"]


def test_rent_post_timeout_leaves_a_name_claim():
    """`rent()` learns the id from the create response, so a create that succeeds server-side and
    times out client-side leaves a correctly-named H100 whose id nobody ever knew. The NAME
    written before the POST is the only claim that survives."""
    name = "ralph-round-9-11111"
    lv = LogView(mtime=NOW, inode=1, banner_pid=1, trusted=True,
                 claims_dead=frozenset([name]))
    v = classify(NOW, _unit(running=False), [], lv, [_inst(name, "i-unknown", NOW - 600)],
                 _cfg(), {"orphan": {"key": "i-unknown", "since": NOW - 600, "obs": 1}})
    assert v.state == "ORPHAN", v.state
    assert [d.name for d in v.destroy] == [name]


def test_protected_and_stray_are_reported_never_killed():
    """The account is shared: a live miner box, another project, and — observed on 2026-08-06 —
    a third H100 at $3.30/hr matching neither. Kill narrow, report wide."""
    rentals = [_inst("epago-4090", "i-a", NOW - 90 * 3600),
               _inst("ralph-v2-miner-m19", "i-b", NOW - 90 * 3600),
               _inst("kimi-ralph-sub4-16569", "i-c", NOW - 5 * 3600)]
    v = classify(NOW, _unit(running=False), [], _log(NOW - 3600), rentals, _cfg(), {})
    assert not v.destroy, [d.name for d in v.destroy]
    assert any("kimi-ralph-sub4" in r for r in v.report), v.report
    assert not any("epago" in r or "miner-m19" in r for r in v.report), v.report


def test_the_ceiling_needs_no_log_and_no_process():
    v = classify(NOW, _unit(running=False), [], _log(NOW),
                 [_inst("ralph-round-1-1", "i-1", NOW - RENTAL_CEILING_S - 60)], _cfg(),
                 {"overdue": {"key": "i-1", "since": NOW - 600, "obs": 1}})
    assert v.state == "OVERDUE", v.state
    assert [d.id for d in v.destroy] == ["i-1"]


def test_age_fails_to_infinity_not_zero():
    """A money guard must fail SAFE, and safe is 'assume it is old'. Returning 0 on an unparseable
    timestamp makes a leak permanently un-overdue and prints $0.00 next to a burning H100."""
    bad = {"name": "ralph-round-1-1", "id": "i-1", "status": "active", "hourly_price": RATE,
           "created_at": None}
    v = classify(NOW, _unit(running=False), [], _log(NOW), [bad], _cfg(),
                 {"overdue": {"key": "i-1", "since": NOW - 600, "obs": 1}})
    assert v.state == "OVERDUE", v.state


# --- degraded views ---------------------------------------------------------------------------

def test_an_api_failure_is_degraded_and_still_signals():
    """Withholding the one free remedy precisely when the money is invisible is attempt 7 with
    extra steps. SIGTERM costs nothing and delegates teardown to the round's own `finally`."""
    log = _log(NOW - 4000, "scoring", live=["i-1"])
    latches = {}
    classify(NOW - POLL_S, _unit(), [_owner()], log, None, _cfg(), latches)
    v = classify(NOW, _unit(), [_owner()], log, None, _cfg(), latches)
    assert v.state == "DEGRADED_HUNG", v.state
    assert v.signal_pid == 4242
    assert not v.destroy, "nothing can be destroyed through an unreachable API"


def test_an_api_failure_with_no_owner_is_not_all_clear():
    v = classify(NOW, _unit(running=False), [], _log(NOW - 3600), None, _cfg(), {})
    assert v.state == "NO_VERDICT", v.state
    assert "could not ask" in " ".join(v.report)


def test_an_untrusted_log_disables_only_the_log_rung():
    """A rotated log, or a round started by hand with a redirect. The money rungs never needed the
    log and must keep running."""
    v = classify(NOW, _unit(running=False), [], _log(NOW, trusted=False, dead=["i-1"]),
                 [_inst("ralph-round-1-1", "i-1", NOW - 3600)], _cfg(),
                 {"orphan": {"key": "i-1", "since": NOW - 600, "obs": 1}})
    assert v.state == "ORPHAN", v.state


def test_the_confirm_latch_needs_two_observations_and_a_still_log():
    latches = {}
    r = [_inst("ralph-round-1-1", "i-1", NOW - 3000)]
    a = classify(NOW - POLL_S, _unit(), [_owner()], _log(NOW - 2000, "scoring", live=["i-1"]),
                 r, _cfg(), latches)
    assert not a.acting
    # a single fresh write clears it — the key carries the mtime
    b = classify(NOW - 10, _unit(), [_owner()], _log(NOW - 20, "scoring", live=["i-1"]),
                 r, _cfg(), latches)
    assert not b.acting
    c = classify(NOW, _unit(), [_owner()], _log(NOW - 3000, "scoring", live=["i-1"]),
                 r, _cfg(), latches)
    assert not c.acting, "a new silence episode must start its own confirmation"


def test_milestone_survives_a_flood_of_streamed_lines():
    """A torch install writes far more than any fixed tail would hold. Scanning a window rather
    than the whole run slice silently demotes `  installing…` to the fallback budget."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "log")
        with open(p, "w") as fh:
            fh.write(f"{RUN_BANNER}pid=4242 t=1 ===\n  installing…\n")
            fh.write("".join(f"    · collecting package-{i}\n" for i in range(6000)))
        lv = read_log(p, _unit(pid=4242))
        assert lv.milestone == "installing", lv.milestone


# --- behaviour, not just classification -------------------------------------------------------

class _FakeProvider:
    def __init__(self, rentals=()):
        self.rentals = list(rentals)
        self.destroyed, self.swept = [], []

    def list_all(self):
        return [r for r in self.rentals if r["id"] not in self.destroyed]

    def list_active(self):
        return self.list_all()

    def destroy(self, inst):
        self.destroyed.append(inst.id)

    def sweep(self, **kw):
        self.swept.append(kw)
        return []


def test_act_is_a_noop_with_an_empty_plan():
    """The reviewed design ran its destroy-verification on every idle pass and printed the loudest
    string in the file 288 times a day."""
    p = _FakeProvider()
    buf = io.StringIO()
    from eval.watchdog import Verdict
    assert act(Verdict(state="IDLE"), _cfg(), p, out=buf) == 0
    assert buf.getvalue() == ""
    assert not p.destroyed


def test_it_never_sweeps_and_never_sigkills():
    """`sweep()` is prefix-scoped but age- and claim-blind; this process knows which rentals the
    running round claimed, and destroying by claim is the whole point. SIGKILL is the one signal
    `_teardown_on_signal` cannot catch."""
    import inspect

    import ast

    from eval import watchdog
    # PARSED, NOT GREPPED. Both of these names appear in prose in the module header explaining why
    # they are never used, and a substring search would fail on the documentation of its own rule.
    tree = ast.parse(inspect.getsource(watchdog))
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "sweep" not in called, "the watchdog must never call sweep(): it is claim-blind"
    named = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "SIGKILL" not in named, "SIGKILL is the one signal _teardown_on_signal cannot catch"

    p = _FakeProvider([_inst("ralph-round-1-1", "i-1", NOW - 3600)])
    from eval.watchdog import Target, Verdict
    v = Verdict(state="ORPHAN", destroy=(Target(id="i-1", name="ralph-round-1-1", age=3600,
                                                rate=3.30),))
    act(v, _cfg(), p, out=io.StringIO())
    assert p.destroyed == ["i-1"]
    assert p.swept == []


def test_a_name_we_did_not_mint_is_refused_at_the_last_moment():
    """The guard is re-checked immediately before the API call rather than inherited, because this
    process holds a key that can create and delete."""
    p = _FakeProvider()
    from eval.watchdog import Target, Verdict
    buf = io.StringIO()
    act(Verdict(state="ORPHAN", destroy=(Target(id="i-x", name="ralph-v2-miner-m19"),)),
        _cfg(), p, out=buf)
    assert p.destroyed == [], "it tried to destroy a protected box"
    assert "REFUSED" in buf.getvalue()


def test_hung_is_observed_before_it_is_armed():
    """Ship with the rung that cannot false-positive armed and the rung that can observed."""
    p = _FakeProvider()
    from eval.watchdog import Target, Verdict
    buf = io.StringIO()
    rc = act(Verdict(state="HUNG", signal_pid=999,
                     destroy=(Target(id="i-1", name="ralph-round-1-1"),)),
             _cfg(hung_act=False), p, out=buf)
    assert rc == 3 and not p.destroyed
    assert "OBSERVE" in buf.getvalue()


# --- the ladder -------------------------------------------------------------------------------

def test_the_budget_ladder_is_ordered():
    budgets = [b for _n, b, _w in GRAMMAR]
    from eval.orchestrator import HEARTBEAT_S
    assert HEARTBEAT_S <= POLL_S < SILENCE_S, \
        "a poll slower than the in-process killer cannot be its backstop"
    assert HEARTBEAT_S * 3 <= min(budgets), \
        "the log must advance several times inside the smallest budget"
    assert max(budgets) <= STALE_S, \
        "the grammar may only make the watchdog FASTER; a reworded line must fall back to the " \
        "conservative budget, never a shorter one"
    assert min(budgets) >= 900.0, "destroy()'s retry loop alone is 380 s"
    g = {n: b for n, b, _ in GRAMMAR}
    assert g["installing"] > INSTALL_SILENCE_S
    assert g["scoring"] > SILENCE_S
    # The supervisor's own timeout, from deploy/ralph-validator.service.d/40-timeout.conf. It is
    # named rather than inlined because the whole ladder is only meaningful relative to it: whoever
    # kills the process first decides whether the rental is destroyed or abandoned, and that must
    # always be us. Raised to 6 h with max_hours when ten miners were measured at ~3.2 h of scoring.
    SUPERVISOR_TIMEOUT_S = 21600
    assert (STALE_S < GpuSpec().max_hours * 3600 < RUN_CEILING_S < RENTAL_CEILING_S
            < SUPERVISOR_TIMEOUT_S)


def test_the_watchdog_mirrors_the_defaults_it_cannot_import():
    """`wait_ready(timeout_s=900)` and `_scp(timeout=1800)` are default arguments the budget table
    models by hand. Raise wait_ready's timeout and the watchdog starts killing healthy rentals at
    the moment they begin billing, with nothing failing to say so."""
    import inspect

    from eval.orchestrator import ShadeformProvider, _scp
    assert inspect.signature(ShadeformProvider.wait_ready).parameters["timeout_s"].default == 900
    assert inspect.signature(_scp).parameters["timeout"].default == 1800


def test_the_mint_prefix_cannot_match_the_miner_box():
    """`sweep()` was once scoped to `ralph-`, which matches `ralph-v2-miner-m19`. Destroying
    somebody's live work is a far worse failure than the leak it prevents."""
    for name in PROTECTED:
        assert not name.startswith(MINT_PREFIX), name
    assert MINT_PREFIX == "ralph-round-"


# --- through the REAL reader ------------------------------------------------------------------
#
# Everything above this line hands `classify()` a LogView built by hand, and that is how a blocker
# hid behind a green suite: the two tests that appeared to cover ORPHAN passed `running=False`
# together with `claims_dead={...}`, which is the exact INVERSE of what `read_log` produces for a
# crashed run. It filed the dead run's rental as `claims_live`, which excluded it from every bucket
# ORPHAN draws from, so the rung written for the SIGKILL / OOM / reboot class could not fire for
# it. These go through the real reader over a real file.

def _real_log(d, body):
    p = os.path.join(d, "ralph-validator.log")
    with open(p, "w") as fh:
        fh.write(body)
    return p


_CRASHED = (f"{RUN_BANNER}pid=111 t=1 ===\n"
            "  renting ralph-round-7-11111…\n"
            "  rented 11111111-1111-1111-1111-111111111111 H100 @ $3.30/hr\n"
            "  destroyed 11111111-1111-1111-1111-111111111111 after 5.5 min\n"
            f"{RUN_BANNER}pid=3851559 t=2 ===\n"
            "  renting ralph-round-8-69408…\n"
            "  rented 22222222-2222-2222-2222-222222222222 H100 @ $3.30/hr\n"
            "  scoring (this is the expensive part)…\n")


def test_a_crashed_run_leaves_an_orphan_the_real_reader_can_see():
    """THE INCIDENT'S OWN SHAPE. The round rented, was SIGKILLed (or the box rebooted), and the
    unit is `failed` with no process left. Nothing on this box can destroy that rental, so if this
    rung does not fire, nothing does — and it bills to the 2.75 h ceiling instead of ~8 min."""
    with tempfile.TemporaryDirectory() as d:
        lv = read_log(_real_log(d, _CRASHED), _unit(running=False))
        assert "22222222-2222-2222-2222-222222222222" in lv.claims_stale, lv
        assert not lv.claims_live, "a dead run's rental is not live"
        r = [_inst("ralph-round-8-69408", "22222222-2222-2222-2222-222222222222", NOW - 3600)]
        latches = {}
        classify(NOW - POLL_S, _unit(running=False), [], lv, r, _cfg(), latches)
        v = classify(NOW, _unit(running=False), [], lv, r, _cfg(), latches)
        assert v.state == "ORPHAN", v.state
        assert [d_.id for d_ in v.destroy] == ["22222222-2222-2222-2222-222222222222"]
        assert v.signal_pid == 0, "there is no process to signal"


def test_a_hand_started_round_is_not_an_orphan():
    """The other direction, and the reason the stale bucket is gated rather than destroyed on
    sight: a round started outside systemd looks identical to a corpse by log evidence alone.
    Killing somebody's live round is worse than the leak it prevents."""
    with tempfile.TemporaryDirectory() as d:
        lv = read_log(_real_log(d, _CRASHED), _unit(running=False))
        v = classify(NOW, _unit(running=False), [_owner()], lv,
                     [_inst("ralph-round-8-69408", "22222222-2222-2222-2222-222222222222",
                            NOW - 3600)], _cfg(), {})
        assert not v.destroy, [x.name for x in v.destroy]
        assert "HELD" in v.alarms or v.state == "IDLE", v


def test_a_bannerless_log_cannot_destroy_a_live_rental():
    """A rotated log, or one whose banner fell outside the read window. Nothing in the file can be
    attributed to anyone — and a LIVE round's rental is sitting in that blob."""
    with tempfile.TemporaryDirectory() as d:
        body = ("  rented 33333333-3333-3333-3333-333333333333 H100 @ $3.30/hr\n"
                "  scoring (this is the expensive part)…\n")
        lv = read_log(_real_log(d, body), _unit())
        assert not lv.trusted and not lv.claims_dead, lv
        v = classify(NOW, _unit(), [_owner()], lv,
                     [_inst("ralph-round-8-1", "33333333-3333-3333-3333-333333333333", NOW - 600)],
                     _cfg(), {})
        assert not v.destroy, "an unattributable log destroyed a live round's GPU"


def test_the_renting_name_claim_is_released_by_the_id():
    """`renting <name>` exists only to survive a create whose response was lost. Left behind after
    the id is known, it was never released by `destroyed <uuid>` — so the claim outlived the
    rental and kept the log rung armed over the publish tail, which is free and unbounded."""
    with tempfile.TemporaryDirectory() as d:
        body = (f"{RUN_BANNER}pid=4242 t=1 ===\n"
                "  renting ralph-round-8-1…\n"
                "  rented 44444444-4444-4444-4444-444444444444 H100 @ $3.30/hr\n"
                "  destroyed 44444444-4444-4444-4444-444444444444 after 30.0 min\n"
                "  signed by abc…\n")
        lv = read_log(_real_log(d, body), _unit(pid=4242))
        assert not lv.claims_live, f"the round's claim outlived its rental: {lv.claims_live}"
        v = classify(NOW, _unit(pid=4242), [_owner()], lv, [], _cfg(), {})
        assert v.state != "HUNG" and not v.acting, v


def test_an_undated_rental_of_this_run_is_reported_not_destroyed():
    """`instance_age_s` fails to +inf so a leak is never mistaken for something fresh. Applied to a
    rental THIS run is holding, +inf would mean "destroy the healthy round's GPU because the
    provider reformatted a timestamp"."""
    bad = {"name": "ralph-round-8-1", "id": "i-live", "status": "active", "hourly_price": RATE,
           "created_at": None}
    v = classify(NOW, _unit(), [_owner()], _log(NOW - 10, "scoring", live=["i-live"]), [bad],
                 _cfg(), {})
    assert not v.destroy, "an unparseable timestamp destroyed the live round's rental"
    assert "UNDATED" in v.alarms, v.alarms


def test_the_latch_key_is_order_independent():
    """The provider returns instances in whatever order it likes. An unsorted key changed on every
    poll, so the latch reset each time and the money rungs could never reach two observations."""
    from eval.watchdog import _confirmed
    a, b = {}, {}
    assert not _confirmed(a, "orphan", ["i-1", "i-2"], NOW - 600, 300.0)
    assert _confirmed(a, "orphan", ["i-2", "i-1"], NOW, 300.0), \
        "a reordered instance list reset the confirmation"
    assert not _confirmed(b, "orphan", ["i-1"], NOW - 600, 300.0)
    assert _confirmed(b, "orphan", ["i-1"], NOW, 300.0)


def test_a_recorded_acting_pass_replays_to_the_same_verdict():
    """Without the latches in the record, every verdict that ACTED replayed as a non-action — the
    forensic case the events file exists for was the one case it could not reproduce."""
    import eval.watchdog as W
    with tempfile.TemporaryDirectory() as d:
        cfg = _cfg(work_dir=d, log_path=_real_log(d, _CRASHED))
        r = [_inst("ralph-round-8-69408", "22222222-2222-2222-2222-222222222222", NOW - 3600)]
        p = _FakeProvider(r)
        W.once(cfg, p, out=io.StringIO(), now=NOW - POLL_S)
        W.once(cfg, p, out=io.StringIO(), now=NOW)
        events = os.path.join(d, "watchdog-events.jsonl")
        lines = [json.loads(x) for x in open(events)]
        assert lines[-1]["verdict"]["state"] == "ORPHAN", lines[-1]["verdict"]["state"]
        assert lines[-1]["verdict"]["acting"] is True
        buf = io.StringIO()
        assert W.replay(events, -1, cfg, out=buf) == 0, buf.getvalue()
        assert json.loads(buf.getvalue())["match"] is True


def test_the_events_file_is_valid_json_when_an_age_is_unknown():
    """`json.dumps` emits a bare `Infinity` token, which nothing can read back — in exactly the
    degraded passes this file exists to preserve."""
    import eval.watchdog as W
    with tempfile.TemporaryDirectory() as d:
        cfg = _cfg(work_dir=d, log_path=_real_log(d, _CRASHED))
        bad = {"name": "ralph-round-8-69408", "id": "i-x", "status": "active",
               "hourly_price": RATE, "created_at": "not a date"}
        W.once(cfg, _FakeProvider([bad]), out=io.StringIO(), now=NOW)
        for line in open(os.path.join(d, "watchdog-events.jsonl")):
            json.loads(line)          # raises on `Infinity`


def test_the_provider_deadline_is_the_outermost_ring():
    """THE LADDER, end to end. Each ring must be looser than the one inside it, or the outer one
    pre-empts a legitimate round; and the outermost must be the provider's, because it is the only
    one that does not need this box to be alive. Inside-out: the in-process silence killer, the
    rental's own kill_at, the watchdog's ceiling, then Shadeform's auto_delete."""
    spec = GpuSpec()
    kill_at_s = spec.max_hours * 3600
    provider_s = (spec.max_hours + spec.provider_deadline_slack_h) * 3600
    assert SILENCE_S < kill_at_s < RENTAL_CEILING_S < provider_s, \
        "a ring is tighter than the one it is supposed to back up"
    assert spec.provider_deadline_slack_h > 0, \
        "auto_delete at exactly kill_at would race the round's own teardown"


def test_rent_sets_a_provider_side_deadline_and_verifies_it():
    """`kill_at`, the `finally` and the watchdog all need something on this box to be alive. The
    incident ended with none of them able to run. This is the only bound that survives that, so it
    is checked back rather than assumed — the same lesson destroy() learned the expensive way."""
    from eval.orchestrator import ShadeformProvider

    sent = {}

    class _P(ShadeformProvider):
        def __init__(self):
            super().__init__(key_file="/dev/null")

        def _key(self):
            return "x"

        def _api(self, method, path, body=None):
            if path.startswith("/instances/types"):
                return {"instance_types": [{"cloud": "c", "shade_instance_type": "H100",
                                            "hourly_price": 330,
                                            "availability": [{"region": "r", "available": True}]}]}
            if path == "/instances/create":
                sent.update(body)
                return {"id": "i-new"}
            return {"auto_delete": sent.get("auto_delete")}

    inst = _P().rent(GpuSpec(), "ralph-round-8-12345")
    ad = sent.get("auto_delete")
    assert ad and ad.get("date_threshold") and ad.get("spend_threshold"), sent
    assert ad["date_threshold"].endswith("Z"), ad["date_threshold"]
    assert float(ad["spend_threshold"]) >= 3.30 * GpuSpec().max_hours, ad["spend_threshold"]
    assert "ralph-v2" in sent.get("tags", []), sent.get("tags")
    assert inst.price_per_hour == 3.30


def test_a_region_that_never_delivers_is_retried_elsewhere_and_never_leaked():
    """Attempt 9 sat in `pending_provider` for the full 900 s at scaleway/warsaw and was charged
    $0.84 for hardware that never booted — and the next attempt would have picked the identical
    cheapest candidate. Retrying is safe here in a way substitution never is: the GPU MODEL is
    unchanged and the round has not started. What must NOT happen is a retry loop that leaves a
    billing box behind on every lap."""
    import io as _io

    from eval.orchestrator import Instance, RemoteRoundError, RoundPlan, run_remote_round

    seen, destroyed = [], []

    class _Flaky:
        def rent(self, spec, name, exclude=()):
            seen.append(tuple(exclude))
            n = len(seen)
            return Instance(id=f"i-{n}", ip="10.0.0.1", price_per_hour=3.30,
                            cloud="scaleway", region=f"region-{n}")

        def wait_ready(self, inst, timeout_s=900, out=None):
            raise RuntimeError(f"instance {inst.id} not ready within {timeout_s:.0f}s")

        def destroy(self, inst):
            destroyed.append(inst.id)

        def list_all(self):
            return []

        def sweep(self, **kw):
            return []

    with tempfile.TemporaryDirectory() as d:
        try:
            run_remote_round(RoundPlan(round=1, commit_root="a", round_nonce="b", prev_anchor=""),
                             _Flaky(), GpuSpec(), os.path.join(d, "w"), out=_io.StringIO())
            raise AssertionError("a round that never got a box must not return quietly")
        except RemoteRoundError as e:
            assert "became usable" in str(e), e

    assert len(seen) == 3, f"expected three regions tried, got {seen}"
    assert seen[1] == (("scaleway", "region-1"),), seen[1]
    assert seen[2] == (("scaleway", "region-1"), ("scaleway", "region-2")), seen[2]
    assert destroyed == ["i-1", "i-2", "i-3"], \
        f"a retry loop that leaks a billing box per lap is worse than the bug: {destroyed}"


def test_a_fetched_file_is_a_real_file_not_a_dangling_pointer():
    """THE HF CACHE HANDS BACK A SYMLINK. `hf_hub_download` returns a path under `snapshots/<rev>/`
    which is a RELATIVE symlink (`../../blobs/<sha>`). Hardlinking to save disk linked the POINTER
    rather than the blob — `os.link` does not follow symlinks despite the documented default — so
    every file "arrived" and every read of one raised FileNotFoundError. A real round caught this;
    nothing in the suite could, because the download path had no test at all.

    The destination is deliberately at a different depth from the cache, which is what makes the
    relative target dangle. A test that puts them at the same depth passes on a broken build."""
    import eval.fetch as F

    with tempfile.TemporaryDirectory() as d:
        blobs = os.path.join(d, "cache", "blobs")
        snap = os.path.join(d, "cache", "models--x--y", "snapshots", "rev")
        os.makedirs(blobs)
        os.makedirs(snap)
        with open(os.path.join(blobs, "sha"), "w") as fh:
            fh.write('{"model_type": "qwen3"}')
        os.symlink(os.path.join("..", "..", "..", "blobs", "sha"),
                   os.path.join(snap, "config.json"))

        dest_root = os.path.join(d, "out", "artifacts")
        os.makedirs(dest_root)
        out = os.path.join(dest_root, "config.json")

        # THE REAL FUNCTION, with only the network call stubbed. Re-implementing its body in the
        # test is how the last three bugs stayed hidden behind a green suite.
        import sys as _sys
        import types as _types
        fake = _types.ModuleType("huggingface_hub")
        fake.hf_hub_download = lambda repo, name, revision=None: os.path.join(snap, "config.json")
        prev = _sys.modules.get("huggingface_hub")
        _sys.modules["huggingface_hub"] = fake
        try:
            F._hf_download("x/y", "rev", "config.json", out)
        finally:
            if prev is None:
                _sys.modules.pop("huggingface_hub", None)
            else:
                _sys.modules["huggingface_hub"] = prev

        assert not os.path.islink(out), "the destination is a pointer, not a file"
        with open(out) as fh:
            assert "qwen3" in fh.read()
        assert os.stat(out).st_ino == os.stat(os.path.join(blobs, "sha")).st_ino, \
            "hardlink expected: one inode, one copy on disk"


def test_a_record_naming_no_gpu_is_rejected_even_with_no_pin():
    """THE HOLE THIS CLOSES. `_gpu_name()` returns "" when torch cannot see a GPU, and the
    `require_gpu` pin is skipped on the FIRST round by design — the round whose job is to learn the
    device name. Together those said: on the one round that establishes the hardware, any hardware
    will do. A cu130 torch on a CUDA 12.8 driver makes is_available() False, device_map="auto"
    falls back to CPU, and CPU inference is PERFECTLY DETERMINISTIC — so the identity canary passes
    and nothing downstream ever objects. Observed live on 2026-08-06."""
    from eval.orchestrator import GpuSpec, verify_returned_record

    class _Rec:
        manifest = {"versions": {"gpu": ""}, "identity": {"score": 1.0}}
        signature = ""

        def __getattr__(self, k):
            return ""

    bad = verify_returned_record(_Rec(), _Plan(), None, GpuSpec(require_gpu=""), {"gpu": ""})
    assert any("scored on CPU" in b for b in bad), bad


class _Plan:
    round = ""
    commit_root = ""
    round_nonce = ""
    prev_anchor = ""


def test_the_torch_wheel_never_outruns_the_driver():
    """A driver reporting CUDA X.Y runs any build up to X.Y and none above it. Picking the newest
    wheel regardless is what put an 8B parent on 12 CPU cores for 23 minutes."""
    from eval.orchestrator import _torch_index

    assert _torch_index("12.8").endswith("cu128")
    assert _torch_index("13.0").endswith("cu130")
    assert _torch_index("12.2").endswith("cu121"), "picked a wheel newer than the driver"
    # An unknown driver must NOT be guessed at — score_job refuses in seconds instead.
    assert _torch_index("") == "" and _torch_index("nonsense") == ""


def test_a_rental_deleted_under_a_live_round_is_reported_at_once():
    """EVERY OTHER RUNG ASKS "does something exist that should not". This asks the opposite.

    On 2026-08-06 a console-side bulk delete took a round's H100 out from under it mid-canary,
    along with a miner's box. The round did not notice for minutes — it only found out when ssh
    failed — and the watchdog said nothing, because a vanished instance matches no prefix and
    appears in no list. The gap was VISIBILITY, not remedy.

    Alarm only, and that is deliberate: there is no rental left to destroy, the round's own silence
    budget ends it, and acting here could only ever be a false kill on a transient list blip."""
    log = _live_claim()
    r = [_inst("ralph-round-1-1", "i-live", NOW - 600)]

    ok = classify(NOW, _unit(), [_owner()], log, r, _cfg(), {})
    assert "VANISHED" not in ok.alarms, "fired while the rental was present"

    gone = classify(NOW, _unit(), [_owner()], log, [], _cfg(), {})
    assert "VANISHED" in gone.alarms, gone.alarms
    assert not gone.acting and not gone.destroy, "there is nothing left to destroy"
    assert any("no longer exists" in x for x in gone.report), gone.report

    # "could not ask" is not "gone" — the same distinction the rest of the file insists on
    blind = classify(NOW, _unit(), [_owner()], log, None, _cfg(), {})
    assert "VANISHED" not in blind.alarms, "an unreachable provider was read as a deletion"


def _live_claim():
    return LogView(mtime=NOW - 30, milestone="scoring", banner_pid=4242, trusted=True,
                   claims_live=frozenset({"i-live"}))


def test_the_run_banner_is_one_string_in_both_modules():
    """The watchdog's attribution collapses to the bannerless state if these ever diverge, and the
    bannerless state is the one where nothing in the log can be attributed to anyone."""
    from eval.run_orchestrated import RUN_BANNER as WRITTEN
    assert WRITTEN == RUN_BANNER


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
