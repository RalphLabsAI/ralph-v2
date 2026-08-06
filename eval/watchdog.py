"""Is the round making PROGRESS, or only existing?

THE QUESTION THIS FILE EXISTS TO ASK CORRECTLY. During round 1 attempt 7 the operator checked
`pgrep`, saw a pid, and concluded the round was fine. The pid was there for all 180 minutes and the
round had been dead for 175 of them. Existence is not liveness, and a hung process is
indistinguishable from a working one if the only thing you test is that it is still there. What
that cost: systemd eventually SIGTERM'd the orchestrator at its three-hour start timeout, Python's
default SIGTERM disposition killed the interpreter without unwinding, the `finally` that destroys
the rental never ran, and an H100 billed unattended at $3.30/hr until a human noticed 226 minutes
in. ~$12.50 for a round that produced nothing.

THE PROVIDER IS GROUND TRUTH; THE LOG IS AN ACCELERATOR. Those are not interchangeable and the
ordering is the design. An instance billing on somebody's account is a fact about money that
survives this box being unable to execute code at all — which is the state the incident actually
ENDED in, with 52 minutes of billing and no process left to signal. Log mtime, by contrast, is a
proxy: it can only be read when the box is healthy, and it is only meaningful about a round that is
currently holding a rental. So the rungs are ordered by how certain they are:

    OVERDUE / ORPHAN   a ralph-round-* instance nothing here can account for   provider only
    HUNG               THIS run's rental is up and its log stopped moving      provider + log + unit
    alarm-only         money or a wedge we can see but must not touch          varies

WHY THE LOG RUNG IS GATED ON A LIVE RENTAL. It is the only thing that makes the thresholds
defensible. Outside `[rented … destroyed]` the orchestrator does genuinely unbounded work — a
lineage replay that force-downloads one record per published round and grows forever, a
`read_commitments` fallback that walks 256 uids, a publish tail of ~18 HF calls — and none of it
costs a cent per hour. Applying a silence budget there means inventing a number that must be
re-tuned for the rest of the subnet's life, and being wrong kills a healthy round. Applying no
number at all is not a gap: those windows are free, and the ceiling still catches them.

WHAT IT WILL NEVER DO. It never calls `provider.sweep()` — that is prefix-scoped but this process
knows which rentals the running round has CLAIMED, and destroying by claim is the whole point. It
never SIGKILLs: SIGKILL is the one signal `orchestrator._teardown_on_signal` cannot catch, so using
it would recreate the exact bug this file exists to close. And it never destroys an instance whose
name it did not mint — the account is shared with a live miner box and at least one other project.
Kill narrow, report wide.

    python -m eval.watchdog status            # what a human runs INSTEAD of pgrep. Always prints.
    python -m eval.watchdog once              # one pass; this is what the timer runs
    python -m eval.watchdog once --dry-run    # classify against the live API, touch nothing
    python -m eval.watchdog replay FILE -n N  # re-run classify() on a recorded pass

Exit codes: 0 clear, 1 acted OR would act (a dry run that found something is not "clear"),
2 refused (nothing can be answered), 3 degraded or alarm-without-action.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field

from .orchestrator import (HEARTBEAT_S, INSTALL_SILENCE_S, SILENCE_S, GpuSpec, Instance,
                           ShadeformProvider, instance_age_s)

# ---------------------------------------------------------------------------------------------
# Constants. Each is justified where it is used; the ladder between them is asserted by a test.

POLL_S = 300.0            # $0.28 of exposure per poll at $3.30/hr, and < SILENCE_S so this can
                          # actually backstop the in-process killer rather than trail it
CONFIRM_S = 300.0         # one stat() is one syscall's opinion; two observations across a poll
                          # buys immunity to a clock step, an NTP correction, a filesystem that
                          # lies once — for the price of one more poll
GRACE_S = 180.0           # API list lag plus the microseconds between rent() returning and the
                          # claim reaching disk
TEARDOWN_S = 480.0        # > destroy()'s worst case: 3 x (60s POST + 3s + 60s GET) = 380s
STALE_S = 2400.0          # the fallback budget for a milestone the grammar does not recognise
RENTAL_CEILING_S = 29700.0 # GpuSpec.max_hours*3600 = 16200, + 900 slack, still under the
                           # unit's TimeoutStartSec. Raised with max_hours: a watchdog ceiling
                           # below the rental's would destroy every healthy long round.
RUN_CEILING_S = 29400.0    # report-only: a wedged head or tail, which costs $0/hr
STRAY_ALARM_S = 4 * 3600.0
DEADLINE_S = 900.0        # this process's own budget. A watchdog that can hang is the joke that
                          # writes itself.

MINT_PREFIX = "ralph-round-"

# NEVER DESTROYED, NEVER ALARMED. `ralph-v2-miner-m19` is a live miner's box and `epago-4090` is a
# different project; both sit on this same account. This list is the reason `sweep()` is scoped to
# the names we mint rather than to `ralph-`, which matches the miner box.
#
# An entry ending in `*` is a PREFIX. `m4-h6-*` needs that: those boxes are named with a timestamp
# (`m4-h6-quant-20260806-131754`), so an exact-name list protects the three that exist today and
# none of the ones minted tomorrow — and the failure mode of a stale entry here is destroying
# somebody else's H200. Not ours, explicitly hands-off, and nothing in this repo mints that name.
PROTECTED = ("epago-4090", "ralph-v2-miner-m19", "m4-h6-*")


def is_protected(name: str, protected) -> bool:
    """Exact match, or prefix match for entries ending in `*`.

    Every caller went through `name in cfg.protected` before this existed, which is why the whole
    list had to be exact names."""
    name = str(name or "")
    for p in protected:
        if p.endswith("*"):
            if name.startswith(p[:-1]):
                return True
        elif name == p:
            return True
    return False

RUN_BANNER = "=== ralph round start "

# One latch per rung. They must NOT share: a rung that fails its first observation would otherwise
# overwrite the neighbouring rung's latch and neither could ever reach two in a row.
_LATCHES = ("hung", "orphan", "overdue")

# THE GRAMMAR MAY ONLY MAKE THIS FASTER, NEVER MORE AGGRESSIVE. These needles are ordinary prose in
# another module and they WILL be reworded; when they are, the last line matches nothing,
# `budget_for` returns STALE_S, and the watchdog becomes SLOWER. That asymmetry is deliberate and
# a test asserts it: a rename must never be able to shorten a budget.
#
# Only milestones INSIDE the rental window appear here, because the rung is gated on a claimed live
# rental. The head (chain reads, lineage replay) and the tail (publish, anchor, weights) have no
# entries and need none — they are free, and RENTAL_CEILING_S still bounds anything holding a GPU.
GRAMMAR = (
    ("renting", 1200.0,
     "rent() is two API calls at urlopen(timeout=60) each; ~2s real"),
    ("rented", 1500.0,
     "wait_ready's hard cap is 900s and it now writes a line per 15s poll. The most expensive "
     "place in the round to be wrong: the meter has started and nothing has been produced yet"),
    ("ready at", 1200.0,
     "between wait_ready returning and the first push line; ~0s real"),
    ("pushing repo", 2400.0,
     "ssh mkdir + rsync + scp. ~60s real, but structurally large; each is now clamped to what is "
     "left of kill_at, and this keeps the conservative fallback anyway"),
    ("installing", INSTALL_SILENCE_S + 300.0,
     "the orchestrator hands this leg INSTALL_SILENCE_S on purpose because pip reports per "
     "package, not per megabyte. Anything under that would have the watchdog and the code it "
     "watches holding contradictory definitions of stuck, and the watchdog would win"),
    ("scoring", SILENCE_S + 300.0,
     "must be strictly greater than SILENCE_S so _stream_until_silent always wins the race: when "
     "it fires it destroys the rental itself and writes the cost line. This is the backstop for a "
     "WEDGED detector, not a second detector"),
    ("pulling results", 1800.0,
     "three scp then a CPU-only L0+L1 audit over a 900-line pool"),
    ("audited", 900.0,
     "only destroy()'s retry loop is left, and that is 380s worst case"),
    ("REJECT", 900.0, "same as audited"),
)

_OWNER_RE = re.compile(r"eval[./](run_orchestrated|run_round)\b")

# `[progress +   292s] generate Qwen/Qwen3-8B 8/72 @256 tok` — emitted by eval/progress.py on the
# RENTED box and streamed into this log. The elapsed figure is the remote's own clock, so it is
# carried as a number the remote reported, never as one this box can extrapolate from.
_PROGRESS_RE = re.compile(r"\[progress \+\s*(\d+)s\]\s+(\S+)\s*(.*)$")


# ---------------------------------------------------------------------------------------------
# Frozen observations. `classify` is pure over these, which is what lets a forty-minute budget be
# tested in a millisecond and lets a recorded pass be replayed exactly.

@dataclass
class UnitView:
    load_state: str = ""
    active_state: str = ""
    sub_state: str = ""
    pid: int = 0
    start_t: float = 0.0
    invocation: str = ""
    exec_start: str = ""
    pid_alive: bool = False

    @property
    def running(self) -> bool:
        """A oneshot with RemainAfterExit=no is `activating` while it runs and NEVER `active`, so
        `systemctl is-active` reads a healthy round as dead. `LoadState` is checked because a
        typo'd unit name returns rc=0 with `not-found` — a watchdog that has never worked and has
        never said so."""
        return (self.load_state == "loaded" and self.active_state == "activating"
                and self.pid > 0 and self.pid_alive)


@dataclass
class LogView:
    """THREE CLAIM BUCKETS, NOT TWO, and the third one is the whole correctness story.

    The first version had `claims_live` and `claims_dead` and decided between them with
    `fresh = (not unit.running) or (banner_pid == unit.pid)`. That is wrong in both directions at
    once, which is how it passed 27 tests:

      * A run SIGKILLed while holding a rental leaves `unit.running` False, so its slice was filed
        as `claims_live` — which put the leaked instance in `mine`, and `mine` is excluded from
        every bucket ORPHAN draws from. The rung written for the SIGKILL / OOM / reboot class could
        not fire for it. The leak survived to the 2.75 h ceiling: ~$9 of the original $12.50.
      * And the opposite: `claims_dead` was destroyed with no check that any orchestrator was
        alive, so a log the reader could not attribute — no banner, or a banner pushed out of the
        read window — put a LIVE round's H100 in the destroy set.

    So attribution is now explicit about what it does not know:

        claims_live   this run wrote it and this run is alive     never destroyed
        claims_dead   an EARLIER banner wrote it                   destroyed on sight
        claims_stale  written by a run that is gone, or by a slice
                      we cannot attribute at all                   destroyed only if no
                                                                   orchestrator process is alive
    """
    mtime: float = 0.0
    inode: int = 0
    banner_pid: int = 0
    milestone: str = ""
    line: str = ""
    claims_live: frozenset = frozenset()
    claims_dead: frozenset = frozenset()
    claims_stale: frozenset = frozenset()
    # The remote scorer's own heartbeat lines from THIS run's slice, newest last:
    # (elapsed_s, stage, detail). This is the only view of what a round is actually DOING —
    # the milestone says "scoring", these say which of eight artifacts is being fetched.
    progress: tuple = ()
    trusted: bool = False
    note: str = ""


@dataclass
class Target:
    id: str = ""
    name: str = ""
    age: float = 0.0
    rate: float = 0.0

    @property
    def cost(self) -> float:
        return self.rate * self.age / 3600.0 if self.age != float("inf") else float("inf")


@dataclass
class Verdict:
    state: str = "IDLE"
    alarms: tuple = ()
    signal_pid: int = 0
    destroy: tuple = ()
    report: tuple = ()
    silence: float = 0.0
    silence_since_write: float = 0.0
    budget: float = 0.0
    milestone: str = ""
    why: str = ""

    @property
    def acting(self) -> bool:
        return bool(self.signal_pid or self.destroy)

    def to_event(self) -> dict:
        """Built EXPLICITLY, not from __dict__ — `acting` is a property and would be silently
        absent from every forensic record."""
        return {"state": self.state, "alarms": list(self.alarms), "signal_pid": self.signal_pid,
                "destroy": [vars(d) for d in self.destroy], "report": list(self.report),
                "silence": round(self.silence, 1),
                "silence_since_write": round(self.silence_since_write, 1),
                "budget": self.budget, "milestone": self.milestone, "why": self.why,
                "acting": self.acting}


@dataclass
class WatchdogConfig:
    log_path: str = "/var/log/ralph-validator.log"
    unit: str = "ralph-validator.service"
    work_dir: str = "/workspace/ralph-v2-work"
    protected: tuple = PROTECTED
    act: bool = True          # the rung that cannot false-positive
    hung_act: bool = False    # the rung that can — observed before it is armed
    dry_run: bool = False
    confirm_s: float = CONFIRM_S
    grace_s: float = GRACE_S
    stale_s: float = STALE_S
    rental_ceiling_s: float = RENTAL_CEILING_S
    run_ceiling_s: float = RUN_CEILING_S
    stray_alarm_s: float = STRAY_ALARM_S

    @classmethod
    def from_env(cls) -> "WatchdogConfig":
        e = os.environ.get
        extra = tuple(x.strip() for x in e("RALPH_WATCHDOG_IGNORE", "").split(",") if x.strip())
        return cls(
            log_path=e("RALPH_WATCHDOG_LOG", "/var/log/ralph-validator.log"),
            unit=e("RALPH_WATCHDOG_UNIT", "ralph-validator.service"),
            work_dir=e("RALPH_WORK_DIR", "/workspace/ralph-v2-work"),
            protected=PROTECTED + extra,
            act=e("RALPH_WATCHDOG_ACT", "1") == "1",
            hung_act=e("RALPH_WATCHDOG_HUNG_ACT", "0") == "1")


# ---------------------------------------------------------------------------------------------
# Reading the world.

def read_unit(unit: str) -> UnitView:
    props = ("LoadState", "ActiveState", "SubState", "ExecMainPID", "ExecMainStartTimestamp",
             "InvocationID", "ExecStart")
    try:
        r = subprocess.run(["systemctl", "show", unit, "--timestamp=unix",
                            *[f"-p{p}" for p in props]],
                           capture_output=True, text=True, timeout=30)
        kv = dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l)
    except Exception:
        kv = {}
    pid = int(kv.get("ExecMainPID", "0") or 0)
    return UnitView(
        load_state=kv.get("LoadState", ""), active_state=kv.get("ActiveState", ""),
        sub_state=kv.get("SubState", ""), pid=pid,
        start_t=float((kv.get("ExecMainStartTimestamp", "") or "@0").lstrip("@").split()[0] or 0),
        invocation=kv.get("InvocationID", ""), exec_start=kv.get("ExecStart", ""),
        pid_alive=bool(pid) and os.path.exists(f"/proc/{pid}"))


def read_owners() -> list:
    """Every orchestrator process on this box, by scanning the WHOLE joined cmdline.

    Matching argv[0] would miss `python -m eval.run_orchestrated` entirely, which is the form the
    unit uses. `--dry-run` is excluded because a backgrounded dry-run rents nothing and must not
    veto orphan cleanup for as long as it happens to live."""
    out = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as fh:
                cmd = fh.read().decode("utf-8", "replace").replace("\0", " ").strip()
        except Exception:
            continue
        if not cmd or "eval.watchdog" in cmd or "eval/watchdog" in cmd or "--dry-run" in cmd:
            continue
        if _OWNER_RE.search(cmd):
            # The start time is READ HERE, not in classify(): classify is pure over frozen
            # observations, and a /proc stat inside it meant replay() silently dropped every
            # ZOMBIE_OWNER finding because a recorded pass has no /proc to consult.
            try:
                started = os.path.getmtime(f"/proc/{entry}")
            except Exception:
                started = 0.0
            out.append((int(entry), cmd, started))
    return out


def _claims(text: str) -> frozenset:
    """Rentals this stretch of log claims and does not release.

    `renting <name>` and `rented <uuid>` claim; `destroyed <uuid>` and `sweeping orphan <name>`
    release. The name-claim matters because `rent()` learns the id only from the create response,
    so a create that succeeds server-side and times out client-side leaves a correctly-named
    instance whose id was never returned to anybody."""
    held: set = set()
    pending_name = ""
    # SPLIT ON \n ONLY. `str.splitlines()` also breaks on \r, and `_stream_until_silent`
    # deliberately forwards \r-separated progress bars — so a bar fragment could be mistaken for a
    # line of the log's own grammar.
    for raw in text.split("\n"):
        p = raw.split()
        if len(p) >= 2 and p[0] == "renting":
            pending_name = p[1].rstrip("…").rstrip(".")
            held.add(pending_name)
        elif len(p) >= 2 and p[0] == "rented" and len(p[1]) > 30:
            # THE NAME CLAIM IS SUPERSEDED BY THE ID. It exists only to cover a create whose
            # response was lost; once the id is known it is the claim, and leaving the name behind
            # meant a round's claim was NEVER released — `destroyed <uuid>` releases the uuid, and
            # the orphaned name kept the log rung armed over the free publish tail for the rest of
            # the run.
            held.discard(pending_name)
            pending_name = ""
            held.add(p[1])
        elif len(p) >= 2 and p[0] == "destroyed" and len(p[1]) > 30:
            held.discard(p[1])
        elif len(p) >= 3 and p[0] == "sweeping" and p[1] == "orphan":
            held.discard(p[2])
    return frozenset(held)


def read_log(path: str, unit: UnitView) -> LogView:
    """Everything the log can say about the CURRENT invocation, and nothing about any other.

    The file is opened `append:` and never truncated, so position proves nothing on its own: six
    runs share it today with no separator. `run_orchestrated` now writes an unindented banner as
    its first act, and everything below is scoped to the slice after the last one. If the unit is
    running and the last banner's pid is not this invocation's main pid, then the current run has
    not written its banner yet and owns NO line in this file whatever the tail says — that is the
    window in which a naive tail inherits the previous round's `scoring…`, the tightest budget
    there is, and kills a round that is three seconds old."""
    try:
        st = os.stat(path)
    except OSError as e:
        return LogView(trusted=False, note=f"cannot stat {path}: {e}")
    # A GENEROUS WINDOW, because losing the banner is not a small error — it is the state in which
    # nothing in the file can be attributed to anyone. A single round's streamed output can be tens
    # of megabytes once \r-separated progress bars are forwarded.
    try:
        with open(path, "rb") as fh:
            fh.seek(max(0, st.st_size - 64 * 1024 * 1024))
            blob = fh.read().decode("utf-8", "replace")
    except OSError as e:
        return LogView(mtime=st.st_mtime, inode=st.st_ino, trusted=False,
                       note=f"cannot read {path}: {e}")

    cut = blob.rfind(RUN_BANNER)
    if cut < 0:
        # UNATTRIBUTABLE, NOT DEAD. Claims found here go to the guarded bucket: this is the state
        # a fresh deploy is in (no round has written a banner yet) and also the state a rotated or
        # overflowed log is in, and in the latter a LIVE round's rental is sitting in this blob.
        # Destroying on that evidence is how a monitor takes down the thing it was watching.
        return LogView(mtime=st.st_mtime, inode=st.st_ino, claims_stale=_claims(blob),
                       trusted=False,
                       note="no run banner in the log — nothing here can be attributed to a run, "
                            "so the log rung is off and claims need a dead box to be actioned")
    head, cur = blob[:cut], blob[cut:]
    m = re.search(r"pid=(\d+)", cur.split("\n", 1)[0])
    bpid = int(m.group(1)) if m else 0
    # THIS run wrote the newest banner AND is still alive. Anything less is not `live`: a slice
    # whose author is gone is exactly the leak case, and calling it live was what made ORPHAN
    # unreachable for every way a process can die without unwinding.
    fresh = unit.running and bpid == unit.pid

    live = _claims(cur) if fresh else frozenset()
    # An EARLIER banner is by definition an earlier run: unconditionally actionable. The newest
    # slice, when its author is gone, is only PROBABLY dead — a round started by hand outside the
    # unit looks identical from here — so it waits on there being no orchestrator process at all.
    dead = _claims(head)
    stale = frozenset() if fresh else _claims(cur)

    prog: list = []
    if fresh:
        for raw in cur.split("\n"):
            mm = _PROGRESS_RE.search(raw)
            if mm:
                prog.append((float(mm.group(1)), mm.group(2).strip(), mm.group(3).strip()))

    milestone, line = "", ""
    if fresh:
        # EXACTLY two spaces. Streamed remote output is four-space ("    | ", "    · ") and is
        # liveness — it moves the mtime — but never changes which milestone is active. The whole
        # slice is scanned rather than a fixed tail because a torch install writes far more than
        # any window would hold and would evict `  installing…`, silently demoting the budget.
        for raw in cur.splitlines():
            if raw[:2] == "  " and raw[2:3] != " ":
                line = raw.strip()
        for needle, _b, _why in GRAMMAR:
            if line.startswith(needle):
                milestone = needle
                break
    return LogView(mtime=st.st_mtime, inode=st.st_ino, banner_pid=bpid, milestone=milestone,
                   line=line, claims_live=live, claims_dead=dead, claims_stale=stale,
                   progress=tuple(prog[-400:]), trusted=True)


def read_rentals(provider):
    """The raw list, or None if the provider could not be reached. None is NOT an empty list —
    "nothing is billing" and "I could not ask" are different answers and conflating them is how a
    monitor reports all-clear through an outage."""
    try:
        return provider.list_all()
    except Exception:
        return None


def budget_for(milestone: str, stale_s: float) -> tuple:
    for needle, b, why in GRAMMAR:
        if needle == milestone:
            return b, why
    return stale_s, "unrecognised milestone — the conservative fallback, deliberately the slowest"


# ---------------------------------------------------------------------------------------------

def classify(now: float, unit: UnitView, owners: list, log: LogView, rentals, cfg: WatchdogConfig,
             latches: dict) -> Verdict:
    """PURE. No network, no clock, no /proc, no filesystem — everything it reasons about was frozen
    by the readers above. That is what makes a 2.75-hour ceiling testable in a millisecond and what
    makes `replay` able to reproduce a verdict exactly from a recorded line."""
    protected = tuple(cfg.protected)
    alarms: list = []
    report: list = []

    silence_since_write = now - log.mtime if log.mtime else float("inf")
    run_start = unit.start_t if unit.running else 0.0
    # BOTH numbers are carried, and the reason is trust: the decision uses the run-scoped one, but
    # an operator running `stat` sees the other, and a tool whose headline number nobody can
    # reproduce by hand is a tool people stop believing.
    silence = now - max(log.mtime, run_start) if log.mtime else float("inf")
    budget, why = budget_for(log.milestone, cfg.stale_s)

    if rentals is None:
        mint, strays = [], []
    else:
        mint = [r for r in rentals
                if str(r.get("name", "")).startswith(MINT_PREFIX)
                and not is_protected(r.get("name", ""), protected)]
        strays = [r for r in rentals
                  if not str(r.get("name", "")).startswith(MINT_PREFIX)
                  and not is_protected(r.get("name", ""), protected)]

    def _t(r) -> Target:
        return Target(id=str(r.get("id", "")), name=str(r.get("name", "")),
                      age=instance_age_s(r, now),
                      rate=float(r.get("hourly_price", 0) or 0) / 100.0)

    # A rental is OURS only if THIS run's log slice claims it. Keying on the prefix alone was the
    # worst defect in the reviewed designs: any leaked corpse from a previous crash would then
    # re-arm the log rung over the head and the tail, and a stale mtime there would kill a healthy
    # round that had not even rented yet.
    def _in(r, bucket) -> bool:
        return str(r.get("id", "")) in bucket or str(r.get("name", "")) in bucket

    mine = [_t(r) for r in mint if _in(r, log.claims_live)]
    claimed_dead = [_t(r) for r in mint if not _in(r, log.claims_live) and _in(r, log.claims_dead)]
    claimed_stale = [_t(r) for r in mint
                     if not _in(r, log.claims_live) and not _in(r, log.claims_dead)
                     and _in(r, log.claims_stale)]
    unclaimed = [_t(r) for r in mint
                 if not _in(r, log.claims_live) and not _in(r, log.claims_dead)
                 and not _in(r, log.claims_stale)]

    for s in strays:
        t = _t(s)
        if t.age > cfg.stray_alarm_s:
            report.append(f"STRAY {t.name} ({t.id[:12]}) has been up {t.age / 3600:.1f} h at "
                          f"${t.rate:.2f}/hr (~${t.cost:.2f}) — NOT ours to destroy; if it is "
                          f"expected add it to RALPH_WATCHDOG_IGNORE")
            alarms.append("STRAY")

    for pid, cmd, started in owners:
        if pid != unit.pid:
            age = now - started if started else 0.0
            if age > 10800:
                report.append(f"ZOMBIE_OWNER pid {pid} has run {age / 3600:.1f} h outside the "
                              f"unit: {cmd[:80]}")
                alarms.append("ZOMBIE_OWNER")

    def _mk(state, sig=0, destroy=(), extra=""):
        return Verdict(state=state, alarms=tuple(alarms), signal_pid=sig, destroy=tuple(destroy),
                       report=tuple(report), silence=silence,
                       silence_since_write=silence_since_write, budget=budget,
                       milestone=log.milestone, why=extra or why)

    # 1. OVERDUE — past every legitimate ceiling in the system. Needs no log, no unit, no pid, and
    #    is hoisted above the owner split on purpose: in the reviewed designs the absolute ceiling
    #    lived inside `if not owners:` and was therefore unreachable in the one quadrant where a
    #    dollar was actually burning.
    # AN UNKNOWN AGE IS NOT AN OLD AGE WHEN THE RENTAL IS OURS AND LIVE. `instance_age_s` fails to
    # +inf so that a leak is never mistaken for something fresh — right for a rental nobody owns,
    # and wrong here: applied to a rental THIS run is holding, +inf means "destroy the healthy
    # round's GPU because the provider reformatted a timestamp". Report it instead.
    for t in mine:
        if t.age == float("inf"):
            report.append(f"UNDATED {t.name} ({t.id[:12]}) is claimed by this run but the provider "
                          f"gave no parseable created_at — not applying the ceiling to it")
            alarms.append("UNDATED")
    over = [t for t in (claimed_dead + claimed_stale + unclaimed) if t.age > cfg.rental_ceiling_s]
    over += [t for t in mine if cfg.rental_ceiling_s < t.age < float("inf")]
    # A LATCH OF ITS OWN. Sharing "orphan" meant that when this rung failed its first observation
    # the ORPHAN check below immediately overwrote the latch with a different key, so neither rung
    # could ever reach two consecutive observations and the ceiling was unreachable.
    if over and _confirmed(latches, "overdue", [t.id for t in over], now, cfg.confirm_s):
        alarms.append("OVERDUE")
        sig = unit.pid if (unit.running and any(t.id == m.id for t in over for m in mine)) else 0
        return _mk("OVERDUE", sig=sig, destroy=over,
                   extra=f"a rental past {cfg.rental_ceiling_s / 3600:.2f} h is past max_hours, "
                         f"past the unit's start timeout, and past anything legitimate")

    # 2. ORPHAN — the orchestrator is gone and only the provider can stop the meter. This is the
    #    incident's last 52 minutes, and the whole SIGKILL / OOM / panic / reboot class. It is the
    #    only rung whose guarantee survives this box being unable to run code.
    # An EARLIER banner's rental is actionable whatever else is true: that run is over by
    # definition, because a later one announced itself.
    orphans = [t for t in claimed_dead if t.age > cfg.grace_s]
    # The newest slice's rental, and anything nobody claimed, need the box to be genuinely empty of
    # orchestrators — a round started by hand outside the unit is indistinguishable from a corpse
    # by log evidence alone, and killing it is worse than the leak.
    if not owners:
        orphans += [t for t in (claimed_stale + unclaimed) if t.age > cfg.grace_s]
    if orphans and _confirmed(latches, "orphan", [t.id for t in orphans], now, cfg.confirm_s):
        alarms.append("ORPHAN")
        return _mk("ORPHAN", destroy=orphans,
                   extra="claimed by a run that is no longer here — nothing on this box can "
                         "destroy it, so nothing else will")

    # THE RENTAL A LIVE ROUND CLAIMED, WHICH IS NO LONGER THERE. Every other rung asks "does
    # something exist that should not"; this asks the opposite, and nothing did until a bulk console
    # delete took a round's H100 out from under it mid-canary. The round did not notice for minutes
    # — it only found out when ssh failed — and the watchdog said IDLE-adjacent nothing, because an
    # instance that has vanished matches no prefix and appears in no list.
    #
    # ALARM ONLY, DELIBERATELY. The remedy is nothing this process can perform: there is no rental
    # left to destroy, and the round's own silence budget will end it. What was missing was
    # VISIBILITY — the operator and the dashboard learning within one poll instead of twenty
    # minutes. Acting here could only ever be a false kill on a transient list blip.
    if unit.running and log.trusted and log.claims_live and not mine and rentals is not None:
        report.append(f"VANISHED this run claims rental(s) {sorted(log.claims_live)[:2]} but the "
                      f"provider lists none of them — the round is holding a GPU that no longer "
                      f"exists, and will not notice until ssh fails")
        alarms.append("VANISHED")

    if unclaimed and owners:
        for t in unclaimed:
            report.append(f"HELD {t.name} ({t.id[:12]}, {t.age / 60:.0f} min) is unclaimed by this "
                          f"run's log while an orchestrator is alive — not touching it; the "
                          f"{cfg.rental_ceiling_s / 3600:.2f} h ceiling still applies")
        alarms.append("HELD")

    # 3. HUNG — the log rung, and the only one that can be wrong. Gated on a rental THIS run
    #    claimed, so the free head and the free tail are out of jurisdiction by construction
    #    rather than by a threshold somebody has to keep re-tuning.
    if rentals is None:
        if (unit.running and log.trusted and log.claims_live and silence > budget
                and _confirmed(latches, "hung", [unit.invocation, log.milestone,
                                                 str(int(log.mtime))], now, cfg.confirm_s)):
            alarms.append("DEGRADED_HUNG")
            # SIGTERM ANYWAY. Withholding the one free remedy precisely when the money is invisible
            # is attempt 7 with extra steps — and SIGTERM costs nothing, because it hands teardown
            # to the round's own `finally`.
            return _mk("DEGRADED_HUNG", sig=unit.pid,
                       extra="the provider is unreachable, so nothing can be destroyed from here; "
                             "signalling the round so its own teardown runs")
        alarms.append("NO_VERDICT")
        report.append("the provider could not be reached — this is NOT 'nothing is billing', it "
                      "is 'I could not ask'")
        return _mk("NO_VERDICT")

    if unit.running and not log.trusted:
        report.append(f"the log rung is disabled: {log.note or 'log not trusted'} — money rungs "
                      f"still ran")
        alarms.append("LOG_UNTRUSTED")

    if unit.running and mine and log.trusted:
        if log.banner_pid != unit.pid or silence <= 0:
            return _mk("STARTING", extra="this run has not written its banner yet, so no line in "
                                         "the log is its and no budget applies")
        if silence > budget and _confirmed(latches, "hung",
                                           [unit.invocation, log.milestone, str(int(log.mtime))],
                                           now, cfg.confirm_s):
            alarms.append("HUNG")
            return _mk("HUNG", sig=unit.pid, destroy=mine)
        return _mk("RUNNING")

    if unit.running:
        run_age = now - unit.start_t if unit.start_t else 0.0
        if not mine and run_age > cfg.run_ceiling_s:
            report.append(f"STUCK_FREE the run has been up {run_age / 3600:.1f} h with no rental "
                          f"in flight — costs nothing per hour, but it is wedged and it blocks "
                          f"the next round")
            alarms.append("STUCK_FREE")
        return _mk("RUNNING" if mine or log.trusted else "STARTING",
                   extra="no rental is in flight, so the log rung does not apply — the head and "
                         "the tail of a round are free and unbounded by design")

    return _mk("IDLE", extra="no round in flight; a stale log is the CORRECT resting state here")


def _confirmed(latches: dict, kind: str, key_parts, now: float, confirm_s: float) -> bool:
    """Two observations of the SAME situation, at least `confirm_s` apart.

    Keyed on more than mtime: keying on mtime alone is vacuous for every false positive, because a
    false positive is precisely the case where mtime is frozen. Any contradicting observation
    clears the latch, so a round that writes one line resets the clock."""
    # SORTED. The instance list comes back in whatever order the provider chose, so an unsorted
    # join produced a different key on every poll — the latch reset each time and the money rungs
    # could never reach two consecutive observations.
    key = "|".join(sorted(str(p) for p in key_parts))
    st = latches.get(kind) or {}
    if st.get("key") != key:
        latches[kind] = {"key": key, "since": now, "obs": 1}
        return False
    st["obs"] = int(st.get("obs", 0)) + 1
    latches[kind] = st
    return st["obs"] >= 2 and now - float(st.get("since", now)) >= confirm_s


def clear_latch(latches: dict, kind: str) -> None:
    latches.pop(kind, None)


# ---------------------------------------------------------------------------------------------

def _wait_gone(provider, ids: list, budget_s: float, now=time.time) -> bool:
    if not ids:
        return True
    deadline = now() + budget_s
    while now() < deadline:
        try:
            live = {str(i.get("id", "")) for i in provider.list_all()}
        except Exception:
            return False
        if not (set(ids) & live):
            return True
        time.sleep(10)
    return False


def act(v: Verdict, cfg: WatchdogConfig, provider, out=sys.stdout, now=time.time) -> int:
    """Separate from reporting, and it returns immediately on an empty plan — otherwise every idle
    pass runs a destroy-verification over nothing and prints the loudest string in the file 288
    times a day, which is how an alarm becomes something people filter."""
    w = out.write
    if not v.destroy and not v.signal_pid:
        return 0
    if v.state in ("HUNG", "DEGRADED_HUNG") and not cfg.hung_act:
        w(f"  OBSERVE  would act on {v.state} (RALPH_WATCHDOG_HUNG_ACT=0): signal pid "
          f"{v.signal_pid or '-'}, destroy {[d.name for d in v.destroy] or '-'}\n")
        return 3
    if not cfg.act or cfg.dry_run:
        # 1, NOT 0. A pass that found a live leak and declined to touch it has not found nothing,
        # and a monitor whose exit code says "clear" while its alarm file says otherwise is a
        # monitor that gets wired into something that only reads the exit code.
        w(f"  {'DRY RUN' if cfg.dry_run else 'DISARMED'}  would signal pid "
          f"{v.signal_pid or '-'} and destroy {[d.name for d in v.destroy] or '-'}\n")
        return 1

    if v.signal_pid:
        # SIGTERM, NEVER SIGKILL, and never `systemctl stop`.
        #  * SIGKILL is the one signal `_teardown_on_signal` cannot catch; using it here recreates
        #    the bug this file exists to close. v1's vali_restart_audit.sh does `kill -9` — that is
        #    exactly the inheritance to refuse.
        #  * `systemctl stop` inherits DefaultTimeoutStopUSec=90s and would SIGKILL the round 90s
        #    in, mid-teardown, while destroy()'s retry loop needs up to 380s. The stop that exists
        #    to end a leak would cause one. (A drop-in raises TimeoutStopSec anyway, because a
        #    human will type the command.)
        #  * The pid is ExecMainPID and nothing else. Signalling an arbitrary /proc match can hit a
        #    wrapper shell, after which the round keeps running and we yank its GPU out from under
        #    it.
        w(f"  SIGTERM  pid {v.signal_pid} — the orchestrator turns this into a RemoteRoundError so "
          f"its own `finally` destroys the rental. Waiting up to {TEARDOWN_S:.0f}s.\n")
        try:
            os.kill(v.signal_pid, signal.SIGTERM)
        except Exception as e:
            w(f"  ERROR    could not signal {v.signal_pid}: {type(e).__name__}: {e}\n")
        # ONLY IF THERE WAS SOMETHING TO VERIFY. `_wait_gone([])` is vacuously true, so on
        # DEGRADED_HUNG — where the destroy list is empty BY CONSTRUCTION because the provider is
        # unreachable — this printed "the round destroyed its own rental" having checked nothing,
        # on the one path where we know we cannot see the money.
        if v.destroy and _wait_gone(provider, [d.id for d in v.destroy], TEARDOWN_S, now):
            w("  clean    the round destroyed its own rental. Nothing left to do.\n")
            return 1
        if not v.destroy:
            w("  signalled — the provider is unreachable from here, so whether the rental "
              "actually went cannot be confirmed. Check the console.\n")
            return 3

    for d in v.destroy:
        # BY ID, NEVER provider.sweep(). Between classification and here a new round may have
        # legitimately started, and sweep() would take its GPU. The prefix and protected-name
        # guard is re-checked immediately before the call rather than inherited, because this
        # process holds a key that can create and delete.
        if not d.name.startswith(MINT_PREFIX) or is_protected(d.name, cfg.protected):
            w(f"  REFUSED  {d.name} is not a name this system mints — not destroying it\n")
            continue
        w(f"  DESTROY  {d.name} ({d.id[:12]}) up {d.age / 60:.0f} min, ~${d.cost:.2f} spent\n")
        try:
            provider.destroy(Instance(id=d.id))
        except Exception as e:
            w(f"  ERROR    destroy {d.id[:12]}: {type(e).__name__}: {e}\n")
    if not _wait_gone(provider, [d.id for d in v.destroy], 90.0, now):
        w("  ALARM    a rental we tried to destroy is STILL LISTED (raw list, not list_active) — "
          "it is billing right now. Kill it in the Shadeform console.\n")
    return 1


# ---------------------------------------------------------------------------------------------

def _state_path(cfg) -> str:
    return os.path.join(cfg.work_dir, "watchdog-state.json")


def _load_state(cfg) -> dict:
    try:
        with open(_state_path(cfg)) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_state(cfg, state: dict) -> None:
    """Written on EVERY pass, including passes that errored — that is the whole point of the
    heartbeat `run_orchestrated`'s preflight reads. A watchdog that only records its successes
    looks identical, from outside, to one that has not run since a key rotation."""
    try:
        os.makedirs(cfg.work_dir, exist_ok=True)
        tmp = _state_path(cfg) + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(state, fh, indent=1)
        os.replace(tmp, _state_path(cfg))
    except Exception:
        pass


def _finite(o):
    """inf/nan are not JSON. `json.dumps` emits bare `Infinity`, which nothing can read back — in
    exactly the degraded passes this file exists to preserve."""
    if isinstance(o, float):
        return o if o == o and abs(o) != float("inf") else str(o)
    if isinstance(o, dict):
        return {k: _finite(x) for k, x in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(x) for x in o]
    return o


def _record(cfg, payload: dict) -> None:
    try:
        os.makedirs(cfg.work_dir, exist_ok=True)
        path = os.path.join(cfg.work_dir, "watchdog-events.jsonl")
        if os.path.exists(path) and os.path.getsize(path) > 5 * 1024 ** 2:
            os.replace(path, path + ".1")
        with open(path, "a") as fh:
            fh.write(json.dumps(_finite(payload), allow_nan=False) + "\n")
    except Exception:
        pass


def _alarm_file(cfg, open_: bool) -> None:
    """A file an operator can `stat`, which is the reflex the handoff already trains."""
    path = os.path.join(cfg.work_dir, "watchdog-alarm")
    try:
        if open_:
            os.makedirs(cfg.work_dir, exist_ok=True)
            with open(path, "w") as fh:
                fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))
        elif os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def observe(cfg: WatchdogConfig, provider, now: float):
    unit = read_unit(cfg.unit)
    owners = read_owners()
    log = read_log(cfg.log_path, unit)
    # ROTATION AND REDIRECTION DISABLE ONLY THE LOG RUNG. If the unit's stdout is not the file we
    # are watching, this file cannot say anything about that round — but the money rungs never
    # needed it, and the reviewed designs switched the leak detector off over exactly this.
    if unit.running and log.trusted:
        try:
            if os.stat(f"/proc/{unit.pid}/fd/1").st_ino != log.inode:
                log.trusted = False
                log.note = ("the running round's stdout is not this file (inode mismatch — "
                            "rotated, or started by hand with a redirect)")
        except Exception:
            pass
    return unit, owners, log, read_rentals(provider)


def once(cfg: WatchdogConfig, provider, out=sys.stdout, now=None) -> int:
    # THE WATCHDOG'S OWN BUDGET, actually enforced. A watchdog that can itself hang is the joke
    # that writes itself, and the unit's TimeoutStartSec would SIGKILL this process mid-destroy —
    # abandoning the very instance it was rescuing, which is the incident one level up.
    started = time.monotonic()
    signal.signal(signal.SIGALRM, _too_slow) if hasattr(signal, "SIGALRM") else None
    if hasattr(signal, "alarm"):
        signal.alarm(int(DEADLINE_S))
    try:
        return _once(cfg, provider, out, now, started)
    finally:
        if hasattr(signal, "alarm"):
            signal.alarm(0)


def _too_slow(_sig, _frame):
    raise TimeoutError(f"the watchdog exceeded its own {DEADLINE_S:.0f}s budget")


def _once(cfg: WatchdogConfig, provider, out, now, started) -> int:
    now = now if now is not None else time.time()
    state = _load_state(cfg)
    latches = {k: state.get(k) for k in _LATCHES if state.get(k)}
    import copy
    latches_in = copy.deepcopy(latches)     # the INPUT state, before classify advances it
    unit, owners, log, rentals = observe(cfg, provider, now)
    v = classify(now, unit, owners, log, rentals, cfg, latches)

    rc = 0
    if v.acting:
        rc = act(v, cfg, provider, out=out, now=lambda: time.time())
    elif v.alarms:
        rc = 3
    for line in v.report:
        out.write(f"  {line}\n")
    if not v.report and not v.acting:
        # ONE LINE, EVERY PASS. journalctl timestamps it, so this is also how an operator confirms
        # the watchdog is alive without running the thing this file exists to replace.
        out.write(f"  {v.state}: {v.why}\n")
    if v.alarms or v.acting:
        _alarm_file(cfg, True)
    else:
        _alarm_file(cfg, False)

    state.update({"last_pass": now, "last_state": v.state, "last_alarms": list(v.alarms)})
    for kind in _LATCHES:
        state[kind] = latches.get(kind)
    _save_state(cfg, state)
    # THE LATCHES ARE PART OF THE INPUT. classify() is pure over (observations, latches), so a
    # record without them cannot reproduce any verdict that ACTED — every real incident replayed
    # as `match: false`, which is exactly the forensic case the file exists for.
    _record(cfg, {"t": now, "verdict": v.to_event(), "latches": latches_in,
                  "unit": vars(unit), "owners": [[p, c, st] for p, c, st in owners],
                  "log": {k: (sorted(x) if isinstance(x, frozenset) else x)
                          for k, x in vars(log).items()},
                  "rentals": rentals})
    return rc


def status(cfg: WatchdogConfig, provider, out=sys.stdout, now=None) -> int:
    """What a human runs INSTEAD of pgrep, and it is never silent — a tool that prints nothing when
    things are fine teaches people to run the thing that always prints, which was pgrep."""
    now = now if now is not None else time.time()
    w = out.write
    unit, owners, log, rentals = observe(cfg, provider, now)
    # THE SAME LATCHES `once` WOULD SEE, and only those — handing classify() the whole state file
    # lets unrelated keys ride along, and status must answer the question `once` would answer.
    _st = _load_state(cfg)
    v = classify(now, unit, owners, log, rentals, cfg,
                 {k: _st.get(k) for k in _LATCHES if _st.get(k)})

    w(f"\n  state     {v.state}\n")
    w(f"  unit      {cfg.unit}: load={unit.load_state} active={unit.active_state}"
      f"/{unit.sub_state} pid={unit.pid or '-'} alive={unit.pid_alive}\n")
    w(f"  owners    {len(owners)}"
      + ("".join(f"\n              pid {p}: {c[:88]}" for p, c, _s in owners) if owners else "")
      + "\n")
    if log.mtime:
        w(f"  log       {cfg.log_path}\n")
        w(f"            last write {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(log.mtime))}"
          f" ({v.silence_since_write / 60:.1f} min ago by stat)\n")
        w(f"            milestone {v.milestone or '(none matched)'} — "
          f"{log.line[:70] or '(no milestone line in this run)'}\n")
        w(f"            silence used for the decision: {v.silence / 60:.1f} min "
          f"vs budget {v.budget / 60:.1f} min\n")
    if not log.trusted:
        w(f"  LOG RUNG  DISABLED — {log.note}\n")
    if rentals is None:
        w("  rentals   COULD NOT ASK — the provider is unreachable. This is not 'nothing is "
          "billing'.\n")
    else:
        w(f"  rentals   {len(rentals)} instance(s) on the account\n")
        for r in rentals:
            name = str(r.get("name", ""))
            t = Target(id=str(r.get("id", "")), name=name, age=instance_age_s(r, now),
                       rate=float(r.get("hourly_price", 0) or 0) / 100.0)
            if is_protected(name, cfg.protected):
                tag = "protected"
            elif not name.startswith(MINT_PREFIX):
                tag = "stray"
            elif any(d.id == t.id for d in v.destroy):
                tag = "TO DESTROY"
            elif t.id in log.claims_live or name in log.claims_live:
                tag = "RENTAL (this run)"
            else:
                tag = "held"
            w(f"            {name:34} {tag:18} {t.age / 60:7.0f} min  "
              f"${t.rate:.2f}/hr  ~${t.cost:.2f}\n")
    for line in v.report:
        w(f"  ALARM     {line}\n")
    st = _load_state(cfg)
    lp = st.get("last_pass")
    w(f"  last pass {(now - float(lp)) / 60:.1f} min ago\n" if lp else
      "  last pass NEVER — the timer has not run; nothing is watching this box\n")
    w(f"  verdict   {v.why}\n\n")
    return 1 if v.acting else (3 if v.alarms else 0)


def replay(path: str, n: int, cfg: WatchdogConfig, out=sys.stdout) -> int:
    """Re-run `classify()` over a recorded pass. Without this, the rational response to one false
    kill is `systemctl disable`, after which the real hang goes unwatched."""
    lines = [l for l in open(path) if l.strip()]
    rec = json.loads(lines[n if n >= 0 else len(lines) + n])
    log = LogView(**{**rec["log"],
                     "claims_live": frozenset(rec["log"].get("claims_live") or []),
                     "claims_dead": frozenset(rec["log"].get("claims_dead") or []),
                     "claims_stale": frozenset(rec["log"].get("claims_stale") or [])})
    owners = [tuple(o) if isinstance(o, list) else (0, o, 0.0) for o in rec["owners"]]
    v = classify(rec["t"], UnitView(**rec["unit"]), owners, log, rec["rentals"], cfg,
                 dict(rec.get("latches") or {}))
    out.write(json.dumps({"recorded": rec["verdict"]["state"], "replayed": v.state,
                          "match": rec["verdict"]["state"] == v.state}, indent=1) + "\n")
    return 0 if rec["verdict"]["state"] == v.state else 1


def main(argv: list) -> int:
    cfg = WatchdogConfig.from_env()
    cfg.dry_run = "--dry-run" in argv
    cmd = next((a for a in argv if not a.startswith("-")), "status")
    try:
        provider = ShadeformProvider()
        provider._key()
    except Exception as e:
        sys.stdout.write(f"  REFUSING: no provider key ({e}). Nothing can be answered without "
                         f"it — the provider is the only ground truth about money.\n")
        return 2
    if cmd == "status":
        return status(cfg, provider)
    if cmd == "once":
        return once(cfg, provider)
    if cmd == "replay":
        rest = [a for a in argv if not a.startswith("-")][1:]
        n = int(argv[argv.index("-n") + 1]) if "-n" in argv else -1
        return replay(rest[0], n, cfg)
    sys.stdout.write(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
