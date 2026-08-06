"""The CPU orchestrator: rent a GPU, score one round on it, sign and publish here, tear it down.

WHY SPLIT THE VALIDATOR IN TWO. The obvious reason is cost — an idle H100 bills all month to do a
few minutes of work per round. The reason that actually matters is that the June 2026 compromise of
a persistent GPU box gave an attacker root for seven hours, and with root came the signing keys: a
rigged crowning and wiped logs. Renting makes that worse on its face, because it is somebody else's
hardware. So the split is drawn where the keys are:

    CPU orchestrator (ours, long-lived)     rented GPU (ephemeral, keyless)
    -------------------------------------   ---------------------------------
    wallet, record seed, HF write token     a READ token, at most
    reads the chain, draws the nonce        loads parent + observers
    decides the round's identity            fetches miner artifacts (60 GB ceiling)
    AUDITS what comes back                  generates, scores, runs the canary
    signs, publishes, anchors, weights      returns an UNSIGNED record + its pool

THE ORCHESTRATOR AUDITS ITS OWN SCORER. This is the part that keeps the split honest. A signature
applied to whatever the rented box returned would simply launder it — the box could hand back a
record for a different round, a pruned exam, or a crown it invented, and our key would make it
authoritative. So before signing we run the SAME audit an outsider runs (`eval.rerun`, L0 + L1)
against the returned record, and we check that the round's identity fields are the ones WE supplied
rather than values the box chose. Only then does the key touch it.

THE GPU MODEL IS PART OF THE MEASUREMENT, NOT A DETAIL. Measured across three architectures on
byte-identical items: within a box a round is bit-exact, across boxes it moves ~0.03 retention on a
genuine compression and ~0.17 on a foreign control. The dethrone margin is 0.05. So a crown scored
on an L40S is not comparable to one defended on an H100, and a rental flow that takes "whatever is
available" would silently make the crown a function of the spot market. `require_gpu` is therefore
a HARD gate: the wrong hardware aborts the round rather than scoring on it. Renting the cheapest
available GPU is the correct behaviour for a training job and the wrong behaviour for a referee.

TEARDOWN IS IN A `finally`, AND A `finally` IS NOT ENOUGH. This is the most expensive lesson in the
module. Round 1 attempt 7 reached `scoring…` and stopped: nothing streamed the remote scorer, so
nothing could tell a slow round from a dead one, and the first deadline to fire was systemd's
`TimeoutStartSec=10800`. SIGTERM does not raise in Python — it terminates the process outright — so
the `finally` below never ran, the H100 was never destroyed, and it billed at $3.30/hr until a human
noticed 226 minutes later. ~$12.50 for a round that produced nothing. Three things come out of that:

  * A ROUND THAT STOPS TALKING IS STUCK. The scorer is streamed, not awaited, and twenty minutes of
    silence kills it (`SILENCE_S`) — with `eval/progress.py` on the far end so silence is a real
    signal rather than an artefact of nobody having printed anything.
  * OUR DEADLINE MUST BEAT THE SUPERVISOR'S. Every remote step is bounded well inside
    `spec.max_hours`, so the process that ends a stuck round is this one, which can tear down,
    rather than an external killer, which cannot.
  * A SIGNAL HAS TO BECOME AN EXCEPTION. `_teardown_on_signal` turns SIGTERM/SIGINT into a raise so
    the `finally` runs anyway, because the day it matters is the day something else kills us.

Belt and braces on top: `sweep()` at the head of the next round is the backstop for a process that
dies where even a handler cannot run (SIGKILL, power loss).
"""
from __future__ import annotations

import contextlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field

# THE SILENCE BUDGET. Twenty minutes with nothing on the wire from the scorer. Sized against the
# slowest step that legitimately reports at one point only — a checkpoint download on a cold box —
# with room to spare, because the cost of being wrong in one direction is a killed healthy round
# and in the other is $3.30 an hour.
SILENCE_S = 1200.0

# THE HEARTBEAT, and it is what makes an EXTERNAL watchdog possible at all. `SILENCE_S` measures
# bytes on the PIPE; `eval/watchdog.py` measures bytes in the LOG, and those are not the same fact.
# They diverge on a perfectly healthy round in two ways: `max_lines` stops `_emit` dead after 5000
# lines while the reader thread keeps resetting the silence clock, and a `\r`-only tqdm bar only
# reaches `_emit` once per 2000 characters. Either one freezes the log's mtime for the rest of the
# leg while the round is working — which from outside is indistinguishable from attempt 7, and
# would have the watchdog SIGTERM the first round that ever got deep enough into scoring to trip
# it. So the stream writes a line on a timer regardless of how much the far end said.
HEARTBEAT_S = 300.0

# THE INSTALL LEG'S BUDGETS, NAMED BECAUSE eval/watchdog.py IMPORTS THEM. pip reports per package,
# not per megabyte, so a single 2.5 GB torch wheel on a slow link is legitimately quiet for a long
# time — this leg gets a looser silence budget than the scorer on purpose. If you tune it, the
# external watchdog re-tunes with it; a literal here and a copy over there is a rule that drifts
# silently and ends with two components holding contradictory definitions of "stuck".
INSTALL_SILENCE_S = 1800.0
INSTALL_HARD_S = 2400.0

# EPHEMERAL HOSTS GET NO known_hosts. Providers recycle IPs: scaleway handed us 51.159.162.43 twice
# with different host keys, and `accept-new` correctly refuses a CHANGED key — so the second rental
# at a reused IP failed after the GPU was paid for, and every stale entry would poison the next
# reuse forever. Pinning buys nothing here anyway: we have no prior key for a box that did not exist
# a minute ago. What makes this safe is the trust model, not the host key — the box gets no write
# credentials, and everything it returns is audited before our key touches it.
#
# THE KEEPALIVES ARE NOT COSMETIC. ssh's default behaviour on a half-open TCP is to wait forever,
# which is indistinguishable from a working round and is one of the two ways attempt 7 could have
# gone quiet. Six missed 30-second probes ends the connection in ~3 minutes and turns a dead network
# into a nonzero exit status, which the code above already knows how to handle.
_SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
             "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=15",
             "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6"]


def instance_age_s(inst: dict, now: float) -> float:
    """How long a provider instance has been up, from its own `created_at`.

    FAILS TO +INF, NEVER TO 0. This feeds decisions about destroying things that cost money, and
    the safe assumption when a timestamp cannot be parsed — a null, a reformatted field, a provider
    that renames it — is that the instance is OLD. Returning 0.0 makes a leak permanently
    un-overdue and prints "$0.00 spent" next to an H100 that has been billing all night."""
    raw = str(inst.get("created_at", "") or "")
    if not raw:
        return float("inf")
    try:
        from datetime import datetime
        return now - datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except Exception:
        return float("inf")


@dataclass
class Instance:
    id: str = ""
    ip: str = ""
    ssh_user: str = "shadeform"
    ssh_port: int = 22
    cloud: str = ""
    region: str = ""
    instance_type: str = ""
    price_per_hour: float = 0.0
    status: str = ""
    # THE RENTAL'S DEADLINE, as a wall-clock stamp rather than a per-step budget. Per-step budgets
    # add up: rent + wait + rsync + a 40-minute install + a 2.5-hour score is over three hours, and
    # three hours is when the supervisor kills us without teardown. Every step clamps to what is
    # left of this, so the total is what is actually bounded.
    kill_at: float = 0.0


@dataclass
class GpuSpec:
    """What the round needs. `require_gpu` is matched against `torch.cuda.get_device_name(0)`."""
    gpu_type: str = "H100"
    require_gpu: str = "NVIDIA H100 PCIe"
    cloud: str = ""                  # "" = any, but see require_gpu — the NAME is what binds
    # THE HARD CEILING ON A RENTAL, and it must stay under whatever supervises this process
    # (currently systemd's TimeoutStartSec=10800). Whoever's deadline fires first decides whether
    # the instance gets destroyed or leaked, so it has to be ours.
    max_hours: float = 2.5
    # ...and the ceiling that actually fires, on every remote step. See SILENCE_S.
    silence_s: float = SILENCE_S
    max_price_per_hour: float = 0.0  # 0 = no cap
    # THE PROVIDER-SIDE BACKSTOP, in hours past our own ceiling. Everything else in this module
    # bounds a rental by running code on this box: `kill_at` needs the process, the `finally` needs
    # an exception, the watchdog needs the box to be up. All three die together when the box does,
    # and a leak with nobody watching is unbounded — the incident was ended by a human noticing.
    # `auto_delete` is enforced by Shadeform, so it is the only bound that survives that.
    provider_deadline_slack_h: float = 0.5
    ssh_key: str = "/root/.ssh/id_bitzic"


class Provider:
    """Rent / wait / destroy. A protocol so the whole flow is testable without spending money."""

    # `exclude` is a tuple of (cloud, region) pairs already tried and failed THIS round. A
    # provider that ignores it still works; one that honours it stops a single broken datacentre
    # from wedging the subnet.
    def rent(self, spec: GpuSpec, name: str, exclude: tuple = ()) -> Instance: ...
    # `out` is part of the protocol, not an implementation detail: this call blocks for up to
    # fifteen minutes with the meter running, and a provider that cannot say so leaves the
    # single most expensive silent gap in the round.
    def wait_ready(self, inst: Instance, timeout_s: float = 900, out=None) -> Instance: ...
    def destroy(self, inst: Instance) -> None: ...
    def list_active(self) -> list: ...
    def list_all(self) -> list: ...


class ShadeformProvider(Provider):
    """Shadeform's REST API. The key lives in a file, never in the repo or a job spec."""

    API = "https://api.shadeform.ai/v1"

    def __init__(self, key_file: str = "/root/.shadeform_api_key", ssh_key_id: str = ""):
        self.key_file = key_file
        self.ssh_key_id = ssh_key_id or os.environ.get("SHADEFORM_SSH_KEY_ID", "")

    def _key(self) -> str:
        """Accept a bare key OR a `SHADEFORM_API_KEY=...` line OR the env var.

        A key file is hand-written, and writing the whole assignment line into it is the obvious
        thing to do. The bare-`.strip()` version sent the literal string `SHADEFORM_API_KEY=...`
        as the header and got back an unexplained 401 — a config shape that common deserves parsing,
        not a diagnosis session."""
        raw = os.environ.get("SHADEFORM_API_KEY", "")
        if not raw:
            try:
                with open(self.key_file) as fh:
                    raw = fh.read()
            except OSError as e:
                raise RuntimeError(
                    f"no Shadeform API key: set SHADEFORM_API_KEY or put it in {self.key_file} "
                    f"({e}). Renting is the only step of the round that needs it.") from e
        key = raw.strip()
        if "=" in key.split("\n")[0]:
            key = key.split("\n")[0].split("=", 1)[1].strip()
        key = key.strip().strip('"').strip("'")
        if not key or any(c.isspace() for c in key):
            raise RuntimeError(f"the Shadeform key in {self.key_file} does not look like a key "
                               f"(len={len(key)}); expected a bare token or NAME=token")
        return key

    def _api(self, method: str, path: str, body: dict | None = None) -> dict:
        import urllib.error
        import urllib.request
        req = urllib.request.Request(
            f"{self.API}{path}", method=method,
            data=json.dumps(body).encode() if body else None,
            headers={"X-API-KEY": self._key(), "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"shadeform {method} {path} -> {e.code}: "
                               f"{(e.read().decode() if e.fp else '')[:300]}") from e

    def rent(self, spec: GpuSpec, name: str, exclude: tuple = ()) -> Instance:
        r = self._api("GET", f"/instances/types?gpu_type={spec.gpu_type}"
                             f"&available=true&sort=price&num_gpus=1")
        types = r.get("instance_types", [])
        if spec.cloud:
            types = [t for t in types if t.get("cloud") == spec.cloud] or []
        if spec.max_price_per_hour:
            types = [t for t in types
                     if (t.get("hourly_price", 0) / 100.0) <= spec.max_price_per_hour]
        # NO FALLBACK TO "WHATEVER IS AVAILABLE". The existing v1 helper falls back to the cheapest
        # cloud when the preferred one is unavailable, which is right for a training job and wrong
        # for a referee: the GPU model is inside the measurement.
        if not types:
            raise RuntimeError(
                f"no {spec.gpu_type} available matching cloud={spec.cloud or 'any'} "
                f"price<={spec.max_price_per_hour or 'any'}. Refusing to substitute a different "
                f"GPU: cross-box spread is ~0.03 retention against a 0.05 dethrone margin.")
        # ...BUT A DIFFERENT REGION IS NOT A DIFFERENT MEASUREMENT. What binds the round is the
        # device NAME, which `require_gpu` gates on; where the box physically sits does not change
        # it. That distinction is load-bearing, because a region can simply fail to deliver:
        # scaleway/warsaw sat in `pending_provider` for the full 900 s and charged $0.84 for a box
        # that never booted. Without `exclude` the retry picks the identical cheapest candidate and
        # the subnet is wedged on one bad datacentre for as long as it stays broken.
        cands = [(t, a["region"]) for t in types for a in t.get("availability", [])
                 if a.get("available") and (t.get("cloud"), a.get("region")) not in exclude]
        if not cands:
            raise RuntimeError(
                f"every available {spec.gpu_type} (cloud, region) has already failed this round: "
                f"{sorted(exclude)}. Still not substituting a different GPU model.")
        best, region = cands[0]
        rate = best.get("hourly_price", 0) / 100.0
        body = {"cloud": best["cloud"], "region": region,
                "shade_instance_type": best["shade_instance_type"], "shade_cloud": True,
                "name": name}
        if self.ssh_key_id:
            body["ssh_key_id"] = self.ssh_key_id
        # THE ONE BOUND THAT SURVIVES THIS BOX DYING. Shadeform enforces both thresholds itself, so
        # unlike `kill_at`, the `finally`, and the watchdog — all of which need something here to be
        # alive — this one holds through a SIGKILL, an OOM, a panic, a reboot, or the orchestrator
        # being reclaimed entirely. That was the largest uncovered risk in the design: with nothing
        # off-box watching, a leak is bounded only by a human noticing, which last time took 226
        # minutes. Set LOOSER than every local deadline so it never pre-empts a legitimate round,
        # and paired with a spend cap because a date is the wrong unit for the thing being risked.
        hours = spec.max_hours + spec.provider_deadline_slack_h
        body["auto_delete"] = {
            "date_threshold": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime(time.time() + hours * 3600.0)),
            "spend_threshold": f"{max(1.0, rate * hours):.2f}",
        }
        # Self-describing ownership: the account is shared with a live miner box and other
        # projects, and an instance that cannot say whose it is has to be reasoned about by name.
        body["tags"] = ["ralph-v2", f"round-{name.split('-')[2] if name.count('-') >= 2 else '?'}",
                        f"host-{os.uname().nodename[:16]}"]
        got = self._api("POST", "/instances/create", body)
        inst = Instance(id=str(got.get("id", "")), cloud=best["cloud"], region=region,
                        instance_type=best["shade_instance_type"],
                        price_per_hour=rate, status="pending")
        # VERIFIED, NOT ASSUMED — this module's own lesson, learned when destroy() used the wrong
        # HTTP verb and reported success. A silently-ignored auto_delete is worse than none,
        # because the whole point is that it is the guarantee nobody is around to check.
        try:
            got_ad = (self._api("GET", f"/instances/{inst.id}/info") or {}).get("auto_delete")
        except Exception:
            got_ad = None
        if not got_ad or not got_ad.get("date_threshold"):
            sys.stdout.write(
                f"  WARNING the provider did not record an auto_delete deadline on {inst.id[:12]} "
                f"— if this box dies holding it, nothing off-box will stop the billing\n")
        return inst

    def wait_ready(self, inst: Instance, timeout_s: float = 900, out=None) -> Instance:
        """`out` IS NOT DEBUG NOISE. This loop is up to fifteen minutes long, the meter is already
        running, and until it reported one line at the end it was the single most expensive silent
        gap in the round — the most expensive place in the whole flow to be unable to tell a slow
        provider from a dead one. A line per poll turns a 900-second hole into a 15-second one."""
        t0 = time.time()
        deadline = t0 + timeout_s
        while time.time() < deadline:
            info = self._api("GET", f"/instances/{inst.id}/info")
            inst.status = info.get("status", "unknown")
            inst.ip = info.get("ip", "") or inst.ip
            inst.ssh_user = info.get("ssh_user", inst.ssh_user)
            inst.ssh_port = int(info.get("ssh_port", inst.ssh_port) or 22)
            if inst.status == "active" and inst.ip:
                return inst
            if inst.status in ("error", "deleted"):
                raise RuntimeError(f"instance {inst.id} entered status {inst.status}")
            if out is not None:
                out.write(f"    · waiting for {inst.id[:12]} ({inst.status}) "
                          f"{time.time() - t0:.0f}s\n")
                _flush(out.write)
            time.sleep(15)
        raise RuntimeError(f"instance {inst.id} not ready within {timeout_s:.0f}s")

    def destroy(self, inst: Instance) -> None:
        """POST, not DELETE — and then CHECK IT ACTUALLY WENT.

        Both halves of this are scar tissue. The first version used the HTTP verb the endpoint name
        suggests; the API wants `POST /instances/{id}/delete` and answers `DELETE` with a 405. I
        found that by leaking a $2.73/hr instance, which is the worst possible place in this module
        for a bug — teardown is the thing that stops billing, and it would have failed on every
        real round.

        The second half is the same lesson publish.py learned: a write whose return value you trust
        is a write you have not verified. So the instance is re-listed afterwards, and a failure to
        disappear is raised rather than assumed away. The empty body on success is expected and is
        NOT an error."""
        last = None
        for _ in range(3):
            try:
                self._api("POST", f"/instances/{inst.id}/delete")
            except Exception as e:
                # success returns an empty body, which the JSON decode reports as a value error
                if "Expecting value" not in str(e) and "204" not in str(e):
                    last = e
            time.sleep(3)
            # AGAINST list_all, NOT list_active — see below. Verifying teardown against a view that
            # hides `error` instances means an instance that merely flipped to `error` passes this
            # check, we print "destroyed", and it goes on billing.
            if not any(str(i.get("id", "")) == inst.id for i in self.list_all()):
                return
        raise RuntimeError(
            f"instance {inst.id} is STILL ACTIVE after three delete attempts — it is billing right "
            f"now, go and kill it in the console" + (f" (last error: {last})" if last else ""))

    def list_active(self) -> list:
        """What is RENTABLE-adjacent: healthy instances only. Correct for deciding where to rent.

        Do not use it to answer a money question — see `list_all`."""
        r = self._api("GET", "/instances")
        return [i for i in r.get("instances", [])
                if str(i.get("status", "")).lower() not in ("deleted", "error")]

    def list_all(self) -> list:
        """Everything the account holds that is not deleted — INCLUDING `status == "error"`.

        The distinction is worth a method because it is worth money. `list_active` drops errored
        instances, which is right when choosing where to rent and wrong for every question of the
        form "is anything still billing": an instance that flipped to `error` still exists, still
        costs, and — until this existed — passed `destroy()`'s own verification, so the orchestrator
        announced a teardown that had not happened. Money questions use this one."""
        r = self._api("GET", "/instances")
        return [i for i in r.get("instances", [])
                if str(i.get("status", "")).lower() != "deleted"]

    def sweep(self, prefix: str = "ralph-round-", out=sys.stdout, min_age_s: float = 0.0) -> list:
        """Kill anything we named that is still up. The backstop for a process that died between
        renting and destroying — which is exactly how the bug above was found.

        SCOPED TO THE NAMES WE MINT, and narrowly. `run_remote_round` names its rentals
        `ralph-round-<n>-<ts>`, so that is the prefix — NOT `ralph-`, which was the first version
        and which matches `ralph-v2-miner-m19`, a live miner box on this same account that this
        code has no business touching. A sweep that kills someone else's running work is a far
        worse failure than the leak it exists to prevent, so the prefix must match what we create
        rather than what we are called.

        AND SCOPED BY AGE. This runs at the head of a round, before `rent()`, and it used to be
        age-blind — so starting a round while another was still scoring destroyed the live one's
        GPU mid-round. That was survivable while rounds were started by hand one at a time and
        stops being survivable the moment the timer is enabled. Callers pass a `min_age_s` past
        every legitimate round's ceiling; anything younger than that may still belong to somebody,
        and `eval/watchdog.py` — which polls every five minutes and knows which rentals the running
        round has CLAIMED — does the fast cleanup this was overreaching to do."""
        killed = []
        now = time.time()
        for i in self.list_all():
            name = str(i.get("name", ""))
            if not name.startswith(prefix):
                continue
            age = instance_age_s(i, now)
            if age < min_age_s:
                out.write(f"  leaving {name} alone ({age / 60:.0f} min old, under the "
                          f"{min_age_s / 60:.0f} min floor) — it may be a live round\n")
                continue
            iid = str(i.get("id", ""))
            out.write(f"  sweeping orphan {name} ({iid[:12]}, {age / 60:.0f} min old)\n")
            self.destroy(Instance(id=iid))
            killed.append(name)
        return killed


@dataclass
class RoundPlan:
    """Everything the CPU decided, which the GPU is merely told."""
    round: int
    commit_root: str
    round_nonce: str
    prev_anchor: str
    committed: list = field(default_factory=list)
    # tier -> Reign, replayed from the published trail. Without it the rented box opens every
    # throne and crowns max(retention) outright, and the dethrone margin never runs.
    kings: dict = field(default_factory=dict)
    parent_key: str = "qwen3-8b"
    observers: list = field(default_factory=list)
    tiers: list = field(default_factory=list)
    n_items: int = 72
    pool_size: int = 900
    margin: float = 0.05

    def as_job(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        return d


class RemoteRoundError(RuntimeError):
    """Raised INSTEAD of signing. Nothing is published and no weights are set."""


def _ssh(inst: Instance, spec: GpuSpec, cmd: str, timeout: int = 3600, stdin: str = "") -> str:
    """A SHORT remote command, awaited. Anything that can run for minutes goes through
    `_ssh_stream` instead — a wall-clock timeout on a long step is exactly the bug that burned an
    H100 for 226 minutes.

    `stdin` exists so a secret can reach the box WITHOUT appearing in argv — the remote process
    table is readable by any co-tenant, and argv lands verbatim in the exception text below."""
    r = subprocess.run(
        ["ssh", "-i", spec.ssh_key, "-p", str(inst.ssh_port), *_SSH_OPTS,
         f"{inst.ssh_user}@{inst.ip}", cmd],
        input=stdin if stdin else None,
        capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RemoteRoundError(f"remote command failed ({r.returncode}): {cmd[:80]}\n"
                               f"{r.stderr[-2000:]}")
    return r.stdout


def _flush(w) -> None:
    """Flush the stream behind a bound `.write`. Unlovely, and load-bearing: the watchers that
    decide a round is alive read the log's MTIME, so output that sits in a buffer is a round that
    reads as hung. The runner is handed a `write` and not the stream, hence `__self__`."""
    s = getattr(w, "__self__", None)
    if s is not None and hasattr(s, "flush"):
        try:
            s.flush()
        except Exception:
            pass


class _Out:
    """A stream-shaped view of a bare `write` callable, so a helper that wants `out.write(...)` can
    be handed the runner's `w` without every caller threading the real stream through."""

    def __init__(self, w):
        self._w = w

    def write(self, s: str) -> None:
        self._w(s)

    def flush(self) -> None:
        _flush(self._w)


def _stream_until_silent(proc, silence_s: float, hard_s: float, w, what: str,
                         prefix: str = "    | ", max_lines: int = 5000,
                         heartbeat_s: float = HEARTBEAT_S) -> str:
    """Drain `proc`'s merged output, echo it live, and kill it when it stops arriving.

    Split out from `_ssh_stream` because it is the whole safety property and it must be testable
    without renting anything: point it at any Popen and it behaves identically.

    LIVENESS IS MEASURED IN BYTES, NOT LINES. `readline` would block for the entire duration of a
    `tqdm` bar or an `hf_hub_download` — they separate updates with \\r and emit no \\n until they
    finish — so a line-based detector would read a perfectly healthy 16 GB checkpoint download as
    twenty minutes of silence. Any byte off the pipe proves the far end is alive; lines are only
    how it gets displayed.

    Two budgets, both real: `silence_s` catches the round that stopped working, `hard_s` catches
    the round that is still working and can no longer be afforded. `heartbeat_s` is neither — it
    guarantees the LOG advances even when this function has decided to stop echoing, so that an
    external watcher measuring mtime and this function measuring the pipe agree about liveness.
    """
    q: queue.Queue = queue.Queue()

    def _reader():
        try:
            while True:
                b = proc.stdout.read(65536)     # bufsize=0: one read syscall, returns what is there
                if not b:
                    break
                q.put(b)
        except Exception:
            pass
        finally:
            q.put(None)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    tail: deque = deque(maxlen=400)      # the caller only ever displays the end of this
    pending, echoed = "", 0
    t0 = last = last_write = time.monotonic()
    done = False

    def _emit(line: str) -> None:
        # `last_write` ADVANCES ONLY WHEN SOMETHING REACHES THE LOG. Setting it on every call —
        # including the suppressed ones past `max_lines` — silently disarms the heartbeat in
        # exactly the case it exists for: a chatty remote whose output we have stopped echoing,
        # where the file's mtime freezes while the round is working perfectly.
        nonlocal echoed, last_write
        echoed += 1
        if echoed <= max_lines:
            tail.append(line)
            w(f"{prefix}{line}\n")
            _flush(w)
            last_write = time.monotonic()
        elif echoed == max_lines + 1:
            w(f"{prefix}… further output suppressed after {max_lines} lines "
              f"(the heartbeat continues)\n")
            _flush(w)
            last_write = time.monotonic()

    while not done:
        try:
            # The poll interval keeps the hard deadline responsive AND keeps the main thread in an
            # interruptible wait, which is what lets `_teardown_on_signal` land at all.
            chunk = q.get(timeout=5.0)
        except queue.Empty:
            chunk = b""
        if chunk is None:
            done = True
        elif chunk:
            last = time.monotonic()
            pending += chunk.decode("utf-8", "replace")
            while True:
                i = pending.find("\n")
                if i < 0:
                    break
                _emit(pending[:i].rstrip("\r"))
                pending = pending[i + 1:]
            # A \r-only progress bar never terminates a line; flush it in bounded pieces so the log
            # shows movement instead of holding a megabyte hostage until the download ends.
            while len(pending) > 2000:
                _emit(pending[:2000])
                pending = pending[2000:]

        now = time.monotonic()
        # THE HEARTBEAT GOES BEFORE THE SILENCE CHECK, and it writes with `w` directly rather than
        # through `_emit` — the whole point is that it survives `_emit`'s own suppression cap.
        if not done and now - last_write >= heartbeat_s:
            w(f"{prefix}[heartbeat] {what}: {now - t0:.0f}s in, "
              f"last byte {now - last:.0f}s ago, {echoed} lines\n")
            _flush(w)
            last_write = now
        if not done and now - last > silence_s:
            _kill(proc)
            raise RemoteRoundError(
                f"{what} produced no output for {(now - last) / 60:.1f} min — a round that stops "
                f"talking is stuck, and a stuck round on a rented GPU is only ever more expensive. "
                f"Killed; the instance is being destroyed. Last output:\n" +
                "".join(f"    | {l}\n" for l in list(tail)[-10:]))
        if not done and now - t0 > hard_s:
            _kill(proc)
            raise RemoteRoundError(
                f"{what} exceeded its {hard_s / 3600:.1f} h ceiling (still producing output). "
                f"Killed; the instance is being destroyed.")

    if pending.strip():
        _emit(pending.rstrip("\r\n"))
    try:
        rc = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        _kill(proc)
        raise RemoteRoundError(f"{what} closed its output but would not exit")
    if rc != 0:
        raise RemoteRoundError(f"{what} failed ({rc}). Last output:\n" +
                               "".join(f"    | {l}\n" for l in list(tail)[-25:]))
    return "\n".join(tail)


def _kill(proc) -> None:
    """Kill the LOCAL ssh. The remote process outlives it — there is no TTY to hang up — and that
    is deliberately not chased: teardown is what stops the billing, it runs in the `finally`
    directly above every caller, and a second ssh into a box that just went unreachable is the one
    call most likely to hang next."""
    try:
        proc.kill()
        proc.wait(timeout=10)
    except Exception:
        pass


def _ssh_stream(inst: Instance, spec: GpuSpec, cmd: str, w, what: str,
                silence_s: float = 0.0, hard_s: float = 0.0, prefix: str = "    | ",
                heartbeat_s: float = HEARTBEAT_S) -> str:
    """`_ssh` for the steps that take minutes: the output is watched as it arrives rather than
    collected at the end, so the round can be ended while it is still cheap to end it."""
    # CLAMPED TO WHAT IS LEFT OF THE RENTAL, never just to this step's own budget — see
    # Instance.kill_at. A step that starts with ten minutes left on the clock gets ten minutes.
    hard = hard_s or spec.max_hours * 3600.0
    if inst.kill_at:
        hard = min(hard, max(0.0, inst.kill_at - time.time()))
    proc = subprocess.Popen(
        ["ssh", "-i", spec.ssh_key, "-p", str(inst.ssh_port), *_SSH_OPTS,
         f"{inst.ssh_user}@{inst.ip}", cmd],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    try:
        return _stream_until_silent(proc, silence_s or spec.silence_s, hard, w, what,
                                    prefix=prefix, heartbeat_s=heartbeat_s)
    finally:
        if proc.poll() is None:
            _kill(proc)


@contextlib.contextmanager
def _teardown_on_signal():
    """Turn SIGTERM/SIGINT into an exception so the `finally` that destroys the rental RUNS.

    THIS IS THE $12.50 LINE. Python's default SIGTERM disposition terminates the interpreter
    outright — no unwinding, no `finally`, no teardown — so when systemd's TimeoutStartSec fired on
    attempt 7 the H100 was simply abandoned mid-round and billed until a human noticed. Every
    deadline in this module is now set to fire before an external one does, but that ordering is an
    argument and this is a guarantee: whatever kills us, we get to destroy the instance first.

    A best-effort install. Off the main thread `signal.signal` raises, and a runner under test has
    nothing to tear down anyway."""
    def _raise(signum, _frame):
        raise RemoteRoundError(f"signal {signum} — ending the round so the rental can be destroyed")

    prev = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            prev[sig] = signal.signal(sig, _raise)
        except (ValueError, OSError, AttributeError):
            pass
    try:
        yield
    finally:
        for sig, handler in prev.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass


def _scp(spec: GpuSpec, inst: Instance, src: str, dst: str, to_remote: bool = True,
         timeout: float = 1800) -> None:
    a = f"{inst.ssh_user}@{inst.ip}:{dst}" if to_remote else src
    b = dst if to_remote else f"{inst.ssh_user}@{inst.ip}:{src}"
    args = ["scp", "-i", spec.ssh_key, "-P", str(inst.ssh_port), "-r", *_SSH_OPTS]
    args += [src, a] if to_remote else [b, dst]
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RemoteRoundError(f"scp failed: {r.stderr[-500:]}")


def verify_returned_record(rec, plan: RoundPlan, pool, spec: GpuSpec, summary: dict) -> list:
    """AUDIT THE SCORER BEFORE SIGNING IT. Returns a list of fatal reasons; empty means sign.

    Signing whatever came back would launder it: a rented box could return a record for a different
    round, a pruned exam, or an invented crown, and our key would make it authoritative. So the
    round's identity has to be the one WE issued, the hardware has to be the one we required, and
    the arithmetic and the selection have to reproduce — the same L0/L1 an outsider runs, pointed at
    our own scorer."""
    from .rerun import FAIL, audit_loaded

    bad = []
    if rec is None:
        return ["the remote produced no record"]

    # 1. IDENTITY. These are the fields the CPU decided; a box that returns different ones scored
    #    some other round, whatever else is true of it.
    for field_, want in (("round", plan.round), ("commit_root", plan.commit_root),
                         ("round_nonce", plan.round_nonce), ("prev_anchor", plan.prev_anchor)):
        got = getattr(rec, field_, None)
        if got != want:
            bad.append(f"{field_} came back as {str(got)[:24]!r}, we issued {str(want)[:24]!r} — "
                       f"this record is not for the round we asked for")

    # 2. HARDWARE. The GPU model is inside the measurement (see the module header).
    got_gpu = ((rec.manifest or {}).get("versions") or {}).get("gpu") or summary.get("gpu") or ""
    # AN ABSENT DEVICE IS ALWAYS FATAL, INDEPENDENT OF `require_gpu`. `_gpu_name()` returns "" when
    # torch cannot see a GPU, and the pin below is skipped on the FIRST round by design — the round
    # whose whole job is to learn the device name. So the two rules combined said: on the one round
    # that establishes the hardware, any hardware will do. That is not hypothetical. A cu130 torch
    # against a CUDA 12.8 driver makes `is_available()` False, `device_map="auto"` falls back to
    # CPU, and the round scores an 8B parent on 12 cores — slowly, and correctly enough that the
    # identity canary PASSES, because CPU inference is perfectly deterministic. It would have been
    # signed, published and crowned, and nothing downstream would ever have said otherwise.
    if not got_gpu:
        bad.append("the record names no GPU at all, which means torch could not see one and the "
                   "round was scored on CPU. A crown measured on CPU is not comparable with one "
                   "defended on an H100, and the identity canary cannot catch it because CPU "
                   "inference is deterministic")
    if spec.require_gpu and got_gpu != spec.require_gpu:
        bad.append(f"scored on {got_gpu or '(unknown)'} but this subnet's records are pinned to "
                   f"{spec.require_gpu} — a crown scored on other hardware is not comparable with "
                   f"one defended on this hardware")

    # 3. THE CANARY. The parent scored against itself must be exactly 1.0 by construction; anything
    #    else means the box is nondeterministic and every number from it is suspect.
    ident = (rec.manifest or {}).get("identity") or summary.get("identity") or {}
    score = ident.get("score")
    if score is None or abs(float(score) - 1.0) > 1e-9:
        bad.append(f"identity canary returned {score} — the parent must score exactly 1.000 "
                   f"against itself, so this box is not deterministic")

    # 4. THE AUDIT ITSELF, at the levels the orchestrator can run without a GPU.
    #
    #    THE SIGNATURE CHECKS ARE EXCLUDED HERE, AND ONLY HERE. `audit_loaded` fails an unsigned
    #    record because for an outsider an unsigned record is attributable to nobody — correct
    #    everywhere except at this exact moment, where the record is unsigned BY DESIGN because the
    #    key is on this machine and has not been applied yet. Excluding them by name rather than
    #    relaxing the audit keeps that narrow: every other check still binds.
    a = audit_loaded(rec, pool=pool)
    for c in a.checks:
        if c.status == FAIL and not c.name.startswith("signature"):
            bad.append(f"[{c.level}] {c.name}: {c.detail}")
    if getattr(rec, "signature", ""):
        bad.append("the remote returned a SIGNED record — the signing key is supposed to live "
                   "only on this machine, so a signature arriving from the rented box means a key "
                   "leaked to it")
    return bad


def run_remote_round(plan: RoundPlan, provider: Provider, spec: GpuSpec, work_dir: str,
                     repo_dir: str = ".", remote_dir: str = "~/ralph-v2",
                     runner=None, out=sys.stdout) -> dict:
    """Rent, score, pull back, AUDIT, and return the unsigned record + pool for the caller to sign.

    Deliberately stops short of signing and publishing: this function's job is to produce a record
    the caller can trust, and the caller's job is to hold the key. Returns
    {record, pool_blob, summary, instance, cost}."""
    w = out.write
    os.makedirs(work_dir, exist_ok=True)
    job_path = os.path.join(work_dir, "job.json")
    with open(job_path, "w") as fh:
        json.dump(plan.as_job(), fh, indent=1)

    # NO SECRETS IN THE JOB SPEC. Asserted rather than assumed, because the spec is written to disk
    # and copied to somebody else's machine.
    # CHECK FOR THE VALUES, NOT FOR WORDS. The first version matched field NAMES — including
    # "coldkey", which the spec legitimately carries as a miner's PUBLIC ss58 address because the
    # economics gate is per-coldkey. It refused every real round while protecting nothing: a secret
    # that happened not to contain one of those five words would have shipped regardless. So this
    # looks for the actual secrets this process holds.
    blob = json.dumps(plan.as_job())
    held = {"RALPH_RECORD_SEED": os.environ.get("RALPH_RECORD_SEED", ""),
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "SHADEFORM_API_KEY": os.environ.get("SHADEFORM_API_KEY", "")}
    for name, val in held.items():
        if val and len(val) >= 16 and val in blob:
            raise RemoteRoundError(f"refusing to ship a job spec containing the value of {name}")
    if "PRIVATE KEY" in blob or "mnemonic" in blob.lower():
        raise RemoteRoundError("refusing to ship a job spec containing key material")

    # SWEEP BEFORE RENTING. `finally` covers exceptions; it does NOT cover the process being
    # killed, and that is not hypothetical — an ssh session timing out took a round down mid-score
    # and left an H100 billing at $3.30/hr with no teardown. There is no server-side TTL to fall
    # back on, so the guarantee has to be "the next round cleans up the last one's corpse", which
    # is what v1's helper did at the top of every command for the same reason.
    #
    # THE AGE FLOOR IS THE POINT. Age-blind, this destroyed the GPU of any round already in flight
    # — fine while rounds were started one at a time by hand, fatal the moment the timer is on. The
    # floor sits past every legitimate round's ceiling, so what it reaps is only ever a corpse.
    try:
        swept = (provider.sweep(out=out, min_age_s=spec.max_hours * 3600.0 + 900.0)
                 if hasattr(provider, "sweep") else [])
        if swept:
            w(f"  swept {len(swept)} orphaned instance(s) from a previous run\n")
    except Exception as e:
        w(f"  WARNING orphan sweep failed: {e} — check the provider console\n")

    name = f"ralph-round-{plan.round}-{int(time.time()) % 100000}"
    # The handler goes on BEFORE the money starts, so that from here to the `finally` there is no
    # signal that can take the rental with it. The window between `rent` returning and entering the
    # `try` is a few microseconds of Python, and `sweep()` covers even that.
    with _teardown_on_signal():
        # THE CLAIM IS WRITTEN BEFORE THE POST, and the name is the whole reason. `rent()` learns
        # the instance id only from the create response, so a create that SUCCEEDS server-side and
        # times out client-side leaves a correctly-named H100 billing whose id nobody ever knew —
        # unfindable in the log, unattributable by any watcher. The name is the only claim that
        # survives that, so it goes to disk first.
        # PROVISIONING IS A SEPARATE FAILURE FROM SCORING, and it needs a separate remedy. A round
        # that never got a box has produced nothing, risked nothing and audited nothing — retrying
        # it elsewhere is free of every concern that makes substitution dangerous, because the GPU
        # MODEL is unchanged and the round has not started. Attempt 9 died exactly here: 900 s in
        # `pending_provider` at scaleway/warsaw, $0.84 for hardware that never booted, and the next
        # attempt would have chosen the same broken datacentre.
        tried: list = []
        inst = None
        for attempt in range(1, 4):
            w(f"  renting {name}…"
              f"{f' (attempt {attempt}, avoiding {tried})' if tried else ''}\n")
            _flush(w)
            inst = provider.rent(spec, name, exclude=tuple(tried))
            inst.kill_at = time.time() + spec.max_hours * 3600.0
            w(f"  rented {inst.id} {inst.instance_type} @ ${inst.price_per_hour:.2f}/hr "
              f"({inst.cloud}/{inst.region}) — ceiling {spec.max_hours:.1f} h "
              f"(~${inst.price_per_hour * spec.max_hours:.2f} worst case), "
              f"silence limit {spec.silence_s / 60:.0f} min\n")
            _flush(w)
            t0 = time.time()
            try:
                inst = provider.wait_ready(inst, out=_Out(w))
                break
            except Exception as e:
                # DESTROY BEFORE RETRYING. A box stuck in `pending_provider` is still billing, and
                # leaving one behind per attempt is how a retry loop becomes the expensive bug.
                w(f"  never became usable ({e}) — trying another region\n")
                tried.append((inst.cloud, inst.region))
                try:
                    provider.destroy(inst)
                    w(f"  destroyed {inst.id} (~${inst.price_per_hour * (time.time() - t0) / 3600.0:.2f})\n")
                except Exception as de:
                    w(f"  WARNING could not destroy {inst.id}: {de} — CHECK THE CONSOLE\n")
                _flush(w)
                inst = None
        if inst is None:
            raise RemoteRoundError(
                f"no {spec.gpu_type} became usable after {len(tried)} region(s): {tried}. "
                f"Nothing was scored and every instance was destroyed.")
        w(f"  ready at {inst.ip} after {time.time() - t0:.0f}s\n")
        _flush(w)
        return _run_on(inst, plan, provider, spec, job_path, work_dir, repo_dir, remote_dir,
                       runner, w, t0)


def _run_on(inst, plan, provider, spec, job_path, work_dir, repo_dir, remote_dir,
            runner, w, t0) -> dict:
    """The rented half, split out only so the signal handler above wraps `rent` itself.

    Enters with a box that is already READY — provisioning and its retries belong to the caller,
    because a box that never booted has nothing to tear down here and nothing to audit."""
    try:
        run = runner or _default_runner
        summary = run(inst, spec, plan, job_path, work_dir, repo_dir, remote_dir, w)
        rec_path = os.path.join(work_dir, "record.json")
        pool_path = os.path.join(work_dir, "pool.jsonl")
        raw = json.loads(open(rec_path).read())
        if raw is None:
            raise RemoteRoundError("the remote scored no record (no usable submissions?)")
        from .rerun import load_pool, record_from_blob
        rec = record_from_blob(json.dumps(raw).encode())
        pool = load_pool(pool_path)
        bad = verify_returned_record(rec, plan, pool, spec, summary)
        if bad:
            for r in bad:
                w(f"  REJECT  {r}\n")
            raise RemoteRoundError(f"the returned record failed {len(bad)} check(s); NOT signing")
        w(f"  audited: L0+L1 reproduce, canary 1.000, gpu {spec.require_gpu}\n")
        with open(pool_path, "rb") as fh:
            pool_blob = fh.read()
        return {"record": rec, "pool_blob": pool_blob, "summary": summary,
                "instance": inst, "cost": inst.price_per_hour * (time.time() - t0) / 3600.0}
    finally:
        # THE POST-MORTEM HAS TO HAPPEN BEFORE THE BODY IS DESTROYED. Teardown is correct and
        # non-negotiable, but it also deletes the only machine that knows why the round died —
        # attempt 8 ended with `ssh exit 255` and nothing else, which says the far end vanished and
        # nothing about why. Disk, memory, the GPU and the kernel ring buffer are four cheap
        # questions, and asking them costs seconds against a rental we are about to stop paying for.
        if sys.exc_info()[0] is not None:
            _capture_diagnostics(inst, spec, w)
        # ALWAYS. A leaked instance bills until somebody notices, and the process that leaked it is
        # by definition the one that already went wrong. `destroy` verifies rather than assumes:
        # the first version of it used the wrong HTTP verb and would have failed silently here.
        try:
            provider.destroy(inst)
            w(f"  destroyed {inst.id} after {(time.time() - t0) / 60:.1f} min "
              f"(~${inst.price_per_hour * (time.time() - t0) / 3600.0:.2f})\n")
        except Exception as e:
            w(f"  WARNING could not destroy {inst.id}: {e} — CHECK THE PROVIDER CONSOLE\n")


# The torch wheel indexes, newest first. A driver reporting CUDA X.Y can run any build up to X.Y
# and none above it, so the rule is "the newest wheel not newer than the driver".
_TORCH_WHEELS = ((13.0, "cu130"), (12.8, "cu128"), (12.6, "cu126"), (12.4, "cu124"),
                 (12.1, "cu121"), (11.8, "cu118"))


def _torch_index(driver_cuda: str) -> str:
    """The pytorch wheel index this host's driver can actually run, or "" to let pip decide.

    Returning "" on an unparseable version is deliberate: guessing a CUDA build from no information
    is how you get a torch that cannot see the GPU, and `score_job._assert_gpu` will refuse the
    round in seconds either way. Better a loud refusal than a quiet wrong wheel."""
    try:
        want = float(".".join(driver_cuda.strip().split(".")[:2]))
    except Exception:
        return ""
    for ver, tag in _TORCH_WHEELS:
        if want >= ver:
            return f"https://download.pytorch.org/whl/{tag}"
    return ""


def _remaining(inst: Instance, default_s: float) -> float:
    """`default_s`, or what is left of the rental, whichever is smaller.

    The awaited steps — the ssh mkdir, the rsync, the scps — are 6300 seconds of structural wall
    clock between them, all of it INSIDE the rental and none of it previously bounded by anything.
    Only the streamed steps clamped to `kill_at`, so "max_hours bounds the rented leg" was true of
    the expensive part and false of the rest."""
    if not inst.kill_at:
        return default_s
    return max(1.0, min(default_s, inst.kill_at - time.time()))


def _capture_diagnostics(inst: Instance, spec: GpuSpec, w) -> None:
    """Ask the dying box the four questions, best-effort, on a short leash.

    Every one of the eight round-1 failures was a seam rather than the mechanism, and the recurring
    shape is "works from the dev box, breaks where it runs". The box IS the evidence, so a teardown
    with no post-mortem converts a diagnosable failure into a guess — which is how attempt 8 ended
    with a bare `ssh exit 255`.

    Short timeout and everything swallowed: this runs while an exception is already propagating,
    on a host that has just proved it may be unreachable, and it must never replace the real error
    with its own."""
    # NOTHING TO ASK IF IT NEVER ANSWERED. An instance that died in `pending_provider` has no IP,
    # so every probe becomes `ssh: Could not resolve hostname` — four confusing 255s that look like
    # the box refusing to talk rather than a box that never existed.
    if not inst.ip:
        w(f"  no post-mortem: {inst.id[:12]} never reached a usable IP "
          f"(status {inst.status!r}), so there is nothing on it to ask\n")
        return
    probes = (
        ("disk", "df -h / /home 2>/dev/null | head -5"),
        ("memory", "free -g 2>/dev/null | head -3"),
        ("gpu", "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader "
                "2>/dev/null"),
        ("kernel", "(dmesg 2>/dev/null || sudo -n dmesg 2>/dev/null) | tail -12"),
    )
    w("  post-mortem on the box before teardown:\n")
    for label, cmd in probes:
        try:
            out = _ssh(inst, spec, cmd, timeout=25)
        except Exception as e:
            out = f"({type(e).__name__}: {str(e)[:80]})"
        for line in (out or "(no output)").strip().splitlines()[:12]:
            w(f"    {label:7} {line}\n")
    _flush(w)


def _default_runner(inst, spec, plan, job_path, work_dir, repo_dir, remote_dir, w) -> dict:
    """Push the repo, install, score, pull the artifacts back. Replaceable for tests."""
    # NAMED SUB-STEPS, because "pushing repo…" covered three calls with a combined structural
    # ceiling of 6300 s and an external watcher cannot tell which one it is sitting in.
    w("  pushing repo: mkdir…\n")
    _flush(w)
    _ssh(inst, spec, f"mkdir -p {remote_dir} && rm -rf {remote_dir}/eval",
         timeout=int(_remaining(inst, 3600)))
    w("  pushing repo: rsync…\n")
    _flush(w)
    # HOME-RELATIVE, because the rented box is not our box. It logs in as `shadeform`, not root,
    # and /workspace is either absent or root-owned there — `mkdir -p /workspace/ralph-v2` failed
    # with Permission denied after the GPU was already rented and paid for. The one directory a
    # login user can always write is their own home.
    subprocess.run(["rsync", "-az", "-e",
                    f"ssh -i {spec.ssh_key} -p {inst.ssh_port} " + " ".join(_SSH_OPTS),
                    # EXPLICIT SECRET EXCLUSIONS. The list used to be runs/ and .git/ only, which
                    # kept .env off the box by accident (it is not in the repo dir) rather than on
                    # purpose. An operator who ever drops a .env or a wallet beside the code would
                    # have shipped it to somebody else's hardware with no warning.
                    "--exclude", "runs/", "--exclude", ".git/",
                    "--exclude", ".env", "--exclude", ".env.*", "--exclude", "*.key",
                    "--exclude", "wallets/", "--exclude", ".venv/", "--exclude", "__pycache__/",
                    "--exclude", ".hf_read", "--exclude", "*.pem",
                    f"{repo_dir}/",
                    f"{inst.ssh_user}@{inst.ip}:ralph-v2/"],
                   check=True, timeout=_remaining(inst, 900))
    w("  pushing repo: job.json…\n")
    _flush(w)
    _scp(spec, inst, job_path, f"{remote_dir}/job.json", to_remote=True,
         timeout=_remaining(inst, 1800))

    # WHICH TORCH THE DRIVER CAN ACTUALLY RUN. `pip install torch` takes the newest build, which
    # in August 2026 is cu130 — and these boxes ship driver 570.x, i.e. CUDA 12.8. The mismatch does
    # not raise: `torch.cuda.is_available()` simply returns False, `device_map="auto"` places the
    # parent on CPU, and the round is ~10x slower and unusable. Asking the box which CUDA its
    # driver speaks costs one ssh and removes an entire class of silent failure.
    cuda = ""
    try:
        cuda = _ssh(inst, spec,
                    "nvidia-smi | sed -n 's/.*CUDA Version: *\\([0-9][0-9.]*\\).*/\\1/p' | head -1",
                    timeout=60).strip()
    except Exception as e:
        w(f"  WARNING could not read the driver's CUDA version ({e}); using pip's default torch\n")
    index = _torch_index(cuda)
    w(f"  driver speaks CUDA {cuda or '?'} -> torch wheel {index or 'pypi default'}\n")
    _flush(w)

    w("  installing…\n")
    # STREAMED, NOT AWAITED, and for two reasons beyond the silence budget. `-q` plus `| tail -2`
    # meant this step produced nothing for up to forty minutes AND — the same bug called out on the
    # scorer below — reported a failed install as success, because a pipeline exits with the status
    # of `tail`. So a broken wheel surfaced later as an unexplained scoring crash. Without the pipe
    # pip talks (one line per package; it draws no progress bars when stdout is not a TTY), which
    # is both the liveness signal and a readable install log.
    # A LOOSER SILENCE BUDGET HERE, ON PURPOSE. pip reports per package, not per megabyte, so a
    # single 2.5 GB torch wheel on a slow link is legitimately quiet for a long time — and unlike
    # the scorer this leg has a real wall-clock ceiling to fall back on. 40 minutes of install is
    # ~$2.20 and bounded; a false kill costs the whole round.
    # TORCH FIRST, FROM ITS OWN INDEX, then everything else from PyPI. `--index-url` REPLACES PyPI
    # rather than adding to it, so a single combined command would fail to find transformers; and
    # `--extra-index-url` leaves the resolver free to pick either, which is how you get the newest
    # build back by accident.
    torch_cmd = (f".venv/bin/pip install --index-url {index} torch && " if index
                 else ".venv/bin/pip install torch && ")
    _ssh_stream(inst, spec, f"cd {remote_dir} && python3 -m venv .venv 2>/dev/null; "
                            f"{torch_cmd}"
                            f".venv/bin/pip install transformers safetensors datasets "
                            f"huggingface_hub pynacl accelerate llama-cpp-python",
                w, what="the remote install", silence_s=INSTALL_SILENCE_S,
                hard_s=INSTALL_HARD_S, prefix="    · ")

    w("  scoring (this is the expensive part)…\n")
    # THE WRITE TOKEN NEVER GOES TO THE RENTED BOX, and the fallback that sent it was the single
    # worst line in this module: it defeated the one thing the whole split exists to guarantee.
    # HF_TOKEN is the token that can publish the round trail and overwrite the artifact repos;
    # `HF_TOKEN_READ` is a read-scoped token, and its only job on that box is gated public models
    # (google/gemma-2-2b-it needs one). If there is no read token we send NOTHING — an ungated
    # observer works fine anonymously, and a missing gated model is a loud failure rather than a
    # leaked credential.
    read_tok = os.environ.get("HF_TOKEN_READ", "")
    if read_tok and read_tok == os.environ.get("HF_TOKEN", ""):
        raise RemoteRoundError(
            "HF_TOKEN_READ is the same value as HF_TOKEN — that is the WRITE token, and shipping "
            "it to a rented box hands whoever holds that box the ability to rewrite the published "
            "round trail. Mint a read-scoped token, or unset HF_TOKEN_READ.")
    env = ""
    if read_tok:
        # ...and not on the COMMAND LINE, where it sits in the remote process table for any
        # co-tenant to read and lands verbatim in any error message this raises.
        _ssh(inst, spec, f"umask 077 && mkdir -p {remote_dir} && cat > {remote_dir}/.hf_read",
             timeout=60, stdin=read_tok)
        env = f"HF_TOKEN=$(cat {remote_dir}/.hf_read) "
    # Eager attention allocates one enormous contiguous score matrix per batch; the default caching
    # allocator fragments around it and then fails a request the card could physically satisfy.
    # This does not change any number — only whether the allocation succeeds.
    env += "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    # NO `| tail`. A pipeline's exit status is the LAST command's, so piping the scorer through
    # tail made every remote crash exit 0 — _ssh saw success, and the operator's first sign of a
    # failed round was `scp failed` on a record that was never written. A file-transfer error is a
    # terrible way to report that scoring died.
    #
    # AND NO WALL-CLOCK WAIT. This one line is what the 226-minute hang was: `_ssh(timeout=10800)`
    # says "give the scorer three hours and tell me how it went", which turns "the far end went
    # quiet" into "bill an H100 until somebody notices". It is watched now — `eval/progress.py`
    # ticks on the far side, `_stream_until_silent` ends the round twenty minutes after those ticks
    # stop, and the `finally` above destroys the instance on the way out.
    _ssh_stream(inst, spec, f"cd {remote_dir} && {env}.venv/bin/python -u -m eval.score_job "
                            f"job.json out/", w, what="the remote scorer")

    w("  pulling results…\n")
    _flush(w)
    for f in ("record.json", "pool.jsonl", "summary.json"):
        _scp(spec, inst, f"{remote_dir}/out/{f}", os.path.join(work_dir, f), to_remote=False,
             timeout=_remaining(inst, 1800))
    return json.loads(open(os.path.join(work_dir, "summary.json")).read())
