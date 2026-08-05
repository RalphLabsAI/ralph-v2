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

TEARDOWN IS IN A `finally`, AND SO IS A DEADLINE. A leaked instance bills until someone notices.
Every rental carries a `kill_at` the provider-side sweep enforces, in case this process dies between
renting and destroying.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field


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


@dataclass
class GpuSpec:
    """What the round needs. `require_gpu` is matched against `torch.cuda.get_device_name(0)`."""
    gpu_type: str = "H100"
    require_gpu: str = "NVIDIA H100 PCIe"
    cloud: str = ""                  # "" = any, but see require_gpu — the NAME is what binds
    max_hours: float = 2.5
    max_price_per_hour: float = 0.0  # 0 = no cap
    ssh_key: str = "/root/.ssh/id_bitzic"


class Provider:
    """Rent / wait / destroy. A protocol so the whole flow is testable without spending money."""

    def rent(self, spec: GpuSpec, name: str) -> Instance: ...
    def wait_ready(self, inst: Instance, timeout_s: float = 900) -> Instance: ...
    def destroy(self, inst: Instance) -> None: ...
    def list_active(self) -> list: ...


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

    def rent(self, spec: GpuSpec, name: str) -> Instance:
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
        best = types[0]
        regions = [a for a in best.get("availability", []) if a.get("available")]
        if not regions:
            raise RuntimeError("no available region for the selected instance type")
        body = {"cloud": best["cloud"], "region": regions[0]["region"],
                "shade_instance_type": best["shade_instance_type"], "shade_cloud": True,
                "name": name}
        if self.ssh_key_id:
            body["ssh_key_id"] = self.ssh_key_id
        got = self._api("POST", "/instances/create", body)
        return Instance(id=str(got.get("id", "")), cloud=best["cloud"],
                        region=regions[0]["region"],
                        instance_type=best["shade_instance_type"],
                        price_per_hour=best.get("hourly_price", 0) / 100.0, status="pending")

    def wait_ready(self, inst: Instance, timeout_s: float = 900) -> Instance:
        deadline = time.time() + timeout_s
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
            if not any(str(i.get("id", "")) == inst.id for i in self.list_active()):
                return
        raise RuntimeError(
            f"instance {inst.id} is STILL ACTIVE after three delete attempts — it is billing right "
            f"now, go and kill it in the console" + (f" (last error: {last})" if last else ""))

    def list_active(self) -> list:
        r = self._api("GET", "/instances")
        return [i for i in r.get("instances", [])
                if str(i.get("status", "")).lower() not in ("deleted", "error")]

    def sweep(self, prefix: str = "ralph-", out=sys.stdout) -> list:
        """Kill anything we named that is still up. The backstop for a process that died between
        renting and destroying — which is exactly how the bug above was found.

        SCOPED BY NAME PREFIX, deliberately and narrowly: this account also carries instances
        belonging to other projects, and a sweep that reached them would be a far worse failure
        than the leak it prevents."""
        killed = []
        for i in self.list_active():
            name = str(i.get("name", ""))
            if not name.startswith(prefix):
                continue
            iid = str(i.get("id", ""))
            out.write(f"  sweeping orphan {name} ({iid[:12]})\n")
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


def _ssh(inst: Instance, spec: GpuSpec, cmd: str, timeout: int = 3600) -> str:
    r = subprocess.run(
        ["ssh", "-i", spec.ssh_key, "-p", str(inst.ssh_port),
         "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
         f"{inst.ssh_user}@{inst.ip}", cmd],
        capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RemoteRoundError(f"remote command failed ({r.returncode}): {cmd[:80]}\n"
                               f"{r.stderr[-2000:]}")
    return r.stdout


def _scp(spec: GpuSpec, inst: Instance, src: str, dst: str, to_remote: bool = True) -> None:
    a = f"{inst.ssh_user}@{inst.ip}:{dst}" if to_remote else src
    b = dst if to_remote else f"{inst.ssh_user}@{inst.ip}:{src}"
    args = ["scp", "-i", spec.ssh_key, "-P", str(inst.ssh_port), "-r",
            "-o", "StrictHostKeyChecking=accept-new"]
    args += [src, a] if to_remote else [b, dst]
    r = subprocess.run(args, capture_output=True, text=True, timeout=1800)
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
                     repo_dir: str = ".", remote_dir: str = "/workspace/ralph-v2",
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
    blob = json.dumps(plan.as_job())
    for leak in ("RALPH_RECORD_SEED", "coldkey", "seed", "private", "mnemonic"):
        if leak.lower() in blob.lower():
            raise RemoteRoundError(f"refusing to ship a job spec containing {leak!r}")

    name = f"ralph-round-{plan.round}-{int(time.time()) % 100000}"
    inst = provider.rent(spec, name)
    w(f"  rented {inst.id} {inst.instance_type} @ ${inst.price_per_hour:.2f}/hr "
      f"({inst.cloud}/{inst.region})\n")
    t0 = time.time()
    try:
        inst = provider.wait_ready(inst)
        w(f"  ready at {inst.ip} after {time.time() - t0:.0f}s\n")
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
        # ALWAYS. A leaked instance bills until somebody notices, and the process that leaked it is
        # by definition the one that already went wrong. `destroy` verifies rather than assumes:
        # the first version of it used the wrong HTTP verb and would have failed silently here.
        try:
            provider.destroy(inst)
            w(f"  destroyed {inst.id} after {(time.time() - t0) / 60:.1f} min "
              f"(~${inst.price_per_hour * (time.time() - t0) / 3600.0:.2f})\n")
        except Exception as e:
            w(f"  WARNING could not destroy {inst.id}: {e} — CHECK THE PROVIDER CONSOLE\n")


def _default_runner(inst, spec, plan, job_path, work_dir, repo_dir, remote_dir, w) -> dict:
    """Push the repo, install, score, pull the artifacts back. Replaceable for tests."""
    w("  pushing repo…\n")
    _ssh(inst, spec, f"mkdir -p {remote_dir} && rm -rf {remote_dir}/eval")
    subprocess.run(["rsync", "-az", "-e",
                    f"ssh -i {spec.ssh_key} -p {inst.ssh_port} -o StrictHostKeyChecking=accept-new",
                    "--exclude", "runs/", "--exclude", ".git/", f"{repo_dir}/",
                    f"{inst.ssh_user}@{inst.ip}:{remote_dir}/"], check=True, timeout=900)
    _scp(spec, inst, job_path, f"{remote_dir}/job.json", to_remote=True)

    w("  installing…\n")
    _ssh(inst, spec, f"cd {remote_dir} && python3 -m venv .venv 2>/dev/null; "
                     f".venv/bin/pip -q install torch transformers safetensors datasets "
                     f"huggingface_hub pynacl accelerate llama-cpp-python 2>&1 | tail -2", timeout=2400)

    w("  scoring (this is the expensive part)…\n")
    env = "HF_TOKEN=%s " % os.environ.get("HF_TOKEN_READ", os.environ.get("HF_TOKEN", ""))
    # NO `| tail`. A pipeline's exit status is the LAST command's, so piping the scorer through
    # tail made every remote crash exit 0 — _ssh saw success, and the operator's first sign of a
    # failed round was `scp failed` on a record that was never written. A file-transfer error is a
    # terrible way to report that scoring died.
    out_txt = _ssh(inst, spec, f"cd {remote_dir} && {env}.venv/bin/python -u -m eval.score_job "
                               f"job.json out/", timeout=10800)
    for line in out_txt.strip().splitlines()[-12:]:
        w(f"    | {line}\n")

    w("  pulling results…\n")
    for f in ("record.json", "pool.jsonl", "summary.json"):
        _scp(spec, inst, f"{remote_dir}/out/{f}", os.path.join(work_dir, f), to_remote=False)
    return json.loads(open(os.path.join(work_dir, "summary.json")).read())
