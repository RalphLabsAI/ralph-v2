"""THE OPERATOR ENTRYPOINT for the split validator. Runs on the CPU box; scores on a rented GPU.

`eval.run_round` is the single-box version: it loads the parent and the observers in-process, so it
needs a GPU wherever it runs. This is the one an orchestrator runs — it never imports torch, never
downloads a checkpoint, and never lets the signing key leave the machine.

    read the chain      ->  rent a GPU  ->  score there  ->  AUDIT what came back
    ->  sign here  ->  publish  ->  anchor on chain  ->  set weights  ->  destroy the GPU

WHAT THIS BOX DOES THAT THE GPU CANNOT. It decides the round's identity — commit window, nonce,
commit_root, prev_anchor — because a box that chose its own nonce could grind it, and because the
anchor has to chain onto the history WE published. It holds the record seed and the wallet. And it
runs the audit that stands between the rented box's output and our signature.

WRITES ARE OFF BY DEFAULT, and `--live` is required for the two that matter (the anchor commitment
and the weight vector). That is not a debug convenience: at the time of writing the same hotkey is
being used by another signer, and two processes signing with one key produce nonce collisions and
failed extrinsics. Bringing this up live has to be a conscious act.

    python -m eval.run_orchestrated --dry-run    # preflight only, no rental, no chain reads
    python -m eval.run_orchestrated              # full round, rents a GPU, no chain WRITES
    python -m eval.run_orchestrated --live       # ... and anchor + set weights
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass


@dataclass
class Config:
    netuid: int = 40
    network: str = "finney"
    wallet: str = "ralph"
    hotkey: str = "owner"
    parent_key: str = "qwen3-8b"
    # THREE UNGATED OBSERVERS, AND UNGATED IS THE LOAD-BEARING WORD.
    #
    # google/gemma-2-2b-it is GATED: a rented box with no credentials gets a 401 and the round dies
    # — which is exactly what happened once the nonce finally drew it, fifteen attempts in, because
    # until then it kept drawing the ungated one. A dependency that fails only on some draws is
    # worse than one that always fails.
    #
    # The fix is not to drop to a single observer. The observer is drawn from the round nonce
    # PRECISELY so it cannot be pre-fitted; with one candidate that draw is a constant and a miner
    # tunes against a known judge. Two candidates is one bit. Three is ~1.6, from three unrelated
    # pretraining lineages (HuggingFaceTB / Microsoft / AllenAI), none sharing the parent's Qwen
    # family — `preflight` rejects an observer that does.
    #
    # And the security dividend: with nothing gated, the rented box needs NO Hugging Face
    # credential at all. HF_TOKEN_READ existed only to reach gemma; the box that scores the round
    # can now hold literally no secret.
    observers: tuple = ("HuggingFaceTB/SmolLM2-1.7B-Instruct",
                        "microsoft/Phi-3-mini-4k-instruct",
                        "allenai/OLMo-2-1124-7B-Instruct")
    records_repo: str = "RalphLabsAI/ralph-v2-rounds"
    # ALL FOUR, because the public README advertises all four and miners clone it. Running two
    # meant a `binary` submission did the whole compression job and was rejected at intake with
    # "not being scored this round" — the spec promised a lane that could not be won. A tier that
    # receives nothing is free: `consider()` returns action "none" and no crown is minted.
    tiers: tuple = ("binary", "ternary", "sub2", "sub4")
    # Whether to write the weight vector on chain. Separate from `live` on purpose: a shakedown
    # wants a real anchored trail without moving anyone's emission.
    set_weights: bool = True
    # Mirror each crowned artifact into one repo under our org, verified against the signed
    # record. On by default: a crown nobody can find is not a product, and the miner repos it
    # lives in are named after miners and can be overwritten at will.
    publish_crowns: bool = True
    crowns_repo: str = "RalphLabsAI/ralph-crowns"
    # (cloud/region) pairs whose IMAGE cannot run a round — e.g. no CUDA toolkit, so llama.cpp
    # falls back to the CPU wheel and every submission scores ~6x slower on a GPU we are paying for.
    exclude_regions: tuple = ()
    # RALPH_REFERENCES="name=tier=hf://repo@rev,name2=tier2=hf://..." — fixed points scored every
    # round and never crowned. Without one, a retention of 0.30 is a number with nothing to be
    # measured against, including for us.
    references: tuple = ()
    n_items: int = 72
    pool_size: int = 900
    commit_window: int = 100
    work_dir: str = "/workspace/ralph-v2-work"
    gpu_type: str = "H100"
    require_gpu: str = ""          # "" on the FIRST round: learn it, then pin it. See below.
    max_price_per_hour: float = 4.50
    live: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        e = os.environ.get
        return cls(
            netuid=int(e("RALPH_NETUID", "40")), network=e("RALPH_NETWORK", "finney"),
            wallet=e("RALPH_WALLET", "ralph"), hotkey=e("RALPH_HOTKEY", "owner"),
            parent_key=e("RALPH_PARENT_KEY", "qwen3-8b"),
            records_repo=e("RALPH_HF_REPO", "RalphLabsAI/ralph-v2-rounds"),
            n_items=int(e("RALPH_N_ITEMS", "72")), pool_size=int(e("RALPH_POOL_SIZE", "900")),
            work_dir=e("RALPH_WORK_DIR", "/workspace/ralph-v2-work"),
            observers=tuple(x.strip() for x in e("RALPH_OBSERVERS", "").split(",") if x.strip())
                      or cls.observers,
            # the ONE field that was not env-readable, so opening a tier needed a code change and
            # a redeploy while every other setting was a config flip
            tiers=tuple(x.strip() for x in e("RALPH_TIERS", "").split(",") if x.strip())
                  or cls.tiers,
            set_weights=e("RALPH_SET_WEIGHTS", "1") != "0",
            publish_crowns=e("RALPH_PUBLISH_CROWNS", "1") != "0",
            crowns_repo=e("RALPH_CROWNS_REPO", "") or cls.crowns_repo,
            exclude_regions=tuple(x.strip() for x in e("RALPH_GPU_EXCLUDE", "").split(",")
                                  if x.strip()),
            references=tuple(x.strip() for x in e("RALPH_REFERENCES", "").split(",") if x.strip()),
            gpu_type=e("RALPH_GPU_TYPE", "H100"), require_gpu=e("RALPH_REQUIRE_GPU", ""),
            max_price_per_hour=float(e("RALPH_MAX_GPU_PRICE", "4.50")))


def _parse_references(specs, out=None) -> list:
    """`name=tier=uri` -> `[{"name", "tier", "artifact_uri"}]`. A malformed entry is SKIPPED with a
    warning, never fatal: a reference is our own yardstick, and a typo in it must not stop eleven
    miners being scored."""
    refs = []
    for raw in (specs or []):
        parts = [p.strip() for p in str(raw).split("=", 2)]
        if len(parts) != 3 or not all(parts):
            if out is not None:
                out.write(f"  reference {raw!r} ignored — expected name=tier=uri\n")
            continue
        refs.append({"name": parts[0], "tier": parts[1], "artifact_uri": parts[2]})
    return refs


def _supervisor_deadline_problems(spec) -> list:
    """Is this process's own supervisor deadline OUTSIDE every deadline the round enforces?

    Reads the running unit rather than a file path, because the drop-in that actually applies is
    whichever systemd merged last. Absent systemd (a hand-started round, a container, a test) this
    says nothing: an unsupervised round has no outer ring to invert."""
    import subprocess

    unit = os.environ.get("RALPH_VALIDATOR_UNIT", "ralph-validator.service")
    try:
        r = subprocess.run(["systemctl", "show", unit, "-p", "TimeoutStartUSec", "--value"],
                           capture_output=True, text=True, timeout=15)
        raw = (r.stdout or "").strip()
    except Exception:
        return []
    if not raw or raw in ("infinity", "0"):
        return []
    # systemd prints things like "10h", "6h", "1h 30min", "21600s"
    units = {"us": 1e-6, "ms": 1e-3, "s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}
    import re
    secs = 0.0
    for n, u in re.findall(r"(\d+(?:\.\d+)?)\s*(us|ms|min|[smhd])", raw):
        secs += float(n) * units.get(u, 0.0)
    if secs <= 0:
        return []
    outer = (spec.max_hours + spec.provider_deadline_slack_h) * 3600.0
    if secs <= outer:
        return [f"{unit} has TimeoutStartSec={raw} ({secs / 3600:.2f} h), which is INSIDE this "
                f"round's own deadlines (kill_at {spec.max_hours:.2f} h, provider auto_delete "
                f"{outer / 3600:.2f} h). systemd would SIGTERM a healthy round before it finished, "
                f"and whoever kills the process decides whether the rental dies or leaks. Raise it "
                f"above {outer / 3600:.2f} h in a drop-in and `systemctl daemon-reload`."]
    return []


def preflight(cfg: Config, out=sys.stdout) -> list:
    """Everything checkable before a cent is spent. NOTE WHAT IS ABSENT: no torch check.

    This box does not score, so requiring torch here — as `eval.run_round`'s preflight does, quite
    correctly for itself — would block the orchestrator on a dependency it exists to avoid."""
    bad, warn = [], []
    if not os.environ.get("HF_TOKEN"):
        bad.append("HF_TOKEN unset: the publisher refuses to construct without one, and finding "
                   "that out after a GPU has already scored wastes the rental")
    if not os.environ.get("RALPH_RECORD_SEED"):
        bad.append("RALPH_RECORD_SEED unset: records would be unsigned and publish_and_gate "
                   "withholds every unsigned round, so the whole rental would be thrown away")
    # WHOEVER KILLS THE PROCESS DECIDES WHETHER THE RENTAL DIES OR LEAKS. Every deadline the round
    # enforces itself is useless if the SUPERVISOR's fires first: systemd SIGTERMs, and only the
    # teardown handler stands between that and an abandoned H100. This was live on 2026-08-07 —
    # `kill_at` was raised 4.5 h -> 8 h and the unit's `TimeoutStartSec` was left at 6 h, so a
    # healthy 6.5 h round would have been killed 30 minutes from the end. Nothing caught it because
    # the ladder test asserts the rings this code owns, and the outermost ring lives in a unit file.
    bad.extend(_supervisor_deadline_problems(GpuSpec(
        gpu_type=cfg.gpu_type, require_gpu=cfg.require_gpu,
        max_price_per_hour=cfg.max_price_per_hour)))

    from .parent import PARENTS
    if cfg.parent_key not in PARENTS:
        bad.append(f"unknown parent {cfg.parent_key!r}")
    fam = [o for o in cfg.observers
           if cfg.parent_key in PARENTS
           and o.split("/")[0] == PARENTS[cfg.parent_key].name.split("/")[0]]
    if fam:
        bad.append(f"observers {fam} share the parent's family — they are not independent")
    # EVERY OBSERVER MUST BE ANONYMOUSLY READABLE, checked BEFORE a cent is spent. The rented box
    # holds no Hugging Face credential, so a gated model is a 401 seventeen minutes into a round —
    # and only on the draws that pick it, which is how gemma-2-2b-it survived fourteen attempts
    # before killing one. One HEAD request each is a rounding error against a $3.30/hr rental.
    import urllib.error
    import urllib.request
    for o in cfg.observers:
        try:
            req = urllib.request.Request(
                f"https://huggingface.co/{o}/resolve/main/config.json", method="HEAD")
            urllib.request.urlopen(req, timeout=20)
        except urllib.error.HTTPError as e:
            bad.append(f"observer {o} is not anonymously readable (HTTP {e.code}) — the scoring box "
                       f"carries no HF credential, so this round would die when the nonce drew it")
        except Exception as e:
            warn.append(f"could not verify observer {o} is reachable ({type(e).__name__})")
        # ...AND IT MUST BE ABLE TO HOLD WHAT WE FEED IT. The observer reads
        # prefix + step + continuation. Overflowing a RoPE model does not raise — it extrapolates
        # and returns degraded distributions, which are the input to the KL that decides crowns.
        try:
            import json as _j
            from .pool import MAX_PREFIX_TOKENS
            with urllib.request.urlopen(
                    f"https://huggingface.co/{o}/resolve/main/config.json", timeout=20) as r:
                win = int(_j.loads(r.read()).get("max_position_embeddings") or 0)
            need = MAX_PREFIX_TOKENS + 256 + 128
            if win and win < need:
                bad.append(f"observer {o} holds {win} positions but a round feeds it up to {need} "
                           f"(prefix {MAX_PREFIX_TOKENS} + step 256 + continuation 128). It would "
                           f"not raise — it would extrapolate and return degraded distributions "
                           f"into the KL that decides crowns")
        except Exception:
            pass

    try:
        from .orchestrator import ShadeformProvider
        ShadeformProvider()._key()
    except Exception as e:
        bad.append(f"no usable GPU provider key: {e}")
    if not cfg.require_gpu:
        # NOT fatal, and this is the one place the strictness is relaxed on purpose. The device
        # name is only knowable by booting one, so the first round learns it and prints the value
        # to pin. Every round after that must pin it, or the crown becomes a function of the spot
        # market — cross-box spread is ~0.03 retention against a 0.05 dethrone margin.
        warn.append("RALPH_REQUIRE_GPU is unset — this round will accept whatever GPU it gets and "
                    "print the name to pin. Set it before the round that awards a real crown.")
    if cfg.live:
        warn.append("LIVE: this round WILL commit an anchor with the validator hotkey"
                    + (" and SET WEIGHTS" if cfg.set_weights else
                       "; weights are WITHHELD (RALPH_SET_WEIGHTS=0)")
                    + ". If another signer holds that key, expect nonce collisions.")
    # NOTHING WATCHES THE WATCHER, and `systemctl is-enabled` would be the pgrep mistake one level
    # up — it proves the unit EXISTS. This reads the heartbeat the watchdog writes on every pass,
    # including passes that errored, because a watchdog blind since a key rotation looks perfect to
    # is-enabled and has not actually run in a week. A warning, never a blocker: a monitor being
    # off is not a reason to refuse work, it is a reason to know you are unmonitored.
    import json as _json
    try:
        st = _json.load(open(os.path.join(cfg.work_dir, "watchdog-state.json")))
        age = time.time() - float(st.get("last_pass", 0))
        if age > 900:
            warn.append(f"the watchdog has not completed a pass in {age / 60:.0f} min — if this "
                        f"round is SIGKILLed, OOM-killed, or the box reboots, no in-process guard "
                        f"runs and the rental leaks. That cost $12.50 once.")
    except Exception:
        warn.append("no watchdog heartbeat at all (watchdog-state.json missing) — nothing outside "
                    "this process will notice a leaked rental. See deploy/install_watchdog.sh")
    for r in bad:
        out.write(f"  BLOCKED  {r}\n")
    for r in warn:
        out.write(f"  warn     {r}\n")
    if not bad:
        out.write(f"  preflight OK — parent {cfg.parent_key}, {len(cfg.observers)} observers, "
                  f"tiers {'+'.join(cfg.tiers)}, records -> {cfg.records_repo}\n")
    return bad


RUN_BANNER = "=== ralph round start "


def _milestone_writer(out):
    """`out.write`, but flushed. The whole external-liveness signal is the log's mtime, and until
    this existed it rode entirely on `-u` in a unit file that is not in this repo and that no test
    can see. A buffered milestone is a round that reads as hung."""
    def w(s: str) -> None:
        out.write(s)
        try:
            out.flush()
        except Exception:
            pass
    return w


def run(cfg: Config, round_no: int | None = None, provider=None, out=sys.stdout) -> int:
    from .chain_bittensor import BittensorChainIO
    from .orchestrator import GpuSpec, RoundPlan, ShadeformProvider, run_remote_round
    from .publish import HFSink, PublishError, RecordPublisher, publish_and_gate
    from .signing import Ed25519Signer

    w = _milestone_writer(out)
    # THE RUN DELIMITER, and it is what makes the log readable by anything but a human.
    # /var/log/ralph-validator.log is opened `append:` once per invocation and never truncated, so
    # nothing in it is attributable by position: six runs currently share one file with no
    # separator and no timestamps of their own. A watcher tailing it reads the PREVIOUS round's
    # last line as this round's state — and that line is currently `  scoring (this is the
    # expensive part)…`, which carries the tightest budget there is. UNINDENTED on purpose: every
    # milestone below starts with exactly two spaces, so this can never be mistaken for one.
    w(f"{RUN_BANNER}pid={os.getpid()} t={int(time.time())} "
      f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===\n")
    if preflight(cfg, out):
        w("\n  refusing to run — nothing rented, nothing spent\n")
        return 2

    chain = BittensorChainIO(netuid=cfg.netuid, network=cfg.network, wallet_name=cfg.wallet,
                             hotkey_name=cfg.hotkey, read_only=not cfg.live)
    # THE HIGH-WATER MARK IS PER-REPO. It is the never-shrink guard: "I have published N rounds
    # before, so an index serving fewer is a shrunken history — refuse." Keying it by work_dir
    # alone means switching trails (shakedown -> production) carries the shakedown's count to an
    # empty production repo, and the very first real round refuses to write with "history has
    # shrunk". The guard would be correct about the count and wrong about the question.
    hwm = os.path.join(cfg.work_dir, f"publish-hwm-{cfg.records_repo.replace('/', '_')}.json")
    publisher = RecordPublisher(HFSink(cfg.records_repo), window=8, state_path=hwm)

    now = chain.current_block()
    lo, hi = now - cfg.commit_window, now
    w(f"\n  block {now}, commit window [{lo}, {hi}]\n")
    # require_local=False: the artifacts belong on the GPU's disk, not on the box with the key.
    commits = chain.read_commitments(lo, hi, require_local=False)
    w(f"  v2 commitments: {len(commits)}"
      f"{'  (skipped: ' + str(len(chain.skipped)) + ')' if chain.skipped else ''}\n")
    if not commits:
        # NOT an error and NOT a round. Renting a GPU to score nobody is pure burn, and publishing
        # an empty record would put a round on the trail that says nothing.
        w("  no v2 submissions this window — not renting. Nothing to score is not a failure.\n")
        return 0

    # THE THRONE IS INHERITED. Derived from the published trail, never from local state: the
    # records are signed, hash-chained and anchored, so the lineage is recoverable by anyone — and
    # a local kings.json would be a second source of truth the operator controls, which is exactly
    # what BittensorChainIO.get_king returns None to avoid.
    from .lineage import replay_from_trail
    kings = replay_from_trail(publisher, out=out)

    idx_round = round_no if round_no is not None else len(publisher.load_index()["rounds"]) + 1
    plan = RoundPlan(
        round=idx_round, commit_root=chain.commit_root(lo, hi), round_nonce=chain.block_hash(now),
        prev_anchor=publisher.head_anchor(), parent_key=cfg.parent_key,
        observers=list(cfg.observers), tiers=list(cfg.tiers),
        n_items=cfg.n_items, pool_size=cfg.pool_size,
        committed=[{"hotkey": c.hotkey, "coldkey": c.coldkey, "tier": c.tier,
                    "artifact_uri": c.artifact_uri, "committed_value": c.committed_value,
                    "revealed_hash": c.revealed_hash, "salt": c.salt,
                    "declared_compute_h100h": c.declared_compute_h100h,
                    "bond_posted": c.bond_posted} for c in commits])
    plan.kings = {t: vars(r) for t, r in kings.items()}
    plan.references = _parse_references(cfg.references, out)

    spec = GpuSpec(gpu_type=cfg.gpu_type, require_gpu=cfg.require_gpu,
                   max_price_per_hour=cfg.max_price_per_hour,
                   exclude_regions=cfg.exclude_regions)
    res = run_remote_round(plan, provider or ShadeformProvider(), spec,
                           os.path.join(cfg.work_dir, f"round-{idx_round}"), out=out)
    rec, summary = res["record"], res["summary"]
    if not cfg.require_gpu:
        w(f"\n  PIN THIS: RALPH_REQUIRE_GPU={summary.get('gpu')!r}\n"
          f"  every later round must match it, or crowns stop being comparable\n\n")

    # THE KEY TOUCHES IT HERE, and only after the audit inside run_remote_round passed.
    seed = os.environ["RALPH_RECORD_SEED"]
    rec.sign(Ed25519Signer(seed=bytes.fromhex(seed)[:32]))
    w(f"  signed by {rec.signer[:16]}…\n")

    publisher.publish_pool(res["pool_blob"], digest=(rec.manifest or {}).get("pool_sha256", ""))
    rep = publish_and_gate(publisher, rec, anchor_fn=chain.publish_record,
                           head_anchor_fn=chain.head_anchor, allow_unanchored=not cfg.live)
    w(f"  published : {rep.published.uri if rep.published else None}\n")
    w(f"  anchored  : {rep.anchor_verified}   reasons={rep.reasons or '-'}\n")
    if not rep.ok:
        # FAIL CLOSED. The previous crown keeps earning; an unauditable new one does not take over.
        w("  WITHHELD — publishing did not verify, so no weights were set\n")
        return 1
    # ANCHORING AND PAYING ARE SEPARATE DECISIONS. `--live` used to mean both, so the only way to
    # publish a verifiable, anchored round during a shakedown was to also move real emission. The
    # trail is the product; the weight vector is the payout. Round 1 wrote no weights only because
    # the 100-block rate limit happened to refuse it, which is luck, not a control.
    if not cfg.set_weights:
        w(f"  weights   : {rec.weights}\n"
          f"  weights   : NOT SET — RALPH_SET_WEIGHTS=0. The round is scored, signed, published "
          f"and anchored; only the payout is withheld.\n")
    else:
        ok = chain.set_weights(rec.weights)
        w(f"  weights   : {rec.weights} -> set={ok} (live={cfg.live})\n")

    # MIRROR THE CROWNS, AFTER the round is anchored and paid. Two rules make this safe to run
    # inside the round rather than as a chore somebody remembers:
    #
    #   * it is LAST. Everything above is the round; a crown that fails to mirror leaves a round
    #     that is still scored, signed, published, anchored and paid.
    #   * it cannot raise. `publish()` swallows and reports, because a mirroring bug must not
    #     convert a successful round into a failed one at the final step.
    #
    # It is also idempotent — a tier whose king did not change is skipped after one small read,
    # so an unchanged crown costs nothing rather than re-uploading gigabytes every round.
    if cfg.publish_crowns:
        from .publish_crowns import publish as _publish_crowns
        w(f"  crowns    : mirroring to {cfg.crowns_repo}…\n")
        rep_c = _publish_crowns(repo=cfg.crowns_repo, push=True, out=w)
        if not rep_c["ok"]:
            w("  crowns    : NOT MIRRORED — the round stands; re-run "
              "`python -m eval.publish_crowns --push` once the cause is fixed\n")
    else:
        w("  crowns    : not mirrored (RALPH_PUBLISH_CROWNS=0)\n")
    w(f"  cost      : ~${res['cost']:.2f}\n")
    return 0


def main(argv: list) -> int:
    cfg = Config.from_env()
    cfg.live = "--live" in argv
    if "--dry-run" in argv:
        return 0 if not preflight(cfg) else 2
    try:
        return run(cfg)
    except Exception as e:
        # A round that dies still has to say so loudly; the GPU is torn down in run_remote_round's
        # own `finally`, so the money is already stopped by the time this prints.
        print(f"\n  ROUND FAILED: {type(e).__name__}: {e}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
