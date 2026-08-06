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
    observers: tuple = ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "google/gemma-2-2b-it")
    records_repo: str = "RalphLabsAI/ralph-v2-rounds"
    tiers: tuple = ("ternary", "sub4")
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
            gpu_type=e("RALPH_GPU_TYPE", "H100"), require_gpu=e("RALPH_REQUIRE_GPU", ""),
            max_price_per_hour=float(e("RALPH_MAX_GPU_PRICE", "4.50")))


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
    from .parent import PARENTS
    if cfg.parent_key not in PARENTS:
        bad.append(f"unknown parent {cfg.parent_key!r}")
    fam = [o for o in cfg.observers
           if cfg.parent_key in PARENTS
           and o.split("/")[0] == PARENTS[cfg.parent_key].name.split("/")[0]]
    if fam:
        bad.append(f"observers {fam} share the parent's family — they are not independent")
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
        warn.append("LIVE: this round WILL commit an anchor and set weights with the validator "
                    "hotkey. If another signer holds that key, expect nonce collisions.")
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

    spec = GpuSpec(gpu_type=cfg.gpu_type, require_gpu=cfg.require_gpu,
                   max_price_per_hour=cfg.max_price_per_hour)
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
    ok = chain.set_weights(rec.weights)
    w(f"  weights   : {rec.weights} -> set={ok} (live={cfg.live})\n")
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
