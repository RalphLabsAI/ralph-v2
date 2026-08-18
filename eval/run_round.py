"""The operator entrypoint — one real round, wired end to end.

Everything else in this repo is a component or a harness. This is the thing an operator actually
runs, and the only place where the pool, the observers, the fetcher, the publisher and the chain
are assembled into a round that can award a crown.

PREFLIGHT COMES FIRST, AND IT IS THE POINT. Loading a pinned 8B parent and an observer is minutes
of wall clock and gigabytes of download. Discovering after that that the HF token is missing, or
that the tier forgot to arm its bit-budget gate, is a bad trade — so every check that can be made
cheaply is made before a single weight loads, and each one names what to do about it. The failure
this is built against is the one that already happened twice in this codebase: a misconfiguration
that surfaced only after the expensive part, wearing the wrong diagnosis.

WRITES ARE OFF BY DEFAULT. `--live` is required to set weights, and without it the round scores,
publishes and anchors exactly as it would in production while leaving the chain untouched. That is
not a debug mode — it is how this runs while a previous validator still holds the hotkey, and it is
the safe default for a subnet where two signers on one key fight each other.

    python -m eval.run_round --dry-run          # preflight only, loads nothing
    python -m eval.run_round                    # full round, no chain writes
    python -m eval.run_round --live             # ... and set weights
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

# A round cannot be smaller than (number of scoring slices) x (per-slice floor). With three
# languages and two depths that is 6 x 8 = 48. Set above it so a source dropping out does not
# immediately make the round unfillable.
DEFAULT_N_ITEMS = 72
DEFAULT_POOL = 900


@dataclass
class Config:
    netuid: int = 40
    network: str = "finney"
    wallet: str = "ralph"
    hotkey: str = "owner"
    parent_key: str = "qwen3-8b"
    # DELIBERATELY NOT QWEN. The observer's whole job is to be a third party: it reads what the
    # parent's step did and what the submission's step did, and reports whether they moved it to
    # the same place. An observer from the parent's own family shares its tokenizer, its training
    # mixture and its blind spots — it would systematically under-detect exactly the failures that
    # family has, which on a Qwen parent is the case that matters most.
    observers: tuple = ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "google/gemma-2-2b-it")
    records_repo: str = "RalphLabsAI/ralph-v2-rounds"
    tiers: tuple = ("ternary", "sub4")
    n_items: int = DEFAULT_N_ITEMS
    pool_size: int = DEFAULT_POOL
    work_dir: str = "/workspace/ralph-v2-work"
    live: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        e = os.environ.get
        return cls(
            netuid=int(e("RALPH_NETUID", "40")),
            network=e("RALPH_NETWORK", "finney"),
            wallet=e("RALPH_WALLET", "ralph"),
            hotkey=e("RALPH_HOTKEY", "owner"),
            parent_key=e("RALPH_PARENT_KEY", "qwen3-8b"),
            records_repo=e("RALPH_HF_REPO", "RalphLabsAI/ralph-v2-rounds"),
            n_items=int(e("RALPH_N_ITEMS", str(DEFAULT_N_ITEMS))),
            pool_size=int(e("RALPH_POOL_SIZE", str(DEFAULT_POOL))),
            work_dir=e("RALPH_WORK_DIR", "/workspace/ralph-v2-work"),
        )


@dataclass
class Check:
    ok: bool
    name: str
    detail: str = ""
    fatal: bool = True


@dataclass
class Preflight:
    checks: list = field(default_factory=list)

    def add(self, ok, name, detail="", fatal=True):
        self.checks.append(Check(bool(ok), name, detail, fatal))
        return ok

    @property
    def fatal_failures(self):
        return [c for c in self.checks if not c.ok and c.fatal]

    def report(self, out=sys.stdout) -> bool:
        for c in self.checks:
            mark = "ok  " if c.ok else ("FAIL" if c.fatal else "warn")
            out.write(f"  {mark}  {c.name}\n")
            if c.detail:
                out.write(f"          {c.detail}\n")
        bad = self.fatal_failures
        out.write(f"\n  preflight: {len(self.checks) - len(bad)}/{len(self.checks)} "
                  f"{'PASS' if not bad else 'BLOCKED'}\n")
        return not bad


def preflight(cfg: Config) -> Preflight:
    """Every check that can be made without loading a model. Cheap, and it runs first."""
    p = Preflight()

    tok = os.environ.get("HF_TOKEN", "")
    p.add(bool(tok), "HF token present",
          "" if tok else "set HF_TOKEN — the publisher refuses to construct without one, because "
                         "a 401 discovered after scoring wastes the whole round")

    from .parent import PARENTS
    spec = PARENTS.get(cfg.parent_key)
    p.add(spec is not None, "pinned parent known",
          f"{spec.name} · {spec.weight_params:,} params" if spec
          else f"{cfg.parent_key!r} is not in eval.parent.PARENTS")

    from .bitrate import TIERS
    known = {t.name for t in TIERS}
    missing = [t for t in cfg.tiers if t not in known]
    p.add(not missing, "bit tiers exist",
          f"unknown tiers {missing}; known: {sorted(known)}" if missing else " · ".join(cfg.tiers))

    # THE SILENT ONE. TierBudget.bit_tier and .parent default to None and intake skips both
    # without a word — the measured bit budget and the pinned-parent check, which are the two
    # gates that make this a compression subnet rather than a general model contest.
    budgets = build_budgets(cfg) if spec else {}
    unarmed = [n for n, b in budgets.items() if b.bit_tier is None or b.parent is None]
    p.add(bool(budgets) and not unarmed, "all six intake gates armed",
          f"tiers with a gate disabled: {unarmed}" if unarmed
          else "economics · safety · tier fit · bit budget · pinned parent · commit-reveal")

    # arithmetic that decides whether the round can score at all
    import inspect
    from .observer_kl import score_miner
    floor = inspect.signature(score_miner).parameters["min_per_slice"].default
    from .pool import DEFAULT_MIX, LANG_OF_SOURCE
    langs = {LANG_OF_SOURCE[n] for n, _ in DEFAULT_MIX}
    slices = len(langs) * 2                      # language x depth
    need = slices * floor
    p.add(cfg.n_items >= need, "round is big enough to fill its slices",
          f"n_items={cfg.n_items}, {slices} slices x floor {floor} = {need} minimum")
    p.add(len(langs) >= 2, "pool is multilingual",
          f"languages: {sorted(langs)} — a single-language pool makes worst-slice a no-op and "
          f"hands the crown to a downloaded artifact")

    p.add(len(cfg.observers) >= 2, "observer pool has choices",
          f"{len(cfg.observers)} observers; one makes the post-commit draw meaningless", fatal=False)
    same_family = [o for o in cfg.observers
                   if spec is not None and o.split("/")[0] == spec.name.split("/")[0]]
    # FATAL, not a warning. The metric is "did you move an INDEPENDENT model the same way the
    # parent did". An observer from the parent's family is not independent, and the failure is
    # silent — the scores look fine, they are just blind in the same places the parent is.
    p.add(not same_family, "observers are independent of the parent",
          f"{same_family} share the parent's family ({spec.name.split('/')[0] if spec else '?'}) — "
          f"pick observers from other model families"
          if same_family else " · ".join(o.split("/")[-1] for o in cfg.observers))

    try:
        os.makedirs(cfg.work_dir, exist_ok=True)
        free = os.statvfs(cfg.work_dir)
        gb = free.f_bavail * free.f_frsize / 1e9
        p.add(gb > 80, "disk headroom", f"{gb:.0f} GB free at {cfg.work_dir} (parent + artifacts)")
    except Exception as e:
        p.add(False, "work dir usable", f"{cfg.work_dir}: {e}")

    try:
        import torch
        cuda = torch.cuda.is_available()
        p.add(cuda, "GPU available",
              torch.cuda.get_device_name(0) if cuda
              else "no CUDA device — the validator runs the parent AND the observer every round; "
                   "on the CPU-orchestrator design this step runs on a rented box")
    except Exception:
        p.add(False, "torch importable", "pip install torch")

    p.add(not cfg.live, "chain writes disabled",
          "LIVE: this round WILL set weights" if cfg.live
          else "read-only — scores, publishes and anchors without touching the chain", fatal=False)
    return p


def build_budgets(cfg: Config) -> dict:
    """Tier budgets with BOTH optional gates armed. Constructing these anywhere else risks the
    silent-skip: a bare TierBudget runs four gates while the operator believes it runs six."""
    from .bitrate import TIERS
    from .gates import TierBudget
    from .parent import PARENTS
    spec = PARENTS[cfg.parent_key]
    by_name = {t.name: t for t in TIERS}
    out = {}
    for name in cfg.tiers:
        bt = by_name[name]
        out[name] = TierBudget(
            name=name,
            max_params=int(spec.weight_params * (1 + spec.tol)),
            max_effective_bits=bt.max_container_bits,
            bit_tier=bt,
            parent=spec,
        )
    return out


def run(cfg: Config, round_no: int = 1) -> int:
    """One round. Returns a process exit code."""
    pf = preflight(cfg)
    if not pf.report():
        print("\n  refusing to run — fix the blocked checks above\n")
        return 2

    from .bitrate import TIERS
    from .chain_bittensor import BittensorChainIO
    from .economics import RegistrationLedger
    from .fetch import resolver
    from .koth import Tier, Tournament
    from .parent import PARENTS
    from .pool import build_pool, check_balance
    from .publish import HFSink, RecordPublisher
    from .runners import HFRunner, SafeStudentRunner
    from .chain import run_v2_observer_epoch

    spec = PARENTS[cfg.parent_key]
    print(f"\n  building pool ({cfg.pool_size} trajectories)…")
    pool, pspec = build_pool(cfg.pool_size)
    ok, why = check_balance(pool)
    if not ok:
        for r in why:
            print(f"  FAIL  pool: {r}")
        return 2
    print(f"  pool: {pspec.as_corpus_spec()}")

    print(f"  loading parent {spec.name}…")
    parent = HFRunner(spec.name)
    # KEYED BY FULL REPO ID. The manifest records this key as the round's observer, and an
    # auditor has to name a downloadable repo to re-derive the grades — a bare model name is
    # not one. The nonce draw is over sorted(keys) either way.
    observers = {o: HFRunner(o) for o in cfg.observers}
    print(f"  observers: {list(observers)}")

    from .bitrate import emission_weight

    budgets = build_budgets(cfg)
    # Difficulty-weighted, not an equal split — see bitrate.TIER_EMISSION_WEIGHT for why an equal
    # split predicts exactly the field the first two rounds drew.
    tiers = [Tier(n, max_params=budgets[n].max_params,
                  weight=emission_weight(n, len(cfg.tiers)))
             for n in cfg.tiers]

    fetch_log: list = []
    chain = BittensorChainIO(netuid=cfg.netuid, network=cfg.network, wallet_name=cfg.wallet,
                             hotkey_name=cfg.hotkey, read_only=not cfg.live)
    chain.fetch_dir_for = resolver(os.path.join(cfg.work_dir, "artifacts"),
                                   reveals=chain.reveals, log=fetch_log)

    publisher = RecordPublisher(HFSink(cfg.records_repo), window=8,
                                state_path=os.path.join(cfg.work_dir, "publish-hwm.json"))

    print(f"\n  running round {round_no} (live={cfg.live})…")
    res = run_v2_observer_epoch(
        chain, round_no, pool, parent, observers, tiers, budgets,
        Tournament(tiers, margin=0.05), RegistrationLedger(), {},
        make_safe_runner=lambda cd: SafeStudentRunner(cd),
        signer=_signer(cfg), n_items=cfg.n_items,
        corpus_spec=pspec.as_corpus_spec(), publisher=publisher,
    )
    o = res.outcome
    print(f"  identity   : {o.identity.get('score')}")
    print(f"  accepted   : {len(o.accepted)}   rejected: {len(o.rejected)}")
    for hk, reasons in o.rejected[:5]:
        print(f"    reject {hk[:12]}…: {reasons[:1]}")
    for entry in fetch_log[:5]:
        print(f"    fetch  {entry}")
    print(f"  events     : {[e.get('action') for e in o.events]}")
    print(f"  published  : {res.publish.published.uri if res.publish and res.publish.published else None}")
    print(f"  weights set: {res.weights_set}")
    if res.withheld:
        print(f"  WITHHELD   : {res.withheld['reason']}")
        return 1
    return 0


def _signer(cfg: Config):
    """The record signer. Distinct from the chain wallet on purpose — a record is signed so it can
    be attributed, which is a different job from authorising a weight write."""
    from .signing import Ed25519Signer
    seed = os.environ.get("RALPH_RECORD_SEED", "")
    if not seed:
        return None
    return Ed25519Signer(seed=bytes.fromhex(seed)[:32])


def main(argv) -> int:
    cfg = Config.from_env()
    cfg.live = "--live" in argv
    if "--dry-run" in argv:
        return 0 if preflight(cfg).report() else 2
    return run(cfg)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
