"""Operator-runnable shadow on the AXIS substrate — the full v2 epoch, chain faked.

This is what an operator runs to see the REAL crown substrate end to end: a FakeChain
supplies committed submissions and the round nonce, and everything downstream is the
production path — intake (economics + safety + tier + commit-reveal), axis_round over the
deterministic-checker cover, the genre-overfit crown precondition, bond settlement, a
SIGNED record, and weights. Swapping FakeChain for the karpa validator's ChainIO is the
only step to a live shadow; nothing else changes.

    python -m eval.shadow_axis_epoch      # real HF models on a GPU; env-configurable

Env: RALPH_TEACHER, RALPH_BASE, RALPH_STUDENTS (comma-sep), RALPH_ITEMS.
"""
from __future__ import annotations

import os

from .axes.code_exec import CodeExec
from .axes.doc_tasks import ConstrainedExtraction, LongContextReal, NumericComposition
from .axes.extractive import ExtractiveQA
from .axes.instruction import InstructionFollowing
from .axes.long_context import LongContext
from .axes.math_gsm import MathGSM
from .axes.multi_constraint import MultiConstraint
from .axes.multihop import MultiHop
from .axis_round import AxisSpec
from .chain import Commitment, run_v2_axis_epoch
from .economics import RegistrationLedger
from .gates import TierBudget
from .identity import commit_value, content_hash
from .koth import Tier, Tournament
from .signing import Ed25519Signer

TEACHER = os.environ.get("RALPH_TEACHER", "THUDM/glm-4-9b-chat-hf")
BASE = os.environ.get("RALPH_BASE", "Qwen/Qwen2.5-0.5B-Instruct")
STUDENTS = os.environ.get(
    "RALPH_STUDENTS", "Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-1.5B-Instruct"
).split(",")
ITEMS = int(os.environ.get("RALPH_ITEMS", "150"))  # dethrone power is PER AXIS; 64 needs ~+0.25 to dethrone


class FakeChain:
    """Minimal ChainIO: hands back the committed submissions + a fixed nonce, records the
    write-back so the operator can inspect the crown/weights/record. NO real chain."""

    def __init__(self, commits):
        self._commits = commits
        self.weights = None
        self.kings = {}
        self.record = None

    def current_block(self):
        return 1000

    def block_hash(self, block):
        return f"nonce-block-{block}"

    def read_commitments(self, lo, hi):
        return self._commits

    def commit_root(self, lo, hi):
        return "commit-root"

    def set_weights(self, weights):
        self.weights = weights
        return True

    def set_king(self, tier, hotkey, model_id):
        self.kings[tier] = (hotkey, model_id)

    def get_king(self, tier):
        return self.kings.get(tier)

    def blacklist(self, hotkey, reason):
        pass

    def publish_record(self, record):
        self.record = record

    def settle_bonds(self, refunds):
        pass


def _tier_specs(fresh_docs, code_sandbox, ml_docs=()):
    # THE PIVOT (const's SN97 fix). CROWN = real∩verifiable capability retention on fresh,
    # post-commit real documents (extractive QA, answer is a verbatim span, exact-match, no
    # judge) — carries information a miner cannot pre-distill. FLOOR = the synthetic
    # generators, demoted to liveness/calibration/fragile-capability probes: they gate
    # (degeneracy + no-negative) but do NOT set the crown, because a specialist can ace a
    # public generator offline without retaining GLM. code axis executes untrusted output ->
    # sandbox REQUIRED (RALPH_CODE_SANDBOX=auto only on a trusted dev box without bwrap).
    # CROWN POOL. Six formats, each stressing a DIFFERENT mechanism over real documents, so
    # covering the pool is general capability rather than one narrow skill in six costumes —
    # that is the property that makes surprise selection (draw k post-commit) a real defense
    # instead of a gesture. Each is chosen against where compression measurably breaks, not by
    # difficulty ("quantization amplifies existing weaknesses" — difficulty is the wrong axis).
    crown = [
        AxisSpec(ExtractiveQA(fresh_docs), "extractive", 1.0, difficulty=1, role="crown"),
        AxisSpec(NumericComposition(fresh_docs), "numeric_composition", 1.0, difficulty=1,
                 role="crown"),
        AxisSpec(LongContextReal(fresh_docs), "long_context_real", 1.0, difficulty=1,
                 role="crown"),
        AxisSpec(ConstrainedExtraction(fresh_docs), "constrained_extraction", 1.0, difficulty=1,
                 role="crown"),
    ]
    if ml_docs:
        # the incumbent's blind spot: zero multilingual in PrismML's 15-benchmark suite while
        # independent testing measured Persian collapsing 79.8% -> 45.2% at 1-bit. Scored as its
        # OWN axis so a strong English model cannot average it away.
        crown.append(AxisSpec(ExtractiveQA(ml_docs, kinds=("number",)), "multilingual", 1.0,
                              difficulty=1, role="crown"))
    return crown + [
        # second CROWN axis: compounding verifiable constraints. This is where compression
        # measurably breaks (PrismML's own numbers: IFBench -15.7 vs MATH-500 -1.4), so it is
        # both the discriminative axis and the hard-to-Goodhart one — the score moves when
        # capability moves. Deterministic parsers, no judge.
        AxisSpec(MultiConstraint(), "multi_constraint", 1.0, difficulty=2, role="crown"),
        AxisSpec(MathGSM(), "math", 1.0, difficulty=3, role="floor"),
        AxisSpec(CodeExec(sandbox=code_sandbox), "code", 1.0, difficulty=2, items=160, role="floor"),
        AxisSpec(InstructionFollowing(), "instruction", 1.0, difficulty=1, role="floor"),
        AxisSpec(LongContext(base_facts=12), "long_context", 1.0, difficulty=1, role="floor"),
        AxisSpec(MultiHop(), "multihop", 1.0, difficulty=1, role="floor"),
    ]


def _tier_ladder():
    """Size/bit rungs the crown is contested on, and the emission split across them.

    Two axes of pressure, deliberately: PARAMS (how big a model) and BITS (how tightly stored).
    Bonsai's own product line is exactly two operating points on the bit axis — 1.125 for phone
    footprint, 1.71 for laptop quality — which exists because a single scalar cannot express the
    tradeoff. `bit_tier` is what makes intake measure the tensor DATA rather than dtype headers,
    so a 1.125-bit model in a BF16 container is credited for its compression and still rejected
    as unshippable.

    RALPH_TIER selects a single rung for a cheap shadow run."""
    from .bitrate import BitTier
    rungs = [
        # name        params      weight  code bpw   container cap
        ("ternary-4b", 4_500_000_000, 0.40, 1.75, 2.6),
        ("binary-4b", 4_500_000_000, 0.35, 1.15, 2.0),
        ("sub4-4b", 4_500_000_000, 0.25, 4.25, 5.0),
    ]
    only = os.environ.get("RALPH_TIER")
    if only:
        rungs = [r for r in rungs if r[0] == only] or rungs[:1]
    total = sum(r[2] for r in rungs)
    tiers = [Tier(n, p, w / total) for n, p, w, _, _ in rungs]
    budgets = {
        n: TierBudget(name=n, max_params=p, max_effective_bits=16.0,
                      bit_tier=BitTier(n, max_code_bits=cb, max_container_bits=cc))
        for n, p, _, cb, cc in rungs
    }
    return tiers, budgets


def _load_corpus(seed):
    """Real docs drawn from the LARGE multi-source pool, seeded by the round nonce.

    Scale is the anti-overfit defense: 400 docs out of ~2.8e9 reachable rows is a ~1e-7
    sampling rate, and the seed picks the crawl snapshot + shard order, so the slice is
    unknowable before the (post-commit) seed exists. Same seed -> same docs, so an auditor
    re-derives it; corpus_fingerprint goes in the round record to prove which slice ran.
    Falls back to synthetic ONLY so the shadow runs offline — never in production."""
    n = int(os.environ.get("RALPH_CORPUS_N", "400"))
    try:
        from .corpus_stream import corpus_fingerprint, load_stream_slice, pool_size
        docs = load_stream_slice(seed, n=n, strict=True)
        if not docs:
            raise RuntimeError("empty corpus slice")
        from .corpus_hf import median_commit_ts
        return (docs, median_commit_ts(docs),
                f"{len(docs)} real docs from a {pool_size():,}-row pool "
                f"(fingerprint {corpus_fingerprint(docs)[:12]})")
    except Exception as e:
        # FAIL CLOSED. Silently substituting synth_corpus turns the real-corpus crown back into
        # the closed, enumerable generator the entire pivot exists to escape — and still crowns,
        # on one printed line of warning. An HF outage must stop the round, not quietly change
        # what the subnet is measuring. RALPH_SYNTH_CORPUS=1 is a dev escape hatch only.
        if not os.environ.get("RALPH_SYNTH_CORPUS"):
            raise RuntimeError(
                f"real corpus unavailable ({e}). Refusing to crown on synthetic text — that is "
                f"the closed generator this design exists to avoid. Set RALPH_SYNTH_CORPUS=1 "
                f"for offline development only."
            ) from e
        from .corpus import synth_corpus
        docs = synth_corpus(n, seed=20260724, commit_ts=1000, span=200)
        return docs, 1000, f"{len(docs)} SYNTHETIC docs — DEV ONLY, NOT CROWNABLE ({e})"


def main() -> int:
    from .runners import HFRunner, SafeStudentRunner

    tiers, budgets = _tier_ladder()
    TIER_NAME = tiers[0].name        # shadow submits into the first rung


    # miners commit real checkpoints; here they are off-the-shelf HF ids resolved to a local
    # dir with a proper commit-reveal so the epoch's fetch-verify runs for real.
    from huggingface_hub import snapshot_download
    commits, runners = [], {}
    for i, mid in enumerate(STUDENTS):
        d = snapshot_download(mid, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"])
        h, salt = content_hash(d), f"salt{i}"
        commits.append(Commitment(hotkey=f"hot{i}", coldkey=f"cold{i}", tier=TIER_NAME,
                                   ckpt_dir=d, declared_compute_h100h=1.0,
                                   revealed_hash=h, salt=salt, committed_value=commit_value(h, salt)))
        runners[d] = mid

    glm = HFRunner(TEACHER, name="glm", trust_remote_code=True, batch_size=16)
    base = HFRunner(BASE, name="base", batch_size=16)
    # (ladder built above)
    # A REAL LADDER. The placeholder was Tier("open", 10**12) + max_effective_bits=32.0 — a cap
    # ABOVE fp16, i.e. no budget at all, which pointed the incentive away from the product: an
    # uncompressed model beat a 1.125-bit one on every axis and paid nothing for its size.
    # Emission is split across rungs so there is somewhere to win by compressing harder rather
    # than only by being bigger. Bit budgets are the two operating points PrismML ship.
    chain = FakeChain(commits)

    # real corpus: FRESH (post-commit) docs bear the crown axis; the genre-overfit gate runs
    # stale-vs-fresh over the whole set. Both are ON by default now that they are the load-
    # bearing anti-overfit defense (RALPH_OVERFIT_OFF=1 disables the gate only).
    from .corpus import split_by_commit
    # the corpus slice is seeded by the round nonce: in production this is post-commit chain
    # entropy, so which documents get scored cannot be known when the miner seals its weights.
    seed_str = f"{chain.commit_root(0, 0)}|{chain.block_hash(chain.current_block())}"
    docs, cut, desc = _load_corpus(seed_str)
    stale, fresh = split_by_commit(docs, cut)
    print(f"corpus: {desc}; {len(stale)} stale / {len(fresh)} fresh (cut={cut})")
    code_sandbox = os.environ.get("RALPH_CODE_SANDBOX", "require")

    # non-Latin slice for the multilingual crown axis (optional: skipped cleanly if offline)
    # NB: import the callables HERE. They were previously only imported inside _load_corpus,
    # so this block raised NameError on every run — swallowed by a broad `except Exception` and
    # printed as "corpus unavailable", silently disabling the one axis a cloned artifact loses
    # on. A bare except around a whole block turns a coding bug into a plausible-looking
    # operational message; keep the try around the FETCH only.
    from .corpus_stream import MULTILINGUAL_MIX, load_stream_slice as _load_slice, pool_size
    ml_docs = []
    try:
        ml_docs = _load_slice(f"{seed_str}|ml", n=int(os.environ.get("RALPH_ML_N", "150")),
                              mix=MULTILINGUAL_MIX, strict=True)
    except Exception as e:
        if not os.environ.get("RALPH_ML_OFF"):
            raise RuntimeError(
                f"multilingual corpus unavailable ({e}). This is the anti-clone axis; running "
                f"without it crowns on axes a cloned model wins. Set RALPH_ML_OFF=1 to override."
            ) from e
        print(f"multilingual axis DISABLED by RALPH_ML_OFF ({e})")
    if ml_docs:
        print(f"multilingual: {len(ml_docs)} non-Latin docs "
              f"({pool_size(MULTILINGUAL_MIX):,}-row pool)")

    overfit_check = None
    if not os.environ.get("RALPH_OVERFIT_OFF"):
        from .axis_round import compose_preconditions
        from .invariance import make_invariance_check
        from .overfit_gate import make_overfit_check

        dind = make_overfit_check(docs, cut, glm, base, min_n=20)
        dind.gate_name = "genre_overfit"
        # CORPUS-SWAP INVARIANCE: the miner picks its own recovery corpus, so score the student
        # on probe sets from genuinely DIFFERENT origins and flag source-dependent retention.
        # Sources must be different origins, not two slices of one distribution — swapping
        # within a distribution tests nothing.
        by_source: dict = {}
        for d in fresh:
            by_source.setdefault(d.id.split(":")[0], []).append(d)
        inv = None
        if len(by_source) >= 2:
            inv = make_invariance_check(by_source, glm, base)
            inv.gate_name = "corpus_swap"
            print(f"corpus-swap invariance armed over {len(by_source)} sources: "
                  f"{ {k: len(v) for k, v in by_source.items()} }")
        else:
            print(f"corpus-swap invariance NOT armed — only {len(by_source)} source(s) in the "
                  f"fresh slice; it needs >= 2 distinct origins")
        overfit_check = compose_preconditions(dind, inv)

    specs = _tier_specs(fresh, code_sandbox, ml_docs)
    n_crown = sum(1 for s in specs if s.role == "crown")
    # Surprise selection only defends in proportion to the pool: with a small crown pool it
    # just halves the signal for nothing. Arm it once the pool is big enough to be worth
    # hiding, and draw roughly half.
    surprise_k = int(os.environ.get("RALPH_SURPRISE_K", "0")) or (
        max(2, n_crown // 2) if n_crown >= 4 else 0)
    print(f"crown pool: {n_crown} axes; surprise_k={surprise_k or 'off'} "
          f"({'drawn post-commit' if surprise_k else 'all score'})")

    result = run_v2_axis_epoch(
        chain, 1, specs, glm, base, tiers, budgets,
        Tournament(tiers, margin=0.03), RegistrationLedger(), {},
        make_safe_runner=lambda cd: SafeStudentRunner(cd, name=runners[cd].split("/")[-1]),
        items_per_axis=ITEMS, overfit_check=overfit_check, surprise_k=surprise_k,
        signer=Ed25519Signer(seed=b"shadow-operator-key-000000000000"[:32]),
    )

    print("\n===== SHADOW AXIS EPOCH =====")
    print("accepted:", result.outcome.accepted)
    print("rejected:", result.outcome.rejected)
    print("weights:", chain.weights)
    print("crown:", chain.kings)
    rec = chain.record
    print(f"record signed+valid: {rec is not None and rec.verify_signature()} "
          f"(signer {rec.signer[:12] if rec else '-'}..., sha {result.record_sha256[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
