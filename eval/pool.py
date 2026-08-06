"""Build the round's trajectory pool — LANGUAGE-BALANCED, because that is the anti-clone axis.

WHY THIS FILE EXISTS. Pinning Qwen3-8B means pinning the same parent PrismML compress, which makes
the cheapest attack on this subnet concrete rather than theoretical: download `prism-ml/Bonsai-8B`,
submit it, and it passes every gate. Same architecture as the parent. Genuinely ~1.71 bpw when
measured from the tensor data. Shipped as GGUF, which intake accepts. Apache-2.0, free, sixty
seconds — against tens of thousands of dollars of honest work.

Gates cannot catch that, and they should not try: it IS a real low-bit compression of the real
parent. The only thing that separates it from honest work is WHERE IT IS WEAK. A cloned artifact
arrives already broken on non-Latin text — independent testing measured Persian collapsing from
79.8% to 45.2% at 1-bit while English and coding held at 97-100%, and the published benchmark suite
it was tuned against contains zero non-Latin coverage.

So the crown has to be decided somewhere a clone loses. The scoring path already supports it:
samples are aggregated WORST-SLICE over (observer x language x depth), so a submission cannot buy a
weak language with a strong one. What was missing is that every harness built its pool from
`glaive_r1` alone, which is English — the language slice existed and only ever had one value in it.

A single-language pool makes the worst-slice aggregation a no-op and hands the first crown to
whoever downloads the fastest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Source -> language. `observer_round._lang_of` derives the slice key from the source name, so this
# table and that function must agree; the test asserts they do.
LANG_OF_SOURCE = {
    "glaive_r1": "en",
    "omr_tir": "en",
    "omr_cot": "en",
    "mini_coder": "en",
    "nemotron_tools": "en",
    "swe_zero": "en",
    "agenttrove": "en",
    "samvaad_hi": "hi",
    "zh_reasoning": "zh",
}

# The production mix. Non-Latin is a THIRD of the pool, not a garnish: worst-slice means the
# smallest live slice decides the crown, so a language present in token quantities would be
# excluded by the per-slice sample floor and the axis would silently stop binding.
DEFAULT_MIX = (
    ("glaive_r1", 0.34),
    ("omr_tir", 0.20),
    ("mini_coder", 0.13),
    ("samvaad_hi", 0.17),
    ("zh_reasoning", 0.16),
)

# Below this many samples a slice is dropped by score_miner's floor, so a language that lands under
# it is not protecting anything. Kept above the scorer's own min_per_slice for headroom.
MIN_PER_LANG = 12


@dataclass
class PoolSpec:
    """What was drawn, in a form the signed record can carry. An index into a pool is meaningless
    without the ordering that produced it."""
    mix: tuple = DEFAULT_MIX
    revision: str = "main"
    dedup: str = "per-source defaults"
    languages: dict = field(default_factory=dict)
    n: int = 0

    def as_corpus_spec(self) -> str:
        parts = "+".join(f"{name}:{frac:.2f}" for name, frac in self.mix)
        langs = ",".join(f"{k}={v}" for k, v in sorted(self.languages.items()))
        # The clamp changes the bytes an index points at, so it belongs in the spec the record
        # pins — an auditor rebuilding the pool without it gets different prefixes.
        return (f"{parts}@rev={self.revision}|dedup={self.dedup}|langs={langs}|n={self.n}"
                f"|maxprefix={MAX_PREFIX_TOKENS}@{PREFIX_TOKENIZER}|side={PREFIX_TRUNCATION_SIDE}")


# THE EXAM'S CONTEXT CEILING. Set by the WEAKEST admissible runtime, not by the parent's: GGUF is
# the only format that can pass the bit tiers, it runs under llama.cpp, and llama.cpp RAISES rather
# than truncates past its window. A real round died mid-scoring on
# `Requested tokens (5990) exceed context window of 4096`.
#
# CLAMPED, NOT FILTERED, and at BUILD time. Filtering by length deletes whole strata — measured on
# the real pool, a 3,000-character bound removed 267 of 900 items and the entire en/deep slice,
# which is precisely the axis a cloned artifact cannot fake. Clamping keeps every item and every
# stratum, and costs only the head of the longest 7%.
#
# BUILD TIME, because five consumers read the prefix — the parent's generate, both of the
# observer's `distributions` calls, the miner's generate, and `_freeze`. Clamping inside a runner
# would hand them different K: the step would be conditioned on a clamped prefix while P_0 was
# conditioned on the full one, and the KL would compare distributions over different conditions.
# Doing it here also means `dump_pool` publishes exactly what the GPU saw.
#
# LEFT, because these prefixes end where the step begins. Right-truncation would delete the only
# context that predicts the thing being generated.
# BOUNDED BY THE TIGHTEST WINDOW OF EVERY MODEL THAT READS THE TEXT — and the observer's is
# tighter than the student's, which I got wrong once. The student runs llama.cpp at N_CTX=8192, but
# the OBSERVER is fed `prefix + step + continuation` and two of the three in the pool
# (Phi-3-mini-4k, OLMo-2) stop at 4096 positions. Overflow there does not raise: a RoPE model
# extrapolates and returns degraded distributions, which feed straight into the KL. Silent
# wrongness in the one number that decides crowns.
#
#     4096 (tightest observer) - 256 (max_step_tokens) - 128 (max_cont_tokens) = 3712
#
# 3500 leaves room for chat-template tokens, which `_render` adds and this count does not see.
# `preflight` refuses any observer whose window cannot hold it, so adding a short-context observer
# fails before a cent is spent rather than silently degrading a round.
MAX_PREFIX_TOKENS = 3500
PREFIX_TOKENIZER = "Qwen/Qwen3-8B"
PREFIX_TRUNCATION_SIDE = "left"


def clamp_prefixes(trajectories, max_tokens: int = MAX_PREFIX_TOKENS,
                   tokenizer_id: str = PREFIX_TOKENIZER, tok=None):
    """Left-clamp every prefix to `max_tokens`. Idempotent, and a no-op without a tokenizer.

    Degrades to unclamped rather than crashing when transformers is absent: the pool is built in
    CPU tests and on the orchestrator, neither of which needs a tokenizer to exercise the rest."""
    if tok is None:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_id)
        except Exception:
            return trajectories
    tok.truncation_side = PREFIX_TRUNCATION_SIDE
    for t in trajectories:
        pre = getattr(t, "prefix", "") or ""
        if not pre:
            continue
        ids = tok(pre, add_special_tokens=False)["input_ids"]
        if len(ids) > max_tokens:
            t.prefix = tok.decode(ids[-max_tokens:], skip_special_tokens=True)
    return trajectories


POOL_FIELDS = ("id", "source", "prefix", "step", "index", "meta")


def dump_pool(trajectories) -> bytes:
    """The pool as the JSONL an auditor loads, in the pool's own order.

    L1 asks whether the operator chose the exam. Answering it requires the POOL the indices point
    into — and until this existed the pool was built in memory, used, and thrown away, so no third
    party could run L1 at all no matter how much they wanted to. `corpus_spec` names the sources
    and revisions, which lets a determined auditor rebuild an equivalent pool; this lets an
    ordinary one just fetch it.

    Order is the identity and must never be sorted: the record stores integer indices."""
    import json
    from dataclasses import asdict
    out = []
    for t in trajectories:
        row = {k: v for k, v in asdict(t).items() if k in POOL_FIELDS}
        out.append(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return ("\n".join(out) + "\n").encode()


def pool_digest(trajectories) -> str:
    """sha256 of `dump_pool`. Goes in the manifest, so it is inside the SIGNED body — which is what
    stops the operator serving one pool to the round and a friendlier one to the auditor."""
    import hashlib
    return hashlib.sha256(dump_pool(trajectories)).hexdigest()


def language_balance(trajectories) -> dict:
    """Count trajectories per language slice, using the same derivation the scorer uses."""
    from .observer_round import _lang_of
    out: dict = {}
    for t in trajectories:
        lang = (t.meta or {}).get("lang") or _lang_of(t.source)
        out[lang] = out.get(lang, 0) + 1
    return out


def check_balance(trajectories, min_per_lang: int = MIN_PER_LANG) -> tuple[bool, list]:
    """Fail-closed on a pool that cannot exercise the anti-clone axis.

    Two failure modes, both silent without this: a pool that is entirely one language (worst-slice
    degenerates to plain mean), and a pool where a language is present but too thin to survive the
    scorer's per-slice sample floor (the slice is dropped, and the aggregate quietly stops covering
    it). Either one hands the crown to a downloaded artifact."""
    bal = language_balance(trajectories)
    reasons = []
    non_latin = {k: v for k, v in bal.items() if k not in ("en",)}
    if len(bal) < 2:
        reasons.append(f"pool is single-language ({bal}) — worst-slice aggregation is a no-op and "
                       f"a cloned artifact loses nothing")
    if not non_latin:
        reasons.append("no non-Latin trajectories — this is the axis a clone is known to fail, "
                       "and the pool does not test it")
    for lang, n in sorted(bal.items()):
        if n < min_per_lang:
            reasons.append(f"language {lang!r} has {n} trajectories, below the {min_per_lang} "
                           f"needed to survive the per-slice floor — the slice will be dropped")
    return (not reasons), reasons


def build_pool(n: int, mix=DEFAULT_MIX, max_steps: int = 2, revision: str = "main",
               loader=None) -> tuple[list, PoolSpec]:
    """Draw `n` trajectories across `mix`, in a deterministic, recorded order.

    `loader(source_name, want) -> list[Trajectory]` is injectable so this is testable without the
    network; production passes the streaming HF loader. Sources are drawn in the fixed order of
    `mix` and concatenated — NOT interleaved randomly — because the record stores integer indices
    into this pool and an index is only meaningful against a reproducible ordering.
    """
    if loader is None:
        loader = _hf_loader
    out: list = []
    for name, frac in mix:
        want = max(1, round(n * frac))
        got = loader(name, want, max_steps)
        out.extend(got[:want])
    out = clamp_prefixes(out)
    spec = PoolSpec(mix=tuple(mix), revision=revision, n=len(out))
    spec.languages = language_balance(out)
    return out, spec


def _hf_loader(name: str, want: int, max_steps: int) -> list:
    """Streaming loader for one source. Verifies the source actually yields steps before trusting
    it — a wrong field name produces an EMPTY list, which is indistinguishable from a quiet source
    and would silently shrink the exam rather than failing it."""
    from datasets import load_dataset
    from .steps import TRAJECTORY_SOURCES, extract_steps

    spec = TRAJECTORY_SOURCES[name]
    ds = load_dataset(spec["dataset"], split=spec["split"], streaming=True)
    trajs, seen, probed = [], 0, []
    for row in ds:
        seen += 1
        if len(probed) < 8:
            probed.append(row)
        trajs += extract_steps(row, spec["step"], spec["text_field"], name,
                               f"{name}-r{seen}", max_steps=max_steps)
        if len(trajs) >= want:
            break
    if not trajs:
        raise RuntimeError(
            f"source {name!r} yielded zero steps from {seen} rows — the field name or step "
            f"boundary is wrong, and an empty source silently shrinks the exam instead of "
            f"failing the round")
    return trajs
