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
        return f"{parts}@rev={self.revision}|dedup={self.dedup}|langs={langs}|n={self.n}"


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
