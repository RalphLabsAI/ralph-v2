"""Observer-KL step scoring — semantic equivalence by downstream effect.

THE IDEA (const, 2026-07-27). Two reasoning steps are equivalent when they move an INDEPENDENT
observer model into the same predictive state. So instead of comparing the miner's words to the
teacher's words, append each step to the same trajectory prefix and ask the observer what it now
expects. Same expectations -> the same information was added, however differently it was phrased.

WHY THIS IS THE RIGHT METRIC HERE, when teacher-KL was the wrong one on SN97. Scoring
student-vs-teacher token agreement rewards STYLE: their KL-crowned king mimicked "wait, let me
reconsider" filler, never answered, and was worse than the un-finetuned base on 5/5 benchmarks.
Downstream-effect scoring has no style channel at all — matching the teacher's wording earns
nothing unless it carries the same information into the observer. And it needs no generated
questions, which is the objection that killed our previous design: probe formats are public, so
answering probes is a narrow skill a miner can fit cheaply.

THE THREE QUANTITIES. Const's spec compares "RollGLMKimi" with "RollAKimi", but those are
different token sequences, so no position-wise KL exists between them. Resolved by scoring a
COMMON continuation C under both prefixes — same vocabulary, same positions, one forward pass
each, no sampling:

    P_G[t] = P_obs(. | K, step_teacher, C[:t])      teacher-conditioned
    P_A[t] = P_obs(. | K, step_miner,   C[:t])      miner-conditioned
    P_0[t] = P_obs(. | K,               C[:t])      unconditioned baseline

    s   = mean_t KL(P_G[t] || P_A[t])     disagreement       -> low is good
    d_G = mean_t KL(P_G[t] || P_0[t])     teacher effect size
    d_A = mean_t KL(P_A[t] || P_0[t])     miner effect size

"Effect similarity" is low `s`; "effect magnitude" is low |d_A - d_G|. A miner that adds nothing
has d_A ~ 0 against a large d_G and is penalised by the magnitude term automatically.

THE DISCARD RULE IS AN ATTACK SURFACE, and this is the one place the spec needed hardening. If a
sample is dropped when NEITHER step moved the observer, a miner emitting bland steps gets its
hard samples discarded and is scored on the easy remainder. So discards are decided by `d_G`
ALONE — the teacher's effect, which the miner cannot influence. A miner whose own steps are
systematically inert fails a liveness check instead of quietly benefiting.

Pure logic: distributions come in as sparse {token: prob} maps, so this is CPU-testable with no
models and an auditor can recompute a verdict from the recorded distributions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Sparse top-k support: real observers return top-k logprobs, not a full 150k-way vocabulary.
# Mass outside the shared support is folded onto this floor so KL stays finite and a token the
# other side never proposed is penalised rather than ignored.
PROB_FLOOR = 1e-6
# Below this the TEACHER barely moved the observer, so the sample carries no signal about
# whether the miner reproduced it. Judged on the teacher only — never on the miner.
MIN_TEACHER_EFFECT = 0.05


def kl(p: dict, q: dict, floor: float = PROB_FLOOR) -> float:
    """KL(p || q) over sparse distributions, on the union of supports.

    Renormalised over the union so two top-k slices are comparable, and floored so a token p
    proposes but q never did contributes a large finite penalty instead of infinity."""
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    zp = sum(max(p.get(k, 0.0), 0.0) for k in keys) or 1.0
    zq = sum(max(q.get(k, 0.0), 0.0) for k in keys) or 1.0
    total = 0.0
    for k in keys:
        pi = max(p.get(k, 0.0), 0.0) / zp
        if pi <= 0.0:
            continue
        qi = max(max(q.get(k, 0.0), 0.0) / zq, floor)
        total += pi * math.log(pi / qi)
    return max(0.0, total)


@dataclass
class StepEffect:
    """One (trajectory, step) sample scored against the teacher's step."""
    s: float = 0.0            # disagreement between teacher- and miner-conditioned observer
    d_teacher: float = 0.0    # how much the teacher's step moved the observer
    d_miner: float = 0.0      # how much the miner's step moved it
    n_positions: int = 0
    discarded: bool = False
    reason: str = ""

    @property
    def magnitude_gap(self) -> float:
        return abs(self.d_miner - self.d_teacher)

    def as_dict(self) -> dict:
        return {"s": round(self.s, 5), "d_teacher": round(self.d_teacher, 5),
                "d_miner": round(self.d_miner, 5), "gap": round(self.magnitude_gap, 5),
                "n": self.n_positions, "discarded": self.discarded, "reason": self.reason}


def step_effect(p_teacher: list[dict], p_miner: list[dict], p_base: list[dict],
                min_teacher_effect: float = MIN_TEACHER_EFFECT) -> StepEffect:
    """The three quantities over a common continuation.

    Each argument is the observer's per-position distribution over the SAME continuation C,
    conditioned respectively on (K + teacher step), (K + miner step), and K alone."""
    n = min(len(p_teacher), len(p_miner), len(p_base))
    if n == 0:
        return StepEffect(discarded=True, reason="no positions")
    s = sum(kl(p_teacher[t], p_miner[t]) for t in range(n)) / n
    d_t = sum(kl(p_teacher[t], p_base[t]) for t in range(n)) / n
    d_m = sum(kl(p_miner[t], p_base[t]) for t in range(n)) / n
    eff = StepEffect(s=s, d_teacher=d_t, d_miner=d_m, n_positions=n)
    if d_t < min_teacher_effect:
        # The TEACHER's own step barely moved the observer, so this sample cannot tell us
        # whether the miner reproduced it. Decided on the teacher alone: making the discard
        # depend on the miner would let a miner drop its hard samples by emitting inert steps.
        eff.discarded = True
        eff.reason = f"teacher effect {d_t:.4f} < {min_teacher_effect}"
    return eff


def sample_score(eff: StepEffect, alpha: float = 1.0, beta: float = 1.0) -> float:
    """Score in [0, 1] for one non-discarded sample.

    Both terms are RELATIVE TO THE TEACHER'S OWN EFFECT, which matters for two reasons:

      * SCALE INVARIANCE. Raw KL magnitudes vary by orders of magnitude across trajectories —
        a step that resolves an ambiguity moves the observer far more than one that adds a
        detail. Unnormalised, a handful of high-effect samples would dominate the average and
        the metric would mostly measure which trajectories were drawn.
      * DISCRIMINATION. Unnormalised, doing NOTHING scored 0.56 out of 1.0 on measured values,
        because exp(-s) barely moves at KL ~ 0.3. Normalised, an inert step has s = d_G and
        gap = d_G, so it lands at exp(-1)exp(-1) ~ 0.14 — which is what "contributed no
        information" should be worth.

      similarity  exp(-alpha * s / d_G)             — moved the observer the same way
      magnitude   exp(-beta * |d_A - d_G| / d_G)    — moved it about as much

    Multiplicative, so a miner must satisfy BOTH: being right about the direction while
    contributing nothing (or vice versa) does not pay. `d_G` is safely positive because any
    sample with a smaller teacher effect was already discarded."""
    if eff.discarded:
        return 0.0
    d_g = max(eff.d_teacher, 1e-9)
    return math.exp(-alpha * eff.s / d_g) * math.exp(-beta * eff.magnitude_gap / d_g)


@dataclass
class MinerScore:
    score: float = 0.0
    n_scored: int = 0
    n_discarded: int = 0
    mean_effect: float = 0.0        # mean d_miner over scored samples — the liveness signal
    per_slice: dict = field(default_factory=dict)
    # raw per-sample scores per slice. The KOTH dethrone test is a paired bootstrap over these
    # vectors (koth.softmin_lcb_diff), exactly as it was over per-axis vectors — so the crown
    # machinery, the copy-guard and the noise margin all carry over unchanged.
    slice_samples: dict = field(default_factory=dict)
    # per-sample StepEffect objects, in scoring order — the record freezes these so a divergence
    # in a re-run localises to one sample instead of just to the aggregate.
    effects: list = field(default_factory=list)
    # (slice_key, StepEffect) in the same order. The key is what makes the aggregate recomputable
    # from the raw measurements alone.
    keyed_effects: list = field(default_factory=list)
    # The miner's generated steps, kept so the record can freeze them WITHOUT a second generation
    # pass. Re-generating doubled the most expensive leg of the round and gave greedy decode a
    # second chance to diverge from the numbers actually scored.
    steps: list = field(default_factory=list)
    reasons: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"score": round(self.score, 5), "n_scored": self.n_scored,
                "n_discarded": self.n_discarded, "mean_effect": round(self.mean_effect, 5),
                "per_slice": {k: round(v, 5) for k, v in self.per_slice.items()},
                "reasons": self.reasons}


def score_miner(samples: list[tuple[str, StepEffect]], alpha: float = 1.0, beta: float = 1.0,
                min_live_effect: float = 0.02, min_per_slice: int = 8) -> MinerScore:
    """Aggregate WORST-SLICE over (observer x language x step-length) keys.

    Worst-slice rather than mean for the same reason the capability crown uses worst-domain: a
    miner must not be able to buy a weak slice with a strong one. A slice with too few scored
    samples is excluded rather than allowed to dominate on noise."""
    out = MinerScore()
    by_slice: dict[str, list[float]] = {}
    live: list[float] = []
    for key, eff in samples:
        if eff.discarded:
            out.n_discarded += 1
            continue
        by_slice.setdefault(key, []).append(sample_score(eff, alpha, beta))
        live.append(eff.d_miner)
        out.effects.append(eff)
        out.keyed_effects.append((key, eff))
        out.n_scored += 1
    if not out.n_scored:
        out.reasons.append("no scorable samples")
        return out
    out.mean_effect = sum(live) / len(live)
    for k, v in by_slice.items():
        if len(v) >= min_per_slice:
            out.per_slice[k] = sum(v) / len(v)
            out.slice_samples[k] = v
    if not out.per_slice:
        out.reasons.append(f"no slice reached {min_per_slice} samples")
        return out
    # LIVENESS: a miner whose steps never move the observer would otherwise score well on the
    # similarity term (two inert steps disagree very little) — the magnitude term catches most
    # of it, and this makes the floor explicit.
    if out.mean_effect < min_live_effect:
        out.reasons.append(f"steps are inert (mean effect {out.mean_effect:.4f} "
                           f"< {min_live_effect}) — contributed no information")
        return out
    out.score = min(out.per_slice.values())
    return out
