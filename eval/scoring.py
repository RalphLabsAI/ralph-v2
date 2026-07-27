"""Retention scoring.

Per axis, retention is normalized against the pinned (teacher, student-base) pair:

    R_a = (s_student - s_base) / (s_teacher - s_base)

so an axis where the base already scores well cannot be farmed for headroom it never
had. Axes aggregate by WORST-DOMAIN soft-min, so sacrificing any one capability to
buy another tanks the score. The cap is 1.25, not 1.0: a hard 1.0 makes the crown
undecidable the moment two miners reach teacher parity, and beating the teacher is
the interesting case, not an error.

Small fresh batches are noisy, so every rate carries a Wilson lower bound and the
score uses the conservative end. Sharpening that does not add capability (pass@1 up,
pass@k flat) is caught by a GATE, not a weight — see `passk_gate`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

RETENTION_CAP = 1.25
SOFTMIN_P = -3.0  # in [-4, -2]; more negative == closer to a hard min. A real run showed
                  # -6 lets one thin/noisy axis dominate; -3 keeps worst-domain pressure
                  # without a single 3-point axis annihilating the aggregate.
MIN_AXIS_N = 30   # teacher-passed items required before an axis may gate/dominate. A real
                  # run had a 6-task axis (~3 points) flip negative on sampling noise and
                  # sink every student's soft-min — below this an axis is not "live".

# SATURATION GUARD: minimum fraction of teacher-passed items the BASE must FAIL for the axis to
# carry information. Retention divides by (teacher - base), so a saturated axis both amplifies
# sampling noise (÷0.05 is a 20x multiplier) and measures nothing — and a non-discriminative
# axis is precisely the one a miner can Goodhart while real capability rots. The compression
# evidence is blunt: PrismML's own table has a 14x-compressed 27B scoring 96.06 on GSM8K vs the
# FP16 teacher's 95.30 (the compressed model WINS) and 99.20 vs 99.40 on MATH-500. Those axes
# cannot distinguish a 54 GB model from a 3.8 GB one, so they must not be allowed to vote.
MIN_HEADROOM = 0.15


def wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval. Returns 0.0 for n == 0."""
    if n <= 0:
        return 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom)


def wilson_upper_bound(successes: int, n: int, z: float = 1.96) -> float:
    """Upper bound of the Wilson score interval. Returns 1.0 for n == 0."""
    if n <= 0:
        return 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return min(1.0, (centre + margin) / denom)


@dataclass
class AxisScore:
    axis: str
    weight: float
    teacher_pass: int
    student_pass: int
    base_pass: int
    n: int
    # rates on the teacher-passed subset
    retention: float
    retention_lb: float
    retention_ub: float = 1.25
    # `live` = this axis carries usable information this round and may gate/dominate.
    # False for: teacher solved nothing (n==0), base already at teacher parity (no
    # headroom), or too few teacher-passed items (n < MIN_AXIS_N). A non-live axis is
    # EXCLUDED from aggregation rather than floored to 0 — "can't measure" must not read
    # the same as "catastrophic failure".
    live: bool = True

    @property
    def negative(self) -> bool:
        # interval test: only a live axis whose retention UPPER bound is still below 0
        # (student CI entirely under base) counts as a genuine capability sacrifice. A
        # point-estimate `retention < 0` fires on pure sampling noise at small n.
        return self.live and self.retention_ub < 0.0


def axis_retention(
    axis: str,
    weight: float,
    teacher_passed: list[bool],
    student_passed: list[bool],
    base_passed: list[bool],
    cap: float = RETENTION_CAP,
) -> AxisScore:
    """Normalized retention on one axis.

    Conditioned on items the TEACHER got right: we measure how much of the teacher's
    demonstrated capability the student kept, not how well the student does at a
    benchmark in the abstract.
    """
    n_all = len(teacher_passed)
    if not (len(student_passed) == len(base_passed) == n_all):
        raise ValueError("teacher/student/base result lists must align item-for-item")

    idx = [i for i, ok in enumerate(teacher_passed) if ok]
    n = len(idx)
    t_pass = n  # by construction
    s_pass = sum(1 for i in idx if student_passed[i])
    b_pass = sum(1 for i in idx if base_passed[i])

    if n == 0:
        # teacher solved nothing — the axis is mis-calibrated for this pair, not a
        # student failure. Not live: excluded from aggregation, not dividing by zero.
        return AxisScore(axis, weight, 0, 0, 0, 0, 0.0, 0.0, 0.0, live=False)

    s_rate, b_rate = s_pass / n, b_pass / n
    t_rate = 1.0
    denom = t_rate - b_rate
    if denom < MIN_HEADROOM:
        # SATURATED: the base is already at (or near) teacher parity here, so this axis cannot
        # separate a real compression from a bad one — and dividing by a tiny denominator would
        # turn sampling noise into a retention swing. Not live: excluded from aggregation, and
        # surfaced so an operator re-calibrates the axis rather than trusting a flat score.
        return AxisScore(axis, weight, t_pass, s_pass, b_pass, n, 0.0, 0.0, 0.0, live=False)

    retention = (s_rate - b_rate) / denom
    s_lb = wilson_lower_bound(s_pass, n)
    s_ub = wilson_upper_bound(s_pass, n)
    retention_lb = (s_lb - b_rate) / denom
    retention_ub = (s_ub - b_rate) / denom

    return AxisScore(
        axis=axis, weight=weight, teacher_pass=t_pass, student_pass=s_pass,
        base_pass=b_pass, n=n,
        retention=min(retention, cap),
        retention_lb=min(retention_lb, cap),
        retention_ub=min(retention_ub, cap),
        # too few teacher-passed items to trust: measurable but not allowed to gate.
        live=(n >= MIN_AXIS_N),
    )


def soft_min(scores: list[AxisScore], p: float = SOFTMIN_P, use_lower_bound: bool = True) -> float:
    """Weighted power-mean over LIVE axes with p < 0 — approaches min() as p -> -inf.

    Only live axes (enough headroom AND enough samples) participate: a "can't measure"
    axis is excluded, not floored, so it can't masquerade as catastrophic failure. A
    genuinely-failed live axis IS floored just above zero (eps) so it dominates the mean
    toward ~0 without producing inf; that near-zero is the correct signal, and the crown
    gates (verdict / koth) must treat it as a fail, not award it on an open throne.
    """
    live = [s for s in scores if s.live and s.weight > 0]
    if not live:
        return 0.0
    eps = 1e-3
    total_w = sum(s.weight for s in live)
    acc = 0.0
    for s in live:
        r = max(s.retention_lb if use_lower_bound else s.retention, eps)
        acc += (s.weight / total_w) * (r ** p)
    return acc ** (1.0 / p)


def passk_gate(student_passk: dict[str, float], base_passk: dict[str, float], tol: float = 0.0) -> tuple[bool, list[str]]:
    """pass@k must not regress against the pinned base on ANY axis.

    A student can lift pass@1 while pass@k stays flat or drops — sharpening that
    reads as progress but has not added capability (arXiv:2505.14216). This is a
    hard gate: fail it and no crown is awarded regardless of score.
    """
    failures = [a for a, v in student_passk.items() if v + tol < base_passk.get(a, 0.0)]
    return (not failures), failures


@dataclass
class Verdict:
    score: float
    axes: list[AxisScore]
    gates_passed: bool
    reasons: list[str]

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 6),
            "gates_passed": self.gates_passed,
            "reasons": self.reasons,
            "axes": [
                {
                    "axis": a.axis, "weight": a.weight, "n_teacher_passed": a.n, "live": a.live,
                    "student_pass": a.student_pass, "base_pass": a.base_pass,
                    "retention": round(a.retention, 6), "retention_lb": round(a.retention_lb, 6),
                    "retention_ub": round(a.retention_ub, 6),
                }
                for a in self.axes
            ],
        }


def verdict(axes: list[AxisScore], passk_ok: bool = True, passk_failures: list[str] | None = None,
            min_live_axes: int = 1) -> Verdict:
    """Crown decision for one student. Fail-closed: a student that regresses on any live
    axis, regresses pass@k, or has too few measurable axes is NOT crownable regardless of
    aggregate score — this is what stops an all-failed student grabbing an open throne."""
    reasons: list[str] = []
    negative = [a.axis for a in axes if a.negative]
    if negative:
        reasons.append(f"negative retention on: {', '.join(negative)}")
    if not passk_ok:
        reasons.append(f"pass@k regressed vs base on: {', '.join(passk_failures or [])}")
    n_live = sum(1 for a in axes if a.live)
    if n_live < min_live_axes:
        reasons.append(f"only {n_live} live axis/axes (< {min_live_axes}) — not enough signal to crown")
    gates = not reasons
    return Verdict(
        score=soft_min(axes) if gates else 0.0,
        axes=axes,
        gates_passed=gates,
        reasons=reasons,
    )
