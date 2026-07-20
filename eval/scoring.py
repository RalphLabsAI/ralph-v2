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
SOFTMIN_P = -6.0  # in [-8, -4]; more negative == closer to a hard min


def wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval. Returns 0.0 for n == 0."""
    if n <= 0:
        return 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom)


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

    @property
    def negative(self) -> bool:
        return self.retention < 0.0


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
        # student failure. Surface it as zero-weight rather than dividing by zero.
        return AxisScore(axis, weight, 0, 0, 0, 0, 0.0, 0.0)

    s_rate, b_rate = s_pass / n, b_pass / n
    t_rate = 1.0
    denom = t_rate - b_rate
    if denom <= 1e-9:
        # base already matches the teacher here: no headroom, so this axis carries
        # no information about compression quality this round.
        return AxisScore(axis, weight, t_pass, s_pass, b_pass, n, 0.0, 0.0)

    retention = (s_rate - b_rate) / denom
    s_lb = wilson_lower_bound(s_pass, n)
    retention_lb = (s_lb - b_rate) / denom

    return AxisScore(
        axis=axis, weight=weight, teacher_pass=t_pass, student_pass=s_pass,
        base_pass=b_pass, n=n,
        retention=min(retention, cap),
        retention_lb=min(retention_lb, cap),
    )


def soft_min(scores: list[AxisScore], p: float = SOFTMIN_P, use_lower_bound: bool = True) -> float:
    """Weighted power-mean with p < 0 — approaches min() as p -> -inf.

    Values are floored just above zero because a true zero would annihilate the whole
    power mean; a zero-retention axis should dominate the score, not erase it.
    """
    live = [s for s in scores if s.n > 0 and s.weight > 0]
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
                    "axis": a.axis, "weight": a.weight, "n_teacher_passed": a.n,
                    "student_pass": a.student_pass, "base_pass": a.base_pass,
                    "retention": round(a.retention, 6), "retention_lb": round(a.retention_lb, 6),
                }
                for a in self.axes
            ],
        }


def verdict(axes: list[AxisScore], passk_ok: bool = True, passk_failures: list[str] | None = None) -> Verdict:
    reasons: list[str] = []
    negative = [a.axis for a in axes if a.negative]
    if negative:
        reasons.append(f"negative retention on: {', '.join(negative)}")
    if not passk_ok:
        reasons.append(f"pass@k regressed vs base on: {', '.join(passk_failures or [])}")
    gates = not reasons
    return Verdict(
        score=soft_min(axes) if gates else 0.0,
        axes=axes,
        gates_passed=gates,
        reasons=reasons,
    )
