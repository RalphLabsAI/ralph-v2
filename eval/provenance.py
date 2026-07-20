"""Provenance gate — is this student actually derived from the teacher?

THE ATTACK (const, 2026-07-20): "GLM gets 90% on X. Your distilled S also gets 90% on
X and is 10x smaller. But S never distilled from GLM — he just overfit on X."

He is right that accuracy alone cannot tell those apart. Retention as we score it is
P(student correct | teacher correct), and a model trained directly on the task
distribution can match that number without ever having touched the teacher.

The signal that DOES separate them is conditional agreement, not accuracy:

    retention = P(S correct | T correct)     <- what we already score
    excess    = P(S correct | T wrong)       <- what provenance shows up in

A genuine compression is a lossy copy: it inherits the teacher's difficulty ordering,
so it rarely solves what the teacher could not. `excess` stays low.

A model trained directly on the eval distribution has correctness driven by its own
training, not by the teacher's competence surface. It solves teacher-failures at
roughly its own base rate, so `excess` is high — and the ratio excess/retention is
the giveaway.

This is a GATE on provenance, not a reward for imitation. We still never pay for
resembling the teacher (that is what produced style-mimics elsewhere); we only refuse
to believe a compression claim whose error pattern is independent of the thing it
claims to compress.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProvenanceReport:
    n_teacher_pass: int
    n_teacher_fail: int
    retention: float          # P(S ok | T ok)
    excess: float             # P(S ok | T wrong)
    independence: float       # excess / retention  -> ~1.0 means "unrelated to teacher"
    verdict: str
    passed: bool


def conditional_agreement(
    teacher_passed: list[bool],
    student_passed: list[bool],
    max_independence: float = 0.55,
    min_teacher_fail: int = 25,
) -> ProvenanceReport:
    """Compare the student's competence surface to the teacher's.

    `max_independence` is the ratio above which we stop believing the student is a
    compression of this teacher. A lossy copy has excess << retention. A model that
    learned the tasks on its own has excess ~ retention (its skill does not care what
    the teacher found hard).
    """
    tp = [i for i, ok in enumerate(teacher_passed) if ok]
    tf = [i for i, ok in enumerate(teacher_passed) if not ok]

    retention = (sum(1 for i in tp if student_passed[i]) / len(tp)) if tp else 0.0
    excess = (sum(1 for i in tf if student_passed[i]) / len(tf)) if tf else 0.0

    if len(tf) < min_teacher_fail:
        return ProvenanceReport(
            len(tp), len(tf), retention, excess, 0.0,
            f"inconclusive: only {len(tf)} teacher-failures (need {min_teacher_fail}); "
            "raise difficulty so the teacher fails enough items to test against",
            True,
        )

    independence = (excess / retention) if retention > 1e-9 else float("inf")
    passed = independence <= max_independence
    verdict = (
        f"consistent with compression of the pinned teacher (independence "
        f"{independence:.2f} <= {max_independence})"
        if passed else
        f"NOT consistent with compression: solves {excess:.0%} of teacher-failures vs "
        f"{retention:.0%} of teacher-successes (independence {independence:.2f} > "
        f"{max_independence}). Student capability looks unrelated to this teacher — "
        "trained on the task distribution rather than derived from the teacher."
    )
    return ProvenanceReport(len(tp), len(tf), retention, excess, independence, verdict, passed)


def weight_derivation_hint() -> str:
    """For the QUANTIZATION tiers there is a stronger, cheaper check than behaviour.

    A quantized student is a lossy ENCODING of the teacher's weights: dequantize it and
    the per-tensor reconstruction error against the pinned teacher is small. A model
    trained from scratch fails this catastrophically and cannot be made to pass it
    without actually starting from the teacher.

    Behavioural provenance (above) is the fallback for tiers where the student is
    allowed a different architecture, where weight comparison is not defined.
    """
    return (
        "quantization tiers: require per-tensor dequantized reconstruction error vs the "
        "pinned teacher below a tier threshold; distillation tiers (architecture may "
        "differ): fall back to conditional-agreement provenance"
    )
