"""Off-policy vs on-policy step-agreement — controlled proof.

const's mechanism scores a student by step-agreement with GLM under GLM's own
trajectory prefix (0->K given, both generate K->K+1). That is OFF-POLICY / teacher-
forced: the student is always handed GLM's context, so its own mistakes never
compound. This is the same regime in which the collapsed SN97 king looked good.

CLAIM: a student with high local (teacher-forced) fidelity but poor error-recovery
scores high off-policy and low on-policy; a robust student scores high on both. So
two students can be INDISTINGUISHABLE under const's design yet clearly ranked once an
on-policy slice is added — and the one off-policy alone can't tell apart is exactly
the "fluent drifter" we must not crown.

This is a controlled simulation of the mechanism, not a claim about real GLM. It
proves the SCORING distinguishes drift. Real-model validation plugs into
../trajectory.py on a GPU box.

    python -m eval.experiments.exposure_bias
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass


@dataclass
class Student:
    """A student defined by two independent properties.

    local_fidelity : P(match GLM's step | on GLM's trajectory)  -> the OFF-POLICY score
    recovery       : how strongly a matched step pulls the student back onto GLM's
                     manifold after it has drifted (1.0 = fully self-correcting,
                     0.0 = drift accumulates forever). Invisible off-policy.
    """
    name: str
    local_fidelity: float
    recovery: float
    drift_penalty: float = 0.55  # each unit of drift multiplies match-prob by this


def _match_prob(s: Student, drift: float) -> float:
    """Agreement probability at a given distance from GLM's trajectory manifold."""
    return s.local_fidelity * (s.drift_penalty ** drift)


def off_policy_score(s: Student, length: int, rollouts: int, seed: int) -> float:
    """Teacher-forced: reset to GLM's state every step (drift == 0 always)."""
    rng = random.Random(seed)
    hits = tot = 0
    for _ in range(rollouts):
        for _ in range(length):
            hits += rng.random() < _match_prob(s, 0.0)
            tot += 1
    return hits / tot


def on_policy_score(s: Student, length: int, rollouts: int, seed: int) -> float:
    """The student runs its OWN trajectory; deviations move it off-manifold and
    degrade future agreement unless it recovers."""
    rng = random.Random(seed)
    hits = tot = 0
    for _ in range(rollouts):
        drift = 0.0
        for _ in range(length):
            matched = rng.random() < _match_prob(s, drift)
            hits += matched
            tot += 1
            if matched:
                drift *= (1.0 - s.recovery)   # a GLM-like step pulls you back
            else:
                drift += 1.0                  # a deviation puts you one step off
    return hits / tot


def ci(fn, s: Student, length: int, rollouts: int, reps: int = 12) -> tuple[float, float]:
    vals = [fn(s, length, rollouts, seed=1000 + i) for i in range(reps)]
    return statistics.mean(vals), (statistics.pstdev(vals) if reps > 1 else 0.0)


def main() -> int:
    ROLL, LEN = 400, 20

    # Two students with the SAME local fidelity (== same off-policy score) but
    # opposite recovery. const's design cannot separate them; the on-policy slice
    # must, because one is the fluent drifter that collapses in deployment.
    fragile = Student("fluent-drifter", local_fidelity=0.85, recovery=0.15)
    robust = Student("robust-distill", local_fidelity=0.85, recovery=0.90)

    print("Two students, identical local (teacher-forced) fidelity 0.85.\n")
    hdr = f"{'student':<18}{'OFF-policy':>16}{'ON-policy':>16}{'gap':>9}"
    print(hdr); print("-" * len(hdr))
    rows = {}
    for s in (fragile, robust):
        off_m, off_sd = ci(off_policy_score, s, LEN, ROLL)
        on_m, on_sd = ci(on_policy_score, s, LEN, ROLL)
        rows[s.name] = (off_m, on_m)
        print(f"{s.name:<18}{off_m:>10.3f}±{off_sd:.3f}{on_m:>10.3f}±{on_sd:.3f}{off_m - on_m:>9.3f}")

    off_gap = abs(rows["fluent-drifter"][0] - rows["robust-distill"][0])
    on_gap = abs(rows["fluent-drifter"][1] - rows["robust-distill"][1])
    print(f"\n  separation between the two students:")
    print(f"    OFF-policy (const's design):  {off_gap:.3f}  <- cannot rank them")
    print(f"    ON-policy  (the addition):    {on_gap:.3f}  <- ranks them decisively")

    # Length dependence: the off-policy blindspot grows with trajectory length —
    # the more agentic/multi-step the task, the more it matters.
    print(f"\n  drifter's on-policy score vs trajectory length (off-policy is flat at ~0.85):")
    print(f"    {'length':>8}{'on-policy':>12}")
    for L in (1, 5, 10, 20, 40, 80):
        on_m, _ = ci(on_policy_score, fragile, L, ROLL)
        print(f"    {L:>8}{on_m:>12.3f}")

    ok = on_gap > 4 * off_gap and rows["robust-distill"][1] > rows["fluent-drifter"][1]
    print()
    if ok:
        print("  RESULT: off-policy scores the two students the same; on-policy separates")
        print("  them by a wide margin, correctly ranking the robust distill above the")
        print("  fluent drifter. The on-policy slice is necessary, not decorative.")
    else:
        print("  RESULT: no clear separation — revisit parameters.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
