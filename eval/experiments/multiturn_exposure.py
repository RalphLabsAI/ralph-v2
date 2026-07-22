"""Exposure-bias, demonstrated on the REAL multi-turn env machinery (CPU, no GPU).

experiments/exposure_bias.py proved the SCORING LOGIC separates a drifter from a solid
student under HAND-SET agreement numbers. This proves the same on the actual multi-turn
gridworld rollout machinery — the env the live protocol would use — with agents whose
competence is generated, not hand-set:

  CONTROL  — a uniform-competence agent (same skill everywhere) has NO exposure bias.
             teacher_state ≈ self_state. If our metric showed a gap here it would be an
             artifact. It must not.
  DISTILLED — a distribution-sensitive agent: skilled ON the teacher's states (where SFT
             trains it) and worse OFF them. This is what a real distilled student is. It
             must show teacher_state >> self_state — the fluent drifter the self_state
             slice exists to catch, which teacher_state-only scoring would crown.

Run: python -m eval.experiments.multiturn_exposure
"""
from __future__ import annotations

from ..multiturn import (author_pile, score_exposure, OracleAgent, SimAgent,
                         DistShiftAgent, familiar_states)

K_BUCKETS = [2, 4, 6, 8, 10, 12, 14]


def _row(label, r):
    ts = r.overall["teacher_state"] or 0.0
    ss = r.overall["self_state"] or 0.0
    print(f"  {label:32} teacher_state={ts:.3f}  self_state={ss:.3f}  gap={ts - ss:+.3f}")
    return ts - ss


def main() -> int:
    pile = author_pile(n=80, difficulty=3, teacher=OracleAgent(), horizon=40)
    fam = familiar_states(pile)
    print(f"pile: {len(pile)} teacher rollouts on the gridworld env, "
          f"{sum(r.success for r in pile)} solved, {len(fam)} per-grid familiar states\n")

    print("exposure-bias gap (teacher_state - self_state):")
    control_gap = _row("CONTROL uniform(drift=0.25)",
                       score_exposure(pile, SimAgent(drift=0.25, seed=1), K_BUCKETS))
    distilled_gap = _row("DISTILLED distshift(.05/.6)",
                         score_exposure(pile, DistShiftAgent(fam, 0.05, 0.6, seed=1), K_BUCKETS))
    distilled_gap2 = _row("DISTILLED distshift(.10/.7)",
                          score_exposure(pile, DistShiftAgent(fam, 0.10, 0.7, seed=1), K_BUCKETS))

    # the claim: uniform control ~ 0 gap; a distribution-sensitive student shows a real one
    ok = (abs(control_gap) < 0.04 and distilled_gap > 0.05 and distilled_gap2 > distilled_gap - 0.02)
    print()
    if ok:
        print("PASS: uniform control shows ~0 gap; the distilled (dist-shift) student is "
              "caught by self_state that teacher_state alone would miss.")
        return 0
    print("FAIL: exposure-bias contrast did not hold "
          f"(control={control_gap:+.3f}, distilled={distilled_gap:+.3f})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
