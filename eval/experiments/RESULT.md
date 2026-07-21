# Off-policy vs on-policy step-agreement — result

**Claim tested.** const's trajectory mechanism scores a student by step-agreement with
GLM under GLM's own prefix (0→K given, both generate K→K+1). That is off-policy /
teacher-forced. Claim: it overstates a student that imitates locally but drifts when
autonomous — the exposure-bias failure that collapsed the previous distillation subnet
— and an on-policy slice (student builds its own prefix, then compare) is needed to
catch it.

**Setup.** Two simulated students with *identical* local (teacher-forced) fidelity
0.85, differing only in error-recovery: a fluent-drifter (recovery 0.15) and a
robust-distill (recovery 0.90). `python -m eval.experiments.exposure_bias`.

**Result.**

| student | off-policy (const's design) | on-policy (the addition) |
|---|---|---|
| fluent-drifter | 0.851 | 0.364 |
| robust-distill | 0.851 | 0.565 |

- Off-policy **cannot separate them** (Δ = 0.000).
- On-policy **ranks them decisively** (Δ = 0.201), correctly placing the robust distill
  above the fluent drifter.
- The blindspot grows with trajectory length — off-policy is flat at ~0.85 while the
  drifter's on-policy score falls 0.85 → 0.58 → 0.36 → 0.10 at length 1 → 10 → 20 → 80.
  The more agentic the task, the more the on-policy slice matters.

**Reading.** This is a controlled simulation of the *scoring*, not a claim about real
GLM. It proves the metric distinguishes drift, and that off-policy alone would crown the
fluent drifter — the exact model class that killed SN97. Real-model validation plugs
into `../trajectory.py` on a GPU box (pinned GLM teacher + a rotated rubric judge).

**What it does NOT prove yet.** That real compressed GLM students actually exhibit this
gap, and how large it is. That is the GPU experiment: run the harness against GLM with
a known-good distill vs a known-drifting one and measure the real off/on divergence.

## How this folds into the build (decision — not a debate to send)

The mechanism is const's trajectory step-agreement, unchanged: sample states from a
large experience pile, GLM takes its genuine next step, compare — no envs on the
validator. This result is an internal refinement, not a counter-argument:

- Default scoring stays **teacher-state** (prefix from the experience pile). Cheap,
  cacheable across miners, no execution.
- The experience pile is **salted with degraded / varied-quality states** so recovery
  is exercised without any rollout.
- A **self-state slice** (student builds its own prefix) is sampled only where env cost
  is low (reasoning traces, where it's just generation).
- Whether the salt alone suffices or the self-state slice is needed is **empirical** —
  measured on the real GLM run, not decided in advance.

Labels in `trajectory.py` use `teacher_state` / `self_state` (not "off/on-policy") to
avoid colliding with the RL sense of those words.
