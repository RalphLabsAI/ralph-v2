# Ralph SN40 v2 — ship status

A compression subnet: miners distill a pinned teacher (GLM) into a small student that
keeps the teacher's *multi-step* behavior; the best student per size tier wears the crown
and ships as an open checkpoint. This doc is the honest map for a fresh review — what is
built, what is validated, and what is not yet real.

## The mechanism (const's design + our addition)

Score a student by how much of the teacher's next STEP it reproduces, over (rollout, step)
points drawn from a large experience pile. The pile is the eval — self-labeling (the label
is the teacher's own step), effectively infinite, never a fixed test set.

- **teacher_state** — score the student's action from a state the teacher reached
  (in-distribution; measures raw retained capability). This is const's design.
- **self_state** — roll the student on its OWN actions, then score the action from the
  state it drifted into (out-of-distribution). This catches the *fluent drifter* — a
  student that looks perfect teacher-forced but compounds error on its own trajectory —
  which is what collapsed the prior distillation subnet (SN97). This is our addition.

The two are aggregated **worst-domain** (soft-min), so a student strong teacher-forced but
drifting on its own is pulled down to its weaker mode and cannot win.

## Why a multi-turn ENV substrate

A chopped chain-of-thought pile does not work: re-prompting an instruct model with a
partial CoT makes it *restart* the whole solution, not continue — so the reference is
corrupted and self_state is unmeasurable (found the hard way on real models, RUN 1).

A **multi-turn environment** dissolves this: the observation is Markov (state fully
described each turn), so every turn conditions on the current observation — the model
continues, never restarts. teacher_state vs self_state then differ only by *whose
trajectory produced the state* — the pure form of exposure bias. And on a **discrete-action
deterministic env, "did the student take the teacher's action here" is a deterministic
check** — no LLM judge in the crown path (so no judge self-preference, no injection, no
gaming the judge). The env oracle also gives an optimal-action set for validation.

## What is validated

- **Off-policy step-agreement substrate on real Qwen models**: distilled students beat an
  untrained base ~0.51 vs 0.10; a math-only student caves off-domain (covering has teeth).
- **The exposure-bias differentiator on real Qwen models** (oracle-reference path): a
  student scoring **0.93 teacher-forced drops to 0.55 on its own rollout, declining
  monotonically 0.79 → 0.38 with depth** — the fluent drifter, caught by self_state, that
  teacher-forced scoring alone would crown. A uniform-competence control shows ~0 gap
  (so the gap is not a metric artifact).
- **The production round catches the drifter in the crown decision**: teacher_state 0.95 /
  self_state 0.36 → soft-min 0.45 → cannot dethrone the honest king.

## What is built (production loop, CPU-tested end to end)

intake (safetensors-only, no remote code, params recomputed from tensors, tier fit) →
ModelAgent → env_round (fresh commit-seeded points, two axes, crown gates) → KOTH
dethrone-on-margin → anti-grind economics (one-eval-per-reg + refundable bond) → signed
reproducible round record → chain write-back. Crown gates live: RCE, teacher-as-student,
negative-axis interval test, min-live-axis fail-closed, MIN_CROWN_LB floor, degeneracy.
The reference is reproduce-GLM (const's design), checked by the deterministic env oracle.

## What is NOT yet real (the honest gaps)

- **Reproduce-GLM on real GLM playing the envs** — validated with sim/oracle references;
  the real GLM as teacher+reference on the envs is not yet a GPU run.
- **One env only** (gridworld) — a toy. Real-world credibility needs a suite of
  deterministic envs exercising capabilities GLM is actually good at.
- **pass@k gate** (needs per-point multi-sampling) and **provenance gate** (needs
  teacher-failure points) — documented TODOs, not wired.
- **Compute-metering reconciliation** — declared compute is stored, not yet enforced
  against a throughput envelope.
- **Chain integration + validator** — the ChainIO protocol is defined and the existing
  Ralph validator satisfies it, but live integration is separate work.
- Aggregation robustness (thin-axis pooling, base Wilson bounds) and throughput (batched
  GLM reference) for real scale.

## Posture

Shadow-first: miners submit and see themselves ranked with zero emission at risk, then
emission flips once the gaps above close. Nothing here promises a date.
