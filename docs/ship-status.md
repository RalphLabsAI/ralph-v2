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

## SHIP VERDICT (adversarial review, 44 agents, 2026-07-21)

**NO-GO for emission. Conditional GO for shadow after ~5 fixes, and only as an
operator-run measurement — not a public competitive board.** The mechanism is sound and
the differentiator is validated, but the shipped env path (a) was validated via the ORACLE
reference while production uses GLM exact-match, (b) uses a STATIC pile that is a public
answer key, and (c) certifies "played one toy gridworld," not "compressed GLM." None are
fatal; all are fixable; three block a meaningful shadow and ~a dozen block emission.

### Hard blockers before a trustworthy SHADOW run
1. **Production path never GPU-run.** Validation used the oracle reference + students SFT'd
   on oracle actions. Production uses the GLM reference + real distilled-GLM students —
   unrun. The point of shadow is to close this; run the *corrected* path.
2. **Pile is oracle-authored but the reference is GLM** → teacher_state scores the student
   on the ORACLE's trajectory, which GLM never visited, breaking the exposure-bias framing.
   **GLM must PLAY the env to author the trajectories.**
3. **The env pile is a static, enumerable public answer key** — the commit-nonce only picks
   *which points* to score, not the grids. A miner memorizes the full (state→GLM-action)
   table offline and maxes both axes. **Author the grids per-round from the commit-nonce.**
4. self_state awards a free 1.0 on early solve (multiturn.py) — exclude solved-before-k.
5. Pick ONE crown metric (exact-match-GLM vs oracle-optimal-set) so validated == production.

### Biggest real-world-credibility risk
The crown certifies "reproduces GLM's move on one Markov-LOCAL gridworld" — a from-scratch
tiny bot with **zero GLM lineage** can win it — yet every crown is slated to publish as
"compressed GLM-4-9B." There is **zero measured correlation** between env-agreement and
broad GLM-capability retention. Until closed **in code** (not posture), the board must be
labelled "multi-step control retention (experimental)," never "compressed GLM."
**Close it with:** (a) a broad GLM-retention gate (held-out LM-agreement / bounded
perplexity gap) as a hard crown precondition; (b) an env SUITE across DISTINCT capabilities
GLM is strong at, aggregated worst-first; (c) publish the env-agreement↔retention correlation.

### Before emission flip (crown integrity + product truth)
- Env suite + wire the existing math/code execution axes (currently dead in adversarial.py).
- Broad GLM-retention gate binding the artifact to GLM.
- `model_id = content hash`, not hotkey; persist crowned hash on-chain; reconstruct the king
  agent from its hash each round (FAIL-CLOSED hold is done; durable reconstruct is TODO).
- Validator-side fetch-verify-hash + commit-reveal enforcement (bait-and-switch open today).
- Compute/effective-bits in the ranking, or drop the capability-per-compute claim until real.
- Batch self_state + cache the teacher_state reference for real scale.
- pass@k gate (undefinable on deterministic-greedy — redefine or drop), Wilson CI on base,
  actually sign the record, nonce strictly after the commit window.

### Fixed this pass
Fail-closed crown (king unrescoreable → HOLD, no more open-throne steal); non-zero bond
default (anti-grind was off); honest record reference label; earlier: crown-path gates
(negative-axis interval, min-live, MIN_CROWN_LB, degeneracy), fresh-seed nonce binding.

## Posture

Shadow-first, operator-run: no emission, no public board, until the shadow blockers close.
No "compressed GLM" claim until the capability-laundering gap is a coded gate. No dates.
