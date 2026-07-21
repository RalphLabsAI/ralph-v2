# Capability-retention harness (Phase 0)

Runnable, CPU-only, no GPU required for the mechanism proof.

```bash
python -m eval.adversarial     # the Phase-2 adversarial proof
```

## What's here

| file | role |
|---|---|
| `core.py` | `Item` / `Axis` / `ModelRunner` types; a model is a black box (prompt in, text out) |
| `seeds.py` | commit-then-generate seeding: `seed = H(commit_root ‖ round_nonce ‖ axis)` |
| `scoring.py` | normalized retention, Wilson lower bound, worst-domain soft-min, gates |
| `axes/math_gsm.py` | GSM-Symbolic-style templated word problems + NoOp distractors, exact numeric checker |
| `axes/code_exec.py` | function-implementation tasks graded by **executing** hidden unit tests |
| `runners.py` | `HFRunner` (real, transformers) + simulated adversaries |
| `adversarial.py` | the proof: honest student must beat every cheater |

## Current result

```
student                     math ret    code ret     SCORE  gates
student-honest                 0.511       0.481     0.357  ok
naive-quant-control            0.174      -0.025     0.000  negative retention on: code
student-style-only            -0.641      -0.815     0.000  negative retention on: math, code
student-narrow-math            0.891      -0.630     0.000  negative retention on: code
student-noop-brittle           0.196       0.370     0.072  ok

PASS: honest 0.357 beats every adversary (best cheater 0.072, margin 0.285)
```

**Laundering sensitivity.** Independently of the negative-retention gate, worst-domain
soft-min (p = −6) means a student that maxes math to 1.25 — i.e. *beats the teacher* —
must still keep code retention ≥ 0.446 to match an honest balanced 0.50/0.50 student.
It can only sell an axis down to ~89% of honest level before losing. Selling one
capability to buy another is not a profitable strategy, which is the central claim.

Note the score is nearly invariant to the strong axis (1.25 vs 1.00 math ≈ identical
score) — it is set by the weakest axis, by design.

## Honest limits of this proof

1. **Simulated students, not real models.** This validates the SCORING LOGIC and the
   aggregation properties. It does not tell us how real compressed models behave —
   per-axis accuracies here were chosen, so the ranking is partly assumed.
2. **Two axes.** The design calls for ~6 plus a rotating surprise pool.
3. **No teacher yet.** `HFRunner` is written but unrun: a 9B teacher needs a GPU.
4. **Sandboxing.** `code_exec` runs candidate code in a subprocess with a timeout —
   fine for our own generated solutions, NOT sufficient for untrusted miner output
   (no network isolation, no fs/resource caps). Must be hardened before real use.

## Next

- run the real teacher (pinned GLM) on a GPU box and replace simulated competence
  with measured per-axis pass rates
- add instruction-following (programmatic checkers) and long-context axes
- calibrate difficulty so the teacher sits near its saturation frontier

---

# Subnet core (the round engine)

The mechanism is **const's trajectory step-agreement**: score a student by how much of
what GLM does at a step the student also does, sampled over (rollout, step) pairs from a
large experience pile. No fixed test set, no RL env on the validator. See `trajectory.py`
and the `teacher_state` / `self_state` note there.

```
python -m eval.sim_round     # full KOTH loop, simulated end-to-end on CPU
```

| file | role |
|---|---|
| `trajectory.py` | the eval substrate: sample points, cache GLM references once, score a student (paired) |
| `koth.py` | tournament state machine: per-tier kings, **dethrone on bootstrap-LCB margin**, weights |
| `round_engine.py` | one full round: points → refs → score every submission + the reigning king on the same points → crown → weights |
| `sim_round.py` | multi-round proof of the dynamics |

## Loop result

```
round 1: crown     king=m_good    (open throne -> best challenger)
round 2: hold      margin_lcb=+0.000   (an EXACT copy ties -> no dethrone)
round 3: dethrone  margin_lcb=+0.072   (a genuine improvement clears the margin)
weights track the crown throughout
```

The `+0.000` is the anti-copy property, rigorously: the reigning king is re-scored every
round on the same fresh points, an identical checkpoint produces identical per-point
agreement, the paired difference is exactly zero, and zero clears no margin. No detector,
no fingerprint — copying is simply unprofitable. Genuine improvement past the noise floor
is the only way to dethrone.

## What's built vs what remains

Built and CPU-runnable: the full scoring + tournament + weight loop, model access behind
a `ModelRunner` interface so it runs against a real pinned GLM on a GPU box unchanged.

Remaining (wraps this core, reuses Ralph's existing validator):
- chain I/O — read commitments, set weights, publish the signed round record
- real pinned GLM teacher + rotated rubric judge on a GPU (replaces the sim models)
- compute metering reconciliation (declared vs throughput envelope), the bond
- the experience pile — real agentic rollouts, salted with degraded states

## Superseded (kept for reference)

`axes/math_gsm.py`, `axes/code_exec.py`, `adversarial.py`, `provenance.py` are the earlier
task-set / covering-eval approach. The mechanism moved to trajectory step-agreement, which
makes covering + provenance structural rather than bolted-on. `scoring.py` (normalized
retention, worst-domain soft-min, bootstrap-LCB) carries over unchanged and sits under both.

---

# First real-model run (H100, ~$2)

Qwen2.5-7B as teacher + judge, 0.5B base, 3B & 1.5B students as stand-ins. Goal: does the
mechanism work outside simulation? `python -m eval.gpu_run`.

**Results / findings:**
1. **Pipeline runs end-to-end on real models** — teacher, grounded judge, base, students,
   the full round + KOTH loop complete on real inference. (c) validated.
2. **The grounded judge works** — genuine YES/NO, discriminating (not rubber-stamping):
   it says NO when a candidate step does something different, YES when it matches.
3. **Segmentation is the #1 practical requirement** (the real finding). Naive line-based
   splitting turned steps into bare markdown headers (`2. **Calculate the average:**`),
   so a student continuing the header with actual content always "mismatched" → retention
   collapsed to ~0. Fixed to content-bearing semantic steps (`rollouts_gen._split_steps`).
   After the fix, discrimination appears: **3B student agrees with teacher steps 100%,
   0.5B base 73%** — real capability separation, judge-verified.
4. **Exposure-bias signal is visible even here**: full-run retention 0.519 sits below the
   teacher-state ceiling because the self-state axis (student's own prefix) drags the
   worst-domain soft-min down — the on-policy gap is real, faintly, even on easy tasks.

**Honest limits of this run:** grade-school tasks are too easy to separate a 3B from a
1.5B (both ceiling on teacher-state → identical 0.519); n=48 points makes the Wilson lower
bound very conservative; judge == teacher (self-preference not controlled). **Next run:**
harder tasks (competition math / real agentic rollouts), more points, a distinct judge
model, and a genuinely distilled-vs-drifter student pair to measure the real exposure-bias
gap. Raw logs in `runs/` (gitignored).

**Takeaway:** the mechanism works on real models and discriminates capability. Its
validity hinges on trajectory segmentation quality — agentic rollouts (the intended
source) have natural step boundaries; reasoning CoT needs semantic chunking, not line
splits. Better to learn that here than in production.
