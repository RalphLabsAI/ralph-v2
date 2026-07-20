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
