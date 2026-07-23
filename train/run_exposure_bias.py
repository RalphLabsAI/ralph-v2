"""Real-model exposure-bias run on the multi-turn gridworld (needs a GPU).

The CPU proof (eval/experiments/multiturn_exposure.py) showed the DESIGN separates a
distribution-sensitive student from a uniform control on the real env machinery. This is
the real-model version: distill actual 0.5B students to play the gridworld and measure
whether the self_state slice catches a drifter on genuine model weights.

  1. the ORACLE authors a pile of optimal (observation -> action) grid traces
  2. train/eval split by rollout (eval grids never trained on)
  3. distill S_broad (all steps) and S_short (only steps k<=DRIFT_MAX_K) with distill.py
  4. score base + each student as a ModelAgent via score_exposure on EVAL grids:
       teacher_state@k (act from the oracle's state) vs self_state@k (roll the student on
       its OWN actions, then act) — reported as a curve over k.

Finding (real Qwen2.5-0.5B, 2026-07-21): the exposure-bias signal is NOT "S_short drifts
more" (that was the wrong hypothesis — S_short is uniformly mediocre, so it has little
room to fall). It is that S_BROAD — the student that looks near-perfect under teacher_state
scoring (0.93) — is revealed by self_state to be a FLUENT DRIFTER: self_state 0.55, and a
MONOTONIC decline with depth (0.79 -> 0.38 over k=2..14). teacher_state-only scoring would
crown it; self_state catches it. base (untrained) can't play (ts 0.19 / ss 0.01), so the
task needs real capability. This is const's fluent-drifter concern, on real weights.

    python -m train.run_exposure_bias 2>&1 | tee runs/exposure_bias.log

Env: RALPH_STUDENT_BASE, RALPH_EPOCHS, RALPH_DRIFT_MAX_K, RALPH_DIFF, RALPH_NTRAIN,
RALPH_NEVAL, RALPH_HORIZON.
"""
from __future__ import annotations

import json
import os
import time

from eval.multiturn import (author_pile, score_exposure, OracleAgent, ModelAgent,
                            sft_examples_from_pile)
from eval.runners import HFRunner, SafeStudentRunner
from train.distill import distill_student

STUDENT_BASE = os.environ.get("RALPH_STUDENT_BASE", "Qwen/Qwen2.5-0.5B-Instruct")
EPOCHS = int(os.environ.get("RALPH_EPOCHS", "4"))
DRIFT_MAX_K = int(os.environ.get("RALPH_DRIFT_MAX_K", "2"))
DIFF = int(os.environ.get("RALPH_DIFF", "3"))
NTRAIN = int(os.environ.get("RALPH_NTRAIN", "300"))
NEVAL = int(os.environ.get("RALPH_NEVAL", "80"))
HORIZON = int(os.environ.get("RALPH_HORIZON", "40"))
BATCH = int(os.environ.get("RALPH_BATCH", "32"))
K_BUCKETS = [2, 4, 6, 8, 10, 12, 14]
OUT = os.environ.get("RALPH_OUT", "runs/exposure_bias")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def _score(name, agent):
    r = score_exposure(EVAL_PILE, agent, K_BUCKETS, self_state=True)
    ts = r.overall["teacher_state"] or 0.0
    ss = r.overall["self_state"] or 0.0
    log(f"{name:10} teacher_state={ts:.3f} self_state={ss:.3f} gap={ts-ss:+.3f}")
    return {"by_k": r.by_k, "overall": r.overall, "gap": round(ts - ss, 4)}


def main() -> int:
    global EVAL_PILE
    os.makedirs(OUT, exist_ok=True)
    log(f"base={STUDENT_BASE} epochs={EPOCHS} drift_max_k={DRIFT_MAX_K} diff={DIFF} "
        f"ntrain={NTRAIN} neval={NEVAL}")

    oracle = OracleAgent()
    train_pile = author_pile(NTRAIN, difficulty=DIFF, teacher=oracle, horizon=HORIZON, seed0=10_000)
    EVAL_PILE = author_pile(NEVAL, difficulty=DIFF, teacher=oracle, horizon=HORIZON, seed0=90_000)
    log(f"authored {len(train_pile)} train + {len(EVAL_PILE)} eval grids (oracle traces)")

    broad = sft_examples_from_pile(train_pile)
    short = sft_examples_from_pile(train_pile, max_k=DRIFT_MAX_K)
    log(f"SFT examples: broad={len(broad)} short(k<={DRIFT_MAX_K})={len(short)}")

    log("distilling S_broad ...")
    d_broad = distill_student(STUDENT_BASE, broad, f"{OUT}/s_broad", epochs=EPOCHS, log=log)
    log("distilling S_short ...")
    d_short = distill_student(STUDENT_BASE, short, f"{OUT}/s_short", epochs=EPOCHS, log=log)

    report = {"student_base": STUDENT_BASE, "epochs": EPOCHS, "drift_max_k": DRIFT_MAX_K,
              "diff": DIFF, "n_eval": len(EVAL_PILE), "k_buckets": K_BUCKETS, "scores": {}}
    report["scores"]["base"] = _score("base", ModelAgent(HFRunner(STUDENT_BASE, name="base", batch_size=BATCH)))
    report["scores"]["S_broad"] = _score("S_broad", ModelAgent(SafeStudentRunner(d_broad, name="S_broad", batch_size=BATCH)))
    report["scores"]["S_short"] = _score("S_short", ModelAgent(SafeStudentRunner(d_short, name="S_short", batch_size=BATCH)))

    print("\n===== EXPOSURE-BIAS (real models) =====")
    print(json.dumps(report, indent=2))
    with open(f"{OUT}/report.json", "w") as f:
        json.dump(report, f, indent=2)

    # The signal is a LARGE teacher_state->self_state gap on a student that scores HIGH on
    # teacher_state (a fluent drifter that naive scoring would crown), plus a self_state
    # curve that declines with depth. Report both; the crown machinery must penalise it.
    bs = report["scores"]["S_broad"]
    ssk = bs["by_k"]["self_state"]
    # score_exposure stores INT depth keys; a JSON round-trip would make them str — accept
    # either so the decline check actually fires (it silently never did with str-only keys).
    curve = [v for v in (ssk.get(k, ssk.get(str(k))) for k in K_BUCKETS) if v is not None]
    declines = len(curve) >= 2 and curve[-1] < curve[0] - 0.1
    log(f"S_broad teacher_state={bs['overall']['teacher_state']:.3f} self_state="
        f"{bs['overall']['self_state']:.3f} gap={bs['gap']:+.3f}; self_state curve "
        f"{'DECLINES with depth (exposure bias confirmed)' if declines else 'flat'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
