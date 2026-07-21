"""Real distillation experiment: a genuinely-distilled student vs a drifter.

exposure_bias.py proved the SCORING LOGIC separates a distilled student from a drifter
when handed hand-set accuracies. This trains BOTH students for real from a pinned
teacher and measures the gap on real inference — the honest version of that claim.

  1. generate a teacher-authored, multi-domain experience pile (steps = teacher's steps)
  2. split rollouts into TRAIN and EVAL (eval rollouts are never trained on)
  3. distill S_broad on all train points; distill S_drift on short prefixes only
  4. score both on EVAL points, split by mode:
       teacher_state  — both students handed the teacher's prefix
       self_state     — each student builds its own prefix, teacher acts in that state
     The claim holds iff  S_broad ~ S_drift on teacher_state  but  S_broad >> S_drift
     on self_state (the drifter compounds error on its own rollout).

    python -m train.run_experiment 2>&1 | tee experiment.log

Env: RALPH_TEACHER, RALPH_JUDGE, RALPH_STUDENT_BASE, RALPH_EPOCHS, RALPH_DRIFT_MAX_K,
RALPH_NPOINTS, RALPH_SELF_FRAC.
"""
from __future__ import annotations

import json
import os
import time

from eval.rollouts_gen import generate_experience
from eval.runners import HFRunner, SafeStudentRunner
from eval.trajectory import GroundedJudge, sample_points, prepare_refs, score_on_points
from train.distill import build_sft_examples, distill_student
from train.tasks import DISTILL_TASKS

TEACHER = os.environ.get("RALPH_TEACHER", "Qwen/Qwen2.5-7B-Instruct")
JUDGE = os.environ.get("RALPH_JUDGE", "Qwen/Qwen2.5-7B-Instruct")
STUDENT_BASE = os.environ.get("RALPH_STUDENT_BASE", "Qwen/Qwen2.5-0.5B")
EPOCHS = int(os.environ.get("RALPH_EPOCHS", "3"))
DRIFT_MAX_K = int(os.environ.get("RALPH_DRIFT_MAX_K", "2"))
NPOINTS = int(os.environ.get("RALPH_NPOINTS", "160"))
SELF_FRAC = float(os.environ.get("RALPH_SELF_FRAC", "0.4"))
OUT = os.environ.get("RALPH_OUT", "runs/experiment")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def _mode_agreement(points, refs, exp, teacher, student, judge):
    """Mean per-point agreement, split teacher_state vs self_state."""
    sps = score_on_points(points, refs, exp, teacher, student, judge, max_new_tokens=200)
    ts = [sp.agreement for sp, p in zip(sps, points) if p.mode == "teacher_state"]
    ss = [sp.agreement for sp, p in zip(sps, points) if p.mode == "self_state"]
    m = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    return {"teacher_state": m(ts), "self_state": m(ss), "n_ts": len(ts), "n_ss": len(ss)}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    log(f"teacher={TEACHER} student_base={STUDENT_BASE} epochs={EPOCHS} drift_max_k={DRIFT_MAX_K}")
    teacher = HFRunner(TEACHER, name="teacher", batch_size=16)
    judge_model = HFRunner(JUDGE, name="judge", batch_size=16)
    judge = GroundedJudge(judge_model)

    log(f"generating teacher pile over {len(DISTILL_TASKS)} tasks...")
    exp = generate_experience(teacher, max_new_tokens=400, tasks=DISTILL_TASKS)
    doms = {}
    for r in exp:
        doms[r.domain] = doms.get(r.domain, 0) + 1
    log(f"  {len(exp)} rollouts, by_domain={doms}, steps={[len(r.steps) for r in exp][:20]}...")
    if len(exp) < 8:
        log("too few rollouts — aborting")
        return 1

    # deterministic train/eval split by rollout (eval never trained on)
    idx = list(range(len(exp)))
    eval_idx = idx[::4]                       # 25% held out
    train_idx = [i for i in idx if i not in set(eval_idx)]
    log(f"  train rollouts={len(train_idx)} eval rollouts={len(eval_idx)}")

    broad = build_sft_examples(exp, train_idx)
    drift = build_sft_examples(exp, train_idx, max_k=DRIFT_MAX_K)
    log(f"  SFT examples: broad={len(broad)} drift(k<={DRIFT_MAX_K})={len(drift)}")

    # teacher/judge are loaded; free VRAM before training the student
    teacher._model = None if False else teacher._model  # keep for scoring; H100 fits both 7B+0.5B

    log("distilling S_broad ...")
    d_broad = distill_student(STUDENT_BASE, broad, f"{OUT}/s_broad", epochs=EPOCHS, log=log)
    log("distilling S_drift ...")
    d_drift = distill_student(STUDENT_BASE, drift, f"{OUT}/s_drift", epochs=EPOCHS, log=log)

    # score both on the SAME fresh eval points (through the real safe loader)
    log(f"scoring on {NPOINTS} eval points (self_frac={SELF_FRAC}) ...")
    eval_exp = [exp[i] for i in eval_idx]
    points = sample_points(eval_exp, NPOINTS, SELF_FRAC, seed=12345)
    refs = prepare_refs(points, eval_exp, teacher, judge, max_new_tokens=200)

    s_broad = SafeStudentRunner(d_broad, name="S_broad", batch_size=16)
    s_drift = SafeStudentRunner(d_drift, name="S_drift", batch_size=16)
    base = HFRunner(STUDENT_BASE, name="base", batch_size=16)

    report = {
        "teacher": TEACHER, "student_base": STUDENT_BASE, "epochs": EPOCHS,
        "drift_max_k": DRIFT_MAX_K, "n_eval_rollouts": len(eval_exp),
        "by_domain": doms,
        "base": _mode_agreement(points, refs, eval_exp, teacher, base, judge),
        "S_broad": _mode_agreement(points, refs, eval_exp, teacher, s_broad, judge),
        "S_drift": _mode_agreement(points, refs, eval_exp, teacher, s_drift, judge),
    }
    print("\n===== DISTILLATION EXPERIMENT =====")
    print(json.dumps(report, indent=2))
    with open(f"{OUT}/report.json", "w") as f:
        json.dump(report, f, indent=2)

    b, d = report["S_broad"], report["S_drift"]
    if b["self_state"] is not None and d["self_state"] is not None:
        ts_gap = (b["teacher_state"] or 0) - (d["teacher_state"] or 0)
        ss_gap = b["self_state"] - d["self_state"]
        log(f"teacher_state gap (broad-drift) = {ts_gap:+.3f}")
        log(f"self_state    gap (broad-drift) = {ss_gap:+.3f}   <- self_state should be the larger gap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
