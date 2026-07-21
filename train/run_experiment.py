"""Real distillation experiment: genuine distillation vs drifters, on real models.

exposure_bias.py proved the SCORING LOGIC separates a distilled student from a drifter
under hand-set accuracies. This trains the students for real from a pinned teacher and
measures two claims on real inference:

  COVERING (const's "hard part"): a student that distills only ONE domain must lose,
    because worst-domain aggregation drags it below a student that covers everything.
  EXPOSURE BIAS (self_state slice): a student never trained on its own long-horizon
    states does the teacher's step fine from the teacher's prefix but drifts on its own.

Three students, one SFT code path:
  S_broad   — all domains, all horizons        (the genuine compression)
  S_short   — all domains, only k<=DRIFT_MAX_K  (exposure-bias drifter)
  S_narrow  — one domain only                   (covering drifter)

Each saves safetensors and is scored through the real SafeStudentRunner on HELD-OUT
eval rollouts, reported by mode (teacher_state/self_state) AND by domain.

    python -m train.run_experiment 2>&1 | tee experiment.log

Env: RALPH_TEACHER, RALPH_JUDGE, RALPH_STUDENT_BASE, RALPH_EPOCHS, RALPH_DRIFT_MAX_K,
RALPH_NARROW_DOMAIN, RALPH_NPOINTS, RALPH_SELF_FRAC, RALPH_STUDENTS (subset to train).
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
STUDENT_BASE = os.environ.get("RALPH_STUDENT_BASE", "Qwen/Qwen2.5-0.5B-Instruct")
EPOCHS = int(os.environ.get("RALPH_EPOCHS", "4"))
DRIFT_MAX_K = int(os.environ.get("RALPH_DRIFT_MAX_K", "2"))
NARROW_DOMAIN = os.environ.get("RALPH_NARROW_DOMAIN", "math")
NPOINTS = int(os.environ.get("RALPH_NPOINTS", "140"))
SELF_FRAC = float(os.environ.get("RALPH_SELF_FRAC", "0.35"))
WHICH = set(os.environ.get("RALPH_STUDENTS", "broad,short,narrow").split(","))
OUT = os.environ.get("RALPH_OUT", "runs/experiment")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def _score(points, refs, exp, teacher, student, judge):
    """Per-point agreement, aggregated by mode and by domain."""
    sps = score_on_points(points, refs, exp, teacher, student, judge, max_new_tokens=200)
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    by_mode, by_dom = {}, {}
    for sp, p in zip(sps, points):
        by_mode.setdefault(p.mode, []).append(sp.agreement)
        by_dom.setdefault(exp[p.rollout_idx].domain, []).append(sp.agreement)
    return {
        "overall": mean([sp.agreement for sp in sps]),
        "by_mode": {k: mean(v) for k, v in by_mode.items()},
        "by_domain": {k: mean(v) for k, v in sorted(by_dom.items())},
    }


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    log(f"teacher={TEACHER} base={STUDENT_BASE} epochs={EPOCHS} drift_k<={DRIFT_MAX_K} "
        f"narrow={NARROW_DOMAIN} students={sorted(WHICH)}")
    teacher = HFRunner(TEACHER, name="teacher", batch_size=16)
    judge = GroundedJudge(HFRunner(JUDGE, name="judge", batch_size=16))

    log(f"generating teacher pile over {len(DISTILL_TASKS)} tasks...")
    exp = generate_experience(teacher, max_new_tokens=400, tasks=DISTILL_TASKS)
    doms = {}
    for r in exp:
        doms[r.domain] = doms.get(r.domain, 0) + 1
    log(f"  {len(exp)} rollouts, by_domain={doms}")
    if len(exp) < 12:
        log("too few rollouts — aborting")
        return 1

    idx = list(range(len(exp)))
    eval_idx = idx[::4]                       # 25% held out, spread across domains
    train_idx = [i for i in idx if i not in set(eval_idx)]
    log(f"  train rollouts={len(train_idx)} eval rollouts={len(eval_idx)}")

    specs = {
        "broad": dict(),
        "short": dict(max_k=DRIFT_MAX_K),
        "narrow": dict(domains={NARROW_DOMAIN}),
    }
    trained = {}
    for name in ["broad", "short", "narrow"]:
        if name not in WHICH:
            continue
        ex = build_sft_examples(exp, train_idx, **specs[name])
        log(f"distilling S_{name}: {len(ex)} examples ...")
        trained[name] = distill_student(STUDENT_BASE, ex, f"{OUT}/s_{name}",
                                        epochs=EPOCHS, log=log)

    log(f"scoring on {NPOINTS} held-out eval points (self_frac={SELF_FRAC}) ...")
    eval_exp = [exp[i] for i in eval_idx]
    points = sample_points(eval_exp, NPOINTS, SELF_FRAC, seed=12345)
    refs = prepare_refs(points, eval_exp, teacher, judge, max_new_tokens=200)

    base = HFRunner(STUDENT_BASE, name="base", batch_size=16)
    report = {
        "teacher": TEACHER, "student_base": STUDENT_BASE, "epochs": EPOCHS,
        "drift_max_k": DRIFT_MAX_K, "narrow_domain": NARROW_DOMAIN,
        "n_eval_rollouts": len(eval_exp), "pile_by_domain": doms,
        "scores": {"base": _score(points, refs, eval_exp, teacher, base, judge)},
    }
    for name, d in trained.items():
        runner = SafeStudentRunner(d, name=f"S_{name}", batch_size=16)
        report["scores"][f"S_{name}"] = _score(points, refs, eval_exp, teacher, runner, judge)

    print("\n===== DISTILLATION EXPERIMENT =====")
    print(json.dumps(report, indent=2))
    with open(f"{OUT}/report.json", "w") as f:
        json.dump(report, f, indent=2)

    s = report["scores"]
    if "S_broad" in s and "S_short" in s:
        bm, dm = s["S_broad"]["by_mode"], s["S_short"]["by_mode"]
        ts_gap = (bm.get("teacher_state") or 0) - (dm.get("teacher_state") or 0)
        ss_gap = (bm.get("self_state") or 0) - (dm.get("self_state") or 0)
        log(f"EXPOSURE BIAS: teacher_state gap={ts_gap:+.3f}  self_state gap={ss_gap:+.3f}"
            f"  ({'self_state larger (claim holds)' if ss_gap > ts_gap else 'self_state NOT larger'})")
    if "S_broad" in s and "S_narrow" in s:
        bd, nd = s["S_broad"]["by_domain"], s["S_narrow"]["by_domain"]
        off = [k for k in bd if k != NARROW_DOMAIN]
        b_off = [bd[k] for k in off if bd[k] is not None]
        n_off = [nd[k] for k in off if nd.get(k) is not None]
        log(f"COVERING: off-domain retention  broad={sum(b_off)/len(b_off):.3f}  "
            f"narrow={sum(n_off)/len(n_off):.3f}  (narrow should collapse off its domain)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
