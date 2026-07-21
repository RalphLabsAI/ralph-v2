"""First real-model validation of the pipeline on a GPU.

Goal (bounded, honest): confirm on REAL models that
  (a) the grounded judge produces sane per-step agreement,
  (b) a stronger student gets higher retention than a weaker one,
  (c) the round engine + KOTH loop runs to completion on real inference,
  (d) teacher-state vs self-state retention are both measurable (first read on the
      exposure-bias gap).

NOT claimed here: a real distilled-GLM student vs a real drifter (needs a trained
student — the next run). Students are off-the-shelf models of varying quality as
stand-ins to check the pipeline discriminates real capability.

    python -m eval.gpu_run 2>&1 | tee gpu_run.log

Env overrides: RALPH_TEACHER, RALPH_JUDGE, RALPH_BASE, RALPH_STUDENTS (comma list),
RALPH_NPOINTS, RALPH_NROLLOUTS.
"""
from __future__ import annotations

import json
import os
import time

from .koth import Submission, Tier, Tournament
from .round_engine import run_round
from .rollouts_gen import generate_experience
from .runners import HFRunner
from .trajectory import GroundedJudge

# defaults chosen to fit one 80GB GPU with room to spare
TEACHER = os.environ.get("RALPH_TEACHER", "Qwen/Qwen2.5-7B-Instruct")
JUDGE = os.environ.get("RALPH_JUDGE", "Qwen/Qwen2.5-7B-Instruct")
BASE = os.environ.get("RALPH_BASE", "Qwen/Qwen2.5-0.5B-Instruct")
STUDENTS = os.environ.get("RALPH_STUDENTS", "Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-1.5B-Instruct").split(",")
NPOINTS = int(os.environ.get("RALPH_NPOINTS", "48"))
NROLLOUTS = int(os.environ.get("RALPH_NROLLOUTS", "12"))
TRC = os.environ.get("RALPH_TRUST_REMOTE", "0") == "1"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main() -> int:
    log(f"teacher={TEACHER} judge={JUDGE} base={BASE} students={STUDENTS}")
    teacher = HFRunner(TEACHER, name="teacher", trust_remote_code=TRC)
    judge_model = HFRunner(JUDGE, name="judge", trust_remote_code=TRC)
    base = HFRunner(BASE, name="base", trust_remote_code=TRC)
    students = [HFRunner(s.strip(), name=f"student-{i}", trust_remote_code=TRC)
                for i, s in enumerate(STUDENTS)]
    judge = GroundedJudge(judge_model)

    log("generating experience pile with the teacher...")
    exp = generate_experience(teacher, n=NROLLOUTS)
    log(f"  {len(exp)} rollouts, steps: {[len(r.steps) for r in exp]}")
    if len(exp) < 3:
        log("too few rollouts generated — aborting")
        return 1

    tiers = [Tier("open", max_params=10_000_000_000, weight=1.0)]
    tour = Tournament(tiers, margin=0.03)

    # sizes are nominal (stand-ins); the point is capability discrimination, not tiers
    subs = []
    for i, st in enumerate(students):
        sub = Submission(miner=f"miner{i}", tier="open", model_id=st.name,
                         params=3_000_000_000, compute_h100h=200.0)
        subs.append((sub, st))

    log(f"scoring {len(subs)} students on {NPOINTS} fresh points...")
    res = run_round(1, commit_seed=20260720, experience=exp, glm=teacher, base=base,
                    judge=judge, tiers=tiers, tournament=tour, submissions=subs,
                    registry={}, n_points=NPOINTS, self_frac=0.25, max_new_tokens=200)

    report = {
        "teacher": TEACHER, "judge": JUDGE, "base": BASE,
        "n_rollouts": len(exp), "n_points": res.n_points,
        "students": {
            mid: {"retention": round(s.retention, 4), "retention_lb": round(s.retention_lb, 4),
                  "gates_ok": s.gates_ok}
            for mid, s in res.scored.items()
        },
        "events": res.events,
        "weights": res.weights,
    }
    print("\n===== RESULT =====")
    print(json.dumps(report, indent=2))

    # sanity check: the bigger student should out-retain the smaller one
    ss = sorted(res.scored.values(), key=lambda s: -s.retention)
    log(f"ranking: " + " > ".join(f"{s.sub.model_id}({s.retention:.3f})" for s in ss))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
