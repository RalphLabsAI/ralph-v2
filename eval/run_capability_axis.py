"""Capability-retention run — does the axis_round crown discriminate REAL capability?

The audit found the crown was never exercised on real models at HEAD (the one real run
tied a 3B and a 1.5B student at 0.5193). This is the capability analog of the S1 check:
run a REAL GLM teacher + a capability LADDER of off-the-shelf students through the actual
axis_round crown path (math + code + instruction, deterministic checkers) and confirm:

  * normalized retention RANKS the students by capability (3B > 1.5B > 0.5B), and
  * the crown goes to the most capable, while a base-equivalent "student" is gate-rejected,
  * GLM (the teacher) is competent enough per axis that each axis is LIVE (>= MIN_AXIS_N).

No distillation here — off-the-shelf Qwen models of different sizes are capability proxies;
the point is to de-risk the CROWN MECHANISM on real models before broadening the cover.
This is covering-first v2 groundwork (const's gate), not the drift experiment.

    python -m eval.run_capability_axis 2>&1 | tee runs/cap.log

Env: RALPH_TEACHER, RALPH_BASE, RALPH_STUDENTS (comma-sep), RALPH_ITEMS, RALPH_OUT.
"""
from __future__ import annotations

import json
import os
import time

from eval.axes.code_exec import CodeExec
from eval.axes.instruction import InstructionFollowing
from eval.axes.long_context import LongContext
from eval.axes.math_gsm import MathGSM
from eval.axes.multihop import MultiHop
from eval.axis_round import AxisSpec, axis_round
from eval.koth import Submission, Tier, Tournament
from eval.runners import HFRunner

TEACHER = os.environ.get("RALPH_TEACHER", "THUDM/glm-4-9b-chat-hf")
BASE = os.environ.get("RALPH_BASE", "Qwen/Qwen2.5-0.5B-Instruct")
STUDENTS = os.environ.get(
    "RALPH_STUDENTS", "Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-0.5B-Instruct"
).split(",")
ITEMS = int(os.environ.get("RALPH_ITEMS", "64"))   # headroom so each axis clears MIN_AXIS_N=30
MAXTOK = int(os.environ.get("RALPH_MAXTOK", "512"))
BATCH = int(os.environ.get("RALPH_BATCH", "16"))
OUT = os.environ.get("RALPH_OUT", "runs/cap")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    log(f"teacher={TEACHER} base={BASE} students={STUDENTS} items/axis={ITEMS} (per-axis difficulty calibrated)")

    # per-axis difficulty CALIBRATED to GLM's frontier (the S1/capability-run lesson): each
    # axis must sit where GLM passes >= MIN_AXIS_N (live) AND the base does NOT saturate it
    # (else the retention denominator (teacher-base) collapses). math up (grade-school was
    # base-solvable); long_context down into GLM's band (48-line was too hard, 19/48).
    # code axis executes student-emitted code -> require a hard sandbox by default (fail
    # closed without bwrap). RALPH_CODE_SANDBOX=auto only on a trusted dev box.
    code_sandbox = os.environ.get("RALPH_CODE_SANDBOX", "require")
    specs = [
        AxisSpec(MathGSM(), "math", weight=1.0, difficulty=3),
        # hard pool kept (broader cover); teacher clears it at ~31%, so sample MORE items
        # rather than dumbing it down: 0.31*160 ~ 50 teacher-passed >> MIN_AXIS_N=30.
        AxisSpec(CodeExec(sandbox=code_sandbox), "code", weight=1.0, difficulty=2, items=160),
        AxisSpec(InstructionFollowing(), "instruction", weight=1.0, difficulty=1),
        AxisSpec(LongContext(base_facts=12), "long_context", weight=1.0, difficulty=1),
        # second fragile axis: COMPOSITION (each hop compounds), distinct from retrieval
        AxisSpec(MultiHop(), "multihop", weight=1.0, difficulty=1),
    ]
    tiers = [Tier("open", max_params=10**12, weight=1.0)]
    tour = Tournament(tiers, margin=0.03)

    glm = HFRunner(TEACHER, name="glm", trust_remote_code=True, batch_size=BATCH)
    base = HFRunner(BASE, name="base", batch_size=BATCH)

    # capability ladder as submissions (off-the-shelf models = capability proxies)
    subs = []
    for i, mid in enumerate(STUDENTS):
        r = HFRunner(mid, name=mid.split("/")[-1], batch_size=BATCH)
        sub = Submission(miner=mid, tier="open", model_id=mid, params=1, compute_h100h=1.0)
        subs.append((sub, r))

    log("running axis_round (GLM authors refs, base + each student scored) ...")
    res = axis_round(1, commit_seed=20260723, specs=specs, glm=glm, base=base, tiers=tiers,
                     tournament=tour, submissions=subs, items_per_axis=ITEMS, max_new_tokens=MAXTOK)

    # report
    report = {"teacher": TEACHER, "base": BASE, "students": STUDENTS, "items": ITEMS,
              "per_axis_difficulty": {sp.name: sp.difficulty for sp in specs}, "n_points": res.n_points, "events": res.events,
              "weights": res.weights, "scored": {}}
    log("=== per-student retention (worst-domain soft-min) ===")
    for mid, s in res.scored.items():
        # per-axis teacher-passed count (len of the paired vector) — a proxy for liveness:
        # an axis with < MIN_AXIS_N(30) teacher-passed items is non-live (GLM not competent
        # enough on it this round) and is excluded from the worst-domain aggregate.
        per_axis = {a.axis: {"ret": round(a.retention, 3), "n": a.n, "live": a.live,
                             "student_pass": a.student_pass, "base_pass": a.base_pass}
                    for a in s.axes}
        report["scored"][mid] = {"retention": round(s.retention, 4),
                                 "retention_lb": round(s.retention_lb, 4),
                                 "gates_ok": s.gates_ok, "reasons": s.reasons,
                                 "valid": s.valid, "per_axis": per_axis}
        pa = " ".join(f"{ax}={d['ret']:.2f}({'L' if d['live'] else 'x'}{d['n']})"
                      for ax, d in per_axis.items())
        log(f"  {mid.split('/')[-1]:22} ret={s.retention:.3f} valid={s.valid} | {pa}"
            f" {'' if s.gates_ok else s.reasons}")

    king = tour.kings.get("open")
    log(f"crown: {king.model_id if king else None}  weights={res.weights}")
    with open(f"{OUT}/report.json", "w") as f:
        json.dump(report, f, indent=2)

    # PASS read-out: does retention rank by capability, and did the most-capable crown?
    ladder = [mid for mid in STUDENTS]  # declared most->least capable order
    rets = {mid: res.scored[mid].retention for mid in ladder if mid in res.scored}
    ranked_ok = all(rets.get(ladder[i], -9) >= rets.get(ladder[i + 1], 9) - 1e-6
                    for i in range(len(ladder) - 1) if ladder[i] in rets and ladder[i + 1] in rets)
    crown_ok = king is not None and king.model_id == ladder[0]
    log(f"VERDICT: retention ranks by capability={ranked_ok}; most-capable crowned={crown_ok}")
    if ranked_ok and crown_ok:
        log("PASS: the crown discriminates real capability on real models.")
    else:
        log("CHECK: ranking or crown did not track capability — inspect per-axis liveness "
            "(GLM competence per axis) and retention above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
