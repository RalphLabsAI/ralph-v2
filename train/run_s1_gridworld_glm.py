"""S1 — the cheap falsification: does the exposure-bias drift signal survive when the
reference is a REAL model (GLM) instead of the perfect BFS oracle?

The prior gridworld validation (run_exposure_bias.py) that produced 0.93 teacher_state ->
0.55 self_state used the ORACLE both to author the pile AND as the scoring reference, and
students SFT'd on ORACLE actions. Production uses GLM as the reference and students
distilled from GLM. This driver takes the cheapest step toward that regime — same toy env,
but:

  * GLM PLAYS the grids to author the pile (teacher trajectories are GLM's, not the oracle's)
  * the student is distilled from GLM's OWN recorded actions (not oracle optimal actions)
  * every point is scored under BOTH references, side by side:
        - oracle-optimal-set   (the old regime — reproduce 0.93->0.55 if it's real)
        - GLM-exact-action     (the production regime — does the gap survive?)

teacher_state@k = student's action from GLM's state[k]      (in-distribution)
self_state@k    = roll the student k steps on its OWN actions, score at the drifted state
                  (out-of-distribution — where a fluent drifter fails)

FALSIFICATION: if the teacher_state->self_state gap that shows up under the oracle
reference COLLAPSES under the GLM reference, the differentiator is fragile — it was an
artifact of the perfect oracle, not a real property of distilled-from-GLM students. If the
gap SURVIVES under the GLM reference, the differentiator is real and S2 (full agentic
in-regime run) is justified.

Fail-fast: reports GLM's solve rate on the eval grids first. If GLM can't play the env at
all, the run is inconclusive (null for the wrong reason) — swap RALPH_TEACHER and rerun.

    python -m train.run_s1_gridworld_glm 2>&1 | tee runs/s1.log

Env: RALPH_TEACHER (default THUDM/glm-4-9b-chat), RALPH_STUDENT_BASE, RALPH_NTRAIN,
RALPH_NEVAL, RALPH_DIFF, RALPH_HORIZON, RALPH_EPOCHS, RALPH_BATCH, RALPH_OUT.
"""
from __future__ import annotations

import json
import os
import time

from eval.env import grid as G
from eval.multiturn import ModelAgent, author_pile
from eval.runners import HFRunner, SafeStudentRunner
from train.distill import SFTExample, distill_student

TEACHER = os.environ.get("RALPH_TEACHER", "THUDM/glm-4-9b-chat-hf")
# probe each in order until one loads AND can emit an action — so a finicky GLM repo
# (trust_remote_code / transformers-version issues) doesn't strand a rented box. The
# real-model-reference question is answered by any capable teacher; GLM is preferred for
# regime fidelity and reported as the one actually used.
TEACHER_FALLBACKS = ["THUDM/glm-4-9b-chat", "Qwen/Qwen2.5-7B-Instruct"]
STUDENT_BASE = os.environ.get("RALPH_STUDENT_BASE", "Qwen/Qwen2.5-0.5B-Instruct")
NTRAIN = int(os.environ.get("RALPH_NTRAIN", "120"))
NEVAL = int(os.environ.get("RALPH_NEVAL", "40"))
DIFF = int(os.environ.get("RALPH_DIFF", "2"))
HORIZON = int(os.environ.get("RALPH_HORIZON", "20"))
EPOCHS = int(os.environ.get("RALPH_EPOCHS", "4"))
BATCH = int(os.environ.get("RALPH_BATCH", "16"))
K_BUCKETS = [int(x) for x in os.environ.get("RALPH_KBUCKETS", "2,4,6,8,10").split(",")]
OUT = os.environ.get("RALPH_OUT", "runs/s1")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def _batch(agent, states):
    """agent.act over many states, using act_batch if available (ModelAgent has it)."""
    if not states:
        return []
    if hasattr(agent, "act_batch"):
        return agent.act_batch(states)
    return [agent.act(s) for s in states]


def build_teacher(batch):
    """Load the first candidate teacher that both constructs AND can emit one action on a
    probe state. Returns (ModelAgent, model_id_used)."""
    probe, _ = G.make_task(1, 1)
    for cand in [TEACHER] + [t for t in TEACHER_FALLBACKS if t != TEACHER]:
        try:
            agent = ModelAgent(HFRunner(cand, name="glm", trust_remote_code=True,
                                        batch_size=batch), name="glm")
            agent.act(probe)   # forces the lazy load + one real generation
            return agent, cand
        except Exception as e:
            log(f"teacher {cand} unusable ({type(e).__name__}: {str(e)[:140]}); next")
    raise RuntimeError("no teacher candidate could be loaded")


def sft_from_teacher_actions(pile):
    """(observation -> the TEACHER's own recorded action) — distill from GLM's play, NOT
    the oracle. This is the regime difference vs sft_examples_from_pile (which uses the
    oracle optimal action as the target)."""
    ex = []
    for r in pile:
        for i in range(len(r.actions)):
            prompt = ModelAgent(None)._prompt(r.states[i])[0]["content"]
            ex.append(SFTExample(prompt=prompt, target=r.actions[i], domain="grid", k=i))
    return ex


def _agree(student_actions, states, glm_actions):
    """Return (oracle_rate, glm_rate) over a batch: oracle = action in the optimal set;
    glm = action exactly matches GLM's action at that state."""
    o = [1.0 if a in G.oracle_optimal_actions(s) else 0.0 for a, s in zip(student_actions, states)]
    g = [1.0 if sa == ga else 0.0 for sa, ga in zip(student_actions, glm_actions)]
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    return mean(o), mean(g), len(states)


def teacher_state(pile, student, glm, k_buckets):
    """Score the student from GLM's visited states. GLM's reference action at state[k] is
    the pile's recorded action[k] — no live GLM query needed here."""
    out_o, out_g, n = {}, {}, {}
    for k in k_buckets:
        idx = [i for i, r in enumerate(pile) if k < len(r.states) and not r.states[k].solved]
        states = [pile[i].states[k] for i in idx]
        if not states:
            continue
        s_acts = _batch(student, states)
        glm_acts = [pile[i].actions[k] for i in idx]   # GLM's recorded action at this state
        o, g, c = _agree(s_acts, states, glm_acts)
        out_o[k], out_g[k], n[k] = o, g, c
    return out_o, out_g, n


def self_state(pile, student, glm, k_buckets):
    """Roll the student on its OWN actions (lockstep across rollouts); at each k score the
    drifted state under both references. The GLM reference here is a LIVE GLM query at the
    student-reached state (GLM never visited it — this is the expensive, honest part)."""
    out_o, out_g, n = {}, {}, {}
    kset, maxk = set(k_buckets), max(k_buckets)
    cur = [r.states[0] for r in pile]
    alive = [True] * len(pile)
    for k in range(maxk + 1):
        if k in kset:
            idx = [i for i in range(len(cur)) if alive[i] and not cur[i].solved]
            states = [cur[i] for i in idx]
            if states:
                s_acts = _batch(student, states)
                glm_acts = _batch(glm, states)    # live GLM at the student's drifted state
                o, g, c = _agree(s_acts, states, glm_acts)
                out_o[k], out_g[k], n[k] = o, g, c
        if k == maxk:
            break
        live = [i for i in range(len(cur)) if alive[i] and not cur[i].solved]
        acts = _batch(student, [cur[i] for i in live])
        for i, a in zip(live, acts):
            cur[i], _, _ = G.step(cur[i], a)
        for i in range(len(cur)):
            if alive[i] and cur[i].solved:
                alive[i] = False
    return out_o, out_g, n


def _overall(d):
    vals = [v for v in d.values() if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def score(name, student, eval_pile, glm):
    ts_o, ts_g, ts_n = teacher_state(eval_pile, student, glm, K_BUCKETS)
    ss_o, ss_g, ss_n = self_state(eval_pile, student, glm, K_BUCKETS)
    r = {
        "oracle_ref": {"teacher_state": _overall(ts_o), "self_state": _overall(ss_o),
                       "ts_by_k": ts_o, "ss_by_k": ss_o},
        "glm_ref": {"teacher_state": _overall(ts_g), "self_state": _overall(ss_g),
                    "ts_by_k": ts_g, "ss_by_k": ss_g},
    }
    for ref in ("oracle_ref", "glm_ref"):
        ts, ss = r[ref]["teacher_state"] or 0.0, r[ref]["self_state"] or 0.0
        log(f"{name:8} [{ref:10}] teacher_state={ts:.3f} self_state={ss:.3f} gap={ts-ss:+.3f}")
    return r


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    log(f"teacher={TEACHER} student_base={STUDENT_BASE} ntrain={NTRAIN} neval={NEVAL} "
        f"diff={DIFF} horizon={HORIZON} epochs={EPOCHS}")

    # GLM as the teacher agent. trust_remote_code allowed for the OPERATOR-CHOSEN teacher
    # (not a miner checkpoint); the student below loads through the locked-down runner.
    glm, teacher_used = build_teacher(BATCH)
    log(f"teacher in use: {teacher_used}")

    log("GLM authoring train + eval grids (GLM plays the env) ...")
    train_pile = author_pile(NTRAIN, difficulty=DIFF, teacher=glm, horizon=HORIZON, seed0=10_000)
    eval_pile = author_pile(NEVAL, difficulty=DIFF, teacher=glm, horizon=HORIZON, seed0=90_000)

    # fail-fast competence probe: can GLM even play this env?
    solved = sum(r.success for r in eval_pile)
    avg_len = round(sum(len(r.actions) for r in eval_pile) / max(1, len(eval_pile)), 1)
    log(f"GLM competence: solved {solved}/{len(eval_pile)} eval grids, "
        f"mean trajectory length {avg_len}")
    if solved < 0.15 * len(eval_pile):
        log("WARNING: GLM solves <15% of grids — trajectories may be too degenerate to "
            "test drift cleanly. Result below is INCONCLUSIVE if the gap is ~0; swap "
            "RALPH_TEACHER to a model that can play the env and rerun.")

    examples = sft_from_teacher_actions(train_pile)
    log(f"distilling S_glm from {len(examples)} (state -> GLM-action) pairs ...")
    d_glm = distill_student(STUDENT_BASE, examples, f"{OUT}/s_glm", epochs=EPOCHS,
                            batch_size=max(8, BATCH), log=log)

    report = {"teacher": teacher_used, "student_base": STUDENT_BASE, "diff": DIFF,
              "horizon": HORIZON, "n_eval": len(eval_pile), "k_buckets": K_BUCKETS,
              "glm_solved": solved, "glm_mean_len": avg_len, "scores": {}}

    report["scores"]["base"] = score(
        "base", ModelAgent(HFRunner(STUDENT_BASE, name="base", batch_size=BATCH)), eval_pile, glm)
    report["scores"]["S_glm"] = score(
        "S_glm", ModelAgent(SafeStudentRunner(d_glm, name="S_glm", batch_size=BATCH)), eval_pile, glm)

    with open(f"{OUT}/report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n===== S1: does the drift gap survive the GLM reference? =====")
    print(json.dumps(report["scores"], indent=2))

    # the falsification read-out
    sg = report["scores"]["S_glm"]
    gap_oracle = (sg["oracle_ref"]["teacher_state"] or 0) - (sg["oracle_ref"]["self_state"] or 0)
    gap_glm = (sg["glm_ref"]["teacher_state"] or 0) - (sg["glm_ref"]["self_state"] or 0)
    log(f"S_glm drift gap: oracle_ref={gap_oracle:+.3f}  glm_ref={gap_glm:+.3f}")
    if gap_glm >= 0.15:
        log("VERDICT: drift gap SURVIVES the GLM reference — differentiator holds off the "
            "oracle; S2 (full agentic in-regime run) is justified.")
    elif gap_oracle >= 0.15 and gap_glm < 0.10:
        log("VERDICT: drift gap COLLAPSES under the GLM reference (present under oracle, "
            "gone under GLM) — the differentiator was an oracle artifact. Rethink before S2.")
    else:
        log("VERDICT: INCONCLUSIVE — no clear gap under either reference (check GLM "
            "competence above; the env may be too easy/short or GLM too weak).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
