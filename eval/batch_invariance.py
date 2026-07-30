"""Does a miner's score depend on WHO ELSE submitted that round? Measured, not assumed.

THE FINDING THIS EXISTS TO CHASE DOWN. The first batch probe was vacuous — it built a 16-sample set
and then sliced back to 8 BEFORE scoring, so the batch never changed — and it reported a reassuring
0.0. Fixed to actually re-partition the work, it reports:

    A100  batch_size 8 -> 0.6244    batch_size 3 -> 0.6703     delta 0.046
    L40S  batch_size 8 -> 0.4837    batch_size 3 -> 0.3865     delta 0.097

Both deltas are at or above the 0.05 dethrone margin. The number of samples in a batch is a function
of how many submissions the round is processing, so as written, A MINER'S SCORE MOVES BY MORE THAN
THE DETHRONE MARGIN DEPENDING ON HOW MANY OTHER MINERS SUBMITTED. That is not a rounding concern; it
means the crown is not a well-defined function of the submitted artifact, and two honest miners can
swap places for reasons neither of them controls.

Padding is the mechanism. HFRunner pads left to the longest sequence IN THE BATCH, so the same
prompt gets a different number of pad tokens depending on its neighbours, the attention mask and
position ids shift, and greedy decoding flips on near-ties. Different text, different score. This is
also why the cross-GPU numbers moved: on identical items the A100 and L40S produced DIFFERENT parent
steps (hashes 4056dac7 vs 152dc2e4), so they were not scoring the same exam at all.

WHAT THIS MEASURES. The same samples scored under several batch sizes, plus batch_size=1 as the
reference — with one sequence there is no padding and no neighbours, so if anything is invariant it
is that. The deliverable is: which configuration makes the score a function of the artifact alone.

    python -m eval.batch_invariance
    env: RALPH_PARENT RALPH_OBSERVER RALPH_N_TRAJ RALPH_BATCHES RALPH_OUT RALPH_TAG
"""
from __future__ import annotations

import json
import os
import time

PARENT = os.environ.get("RALPH_PARENT", "Qwen/Qwen2.5-0.5B-Instruct")
OBSERVER = os.environ.get("RALPH_OBSERVER", "Qwen/Qwen2.5-1.5B-Instruct")
N_TRAJ = int(os.environ.get("RALPH_N_TRAJ", "16"))
BATCHES = [int(x) for x in os.environ.get("RALPH_BATCHES", "1,2,3,5,8").split(",")]
TAG = os.environ.get("RALPH_TAG", "unknown-gpu")
OUT = os.environ.get("RALPH_OUT", "runs/batch_invariance.json")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main() -> int:
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    from .observer_round import build_shared, score_submission
    from .runners import HFRunner
    from .shakedown_round_noise import _gpu, _sha, load_trajectories

    rep = {"tag": TAG, "gpu": _gpu(), "parent": PARENT, "observer": OBSERVER,
           "batches": BATCHES, "n_traj": N_TRAJ}
    log("gpu:", rep["gpu"])
    trajs = load_trajectories(N_TRAJ)
    rep["items_sha"] = _sha(t.prefix for t in trajs)

    observer = HFRunner(OBSERVER)
    parent = HFRunner(PARENT)

    # THE SHARED SET IS BUILT ONCE, at a fixed batch size, so only the MINER's batching varies.
    # Otherwise parent steps and continuations move too and the probe cannot attribute the change.
    shared = build_shared(trajs, parent, observer, "obs")
    rep["usable"] = sum(1 for s in shared if s.usable)
    rep["parent_steps_sha"] = _sha(s.parent_step for s in shared)
    log(f"shared set: {rep['usable']} usable, parent_steps_sha={rep['parent_steps_sha']}")

    student = HFRunner(PARENT, load_in_4bit=True)
    scores, step_hashes = {}, {}
    for bs in BATCHES:
        student._batch_size = bs
        observer._batch_size = bs
        t0 = time.time()
        # capture the miner's generated text as well as the score: if the TEXT moves, the cause is
        # padding-dependent decoding, not float accumulation order, and the fix is different
        steps = student.generate([s.prefix for s in shared if s.usable], 256)
        ms = score_submission(shared, student, observer)
        scores[str(bs)] = round(ms.score, 6)
        step_hashes[str(bs)] = _sha(steps)
        log(f"  batch_size={bs:<2} score={ms.score:.6f} steps_sha={step_hashes[str(bs)]} "
            f"({time.time() - t0:.0f}s)")

    rep["scores"] = scores
    rep["step_hashes"] = step_hashes
    vals = list(scores.values())
    rep["score_spread"] = round(max(vals) - min(vals), 6)
    rep["text_stable_across_batches"] = len(set(step_hashes.values())) == 1
    rep["vs_bs1"] = {k: round(v - scores[str(BATCHES[0])], 6) for k, v in scores.items()}

    # is bs=1 itself stable? repeat it — this is the reference the fix would rely on
    student._batch_size = 1
    observer._batch_size = 1
    rep["bs1_repeat"] = round(score_submission(shared, student, observer).score, 6)
    rep["bs1_stable"] = abs(rep["bs1_repeat"] - scores[str(BATCHES[0])]) < 1e-9

    rep["verdict"] = {
        "score_is_batch_invariant": rep["score_spread"] < 1e-6,
        "spread_exceeds_dethrone_margin": rep["score_spread"] > 0.05,
        "text_moves_with_batching": not rep["text_stable_across_batches"],
        "bs1_reproducible": rep["bs1_stable"],
    }
    json.dump(rep, open(OUT, "w"), indent=1)
    log("")
    log(f"SCORE SPREAD across batch sizes: {rep['score_spread']:.6f}  "
        f"(dethrone margin is 0.05)")
    log(f"  text stable across batching  : {rep['text_stable_across_batches']}")
    log(f"  scores                       : {scores}")
    log(f"  verdict                      : {rep['verdict']}")
    log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
