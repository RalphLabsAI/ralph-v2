"""Whole-round noise floor, measured in the units the crown is actually decided in.

WHY THE EXISTING MEASUREMENT IS NOT ENOUGH. `determinism.measure_noise` probes one thing:
`observer.distributions(prefix, continuation)` repeated on a single sequence. That produces a number
in KL units, and the crown margin — the thing it gates — is a difference of RETENTION scores in
[0, 1]. Gating a retention margin on a KL floor is a category error: the two are not the same
quantity and there is no reason a factor of 3 on one bounds the other. It also leaves out most of
the round. A real round runs FOUR batched greedy generations (parent steps, observer continuations,
each miner's steps, and the incumbent's re-score) and hundreds of batched forward passes, and
batching is exactly where GPU nondeterminism lives — a sequence scored next to different neighbours
can come back with different logits. So the earlier "noise floor is 0.0" was not a finding: it was
the answer to a much smaller question, asked of a stub.

WHAT THIS MEASURES INSTEAD. The full round, K times, on identical inputs, and the spread of the
final scores:

    A. TEXT STABILITY   do the four generations return byte-identical strings across repeats?
                        A text change is categorically worse than a logit wobble — it means the
                        round scored a different exam, not the same exam slightly differently.
    B. SCORE SPREAD     max |retention_i - retention_j| over repeats, IN RETENTION UNITS. This is
                        the number the dethrone margin must clear, and it is directly comparable to
                        it. It is the deliverable of this script.
    C. BATCH SENSITIVITY the same submission scored alone vs alongside decoys. If this moves, a
                        miner's score depends on who else submitted that round, which is a fairness
                        bug independent of its size.
    D. KL FLOOR         the old single-sequence number, kept for continuity and because it is what
                        the current code gates on.

FALSIFIABILITY. Two controls, so a green run means something: a genuine 4-bit compression of the
parent should score HIGH, and a DIFFERENT model of similar size should score LOW. If they score the
same, observer-KL is not measuring compression and the run has failed even though nothing crashed.

    python -m eval.shakedown_round_noise
    env: RALPH_PARENT RALPH_OBSERVER RALPH_OTHER RALPH_N_TRAJ RALPH_REPEATS RALPH_OUT RALPH_TAG
"""
from __future__ import annotations

import json
import os
import time

PARENT = os.environ.get("RALPH_PARENT", "Qwen/Qwen2.5-0.5B-Instruct")
OBSERVER = os.environ.get("RALPH_OBSERVER", "Qwen/Qwen2.5-1.5B-Instruct")
OTHER = os.environ.get("RALPH_OTHER", "HuggingFaceTB/SmolLM2-360M-Instruct")
N_TRAJ = int(os.environ.get("RALPH_N_TRAJ", "24"))
REPEATS = int(os.environ.get("RALPH_REPEATS", "3"))
TAG = os.environ.get("RALPH_TAG", "unknown-gpu")
OUT = os.environ.get("RALPH_OUT", "runs/round_noise.json")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def _gpu() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"device": "cpu"}
        return {"device": torch.cuda.get_device_name(0),
                "capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
                "torch": torch.__version__, "cuda": torch.version.cuda}
    except Exception as e:
        return {"device": f"unknown ({e})"}


def load_trajectories(n: int):
    from datasets import load_dataset
    from .steps import TRAJECTORY_SOURCES, extract_steps
    spec = TRAJECTORY_SOURCES["glaive_r1"]
    ds = load_dataset(spec["dataset"], split=spec["split"], streaming=True)
    trajs, seen = [], 0
    for row in ds:
        seen += 1
        trajs += extract_steps(row, spec["step"], spec["text_field"], "glaive_r1",
                               f"r{seen}", max_steps=2)
        if len(trajs) >= n:
            break
    return trajs[:n]


def main() -> int:
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    from .determinism import measure_noise
    from .observer_round import build_shared, score_submission
    from .runners import HFRunner

    rep: dict = {"tag": TAG, "gpu": _gpu(), "parent": PARENT, "observer": OBSERVER,
                 "other": OTHER, "n_traj": N_TRAJ, "repeats": REPEATS}
    log("gpu:", rep["gpu"])

    log("fetching real trajectories (glaive_r1)…")
    trajs = load_trajectories(N_TRAJ)
    rep["n_loaded"] = len(trajs)
    if not trajs:
        rep["error"] = "no trajectories extracted"
        json.dump(rep, open(OUT, "w"), indent=1)
        return 1

    log(f"loading parent {PARENT}…")
    parent = HFRunner(PARENT)
    log(f"loading observer {OBSERVER}…")
    observer = HFRunner(OBSERVER)

    # the STUDENT is a real compression: same checkpoint, 4-bit storage, architecture untouched
    log("loading 4-bit student (same checkpoint, narrower storage)…")
    try:
        student = HFRunner(PARENT, load_in_4bit=True)
        rep["student"] = f"{PARENT}@4bit"
    except Exception as e:
        log(f"  4-bit load failed ({e}); falling back to the fp16 parent as the student")
        student = parent
        rep["student"] = f"{PARENT}@fp16 (4-bit unavailable)"
        rep["student_warning"] = str(e)
    log(f"loading control {OTHER} (a DIFFERENT model — must score LOW)…")
    other = HFRunner(OTHER)

    # ---- A/B: the whole round, REPEATS times, on identical inputs ------------------------
    runs = []
    for i in range(REPEATS):
        t0 = time.time()
        shared = build_shared(trajs, parent, observer, "obs")
        usable = [s for s in shared if s.usable]
        ms_student = score_submission(shared, student, observer)
        ms_other = score_submission(shared, other, observer)
        runs.append({
            "usable": len(usable),
            "parent_steps": [s.parent_step for s in shared],
            "continuations": [s.continuation for s in shared],
            "student": ms_student.score, "other": ms_other.score,
            "student_slices": {k: list(v) for k, v in ms_student.slice_samples.items()},
            "student_reasons": ms_student.reasons, "other_reasons": ms_other.reasons,
            "secs": round(time.time() - t0, 1),
        })
        log(f"  repeat {i + 1}/{REPEATS}: usable={len(usable)} "
            f"student={ms_student.score:.6f} other={ms_other.score:.6f} "
            f"({runs[-1]['secs']}s)")

    rep["runs"] = [{k: v for k, v in r.items()
                    if k not in ("parent_steps", "continuations", "student_slices")}
                   for r in runs]

    # A. text stability — did the generations come back identical?
    rep["text_identical"] = all(r["parent_steps"] == runs[0]["parent_steps"]
                                and r["continuations"] == runs[0]["continuations"]
                                for r in runs)
    if not rep["text_identical"]:
        diffs = sum(1 for r in runs for a, b in zip(r["parent_steps"], runs[0]["parent_steps"])
                    if a != b)
        rep["text_diff_count"] = diffs

    # B. THE DELIVERABLE: score spread in retention units
    sv = [r["student"] for r in runs]
    ov = [r["other"] for r in runs]
    rep["student_score_spread"] = max(sv) - min(sv)
    rep["other_score_spread"] = max(ov) - min(ov)
    rep["round_noise_retention"] = max(rep["student_score_spread"], rep["other_score_spread"])
    # per-sample spread too: the dethrone test bootstraps over these vectors, so a large
    # per-sample wobble matters even when the means happen to agree
    per_sample = []
    for k, v0 in runs[0]["student_slices"].items():
        for r in runs[1:]:
            v1 = r["student_slices"].get(k, [])
            per_sample += [abs(a - b) for a, b in zip(v0, v1)]
    rep["max_per_sample_delta"] = max(per_sample) if per_sample else 0.0

    # C. batch sensitivity — does a miner's score depend on who else submitted?
    log("batch-composition probe…")
    alone = score_submission([s for s in build_shared(trajs[:8], parent, observer, "obs")],
                             student, observer)
    padded_shared = build_shared(trajs[:8] + trajs[8:16], parent, observer, "obs")
    padded = score_submission(padded_shared[:8], student, observer)
    rep["batch_alone"] = alone.score
    rep["batch_padded"] = padded.score
    rep["batch_sensitivity"] = abs(alone.score - padded.score)

    # D. the old single-sequence KL floor, for continuity with what the code gates on today
    probe = [s for s in build_shared(trajs[:4], parent, observer, "obs") if s.usable]
    if probe:
        n = measure_noise(observer, probe[0].prefix + "\n" + probe[0].parent_step,
                          probe[0].continuation, repeats=REPEATS,
                          batch_variations=[p.prefix for p in probe[1:]])
        rep["kl_floor"] = n.as_dict()

    # ---- discrimination: is the metric measuring compression at all? ---------------------
    rep["student_mean"] = sum(sv) / len(sv)
    rep["other_mean"] = sum(ov) / len(ov)
    rep["discrimination"] = rep["student_mean"] - rep["other_mean"]
    rep["verdict"] = {
        "discriminates": rep["discrimination"] > 0.15,
        "margin_resolvable": rep["round_noise_retention"] < 0.05,
        "text_stable": rep["text_identical"],
    }

    json.dump(rep, open(OUT, "w"), indent=1)
    log("")
    log(f"ROUND NOISE (retention units) : {rep['round_noise_retention']:.6f}")
    log(f"  max per-sample delta        : {rep['max_per_sample_delta']:.6f}")
    log(f"  batch sensitivity           : {rep['batch_sensitivity']:.6f}")
    log(f"  generations byte-identical  : {rep['text_identical']}")
    log(f"  KL floor (old measurement)  : {rep.get('kl_floor', {}).get('max_kl')}")
    log(f"  4-bit student  mean score   : {rep['student_mean']:.4f}")
    log(f"  different model mean score  : {rep['other_mean']:.4f}")
    log(f"  DISCRIMINATION              : {rep['discrimination']:+.4f}")
    log(f"  verdict: {rep['verdict']}")
    log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
