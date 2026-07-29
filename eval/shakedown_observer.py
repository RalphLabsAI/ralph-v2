"""First real-model shakedown of the observer-KL crown.

Answers the two questions nothing so far has: does the pipeline run against real models, and
WHAT IS THE NOISE FLOOR. The second one decides whether the mechanism works at all — every
crown is gated on it, and if it lands where SN97's did (2-5 percentage points on logit-derived
metrics) then observer-KL cannot resolve honest differences and the aggregation needs retuning.

Deliberately small models. The question is "does the mechanism discriminate and is it
reproducible", which a 0.5B parent answers as well as a 9B one and for a fraction of the rent.
GLM-scale comes after the mechanism is proven, not before.

The student is a REAL compression of the pinned parent: the same checkpoint loaded in 4-bit, so
the architecture is unchanged and only the storage width differs — exactly the task. Two controls
make the result falsifiable rather than merely printable:

    quantized parent  -> should score HIGH (it is a genuine compression)
    a DIFFERENT model -> should score LOW  (it is not this parent at all)

If the different model scores as well as the quantized parent, the metric is not measuring
compression and the run has failed even if nothing crashed.

    python -m eval.shakedown_observer          # env: RALPH_PARENT, RALPH_OBSERVER, RALPH_OTHER
"""
from __future__ import annotations

import json
import os
import time

PARENT = os.environ.get("RALPH_PARENT", "Qwen/Qwen2.5-0.5B-Instruct")
OBSERVER = os.environ.get("RALPH_OBSERVER", "Qwen/Qwen2.5-1.5B-Instruct")
OTHER = os.environ.get("RALPH_OTHER", "HuggingFaceTB/SmolLM2-360M-Instruct")
N_TRAJ = int(os.environ.get("RALPH_N_TRAJ", "24"))
OUT = os.environ.get("RALPH_OUT", "runs/shakedown_observer.json")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main() -> int:
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    from .determinism import margin_floor, measure_noise
    from .observer_round import build_shared, score_submission
    from .runners import HFRunner
    from .steps import TRAJECTORY_SOURCES, Trajectory, extract_steps, verify_source

    report: dict = {"parent": PARENT, "observer": OBSERVER, "other": OTHER}

    # ---- 1. real trajectories from a verified source -------------------------------------
    log("fetching trajectories (glaive_r1)...")
    from datasets import load_dataset
    spec = TRAJECTORY_SOURCES["glaive_r1"]
    ds = load_dataset(spec["dataset"], split=spec["split"], streaming=True)
    rows, trajs = [], []
    for row in ds:
        rows.append(row)
        trajs += extract_steps(row, spec["step"], spec["text_field"], "glaive_r1",
                               f"r{len(rows)}", max_steps=2)
        if len(trajs) >= N_TRAJ:
            break
    ok, msg = verify_source("glaive_r1", rows[:8])
    log(f"source check: {msg}")
    assert ok, msg
    trajs = trajs[:N_TRAJ]
    # keep prefixes short enough that the observer forward pass is cheap
    trajs = [Trajectory(id=t.id, source=t.source, prefix=t.prefix[-1200:], step=t.step,
                        index=t.index, meta=t.meta) for t in trajs]
    log(f"{len(trajs)} trajectory steps, mean prefix {sum(len(t.prefix) for t in trajs)//len(trajs)} chars")
    report["n_trajectories"] = len(trajs)

    # ---- 2. models: pinned parent, its 4-bit compression, a control, and the observer ----
    log(f"loading parent {PARENT} (bf16) ...")
    parent = HFRunner(PARENT, name="parent", batch_size=8)
    log(f"loading observer {OBSERVER} ...")
    observer = HFRunner(OBSERVER, name="observer", batch_size=4)

    log(f"loading student = {PARENT} in 4-bit (a real compression of the pinned parent) ...")
    student = HFRunner(PARENT, name="student-4bit", batch_size=8)
    student._quant4 = True                      # see _patch_4bit below
    _patch_4bit(student)

    log(f"loading control = {OTHER} (a DIFFERENT model; must score lower) ...")
    other = HFRunner(OTHER, name="control-other", batch_size=8)

    # ---- 3. shared rollouts (once) --------------------------------------------------------
    t0 = time.time()
    shared = build_shared(trajs, parent, observer, observer_name="qwen1.5b",
                          max_step_tokens=96, max_cont_tokens=48)
    usable = [s for s in shared if s.usable]
    log(f"shared rollouts: {len(usable)}/{len(shared)} usable in {time.time()-t0:.0f}s")
    report["usable"] = len(usable)
    report["unusable_reasons"] = [s.reason for s in shared if not s.usable][:5]
    if not usable:
        log("ABORT: the parent moved the observer nowhere on any sample")
        json.dump(report, open(OUT, "w"), indent=2)
        return 1
    report["d_parent_mean"] = sum(s.d_parent for s in usable) / len(usable)
    log(f"mean parent effect d_G = {report['d_parent_mean']:.4f}")

    # ---- 4. THE NOISE FLOOR — the number this whole run exists to produce ----------------
    log("measuring noise floor (same input, repeated, with batch perturbation) ...")
    probe = usable[0]
    noise = measure_noise(observer, probe.prefix + "\n" + probe.parent_step, probe.continuation,
                          repeats=4, batch_variations=[s.prefix for s in usable[1:4]])
    report["noise"] = noise.as_dict()
    report["margin_floor"] = margin_floor(noise)
    log(f"NOISE: identical={noise.identical} max_kl={noise.max_kl:.3e} "
        f"max_prob_delta={noise.max_prob_delta:.3e} -> margin floor {margin_floor(noise):.6f}")

    # ---- 5. discrimination: quantized parent vs a different model ------------------------
    for label, model in (("student_4bit", student), ("control_other", other)):
        t1 = time.time()
        ms = score_submission(shared, model, observer, max_step_tokens=96)
        report[label] = ms.as_dict()
        log(f"{label:14} score={ms.score:.4f} scored={ms.n_scored} "
            f"mean_effect={ms.mean_effect:.4f} ({time.time()-t1:.0f}s) {ms.reasons}")

    # ---- 6. verdict -----------------------------------------------------------------------
    s_q = report["student_4bit"]["score"]
    s_o = report["control_other"]["score"]
    report["discriminates"] = bool(s_q > s_o)
    report["margin"] = s_q - s_o
    report["margin_clears_noise"] = bool((s_q - s_o) > report["margin_floor"])
    log(f"VERDICT: quantized-parent {s_q:.4f} vs different-model {s_o:.4f} "
        f"| discriminates={report['discriminates']} "
        f"| margin {s_q - s_o:.4f} vs floor {report['margin_floor']:.6f} "
        f"-> usable={report['margin_clears_noise']}")
    json.dump(report, open(OUT, "w"), indent=2)
    log(f"wrote {OUT}")
    return 0


def _patch_4bit(runner):
    """Load this runner's checkpoint in 4-bit. A genuine compression of the pinned parent:
    identical architecture, fewer bits per weight."""
    import types

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._tok.padding_side = "left"
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16),
        )
        self._model.eval()

    runner._load = types.MethodType(_load, runner)


if __name__ == "__main__":
    raise SystemExit(main())
