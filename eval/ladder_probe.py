"""Does observer-KL ORDER compressions by quality? The one question that decides the crown.

The first ladder run answered two of three things and botched the third.

WHAT IT ESTABLISHED. The fp16 parent scores EXACTLY 1.0 — the metric identifies a perfect
compression perfectly, which is the sanity check the whole design needs to pass. And within a box
the round is bit-reproducible.

WHAT IT EXPOSED. 8-bit is strictly less lossy than 4-bit, so it must never score lower. On an A100
it did the right thing (0.7287 > 0.6407); on an L40S the order INVERTED (0.5852 < 0.6114). A metric
that ranks a worse compression above a better one cannot be the basis of a king-of-the-hill crown,
because the crown is exactly a ranking claim.

WHAT IT GOT WRONG. The "damaged" control perturbed a model already loaded in 4-bit, whose weights
are `Params4bit` uint8 — so `dtype.is_floating_point` was False for all of them and exactly ONE
tensor was touched. The alarming result (damaged scoring ABOVE 4-bit) is therefore an artefact of a
broken control, not a finding. It had to be redone before drawing any conclusion from it.

THIS PROBE. A monotone degradation curve on fp16, where every parameter really is floating point:
perturb every 2-D weight by sigma x its own std, for sigma in a sweep. A sound metric must be
monotonically decreasing in sigma. Reported alongside the honest quantizations so the two are
directly comparable — the question is not only "does the score fall" but "does a REAL 4-bit
quantization land above a model damaged by the same amount of KL".

    python -m eval.ladder_probe
    env: RALPH_PARENT RALPH_OBSERVER RALPH_N_TRAJ RALPH_SIGMAS RALPH_OUT RALPH_TAG
"""
from __future__ import annotations

import json
import os
import time

PARENT = os.environ.get("RALPH_PARENT", "Qwen/Qwen2.5-0.5B-Instruct")
OBSERVER = os.environ.get("RALPH_OBSERVER", "Qwen/Qwen2.5-1.5B-Instruct")
N_TRAJ = int(os.environ.get("RALPH_N_TRAJ", "24"))
SIGMAS = [float(x) for x in
          os.environ.get("RALPH_SIGMAS", "0.0,0.01,0.02,0.05,0.10,0.20,0.30").split(",")]
TAG = os.environ.get("RALPH_TAG", "unknown-gpu")
OUT = os.environ.get("RALPH_OUT", "runs/ladder.json")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def damage(runner, sigma: float, seed: int = 0) -> int:
    """Perturb every 2-D floating-point weight by sigma x its own std. Returns tensors touched.

    On an fp16/bf16 model this really does hit every projection; the previous version ran against a
    4-bit model where the packed weights are uint8, silently touched one tensor, and produced a
    'damaged' model that was essentially undamaged."""
    import torch
    runner._load()
    # SEEDED. Without this each sigma drew a fresh unreproducible perturbation, so the points on
    # the curve were not nested draws and a non-monotone result could not be distinguished from
    # sampling luck. The first run reported monotone=False on two of three GPUs and that verdict
    # was therefore uninterpretable.
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = 0
    with torch.no_grad():
        for _, prm in runner._model.named_parameters():
            if prm.dtype.is_floating_point and prm.dim() >= 2:
                std = prm.detach().float().std()
                if torch.isfinite(std) and std > 0:
                    noise = torch.randn(prm.shape, generator=g, dtype=torch.float32)
                    prm.add_((noise.to(prm.device) * (sigma * std)).to(prm.dtype))
                    n += 1
    return n


def main() -> int:
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    from .observer_round import build_shared, score_submission
    from .runners import HFRunner
    from .shakedown_round_noise import load_trajectories, _gpu, _sha

    rep = {"tag": TAG, "gpu": _gpu(), "parent": PARENT, "observer": OBSERVER, "sigmas": SIGMAS}
    log("gpu:", rep["gpu"])
    trajs = load_trajectories(N_TRAJ)
    rep["items_sha"] = _sha(t.prefix for t in trajs)
    log(f"{len(trajs)} trajectories, items_sha={rep['items_sha']}")

    parent = HFRunner(PARENT)
    observer = HFRunner(OBSERVER)
    shared = build_shared(trajs, parent, observer, "obs")
    rep["usable"] = sum(1 for s in shared if s.usable)
    rep["parent_steps_sha"] = _sha(s.parent_step for s in shared)

    curve, step_sha = {}, {}

    def run(name, runner):
        t0 = time.time()
        # hash the generated steps too: it separates "the score moved because the model now says
        # something different" from "the score moved but the text did not", which are different
        # problems with different fixes
        steps = runner.generate([s.prefix for s in shared if s.usable], 256)
        step_sha[name] = _sha(steps)
        ms = score_submission(shared, runner, observer)
        curve[name] = round(ms.score, 6)
        log(f"  {name:<16} {ms.score:.6f}  steps={step_sha[name]}  ({time.time() - t0:.0f}s)"
            + (f"  reasons={ms.reasons}" if ms.reasons else ""))

    run("fp16_parent", parent)
    for bits, kw in (("8bit", {"load_in_8bit": True}), ("4bit", {"load_in_4bit": True})):
        try:
            r = HFRunner(PARENT, **kw)
            run(f"quant_{bits}", r)
            del r
        except Exception as e:
            curve[f"quant_{bits}"] = None
            log(f"  quant_{bits} FAILED: {e}")

    # the monotone curve: fp16 weights, real perturbation, increasing magnitude
    touched = {}
    for sg in SIGMAS:
        d = HFRunner(PARENT)
        touched[str(sg)] = damage(d, sg, seed=1234)
        run(f"noise_{sg}", d)
        del d
    rep["tensors_perturbed"] = touched
    rep["curve"] = curve
    rep["step_sha"] = step_sha
    # SIGMA=0 IS THE CONTROL. Same code path as every other rung — load, call damage(), score —
    # but scaled by zero, so it must return exactly the fp16 parent's 1.0. If it does not, the
    # damage path itself perturbs something and the whole curve is measuring the harness.
    if 0.0 in SIGMAS:
        rep["zero_control_ok"] = abs((curve.get("noise_0.0") or 0) - 1.0) < 1e-9
        rep["zero_control_matches_parent"] = (
            step_sha.get("noise_0.0") == step_sha.get("fp16_parent"))

    # ---- verdicts ------------------------------------------------------------------------
    v = {}
    v["parent_is_one"] = abs((curve.get("fp16_parent") or 0) - 1.0) < 1e-6
    q8, q4 = curve.get("quant_8bit"), curve.get("quant_4bit")
    v["quant_order_correct"] = (q8 is not None and q4 is not None and q8 >= q4)
    v["quant_gap"] = None if (q8 is None or q4 is None) else round(q8 - q4, 6)
    ns = [curve.get(f"noise_{s}") for s in SIGMAS]
    if all(x is not None for x in ns):
        v["noise_monotone"] = all(a >= b - 1e-9 for a, b in zip(ns, ns[1:]))
        v["noise_range"] = round(max(ns) - min(ns), 6)
    # the question that decides whether a crown means anything: does a REAL 4-bit quantization
    # outrank a model damaged badly enough to be useless?
    worst = curve.get(f"noise_{SIGMAS[-1]}")
    v["quant_beats_worst_damage"] = (q4 is not None and worst is not None and q4 > worst + 0.05)
    rep["verdict"] = v

    json.dump(rep, open(OUT, "w"), indent=1)
    log("")
    log(f"curve  : {curve}")
    log(f"steps  : {step_sha}")
    log(f"zero control: ok={rep.get('zero_control_ok')} "
        f"same_text={rep.get('zero_control_matches_parent')}")
    log(f"touched: {touched}")
    log(f"verdict: {v}")
    log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
