"""Capture raw model outputs per axis to diagnose CHECKER robustness.

The capability run validated the crown RANKING but exposed that a checker can zero a
capable model on a formatting quirk (the code axis scored a correct 3B at 0/22, and later
a 1.5B at 0/43). Absolute per-axis retention is only trustworthy if the checkers grade
diverse output STYLES correctly. This dumps, per (model, axis, item): the raw output, the
check() result, and the extracted answer — so failures are diagnosable OFFLINE and the
checkers can be hardened + re-validated on CPU without another GPU run.

No teacher needed: check() grades against the deterministic item.answer, not GLM — so this
runs the small students only (fast/cheap).

    python -m eval.capture_checker_outputs   # writes runs/capture.json
"""
from __future__ import annotations

import json
import os

from eval.axes.code_exec import CodeExec
from eval.axes.instruction import InstructionFollowing
from eval.axes.long_context import LongContext
from eval.axes.math_gsm import MathGSM
from eval.runners import HFRunner

MODELS = os.environ.get(
    "RALPH_MODELS",
    "Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-0.5B-Instruct",
).split(",")
N = int(os.environ.get("RALPH_N", "40"))
OUT = os.environ.get("RALPH_OUT", "runs/capture.json")
# same calibration as run_capability_axis
AXES = [(MathGSM(), "math", 3), (CodeExec(), "code", 2),
        (InstructionFollowing(), "instruction", 1), (LongContext(base_facts=12), "long_context", 1)]


def _extracted(ax, output):
    """Best-effort 'what the checker pulled out' for diagnosis (axis-specific)."""
    try:
        if isinstance(ax, CodeExec):
            c = ax._extract_code(output)
            return (c or "")[:200]
        if isinstance(ax, MathGSM):
            from eval.axes.math_gsm import extract_answer
            return extract_answer(output)
    except Exception as e:
        return f"<err {e}>"
    return None


def main() -> int:
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    out: dict = {}
    for mid in MODELS:
        r = HFRunner(mid, name=mid, batch_size=16)
        out[mid] = {}
        for ax, name, diff in AXES:
            items = ax.generate(123, N, diff)
            outs = r.generate([it.prompt for it in items], 512)
            recs = []
            for it, o in zip(items, outs):
                recs.append({"answer": str(it.answer)[:80], "qtype": it.meta.get("qtype", ""),
                             "passed": ax.check(it, o), "extracted": _extracted(ax, o),
                             "raw": o[:2000]})
            npass = sum(x["passed"] for x in recs)
            out[mid][name] = {"n_pass": npass, "n": N, "recs": recs}
            print(f"{mid.split('/')[-1]:22} {name:14} pass {npass}/{N}", flush=True)
        del r
    json.dump(out, open(OUT, "w"), indent=1)
    print("CAPTURE DONE ->", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
