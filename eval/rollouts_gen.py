"""Generate an experience pile with the teacher.

For a first real run we use reasoning traces (math / short coding) rather than agentic
tool-use, so the whole thing runs on one GPU with no environment. GLM solves each task
step by step; we split its solution into ordered steps. Each becomes a Rollout the eval
samples (rollout, step) points from.
"""
from __future__ import annotations

import re

from .core import ModelRunner
from .trajectory import Rollout

TASKS = [
    "A train travels 60 km in the first hour and 90 km in the second. What is its average speed? Reason step by step.",
    "Compute the sum of all integers from 1 to 50, showing your steps.",
    "A rectangle has perimeter 34 and width 6. Find its area, step by step.",
    "If 3 painters paint 3 fences in 3 hours, how long for 9 painters to paint 9 fences? Explain each step.",
    "Write a Python function is_palindrome(s) that ignores case and spaces, and explain each step before the code.",
    "Solve for x: 2x + 7 = 3x - 4. Show every step.",
    "A shirt costs 40 after a 20% discount. What was the original price? Step by step.",
    "How many 3-digit numbers are divisible by 7? Reason it out step by step.",
    "Write a Python function that returns the second-largest distinct value in a list, explaining each step first.",
    "A tank fills in 12 minutes with pipe A and 18 with pipe B. How long with both open? Step by step.",
    "Find the greatest common divisor of 84 and 126, showing the steps.",
    "Convert 3/8 to a decimal and then to a percentage, step by step.",
]


_MARKER = re.compile(r"^\s*(step\s*\d+\b|[0-9]+[.)]|#{1,6}\s|\*\*)", re.I)


def _split_steps(text: str, max_steps: int = 8, min_chars: int = 25) -> list[str]:
    """Split a solution into CONTENT-BEARING steps.

    A step marker (numbered / "Step N" / heading / bold) opens a step that includes
    everything up to the NEXT marker — so each step carries its actual reasoning, not a
    bare header (a header alone can't be locally compared: the student continues it with
    content, and 'content vs header' always mismatches). Falls back to paragraph blocks,
    then sentence groups, and merges any fragment shorter than `min_chars` into the next.
    """
    lines = text.splitlines()
    blocks, cur = [], []
    for ln in lines:
        if _MARKER.match(ln) and cur and any(c.strip() for c in cur):
            blocks.append("\n".join(cur).strip())
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        blocks.append("\n".join(cur).strip())
    steps = [b for b in blocks if b]

    if len(steps) < 2:  # no markers — paragraphs, then sentence pairs
        steps = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > min_chars]
    if len(steps) < 2:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        steps = ["\n".join(sents[i:i + 2]) for i in range(0, len(sents), 2)]

    # merge too-short fragments forward so every step has real content to compare
    merged: list[str] = []
    for s in steps:
        if merged and len(s) < min_chars:
            merged[-1] = merged[-1] + " " + s
        elif merged and len(merged[-1]) < min_chars:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged[:max_steps]


def generate_experience(teacher: ModelRunner, n: int | None = None,
                        max_new_tokens: int = 400,
                        tasks: list[tuple[str, str]] | None = None) -> list[Rollout]:
    """Teacher-authored rollouts. `tasks` is an optional list of (prompt, domain); the
    teacher's own multi-step answer is segmented into steps (so the steps are valid
    distillation targets). Generation is batched via the runner."""
    pairs = tasks if tasks is not None else [(t, "general") for t in (TASKS[:n] if n else TASKS)]
    prompts = [p for p, _ in pairs]
    sols = teacher.generate(prompts, max_new_tokens=max_new_tokens)
    outs: list[Rollout] = []
    for i, ((task, domain), sol) in enumerate(zip(pairs, sols)):
        steps = _split_steps(sol)
        if len(steps) >= 2:
            outs.append(Rollout(id=f"gen{i}", context=task, steps=steps, success=True, domain=domain))
    return outs
