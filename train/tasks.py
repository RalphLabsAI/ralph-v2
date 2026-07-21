"""Multi-domain task set for the distillation experiment.

The teacher answers each; its segmented answer becomes a rollout whose steps are the
distillation targets. Breadth matters: S_broad trains on every domain, and the
self_state gap is most visible on multi-step reasoning, so the set leans on chains.

Built deterministically (fixed seed) so a re-run reproduces the same pile.
"""
from __future__ import annotations

import random

_STEP = "Reason step by step, one step per line."


def _build() -> list[tuple[str, str]]:
    rng = random.Random(7)
    tasks: list[tuple[str, str]] = []

    # arithmetic / word problems
    for _ in range(22):
        a, b = rng.randint(20, 90), rng.randint(20, 90)
        h1, h2 = rng.randint(40, 100), rng.randint(40, 100)
        tmpl = rng.choice([
            f"A train travels {h1} km in the first hour and {h2} km in the second. "
            f"What is its average speed? {_STEP}",
            f"A shirt costs {a} after a {rng.choice([10,20,25])}% discount. "
            f"What was the original price? {_STEP}",
            f"A tank fills in {a} minutes with pipe A and {b} with pipe B. "
            f"How long with both open? {_STEP}",
            f"Compute the sum of all integers from 1 to {rng.randint(30,80)}, showing steps. {_STEP}",
        ])
        tasks.append((tmpl, "math"))

    # algebra / number theory
    for _ in range(18):
        a, b, c, d = (rng.randint(2, 9) for _ in range(4))
        tmpl = rng.choice([
            f"Solve for x: {a}x + {b} = {c+a}x - {d}. Show every step. {_STEP}",
            f"Find the greatest common divisor of {rng.randint(40,120)} and {rng.randint(40,120)}. {_STEP}",
            f"How many 3-digit numbers are divisible by {rng.choice([3,7,11])}? {_STEP}",
            f"Convert {a}/{b+a} to a decimal and then a percentage. {_STEP}",
        ])
        tasks.append((tmpl, "algebra"))

    # geometry / counting
    for _ in range(14):
        p, w = rng.randint(20, 50), rng.randint(3, 12)
        tmpl = rng.choice([
            f"A rectangle has perimeter {p} and width {w}. Find its area. {_STEP}",
            f"How many ways can {rng.randint(4,7)} people sit in a row? {_STEP}",
            f"A circle has radius {rng.randint(3,9)}. Give its area in terms of pi. {_STEP}",
        ])
        tasks.append((tmpl, "geometry"))

    # logic (parameterized over names/relations so instances stay distinct)
    names = ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Gina", "Hugo"]
    for _ in range(16):
        w, x, y, z = rng.sample(names, 4)
        rel = rng.choice(["taller", "older", "faster", "richer"])
        tmpl = rng.choice([
            f"{w} is {rel} than {x}. {x} is {rel} than {y}. {z} is not {rel} than {y}. "
            f"Order {w}, {x}, {y}, {z} accordingly. {_STEP}",
            f"{w} says {x} lies. {x} says {y} lies. {y} says {w} and {x} both lie. "
            f"Who tells the truth? {_STEP}",
            f"Everyone {rel} than {x} sits to the left of {x}. Given {w} is {rel} than {x} "
            f"and {y} is not, where do {w} and {y} sit relative to {x}? {_STEP}",
        ])
        tasks.append((tmpl, "logic"))

    # code (explanation-first so the teacher produces reasoning steps, not just a blob)
    for _ in range(16):
        fn = rng.choice(["is_palindrome", "second_largest", "count_vowels", "merge_sorted",
                         "flatten", "most_frequent", "run_length_encode", "is_balanced"])
        spec = {
            "is_palindrome": "is_palindrome(s) ignoring case and spaces",
            "second_largest": "second_largest(xs) returning the 2nd-largest distinct value",
            "count_vowels": "count_vowels(s) counting vowels in a string",
            "merge_sorted": "merge_sorted(a, b) merging two sorted lists",
            "flatten": "flatten(xs) flattening one level of nesting",
            "most_frequent": "most_frequent(xs) returning the most common element",
            "run_length_encode": "run_length_encode(s) doing run-length encoding",
            "is_balanced": "is_balanced(s) checking balanced brackets",
        }[fn]
        tasks.append((f"Write a Python function {spec}. Explain each step before the code. {_STEP}", "code"))

    # de-dup while preserving order (templates repeat)
    seen, out = set(), []
    for t, dom in tasks:
        if t not in seen:
            seen.add(t)
            out.append((t, dom))
    return out


DISTILL_TASKS: list[tuple[str, str]] = _build()
