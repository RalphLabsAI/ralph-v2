"""Code axis — implement a function, graded by EXECUTING hidden unit tests.

Execution grading is the strongest deterministic checker available: a response
either makes the assertions pass or it does not. Style, confidence and fluent
reasoning are all worth exactly zero, which is the property that makes imitation
unprofitable.

Tasks are generated from parameterized specs (same principle as the math axis:
the procedure is fixed, the surface is fresh), and each carries hidden tests the
model never sees.

SANDBOXING: this module runs candidate code in a subprocess with a wall-clock
timeout, which is sufficient for OUR OWN generated reference solutions and the
simulated students. Grading real miner-submitted model output must additionally run
with no network, a read-only filesystem, and resource caps — see the RCE lesson in
docs/prior-art-and-lessons.md. Do not point this at untrusted output as-is.
"""
from __future__ import annotations

import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from ..core import Item

_SPECS = [
    {
        "key": "count_divisible",
        "sig": "def count_divisible(nums: list[int], k: int) -> int",
        "doc": "Return how many integers in `nums` are exactly divisible by `k`.",
        "ref": "return sum(1 for x in nums if k != 0 and x % k == 0)",
        "cases": lambda rng: [
            ([rng.randint(-30, 30) for _ in range(rng.randint(4, 9))], rng.randint(2, 7))
            for _ in range(4)
        ],
        "oracle": lambda nums, k: sum(1 for x in nums if k != 0 and x % k == 0),
    },
    {
        "key": "running_max",
        "sig": "def running_max(nums: list[int]) -> list[int]",
        "doc": "Return the running maximum: element i is the max of nums[0..i].",
        "ref": "out=[]\n    m=None\n    for x in nums:\n        m = x if m is None else max(m,x)\n        out.append(m)\n    return out",
        "cases": lambda rng: [([rng.randint(-20, 20) for _ in range(rng.randint(3, 8))],) for _ in range(4)],
        "oracle": lambda nums: [max(nums[: i + 1]) for i in range(len(nums))],
    },
    {
        "key": "collapse_spaces",
        "sig": "def collapse_spaces(s: str) -> str",
        "doc": "Collapse every run of consecutive spaces into a single space, and strip the ends.",
        "ref": "return ' '.join(s.split())",
        "cases": lambda rng: [
            ("  ".join(rng.choice(["ab", "c", "dd", "e f", "gg"]) for _ in range(rng.randint(2, 5))) + "  ",)
            for _ in range(4)
        ],
        "oracle": lambda s: " ".join(s.split()),
    },
]


class CodeExec:
    """Function-implementation axis graded by hidden unit tests."""

    name = "code"
    weight = 0.25

    def __init__(self, timeout_s: float = 5.0):
        self.timeout_s = timeout_s

    def generate(self, seed: int, n: int, difficulty: int = 1) -> list[Item]:
        rng = random.Random(seed)
        items: list[Item] = []
        for i in range(n):
            spec = _SPECS[i % len(_SPECS)]
            args_list = spec["cases"](rng)
            expected = [spec["oracle"](*a) for a in args_list]
            tests = [{"args": list(a), "expect": e} for a, e in zip(args_list, expected)]
            prompt = (
                f"Implement this function in Python.\n\n"
                f"{spec['sig']}:\n    \"\"\"{spec['doc']}\"\"\"\n\n"
                "Return ONLY the complete function definition in a ```python code block."
            )
            items.append(Item(
                axis=self.name, prompt=prompt,
                answer={"fn": spec["key"], "sig": spec["sig"], "tests": tests},
                meta={"template": spec["key"], "idx": i, "difficulty": difficulty, "noop": False},
            ))
        return items

    @staticmethod
    def _extract_code(output: str) -> str | None:
        """Robust code extraction. Real models emit malformed/multiple fences (```python
        code block, a bare ``` then ```python, prose around the block); a strict fence
        regex silently drops a CORRECT answer -> false 0 -> worst-domain soft-min inverts
        the ranking (a capable model scored 0/22 on formatting quirks). So: match a lenient
        fence (any info string), prefer the block that actually contains a def, and on the
        unfenced fallback cut any trailing fence."""
        # lenient fence: ``` + arbitrary info string (may be malformed) + newline + body
        blocks = re.findall(r"```[^\n]*\n(.*?)```", output, re.S)
        with_def = [b for b in blocks if "def " in b]
        if with_def:
            return with_def[-1]
        nonempty = [b for b in blocks if b.strip()]
        if nonempty:
            return nonempty[-1]
        if "def " in output:  # unfenced — take from def, drop any trailing fence/prose
            return re.split(r"\n```", output[output.index("def "):])[0]
        return None

    def check(self, item: Item, output: str) -> bool:
        code = self._extract_code(output)
        if not code:
            return False
        spec = item.answer
        fn = spec["fn"]
        harness = [code, "", "def __run():"]
        for t in spec["tests"]:
            harness.append(f"    assert {fn}(*{t['args']!r}) == {t['expect']!r}")
        harness += ["    return True", "", "__run()", "print('__OK__')"]
        script = "\n".join(harness)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cand.py"
            p.write_text(script)
            try:
                r = subprocess.run([sys.executable, str(p)], capture_output=True,
                                   text=True, timeout=self.timeout_s, cwd=td)
            except subprocess.TimeoutExpired:
                return False
            return r.returncode == 0 and "__OK__" in r.stdout

    def reference_solution(self, item: Item) -> str:
        """The correct answer — used by simulated students to model competence."""
        spec = next(s for s in _SPECS if s["key"] == item.answer["fn"])
        body = spec["ref"]
        return f"```python\n{spec['sig']}:\n    {body}\n```"
