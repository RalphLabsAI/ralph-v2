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
        """Robust extraction across real output styles (checker quality is the crown's
        floor — a fence quirk once scored capable models at 0/N). Handles: a stray leading
        bare ``` + prose BEFORE the real ```python block (odd fence count misaligns naive
        pairing and eats the code as a delimiter — a 1.5B failure, 0/40); imports BEFORE the
        def (dropping them -> NameError); malformed info strings (```python code block).
        Strategy: prefer an explicitly python-tagged block (holds imports+def and survives
        the stray fence), then any def-containing lenient block, then an unfenced run."""
        has_def = lambda b: re.search(r"(?:^|\n)[ \t]*def ", b) is not None
        # 1) python-tagged block — info string may be malformed ("python code block")
        py = re.findall(r"```[ \t]*(?:python|py)\b[^\n]*\n(.*?)```", output, re.S | re.I)
        cand = [b for b in py if has_def(b)]
        if cand:
            return cand[-1]
        # 2) any lenient fenced block containing a def
        blocks = re.findall(r"```[^\n]*\n(.*?)```", output, re.S)
        cand = [b for b in blocks if has_def(b)]
        if cand:
            return cand[-1]
        # 3) unfenced: from the first import/def line, cut a trailing fence/prose
        m = re.search(r"(?:^|\n)(?:import |from |def )", output)
        if m:
            return re.split(r"\n```", output[m.start():].lstrip("\n"))[0].strip("\n") or None
        nonempty = [b for b in blocks if b.strip()]
        return nonempty[-1] if nonempty else None

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
