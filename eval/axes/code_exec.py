"""Code axis — implement a function, graded by EXECUTING hidden unit tests.

Execution grading is the strongest deterministic checker available: a response
either makes the assertions pass or it does not. Style, confidence and fluent
reasoning are all worth exactly zero, which is the property that makes imitation
unprofitable.

Tasks are generated from parameterized specs (same principle as the math axis:
the procedure is fixed, the surface is fresh), and each carries hidden tests the
model never sees.

SANDBOXING: candidate code is student-emitted and hostile by construction (a payload in
the ```python block that passes the asserts AND exfiltrates the validator wallet), so the
subprocess runs through eval.sandbox.run_script — bubblewrap (no network, read-only FS,
$HOME/tmp masked) plus rlimits. In production the code axis is constructed with
sandbox="require" (fail closed if no hard sandbox backend is present); "auto" degrades to
rlimit-only with a loud warning for local dev on trusted models. See the RCE lesson in
docs/prior-art-and-lessons.md.
"""
from __future__ import annotations

import random
import re
import tempfile
from pathlib import Path

from ..core import Item
from ..sandbox import run_script

# PARAMETERIZED FAMILIES. Three fixed specs made this axis a lookup table: a miner
# memorizes 3 function bodies and maxes it forever, and "fresh per round" only reshuffles
# the test inputs. Each family instead MINTS a distinct task from the round seed (different
# predicate / accumulator / operation / parameters), so the procedure space is large enough
# that the axis measures implementing-a-spec rather than recalling three answers.
# Every family returns: key, sig, doc, an executable `oracle`, and a case generator.
_INT = lambda rng, lo=-30, hi=30: rng.randint(lo, hi)


def _f_count_pred(rng):
    kind = rng.choice(["divisible by", "greater than", "less than"])
    k = rng.randint(2, 9)
    name = {"divisible by": "count_divisible", "greater than": "count_greater",
            "less than": "count_less"}[kind]
    oracle = {"divisible by": lambda nums, k: sum(1 for x in nums if k != 0 and x % k == 0),
              "greater than": lambda nums, k: sum(1 for x in nums if x > k),
              "less than": lambda nums, k: sum(1 for x in nums if x < k)}[kind]
    return {"key": name,
            "sig": f"def {name}(nums: list[int], k: int) -> int",
            "doc": f"Return how many integers in `nums` are {kind} `k`.",
            "cases": lambda r: [([_INT(r) for _ in range(r.randint(4, 9))], k) for _ in range(4)],
            "oracle": oracle,
            "ref": {"divisible by": "return sum(1 for x in nums if k != 0 and x % k == 0)",
                    "greater than": "return sum(1 for x in nums if x > k)",
                    "less than": "return sum(1 for x in nums if x < k)"}[kind]}


def _f_running(rng):
    kind = rng.choice(["maximum", "minimum", "sum"])
    name = {"maximum": "running_max", "minimum": "running_min", "sum": "running_sum"}[kind]
    oracle = {"maximum": lambda nums: [max(nums[:i + 1]) for i in range(len(nums))],
              "minimum": lambda nums: [min(nums[:i + 1]) for i in range(len(nums))],
              "sum": lambda nums: [sum(nums[:i + 1]) for i in range(len(nums))]}[kind]
    return {"key": name,
            "sig": f"def {name}(nums: list[int]) -> list[int]",
            "doc": f"Return the running {kind}: element i is the {kind} of nums[0..i].",
            "cases": lambda r: [([_INT(r, -20, 20) for _ in range(r.randint(3, 8))],) for _ in range(4)],
            "oracle": oracle,
            "ref": {"maximum": "return [max(nums[:i+1]) for i in range(len(nums))]",
                    "minimum": "return [min(nums[:i+1]) for i in range(len(nums))]",
                    "sum": "return [sum(nums[:i+1]) for i in range(len(nums))]"}[kind]}


def _f_top_k(rng):
    return {"key": "top_k",
            "sig": "def top_k(nums: list[int], k: int) -> list[int]",
            "doc": "Return the `k` largest integers in `nums`, sorted in DESCENDING order. "
                   "If `k` exceeds the length, return all of them sorted descending.",
            "cases": lambda r: [([_INT(r) for _ in range(r.randint(4, 9))], r.randint(1, 4)) for _ in range(4)],
            "oracle": lambda nums, k: sorted(nums, reverse=True)[:k],
            "ref": "return sorted(nums, reverse=True)[:k]"}


def _f_dedupe(rng):
    return {"key": "dedupe",
            "sig": "def dedupe(items: list[int]) -> list[int]",
            "doc": "Remove duplicates from `items`, PRESERVING first-occurrence order.",
            "cases": lambda r: [([r.randint(0, 6) for _ in range(r.randint(5, 10))],) for _ in range(4)],
            "oracle": lambda items: list(dict.fromkeys(items)),
            "ref": "return list(dict.fromkeys(items))"}


def _f_rotate(rng):
    return {"key": "rotate_left",
            "sig": "def rotate_left(nums: list[int], k: int) -> list[int]",
            "doc": "Rotate `nums` LEFT by `k` positions (k may exceed the length). "
                   "An empty list returns empty.",
            "cases": lambda r: [([_INT(r, 0, 20) for _ in range(r.randint(3, 8))], r.randint(0, 9)) for _ in range(4)],
            "oracle": lambda nums, k: (nums[k % len(nums):] + nums[:k % len(nums)]) if nums else [],
            "ref": "return (nums[k % len(nums):] + nums[:k % len(nums)]) if nums else []"}


def _f_clamp(rng):
    return {"key": "clamp_all",
            "sig": "def clamp_all(nums: list[int], lo: int, hi: int) -> list[int]",
            "doc": "Clamp every value into the inclusive range [lo, hi].",
            "cases": lambda r: [([_INT(r) for _ in range(r.randint(4, 8))], -5, 12) for _ in range(4)],
            "oracle": lambda nums, lo, hi: [min(max(x, lo), hi) for x in nums],
            "ref": "return [min(max(x, lo), hi) for x in nums]"}


def _f_flatten(rng):
    return {"key": "flatten",
            "sig": "def flatten(rows: list[list[int]]) -> list[int]",
            "doc": "Concatenate the sublists of `rows` into a single flat list, in order.",
            "cases": lambda r: [([[r.randint(0, 9) for _ in range(r.randint(0, 3))]
                                  for _ in range(r.randint(2, 4))],) for _ in range(4)],
            "oracle": lambda rows: [x for row in rows for x in row],
            "ref": "return [x for row in rows for x in row]"}


def _f_char_count(rng):
    ch = rng.choice("abcde")
    return {"key": "char_count",
            "sig": "def char_count(s: str, ch: str) -> int",
            "doc": f"Return how many times the character `ch` occurs in `s` (case-sensitive).",
            "cases": lambda r: [("".join(r.choice("abcde ") for _ in range(r.randint(6, 14))), ch)
                                for _ in range(4)],
            "oracle": lambda s, ch: s.count(ch),
            "ref": "return s.count(ch)"}


def _f_string_norm(rng):
    kind = rng.choice(["collapse", "squeeze"])
    if kind == "collapse":
        return {"key": "collapse_spaces",
                "sig": "def collapse_spaces(s: str) -> str",
                "doc": "Collapse every run of consecutive spaces into a single space, and strip the ends.",
                "cases": lambda r: [("  ".join(r.choice(["ab", "c", "dd", "e f", "gg"])
                                               for _ in range(r.randint(2, 5))) + "  ",) for _ in range(4)],
                "oracle": lambda s: " ".join(s.split()),
                "ref": "return ' '.join(s.split())"}
    return {"key": "reverse_words",
            "sig": "def reverse_words(s: str) -> str",
            "doc": "Return the words of `s` in reverse order, separated by single spaces.",
            "cases": lambda r: [(" ".join(r.choice(["ab", "cd", "ef", "gh"])
                                          for _ in range(r.randint(2, 5))),) for _ in range(4)],
            "oracle": lambda s: " ".join(reversed(s.split())),
            "ref": "return ' '.join(reversed(s.split()))"}


def _f_sum_range(rng):
    return {"key": "sum_between",
            "sig": "def sum_between(nums: list[int], lo: int, hi: int) -> int",
            "doc": "Return the sum of values in `nums` that fall in the inclusive range [lo, hi].",
            "cases": lambda r: [([_INT(r) for _ in range(r.randint(4, 9))], -5, 15) for _ in range(4)],
            "oracle": lambda nums, lo, hi: sum(x for x in nums if lo <= x <= hi),
            "ref": "return sum(x for x in nums if lo <= x <= hi)"}


# Difficulty selects the FAMILY POOL, not just the inputs. Measured on the pinned teacher:
# GLM-4-9B is a weak coder on the harder families — it emits genuine syntax errors (a stray
# trailing paren) and wrong logic (running_max appending result[-1] on an empty list), which
# dragged the axis to ~28-50% and BELOW MIN_AXIS_N, so the axis went non-live and the
# all-axes-live gate blocked every crown. An axis only measures compression if the TEACHER
# clears it; keep d1 inside the teacher's competence band and escalate for stronger pairs.
_EASY = [_f_count_pred, _f_char_count, _f_string_norm, _f_sum_range, _f_running]
_HARD = [_f_top_k, _f_dedupe, _f_rotate, _f_clamp, _f_flatten]
_FAMILIES = _EASY + _HARD


def _family_pool(difficulty: int) -> list:
    return _EASY if difficulty <= 1 else _FAMILIES


class CodeExec:
    """Function-implementation axis graded by hidden unit tests."""

    name = "code"
    weight = 0.25

    def __init__(self, timeout_s: float = 5.0, sandbox: str = "auto"):
        # sandbox: "require" (production, fail-closed if no bwrap), "auto" (bwrap if present
        # else rlimit-only + warning), "off" (rlimit-only). Untrusted student output must be
        # graded with "require"; the production spec assemblies set it via RALPH_CODE_SANDBOX.
        self.timeout_s = timeout_s
        self.sandbox = sandbox

    def generate(self, seed: int, n: int, difficulty: int = 1) -> list[Item]:
        rng = random.Random(seed)
        items: list[Item] = []
        for i in range(n):
            pool = _family_pool(difficulty)
            spec = pool[i % len(pool)](rng)
            args_list = spec["cases"](rng)
            expected = [spec["oracle"](*a) for a in args_list]
            tests = [{"args": list(a), "expect": e} for a, e in zip(args_list, expected)]
            # MEASURED, not assumed: showing the literal ```python fence here is the better
            # prompt. Replacing it with a described format + "close the fence, do not
            # repeat" was tried and HURT — same tasks/seed, GLM 16/32 -> 9/32. The teacher
            # repeats the block either way (its own generation degeneracy, not the fence),
            # and extraction handles that fine by taking the first block; the negative
            # constraints just degraded its code. Do not "fix" this prompt without an A/B.
            prompt = (
                f"Implement this function in Python.\n\n"
                f"{spec['sig']}:\n    \"\"\"{spec['doc']}\"\"\"\n\n"
                "Return ONLY the complete function definition in a ```python code block."
            )
            items.append(Item(
                axis=self.name, prompt=prompt,
                answer={"fn": spec["key"], "sig": spec["sig"], "tests": tests,
                        "ref": spec["ref"]},
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
            executed, cp = run_script(str(p), td, timeout=self.timeout_s, mode=self.sandbox)
            if not executed or cp is None:   # refused (no sandbox in 'require' mode) or timed out
                return False
            return cp.returncode == 0 and "__OK__" in cp.stdout

    def reference_solution(self, item: Item) -> str:
        """The correct answer — used by simulated students to model competence."""
        a = item.answer
        return f"```python\n{a['sig']}:\n    {a['ref']}\n```"
