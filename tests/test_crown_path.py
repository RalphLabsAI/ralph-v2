"""Regression tests for the crown decision path (the fixes in commit 2494017 + the
long-context axis). Runnable two ways:

    python -m tests.test_crown_path      # self-running harness, no deps
    pytest tests/test_crown_path.py      # if pytest is installed

Covers: worst-axis dethrone (a fluent drifter that clears the pooled mean is blocked),
the axis_round crown gates (degeneracy DQ, every-declared-axis-live, copy ties,
genuine improvement dethrones), and the deterministic long-context checker.
"""
from __future__ import annotations

import random

from eval.axes.code_exec import CodeExec
from eval.axes.long_context import LongContext
from eval.axis_round import AxisSpec, axis_round
from eval.core import Item
from eval.koth import (Scored, Submission, Tier, Tournament, bootstrap_lcb_diff,
                       worst_axis_lcb_diff)

MARGIN = 0.05


def _scored(mid, per_axis):
    flat = [v for ax in per_axis.values() for v in ax]
    return Scored(sub=Submission(mid, "t", mid, 1, 0.0), retention=0.0, retention_lb=0.5,
                  per_point=flat, gates_ok=True, per_axis=per_axis)


def test_worst_axis_blocks_drifter():
    """A challenger strong on a big axis but weak on a small one clears the POOLED mean
    margin yet is blocked by the worst-axis test; honest-better dethrones; copy ties."""
    big, small = 500, 50
    king = _scored("king", {"A": [0.60] * big, "B": [0.60] * small})
    drifter = _scored("drift", {"A": [0.92] * big, "B": [0.50] * small})
    honest = _scored("honest", {"A": [0.70] * big, "B": [0.71] * small})
    copyk = _scored("copy", {"A": [0.60] * big, "B": [0.60] * small})

    # the drifter WOULD dethrone on the pooled mean (the old, wrong test)...
    assert bootstrap_lcb_diff(drifter.per_point, king.per_point, seed=1) > MARGIN
    # ...but the worst-axis test blocks it, admits the honest-better, and ties the copy.
    assert worst_axis_lcb_diff(drifter, king, seed=1) <= MARGIN
    assert worst_axis_lcb_diff(honest, king, seed=1) > MARGIN
    assert worst_axis_lcb_diff(copyk, king, seed=1) <= MARGIN


# --- axis_round end-to-end with a controllable fake axis ---

class _FakeAxis:
    def __init__(self, name):
        self.name = name

    def generate(self, seed, n, difficulty=1):
        return [Item(axis=self.name, prompt=f"{self.name}:{seed}:{i}", answer="GOOD",
                     meta={"idx": i}) for i in range(n)]

    def check(self, item, output):
        return (output or "").strip() == "GOOD"


class _Sim:
    def __init__(self, name, probs, seed=0, loop=False):
        self.name, self.probs, self.seed, self.loop = name, probs, seed, loop

    def generate(self, prompts, max_new_tokens=512):
        out = []
        for p in prompts:
            if self.loop:
                out.append(("spam " * 400).strip())
                continue
            axis = p.split(":")[0]
            r = random.Random(f"{self.seed}|{p}")
            out.append("GOOD" if r.random() < self.probs.get(axis, 0.0) else "no")
        return out


def _run(round_no, specs, tiers, tour, reg, subs, glm, base):
    # 150 items/axis: worst-axis dethrone (unlike the old pooled test) is powered PER AXIS,
    # so a real improvement needs enough teacher-passed items per axis to clear the margin
    # with statistical confidence — the production knob the red-team flagged (n >= ~150).
    return axis_round(round_no, commit_seed=100 + round_no, specs=specs, glm=glm, base=base,
                      tiers=tiers, tournament=tour, submissions=subs, registry=reg,
                      items_per_axis=150, max_new_tokens=8)


def test_axis_round_gates():
    specs = [AxisSpec(_FakeAxis("x"), "x", 1.0), AxisSpec(_FakeAxis("y"), "y", 1.0)]
    tiers = [Tier("t", 10 ** 12, 1.0)]
    tour, reg = Tournament(tiers, margin=0.03), {}
    glm = _Sim("glm", {"x": 1.0, "y": 1.0})
    base = _Sim("base", {"x": 0.30, "y": 0.30}, seed=9)

    def sub(mid):
        return Submission("m_" + mid, "t", mid, 1, 1.0)

    good = (sub("good"), _Sim("good", {"x": 0.85, "y": 0.85}, seed=1))
    looper = (sub("looper"), _Sim("looper", {}, loop=True))
    r1 = _run(1, specs, tiers, tour, reg, [good, looper], glm, base)
    assert not r1.scored["looper"].gates_ok            # degeneracy DQ
    assert tour.kings["t"].model_id == "good"          # good crowns open throne

    drifter = (sub("drift"), _Sim("drift", {"x": 0.99, "y": 0.40}, seed=2))
    _run(2, specs, tiers, tour, reg, [drifter, good], glm, base)
    assert tour.kings["t"].model_id == "good"          # worst-axis holds vs drifter

    copyk = (sub("copy"), _Sim("good", {"x": 0.85, "y": 0.85}, seed=1))
    _run(3, specs, tiers, tour, reg, [copyk, good], glm, base)
    assert tour.kings["t"].model_id == "good"          # exact copy ties

    better = (sub("better"), _Sim("better", {"x": 0.99, "y": 0.98}, seed=4))
    _run(4, specs, tiers, tour, reg, [better, good], glm, base)
    assert tour.kings["t"].model_id == "better"        # genuine improvement dethrones

    onex = (sub("onex"), _Sim("onex", {"x": 0.9, "y": 0.0}, seed=5))
    r5 = _run(5, specs, tiers, tour, reg, [onex], glm, base)
    assert not r5.scored["onex"].gates_ok              # y non-live -> all-axes-live gate blocks


def test_long_context_checker():
    ax = LongContext()
    items = ax.generate(seed=7, n=16, difficulty=2)
    assert {it.meta["qtype"] for it in items} == {"retrieve", "count_gt", "argmax", "sum"}
    for it in items:
        assert ax.check(it, f"Answer: {it.answer}")                 # correct accepted
        if it.meta["numeric"]:
            assert not ax.check(it, f"Answer: {int(it.answer) + 7}")  # wrong number rejected


def test_code_extractor_robust():
    """Real models emit malformed / doubled code fences; the extractor must still recover
    the correct code. A fence quirk once scored a correct Qwen-3B at 0/22 on the code axis,
    and worst-domain soft-min then ranked it BELOW a 1.5B (real capability run, 2026-07-23)."""
    ax = CodeExec()
    items = {it.answer["fn"]: it for it in ax.generate(seed=1, n=3, difficulty=2)}
    cases = {
        # malformed info string: ```python code block
        "count_divisible": (" ```python code block\ndef count_divisible(nums, k):\n"
                            "    return sum(1 for x in nums if x % k == 0)\n```"),
        # doubled fence: bare ``` then ```python
        "running_max": (" ```\n```python\ndef running_max(nums):\n    out=[]; m=None\n"
                        "    for x in nums:\n        m = x if m is None else max(m,x)\n"
                        "        out.append(m)\n    return out\n```"),
    }
    for fn, out in cases.items():
        assert ax.check(items[fn], out), fn


def main() -> int:
    tests = [test_worst_axis_blocks_drifter, test_axis_round_gates, test_long_context_checker,
             test_code_extractor_robust]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
