"""The dethrone bar must stay inside the metric's measured range, on the path that pays.

Every other crown test passes an explicit margin, so the PRODUCTION number was uncovered and
free to drift. These pin it end to end: the constant, the two call sites that actually decide
emission, and the exam draw that sets how much precision the deciding slice gets."""
from __future__ import annotations

import ast
import random
from pathlib import Path

from eval.koth import DETHRONE_MARGIN, Tier, Tournament

ROOT = Path(__file__).resolve().parents[1] / "eval"


def test_margin_is_inside_the_observed_spread_of_a_tier():
    # Worst-slice retention spread within a tier, measured over the signed records, is ~0.074
    # (binary) and ~0.078 (sub2). The bootstrap gives up ~0.019 of the mean advantage to
    # uncertainty at n_items=144. A challenger must therefore find margin + penalty ~= 0.03,
    # about a third of a tier's range. If the margin ever exceeds the spread again, a contested
    # throne can only be inherited, never taken.
    penalty_at_144 = 0.019
    narrowest_tier_spread = 0.074
    assert DETHRONE_MARGIN + penalty_at_144 < narrowest_tier_spread * 0.75


def test_the_paths_that_pay_use_the_constant_not_a_literal():
    # score_job runs orchestrated rounds; run_round is the single-box path. Both build their
    # own Tournament, so the class default alone never reaches emission.
    for name in ("score_job.py", "run_round.py"):
        tree = ast.parse((ROOT / name).read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Tournament"):
                continue
            kw = {k.arg: k.value for k in node.keywords}
            assert "margin" in kw, f"{name}: Tournament built without an explicit margin"
            expr = kw["margin"]
            src = ast.unparse(expr)
            assert "DETHRONE_MARGIN" in src, f"{name}: margin is a literal ({src}), not the constant"


def test_default_matches_the_constant():
    assert Tournament([Tier("binary", 1.0, 1.0)], ).margin == DETHRONE_MARGIN


def _alloc(k: int, avail: dict, min_per_lang: int = 10) -> dict:
    """Mirror of the remainder split in observer_round.select_trajectories."""
    langs = sorted(avail)
    rng = random.Random(0)
    a = {l: min(min_per_lang, avail[l], max(1, k // len(langs))) for l in langs}
    rem = k - sum(a.values())
    while rem > 0:
        room = [l for l in langs if a[l] < avail[l]]
        if not room:
            break
        low = min(a[l] for l in room)
        a[rng.choice([l for l in room if a[l] == low])] += 1
        rem -= 1
    return a


def test_spare_exam_items_go_to_the_slice_that_decides():
    # The aggregate is a soft-MIN, so items added to the largest slice never reach the decision.
    # Sharing the remainder by pool availability put them in English; levelling puts them in the
    # slice the dethrone test is actually resolved on.
    avail = {("en", "shallow"): 402, ("en", "deep"): 201, ("hi", "shallow"): 108,
             ("hi", "deep"): 54, ("zh", "shallow"): 128, ("zh", "deep"): 7}
    a = _alloc(144, avail)
    scored = [v for v in a.values() if v >= 8]      # score_miner drops slices under min_per_slice
    assert min(scored) >= 24, f"deciding slice too thin: {a}"
    # and no slice may hoard: the biggest scored slice stays close to the smallest
    assert max(scored) - min(scored) <= 2, f"remainder is not levelled: {a}"


def test_levelling_beats_proportional_on_the_deciding_slice():
    avail = {("en", "shallow"): 402, ("en", "deep"): 201, ("hi", "shallow"): 108,
             ("hi", "deep"): 54, ("zh", "shallow"): 128, ("zh", "deep"): 7}
    rng = random.Random(0)
    langs = sorted(avail)
    prop = {l: min(10, avail[l], max(1, 144 // len(langs))) for l in langs}
    rem = 144 - sum(prop.values())
    while rem > 0:
        room = [l for l in langs if prop[l] < avail[l]]
        if not room:
            break
        prop[rng.choices(room, weights=[avail[l] for l in room], k=1)[0]] += 1
        rem -= 1
    assert min(v for v in _alloc(144, avail).values() if v >= 8) > \
           min(v for v in prop.values() if v >= 8)


def test_the_plan_the_gpu_receives_carries_the_protocol_margin():
    """RoundPlan is serialised to job.json and shipped to the scorer, and the scorer used to trust
    job["margin"] over the constant. The plan's dataclass default was a stale 0.05, so every
    orchestrated round scored at a margin two changes had already retired. Round-trip the real
    writer: the job the GPU receives must say what koth says, and a plan that disagrees must abort
    the scorer rather than be obeyed."""
    import json
    from eval.orchestrator import RoundPlan

    plan = RoundPlan(round=7, commit_root="c" * 64, round_nonce="n" * 64, prev_anchor="p" * 64)
    job = json.loads(json.dumps(plan.as_job()))
    assert job["margin"] == DETHRONE_MARGIN

    # the scorer's own guard, exercised the way score_job runs it
    src = (ROOT / "score_job.py").read_text()
    assert 'Tournament(tiers, margin=DETHRONE_MARGIN)' in src
    assert 'job.get("margin", DETHRONE_MARGIN)' not in src, "plan must not override the constant"
