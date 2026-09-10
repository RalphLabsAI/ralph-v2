"""The crown is decided on the displayed metric: a single-round lead of DETHRONE_MARGIN, or
PERSIST_MARGIN on PERSIST_ROUNDS consecutive fresh exams, with the floor rule (no challenger slice
under the incumbent's worst). Events are the real writer; lineage replays streaks from them; the
orchestrator re-scores contenders instead of dropping them as unchanged."""
from __future__ import annotations

import random
import types

from eval.koth import (DETHRONE_MARGIN, PERSIST_MARGIN, PERSIST_ROUNDS, Scored, Tier, Tournament,
                       floor_regression)
from eval.lineage import replay_contenders

AXES = ["obs=o|lang=en|depth=deep", "obs=o|lang=en|depth=shallow", "obs=o|lang=hi|depth=deep",
        "obs=o|lang=hi|depth=shallow", "obs=o|lang=zh|depth=shallow"]


def _scored(mid: str, miner: str, per_axis: dict) -> Scored:
    """Retention as the round computes it: soft-min over slice means (koth._soft_min, p=-3)."""
    from eval.koth import SOFTMIN_P, _soft_min
    sub = types.SimpleNamespace(model_id=mid, miner=miner, tier="t")
    means = [sum(v) / len(v) for v in per_axis.values()]
    return Scored(sub=sub, retention=_soft_min(means, SOFTMIN_P), retention_lb=1.0, per_point=[],
                  gates_ok=True, per_axis=per_axis)


def _model(base: dict, delta: dict | None = None, noise: float = 0.0, seed: int = 0, n: int = 29):
    rng = random.Random(seed)
    return {ax: [base[ax] + (delta or {}).get(ax, 0.0) + rng.gauss(0, noise) for _ in range(n)]
            for ax in AXES}


KING = {AXES[0]: 0.29, AXES[1]: 0.41, AXES[2]: 0.27, AXES[3]: 0.28, AXES[4]: 0.34}


def _tourney():
    return Tournament([Tier("t", 10 ** 12, 1.0)], margin=DETHRONE_MARGIN)


def _king_scored(t: Tournament, seed=0):
    k = _scored("king", "5K", _model(KING, noise=0.02, seed=seed))
    t.kings["t"] = types.SimpleNamespace(miner="5K", model_id="king", retention=k.retention, reign=0)
    return k


def test_single_round_lead_at_margin_dethrones_and_floor_rule_binds():
    t = _tourney(); k = _king_scored(t)
    # uniform +0.03 on every slice: lead ~0.03 >= 0.02
    ch = _scored("ch", "5C", _model(KING, {ax: 0.03 for ax in AXES}, noise=0.02, seed=1))
    e = t.consider("t", [ch], k)
    assert e["action"] == "dethrone" and e["dethrone_by"] == "margin", e

    # a floor-lifter that reshapes: +0.06 on the two weakest, -0.04 on the strongest, nothing
    # under the old floor (0.27) -> allowed
    t = _tourney(); k = _king_scored(t)
    ch = _scored("ch2", "5C", _model(KING, {AXES[2]: 0.06, AXES[3]: 0.06, AXES[1]: -0.04},
                                     noise=0.02, seed=2))
    e = t.consider("t", [ch], k)
    assert e["action"] == "dethrone", e

    # the same reshaping but a slice sinks BELOW the old floor -> refused, whatever the lead
    t = _tourney(); k = _king_scored(t)
    ch = _scored("ch3", "5C", _model(KING, {AXES[2]: 0.08, AXES[3]: 0.08, AXES[1]: -0.16},
                                     noise=0.0, seed=3))
    e = t.consider("t", [ch], k)
    assert e["action"] == "hold" and e.get("regressed_axis") == AXES[1], e


def test_persistence_crowns_a_small_lead_over_two_rounds_and_resets_on_new_bytes():
    t = _tourney(); k = _king_scored(t, seed=10)
    small = {ax: 0.015 for ax in AXES}                     # real, but under the 0.02 margin
    e1 = t.consider("t", [_scored("ch", "5C", _model(KING, small, noise=0.0))], k)
    assert e1["action"] == "hold" and e1["contender_streak"] == 1 and e1["contender"] == "ch", e1
    assert PERSIST_ROUNDS == 2
    t.round += 1; k = _king_scored(t, seed=11)
    e2 = t.consider("t", [_scored("ch", "5C", _model(KING, small, noise=0.0))], k)
    assert e2["action"] == "dethrone" and e2["dethrone_by"] == "persistence", e2
    assert e2["contender_streak"] == 2 and "t" not in t.contenders

    # a different artifact in round 2 starts its own streak from 1
    t = _tourney(); k = _king_scored(t, seed=12)
    t.consider("t", [_scored("ch", "5C", _model(KING, small, noise=0.0))], k)
    t.round += 1; k = _king_scored(t, seed=13)
    e = t.consider("t", [_scored("other", "5D", _model(KING, small, noise=0.0))], k)
    assert e["action"] == "hold" and e["contender_streak"] == 1 and e["contender"] == "other", e


def test_ties_and_clones_never_crown_on_either_path():
    # exact copy: lead is exactly 0 on the same items
    t = _tourney(); k = _king_scored(t, seed=20)
    copy = _scored("copy", "5X", {ax: list(v) for ax, v in k.per_axis.items()})
    for _ in range(4):
        e = t.consider("t", [copy], k); t.round += 1
        assert e["action"] == "hold" and e["contender_streak"] == 0, e
    # copy + English polish (+0.05 on the strongest slice): the soft-min barely moves
    t = _tourney(); k = _king_scored(t, seed=21)
    pol = {ax: [x + (0.05 if ax == AXES[1] else 0.0) for x in v] for ax, v in k.per_axis.items()}
    for _ in range(4):
        e = t.consider("t", [_scored("polish", "5X", pol)], k); t.round += 1
        assert e["action"] == "hold", e
        assert e["lead"] < PERSIST_MARGIN, e            # not even a strike
    # a tied pair with independent noise: two draws in a row must not build a streak
    t = _tourney()
    for rnd in range(6):
        k = _king_scored(t, seed=100 + rnd)
        e = t.consider("t", [_scored("tie", "5T", _model(KING, noise=0.02, seed=200 + rnd))], k)
        t.round += 1
        assert e["action"] == "hold", e


def test_floor_regression_is_point_based_on_the_incumbent_floor():
    k = _scored("k", "5K", _model(KING))
    ok = _scored("a", "5A", _model(KING, {AXES[1]: -0.13}))           # 0.28 > floor 0.27
    bad = _scored("b", "5B", _model(KING, {AXES[1]: -0.15}))          # 0.26 < floor 0.27
    assert floor_regression(ok, k) is None
    assert floor_regression(bad, k) == AXES[1]


def test_lineage_replays_contender_streaks_from_the_real_events():
    t = _tourney(); k = _king_scored(t, seed=30)
    small = {ax: 0.015 for ax in AXES}
    e1 = t.consider("t", [_scored("ch", "5C", _model(KING, small, noise=0.0))], k)
    subs = [types.SimpleNamespace(model_id="ch", miner="5C", artifact_uri="hf://x/y@rev"),
            types.SimpleNamespace(model_id="king", miner="5K", artifact_uri="hf://k/k@rev")]
    rec1 = types.SimpleNamespace(round=1, events=[e1], submissions=subs)
    got = replay_contenders([rec1])
    assert got == {"t": {"model_id": "ch", "streak": 1, "miner": "5C", "artifact_uri": "hf://x/y@rev"}}, got
    # a dethrone clears the contender
    t.round += 1; k = _king_scored(t, seed=31)
    e2 = t.consider("t", [_scored("ch", "5C", _model(KING, small, noise=0.0))], k)
    rec2 = types.SimpleNamespace(round=2, events=[e2], submissions=subs)
    assert replay_contenders([rec2, rec1]) == {}               # order-insensitive, dethrone wins
