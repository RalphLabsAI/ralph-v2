"""Progress below the dethrone margin must pay something, and copying must still pay nothing.

Before this, a challenger who provably beat the king by 0.01 — paired, same exam, statistically
sound — earned exactly what a miner who never submitted earned. With all four thrones defended
that is every miner but four, and it invites the only rational response: stop iterating.

The 0.05 margin is NOT the problem and is not moved. Below it a dethrone is a coin flip, because
0.05 is the paired bootstrap's own resolution at this exam size. What changes is that beating the
king at all is now worth something.
"""
from eval.koth import CHALLENGER_SHARE, King, Tier, Tournament

TIERS = (Tier(name="binary", max_params=10**10, weight=0.40),
         Tier(name="sub4", max_params=10**10, weight=0.15))


def _t():
    t = Tournament(TIERS, margin=0.05)
    t.kings["binary"] = King("king_hk", "king_model", 0.20, 1)
    t.kings["sub4"] = King("k4", "m4", 0.30, 1)
    return t


def test_a_real_improvement_below_the_margin_now_earns():
    t = _t()
    t.runners_up["binary"] = {"miner": "challenger_hk", "model_id": "ch", "lcb": 0.01}
    w = t.weights()
    total = 0.40 + 0.15
    assert abs(w["challenger_hk"] - (0.40 / total) * CHALLENGER_SHARE) < 1e-9
    assert abs(w["king_hk"] - (0.40 / total) * (1 - CHALLENGER_SHARE)) < 1e-9
    assert abs(sum(w.values()) - 1.0) < 1e-9, "the vector must still sum to 1"


def test_a_copy_of_the_king_earns_nothing():
    """THE ATTACK THIS HAS TO SURVIVE. A copied artifact scores what the king scores, so its paired
    lower bound is ~0 — `consider` never records it, and there is nothing here to pay."""
    t = _t()                              # no runner-up recorded: lcb was not > 0
    w = t.weights()
    assert set(w) == {"king_hk", "k4"}
    assert abs(w["king_hk"] - 0.40 / 0.55) < 1e-9, "the king keeps the tier whole"


def test_taking_the_crown_still_pays_far_more_than_camping_second():
    """If second place approached the crown's income the incentive would inverm — the point is a
    ladder, not a comfortable step."""
    t = _t()
    t.runners_up["binary"] = {"miner": "ch", "model_id": "c", "lcb": 0.01}
    second = t.weights()["ch"]
    t2 = _t()
    t2.kings["binary"] = King("ch", "c", 0.30, 2)      # same miner, now king
    first = t2.weights()["ch"]
    assert first > 3 * second, f"crown {first} vs runner-up {second}"


def test_the_king_is_never_paid_twice_for_their_own_challenger_row():
    """A king's own commitment is re-scored as a challenger too; paying that as a runner-up would
    hand them the tier and the runner-up share for one artifact."""
    t = _t()
    t.runners_up["binary"] = {"miner": "king_hk", "model_id": "king_model", "lcb": 0.02}
    w = t.weights()
    assert abs(w["king_hk"] - 0.40 / 0.55) < 1e-9
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_stale_runner_up_evidence_is_cleared(monkeypatch):
    """`consider` must drop a tier's runner-up when nobody re-earns it, or a one-off improvement
    keeps drawing emission for rounds it did not compete in.

    monkeypatch, NOT module assignment: the first version of this test rebound
    `koth.softmin_lcb_diff` permanently and six unrelated crown-path tests failed in the full suite
    while passing alone. A test that leaks its stubs is worse than no test."""
    class _S:
        def __init__(self, mid, ret, miner):
            self.valid, self.retention = True, ret
            self.sub = type("x", (), {"model_id": mid, "miner": miner})()

    t = _t()
    t.runners_up["binary"] = {"miner": "old", "model_id": "old", "lcb": 0.03}
    monkeypatch.setattr("eval.koth.softmin_lcb_diff", lambda a, b, seed=0: -0.01)
    monkeypatch.setattr("eval.koth.axis_regression", lambda a, b, seed=0: None)
    t.consider("binary", [_S("new", 0.10, "new_hk")], _S("king_model", 0.20, "king_hk"))
    assert "binary" not in t.runners_up
    assert abs(t.weights()["king_hk"] - 0.40 / 0.55) < 1e-9


def test_the_record_says_who_was_paid_and_why(monkeypatch):
    """A weight vector paying a hotkey that holds no crown must be explicable from the record
    alone. Without this an auditor has to know CHALLENGER_SHARE and join model_id against the
    submissions list to discover why a non-king was paid."""
    class _S:
        def __init__(self, mid, ret, miner):
            self.valid, self.retention = True, ret
            self.sub = type("x", (), {"model_id": mid, "miner": miner})()

    t = _t()
    monkeypatch.setattr("eval.koth.softmin_lcb_diff", lambda a, b, seed=0: 0.012)
    monkeypatch.setattr("eval.koth.axis_regression", lambda a, b, seed=0: None)
    ev = t.consider("binary", [_S("ch", 0.21, "ch_hk")], _S("king_model", 0.20, "king_hk"))
    assert ev["action"] == "hold"
    assert ev["challenger_miner"] == "ch_hk"
    assert ev["challenger_share"] == CHALLENGER_SHARE
    assert ev["challenger_lcb"] == 0.012
    assert t.weights()["ch_hk"] > 0


def test_no_challenger_fields_when_nobody_qualifies(monkeypatch):
    """Absence must be unambiguous: a hold with no challenger_miner means the king kept the tier."""
    class _S:
        def __init__(self, mid, ret, miner):
            self.valid, self.retention = True, ret
            self.sub = type("x", (), {"model_id": mid, "miner": miner})()

    t = _t()
    monkeypatch.setattr("eval.koth.softmin_lcb_diff", lambda a, b, seed=0: -0.02)
    monkeypatch.setattr("eval.koth.axis_regression", lambda a, b, seed=0: None)
    ev = t.consider("binary", [_S("ch", 0.10, "ch_hk")], _S("king_model", 0.20, "king_hk"))
    assert "challenger_miner" not in ev
    assert "ch_hk" not in t.weights()
