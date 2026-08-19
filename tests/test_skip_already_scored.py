"""A round must not pay to measure bytes it has already measured.

In live round 2 (2026-08-18) ALL THIRTEEN challengers were byte-identical to round 1. The round
cost $4.99 and 69 minutes and produced no new information about any of them — it re-derived numbers
already on the trail, against a freshly drawn exam.

Cost is the smaller half. The exam is redrawn every round, so an unchanged challenger gets a FREE
RE-ROLL forever at the validator's expense, and a weaker model only has to wait for a favourable
draw. Round 2 showed the defence holding — `5CXEMm6u6ono` scored raw-higher than the sub4 king
(0.2784 vs 0.2671) and the paired bootstrap refused it at margin_lcb -0.0426 — but unlimited free
attempts against a fixed margin is the wrong shape.
"""
from eval.lineage import replay_with_history


class _Rec:
    def __init__(self, round_no, model_ids):
        self.round = round_no
        self.submissions = [type("S", (), {"model_id": m})() for m in model_ids]


class _Pub:
    """Enough publisher to walk: an index and a blob sink, same shape lineage reads."""

    def __init__(self, rounds):
        self.rounds = rounds
        self.sink = self

    def load_index(self):
        return {"rounds": [{"round": n, "name": f"r{n}"} for n in sorted(self.rounds)]}

    def get(self, name):
        return name.encode()


def _patched(monkeypatch, rounds, kings=None):
    monkeypatch.setattr("eval.lineage.replay_from_trail", lambda *a, **k: dict(kings or {}))
    # patched on eval.rerun, not eval.lineage: the import is function-local, so it resolves from
    # the source module at call time and there is no lineage attribute to replace
    monkeypatch.setattr("eval.rerun.record_from_blob",
                        lambda blob: _Rec(int(blob.decode()[1:]),
                                          rounds[int(blob.decode()[1:])]))
    return replay_with_history(_Pub(rounds))


def test_history_names_every_artifact_already_measured(monkeypatch):
    _kings, already = _patched(monkeypatch, {1: ["aaa", "bbb"], 2: ["bbb", "ccc"]})
    assert set(already) == {"aaa", "bbb", "ccc"}


def test_it_reports_the_round_the_bytes_were_first_scored(monkeypatch):
    """The message a miner reads says WHEN, so "already scored" is checkable, not an assertion."""
    _kings, already = _patched(monkeypatch, {1: ["aaa", "bbb"], 2: ["bbb", "ccc"]})
    assert already["bbb"] == 1, "must name the FIRST round that measured it, not the latest"
    assert already["ccc"] == 2


def test_an_empty_trail_skips_nobody(monkeypatch):
    """Round 1 has no history, so every commitment must still be scored."""
    _kings, already = _patched(monkeypatch, {})
    assert already == {}


def test_the_king_is_not_in_the_skip_set_by_accident(monkeypatch):
    """The king's own bytes ARE in the history — it was scored to become king. The caller exempts
    it explicitly, because it is re-scored through the incumbent path on this round's exam. If that
    exemption were dropped, a held crown would stop being defended at all."""
    king = type("K", (), {"model_id": "bbb"})()
    kings, already = _patched(monkeypatch, {1: ["aaa", "bbb"]}, kings={"sub4": king})
    assert "bbb" in already                      # history knows it
    crowned = {r.model_id for r in kings.values()}
    repeats = [m for m in already if m not in crowned]
    assert repeats == ["aaa"], "the king must not be skipped as a stale challenger"
