"""A submission dropped BEFORE scoring must still appear in the signed record, with a reason.

Replayed against the REAL published round-1 record rather than a hand-built one. A fixture assembled
here would be built from my belief about the record's shape, and the belief is the thing under test:
these submissions are `SubmissionRecord` objects after deserialization but plain dicts before it, so
a fixture that guessed wrong would prove the opposite of what it claimed to.

The bug this pins: `5DhpPeU1uamKP` committed inside round 1's window, never revealed, and was
dropped at the chain read — which appends to `chain.skipped` on the orchestrator, a field that never
reaches the record. It scored nothing, was rejected by nothing, and left no row. From the miner's
side they committed and then were simply not in the round, with nothing to read and nothing to
appeal. `rejected` exists precisely to make that impossible and did not cover this path.
"""
import json
import pathlib

from eval.chain import Commitment
from eval.rerun import record_from_blob
from eval.run_orchestrated import _merge_dropped

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "round-0001-live.json"
DROPPED = "5DhpPeU1uamKPHFPBqDgmxrCmNJmzcSchmqXgttNhRxdSpr9"


def _rec():
    return record_from_blob(FIXTURE.read_bytes())


def _commit(hk, tier="sub4"):
    return Commitment(hotkey=hk, coldkey="", tier=tier, ckpt_dir="",
                      declared_compute_h100h=0.0)


def test_never_revealed_gets_a_row_and_a_reason():
    rec = _rec()
    assert not any(r[0] == DROPPED for r in rec.rejected), "fixture already carries the drop"

    why = "no local artifact (reveal/fetch pending)"
    added = _merge_dropped(rec, [_commit(DROPPED)], [(DROPPED, why)])

    assert added == [DROPPED]
    row = [r for r in rec.rejected if r[0] == DROPPED]
    assert len(row) == 1 and row[0][1] == [why]


def test_operational_notes_are_not_miners():
    """`skipped` also holds our own notes — `set_weights`, `metagraph` — and the ~109 slots on this
    netuid that never committed a v2 envelope. None of them is a rejected submission."""
    rec = _rec()
    before = len(rec.rejected)
    added = _merge_dropped(rec, [_commit(DROPPED)], [
        ("set_weights", "rate limit: 100 blocks"),
        ("metagraph", "could not read the uid -> hotkey map"),
        ("5FnotCommittedAnythingHereXXXXXXXXXXXXXXXXXXXXXX", "commitment is not JSON"),
    ])
    assert added == [] and len(rec.rejected) == before


def test_a_scored_miner_is_never_also_rejected():
    """A late `skipped` entry for someone who DID get scored would put them in both lists, and an
    auditor reading the record would see one round rejecting and paying the same hotkey."""
    rec = _rec()
    scored = rec.submissions[0].miner
    before = len(rec.rejected)
    assert _merge_dropped(rec, [_commit(scored)], [(scored, "stale worker note")]) == []
    assert len(rec.rejected) == before


def test_existing_rejection_is_not_duplicated():
    rec = _rec()
    already = rec.rejected[0][0]
    n_before = len(rec.rejected)
    assert _merge_dropped(rec, [_commit(already)], [(already, "some other reason")]) == []
    assert len(rec.rejected) == n_before


def test_round_1_drop_would_now_be_visible_end_to_end():
    """The whole point: after the merge, every hotkey that committed is accounted for in the record
    — as a scored submission or as a rejection with a reason. Nobody silently absent."""
    rec = _rec()
    committed = [_commit(s.miner) for s in rec.submissions]
    committed += [_commit(r[0]) for r in rec.rejected]
    committed += [_commit(DROPPED)]

    _merge_dropped(rec, committed, [(DROPPED, "no local artifact (reveal/fetch pending)")])

    accounted = {s.miner for s in rec.submissions} | {r[0] for r in rec.rejected}
    missing = [c.hotkey for c in committed if c.hotkey not in accounted]
    assert missing == [], f"still invisible: {missing}"
