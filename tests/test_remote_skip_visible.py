"""A miner dropped on the RENTED box must appear in the signed record too.

`_merge_dropped` was added for drops on the orchestrator (`chain.skipped`). The recurring case was
on the other box: `score_job` puts an unfetchable — or REFUSED — commitment in its own `skipped`
list, and that list travels home in the SUMMARY, which is neither signed nor published. The record
carried only `out.rejected`.

`5DhpPeU1uamK` was dropped this way in BOTH live rounds. Round 2's fetch log, verbatim:

    refused | content hash 85fc805f41e5dcfe… does not match the revealed cc0b64ec4f8996a6…
            | the bytes are not what was committed

That is a miner serving different bytes than they sealed — the most security-relevant call the
system makes — and it produced no row at all, while a miner who merely picked a bad quant format
got a permanent signed rejection. Exactly backwards.
"""
from eval.rerun import record_from_blob
from eval.run_orchestrated import _merge_dropped
import pathlib

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "round-0001-live.json"
DROPPED = "5DhpPeU1uamKPGtZMt25NbxCyZ22tcQLWB1gG5NvkDQe3poM"
REAL = ("content hash 85fc805f41e5dcfe… does not match the revealed cc0b64ec4f8996a6… "
        "— the bytes are not what was committed")


class _C:
    def __init__(self, hk):
        self.hotkey = hk


def test_a_bait_and_switch_lands_in_the_signed_record():
    rec = record_from_blob(FIXTURE.read_bytes())
    added = _merge_dropped(rec, [_C(DROPPED)], [(DROPPED, REAL)])
    assert added == [DROPPED]
    row = [r for r in rec.rejected if r[0] == DROPPED]
    assert len(row) == 1
    assert "does not match the revealed" in row[0][1][0], \
        "the row must carry the REAL reason, not 'artifact could not be fetched'"


def test_remote_and_orchestrator_skips_merge_together():
    """Both lists feed one merge; neither may shadow the other."""
    rec = record_from_blob(FIXTURE.read_bytes())
    other = "5FsomeOtherMinerHotkeyThatWasDroppedLocally000"
    added = _merge_dropped(rec, [_C(DROPPED), _C(other)],
                           [(other, "no local artifact (reveal/fetch pending)"),
                            (DROPPED, REAL)])
    assert set(added) == {DROPPED, other}


def test_the_generic_message_is_not_what_gets_recorded():
    """Pins the half of the fix that is easy to lose: score_job used to file every resolver refusal
    as 'artifact could not be fetched', which reads as a transport error. If that string is all the
    record ever carries, a substitution is indistinguishable from a flaky download."""
    rec = record_from_blob(FIXTURE.read_bytes())
    _merge_dropped(rec, [_C(DROPPED)], [(DROPPED, REAL)])
    text = " ".join(r[1][0] for r in rec.rejected if r[0] == DROPPED)
    assert "could not be fetched" not in text
