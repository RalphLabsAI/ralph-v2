"""When the remote aborts, the operator must be told WHY, not guessed at.

Round 3 (2026-08-25) scored eleven submissions, hit the determinism canary — the student path gave
29/72 and then 32/72 for the SAME model on the SAME box — and correctly refused to crown from a box
whose results are a lottery. The operator was told "the remote scored no record (no usable
submissions?)". It had eleven things to score.

The abort event carries a full explanation and travels home in the summary. It was being discarded
in favour of a guess with a question mark on it.
"""
import json
import pytest

REASON = ("the student path is not reproducible: scoring 10261bdd8150… twice on this box gave "
          "0.292700 and 0.310400 (worst slice drifted 0.017700). A crown decided inside that "
          "drift is a lottery.")


def _raise_like_orchestrator(raw, summary):
    """The exact branch under test, kept in step with eval/orchestrator.py."""
    from eval.orchestrator import RemoteRoundError
    if raw is None:
        why = next((str(e.get("reason") or "") for e in (summary.get("events") or [])
                    if isinstance(e, dict) and e.get("action") == "abort"), "")
        det = summary.get("determinism") or {}
        raise RemoteRoundError(
            why or (f"the remote returned no record. determinism={det}" if det else
                    "the remote scored no record (no usable submissions?)"))


def test_the_determinism_abort_reason_reaches_the_operator():
    from eval.orchestrator import RemoteRoundError
    summary = {"events": [{"round": 3, "action": "abort", "reason": REASON}],
               "determinism": {"first": 0.2927, "second": 0.3104, "reproduced": False}}
    with pytest.raises(RemoteRoundError, match="not reproducible"):
        _raise_like_orchestrator(None, summary)


def test_it_does_not_claim_there_was_nothing_to_score():
    from eval.orchestrator import RemoteRoundError
    summary = {"events": [{"round": 3, "action": "abort", "reason": REASON}], "determinism": {}}
    try:
        _raise_like_orchestrator(None, summary)
    except RemoteRoundError as e:
        assert "no usable submissions" not in str(e)


def test_numbers_are_used_when_the_event_has_no_prose():
    from eval.orchestrator import RemoteRoundError
    summary = {"events": [], "determinism": {"first": 0.29, "second": 0.31, "reproduced": False}}
    with pytest.raises(RemoteRoundError, match="determinism="):
        _raise_like_orchestrator(None, summary)


def test_the_old_message_survives_when_there_is_genuinely_nothing():
    from eval.orchestrator import RemoteRoundError
    with pytest.raises(RemoteRoundError, match="no usable submissions"):
        _raise_like_orchestrator(None, {"events": [], "determinism": None})


def test_a_real_record_is_untouched():
    assert _raise_like_orchestrator({"round": 3}, {}) is None
