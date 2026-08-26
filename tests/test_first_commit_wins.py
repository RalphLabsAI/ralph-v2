"""One artifact, one owner — the hotkey that committed those bytes first.

Every challenger's artifact is public on HuggingFace the moment it is committed. Without this a
sniper downloads a competitor's file and commits the identical bytes under their own hotkey. It was
survivable while only CROWNS were paid — a duplicate of the king scores what the king scores and
cannot clear the dethrone margin — and it stops being survivable the moment a strictly-improving
challenger is paid, because the sniper would collect for someone else's improvement.

Miners raised exactly this ("cunning miners copy it and resubmit"). They were right that it is
possible; they were wrong that it had happened.
"""
from eval.chain_bittensor import _first_commit_wins


class _C:
    def __init__(self, hotkey, revealed_hash):
        self.hotkey, self.revealed_hash = hotkey, revealed_hash


def test_the_later_commitment_of_identical_bytes_is_dropped():
    a, b = _C("orig", "hash_x"), _C("sniper", "hash_x")
    skipped = []
    kept = _first_commit_wins([a, b], {"orig": 100, "sniper": 200}, skipped)
    assert [c.hotkey for c in kept] == ["orig"]
    assert skipped and skipped[0][0] == "sniper"
    assert "already committed by" in skipped[0][1]


def test_the_winner_is_the_earlier_block_not_the_iteration_order():
    """The sniper appears FIRST in the list and still loses."""
    sniper, orig = _C("sniper", "hash_x"), _C("orig", "hash_x")
    kept = _first_commit_wins([sniper, orig], {"orig": 100, "sniper": 200}, [])
    assert [c.hotkey for c in kept] == ["orig"]


def test_ties_are_broken_deterministically():
    """Same block for both: the outcome must not depend on dict or network ordering."""
    one = _first_commit_wins([_C("aaa", "h"), _C("bbb", "h")], {"aaa": 7, "bbb": 7}, [])
    two = _first_commit_wins([_C("bbb", "h"), _C("aaa", "h")], {"aaa": 7, "bbb": 7}, [])
    assert [c.hotkey for c in one] == [c.hotkey for c in two] == ["aaa"]


def test_distinct_artifacts_are_untouched():
    kept = _first_commit_wins([_C("a", "h1"), _C("b", "h2")], {"a": 1, "b": 2}, [])
    assert [c.hotkey for c in kept] == ["a", "b"]


def test_an_unrevealed_commitment_collides_with_nobody():
    """No revealed hash means there is nothing to compare yet; dropping those would refuse miners
    for the crime of not having revealed."""
    skipped = []
    kept = _first_commit_wins([_C("a", ""), _C("b", "")], {"a": 1, "b": 2}, skipped)
    assert [c.hotkey for c in kept] == ["a", "b"]
    assert skipped == []


def test_one_miner_re_committing_their_own_bytes_is_not_punished():
    """A single hotkey is its own owner — the rule is about a SECOND hotkey taking your work."""
    kept = _first_commit_wins([_C("a", "h")], {"a": 5}, [])
    assert [c.hotkey for c in kept] == ["a"]
