"""Commit-then-generate seeding.

The freshness guarantee is not "we keep the test set secret" — secrecy leaks (the
previous distillation subnet leaked a 30GB teacher-logit cache to public
HuggingFace). It is that the items DO NOT EXIST until after the checkpoints are
locked, and the seed that mints them is unpredictable before that moment.

    seed = H(commit_root || round_nonce || axis_name)

`commit_root` is the on-chain commitment over the round's submitted checkpoint
hashes; `round_nonce` is a chain value (block hash) nobody controls. Publishing
this function is safe and intended — knowing HOW items are minted does not let you
precompute items for a seed you cannot predict.
"""
from __future__ import annotations

import hashlib


def derive_seed(commit_root: str, round_nonce: str, axis_name: str) -> int:
    """Deterministic 64-bit seed for one (round, axis)."""
    h = hashlib.sha256()
    for part in (commit_root, round_nonce, axis_name):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")  # unambiguous separator
    return int.from_bytes(h.digest()[:8], "big")


def derive_surprise_axes(commit_root: str, round_nonce: str, pool: list[str], k: int = 2) -> list[str]:
    """Pick which axes actually SCORE this round, from post-commit entropy.

    THE GAP THIS CLOSES. Corpus scale and item-freshness stop a miner memorizing items; they do
    NOTHING against a miner who pre-fits the TASK FORMAT, because the formats are in the
    validator source everyone reads. Fine-tuning the narrow skill "find the word after an anchor
    phrase" is cheap and buys the crown without retaining much of anything.

    So the format a miner must satisfy is not knowable at seal time: publish the whole pool, but
    draw which k count from the round nonce AFTER checkpoints lock. Combined with worst-axis
    aggregation the two compose multiplicatively — you must cover every format in the pool
    (you cannot know which are drawn) AND you are scored on your weakest drawn one, so buying a
    cheap corner of the space is worthless.

    The honest bar is economic, not absolute: a miner who covers the WHOLE pool is back to
    parity — which is the point, because the pool is diverse real-text task formats, and
    covering all of them is general capability rather than a trick. Cost scales with |pool|/k;
    the payoff is capped by the worst drawn axis.

    Selection is a pure function of published inputs, so an auditor re-derives it exactly; it is
    also recorded in the round record, because a gate nobody can verify is not a gate.
    """
    if k >= len(pool):
        return sorted(pool)
    # INDEPENDENT seed per draw. The old `seed >> (i*13)` exhausted a 64-bit word: for i>=5 the
    # shift is >= 65, the value is 0, and every later draw collapses to index 0 deterministically.
    picked: list[str] = []
    remaining = sorted(pool)
    for i in range(k):
        s = derive_seed(commit_root, round_nonce, f"__surprise__{i}")
        picked.append(remaining.pop(s % len(remaining)))
    return sorted(picked)
