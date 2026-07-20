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
    """Pick which un-announced axes are live this round.

    Miners cannot know which surprise axes count until after they have committed,
    so pre-fitting to the announced axes does not protect them.
    """
    if k >= len(pool):
        return sorted(pool)
    seed = derive_seed(commit_root, round_nonce, "__surprise__")
    picked: list[str] = []
    remaining = sorted(pool)
    for i in range(k):
        # rejection-free index draw from the shrinking pool
        idx = (seed >> (i * 13)) % len(remaining)
        picked.append(remaining.pop(idx))
    return sorted(picked)
