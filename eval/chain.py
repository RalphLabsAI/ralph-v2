"""Chain-I/O boundary for a v2 epoch.

`run_validator_round` (validator_loop.py) is pure: models + pile in, decisions out. The
only thing between it and a live subnet is chain I/O — reading what miners committed,
drawing the round nonce, and writing back weights / crown / the round-record hash.

This module defines that boundary as a small PROTOCOL (`ChainIO`) and wires it to the
round in `run_v2_epoch`. It does NOT import or modify Ralph's live v1 validator — the v1
service (karpa/validator/service.py) already implements every method here
(get_king/set_king/set_weights/current_block/get_commitment/blacklist), so bringing v2 up
is: construct the pile + pinned (glm, base, judge), then call `run_v2_epoch(chain, ...)`
from the existing epoch loop. Keeping it a protocol lets the whole thing be tested against
`FakeChain` with zero chain dependency, and keeps this session off the live signer.

Commit-then-generate ordering enforced here:
  1. read commitments (sealed H(content_hash‖salt)) that locked BEFORE this block
  2. draw round_nonce = hash of a block AFTER the commit window closed
  3. only now are points minted (seed binds commit_root‖nonce) and scoring runs
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .koth import Tier, Tournament
from .economics import RegistrationLedger
from .gates import TierBudget
from .round_record import RoundRecord
from .trajectory import Rollout, StepJudge
from .validator_loop import CommittedSubmission, RoundOutcome, run_validator_round


@dataclass
class Commitment:
    """One miner's sealed on-chain commitment, revealed and resolved to a fetched dir."""
    hotkey: str
    coldkey: str
    tier: str
    ckpt_dir: str              # local path after fetch+verify against the revealed hash
    declared_compute_h100h: float
    bond_posted: float = 0.0


@runtime_checkable
class ChainIO(Protocol):
    """Exactly what a v2 epoch needs from the chain. v1's service satisfies all of it."""

    def current_block(self) -> int: ...
    def block_hash(self, block: int) -> str: ...
    def read_commitments(self, min_block: int, max_block: int) -> list[Commitment]: ...
    def commit_root(self, min_block: int, max_block: int) -> str: ...
    def set_weights(self, weights: dict[str, float]) -> bool: ...
    def set_king(self, tier: str, hotkey: str, model_id: str) -> None: ...
    def get_king(self, tier: str) -> object | None: ...
    def blacklist(self, hotkey: str, reason: str) -> None: ...
    def publish_record(self, record: RoundRecord) -> None: ...


@dataclass
class EpochResult:
    round_no: int
    outcome: RoundOutcome
    record_sha256: str


def run_v2_epoch(
    chain: ChainIO, round_no: int,
    experience: list[Rollout], glm, base, judge: StepJudge,
    tiers: list[Tier], tier_budgets: dict[str, TierBudget],
    tournament: Tournament, ledger: RegistrationLedger, registry: dict,
    commit_window: int = 100, make_safe_runner=None,
    pile_id: str = "pile", n_points: int = 120, self_frac: float = 0.25,
) -> EpochResult:
    """One v2 epoch: read the sealed commit window, draw the nonce, score, write back.

    `make_safe_runner(ckpt_dir) -> ModelRunner` builds the locked-down loader for an
    accepted checkpoint (SafeStudentRunner in prod; a sim factory in tests). It is NOT
    called until intake gates pass — no untrusted weights load on a rejected submission.
    """
    now = chain.current_block()
    lo, hi = now - commit_window, now
    commits = chain.read_commitments(lo, hi)
    commit_root = chain.commit_root(lo, hi)
    round_nonce = chain.block_hash(now)   # drawn AFTER the window closed -> unpredictable

    committed = [
        CommittedSubmission(
            hotkey=c.hotkey, coldkey=c.coldkey, tier=c.tier, ckpt_dir=c.ckpt_dir,
            declared_compute_h100h=c.declared_compute_h100h, bond_posted=c.bond_posted,
            make_runner=(lambda cd=c.ckpt_dir: make_safe_runner(cd)),
        )
        for c in commits
    ]

    outcome = run_validator_round(
        round_no, commit_root, round_nonce, committed, experience, glm, base, judge,
        tiers, tier_budgets, tournament, ledger, registry,
        pile_id=pile_id, n_points=n_points, self_frac=self_frac,
    )

    # write-back: crown per tier, then weights, then the auditable record.
    for tier in tiers:
        king = tournament.kings.get(tier.name)
        if king is not None:
            chain.set_king(tier.name, king.miner, king.model_id)
    chain.set_weights(outcome.weights)
    if outcome.record is not None:
        chain.publish_record(outcome.record)

    return EpochResult(round_no, outcome, outcome.record.sha256() if outcome.record else "")
