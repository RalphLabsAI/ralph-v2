"""One full validator round — the whole pipeline assembled.

    committed submissions
        -> intake gate (economics + safety + tier)     [intake.py]
        -> SafeStudentRunner for each accepted          [runners.py]
        -> score every accepted + reigning king         [round_engine.run_round]
        -> settle bonds (refund on self-improvement)     [economics.py]
        -> build the signed round record                 [round_record.py]
        -> weights                                       [koth.py]

Model/pile/chain access is injected, so this runs on CPU with sims and against real GLM
on a GPU unchanged. The CHAIN I/O (read commitments, set weights, publish the record
hash) is the documented boundary — the karpa validator wraps this and provides:
  read_commitments() -> [(hotkey, coldkey, tier, ckpt_dir, declared_compute, bond)]
  commit_root, round_nonce   (from the on-chain commitment + block hash)
  set_weights(weights); publish(record)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .core import ModelRunner
from .economics import RegistrationLedger
from .gates import TierBudget
from .intake import intake
from .koth import Submission, Tier, Tournament
from .round_engine import run_round
from .round_record import build_round_record, RoundRecord
from .seeds import derive_seed
from .trajectory import Rollout, StepJudge, sample_points


@dataclass
class CommittedSubmission:
    hotkey: str
    coldkey: str
    tier: str
    ckpt_dir: str
    declared_compute_h100h: float
    bond_posted: float = 0.0
    make_runner: object = None   # callable() -> ModelRunner (SafeStudentRunner in prod)
    # commit-reveal (optional): when all three are set, intake fetch-verifies the artifact
    # against the on-chain commitment before it is loaded.
    revealed_hash: str = ""
    salt: str = ""
    committed_value: str = ""


@dataclass
class RoundOutcome:
    accepted: list = field(default_factory=list)
    rejected: list = field(default_factory=list)   # (hotkey, reasons)
    weights: dict = field(default_factory=dict)
    record: RoundRecord | None = None
    refunds: dict = field(default_factory=dict)


def run_validator_round(
    round_no: int, commit_root: str, round_nonce: str,
    committed: list[CommittedSubmission],
    experience: list[Rollout], glm: ModelRunner, base: ModelRunner, judge: StepJudge,
    tiers: list[Tier], tier_budgets: dict[str, TierBudget],
    tournament: Tournament, ledger: RegistrationLedger, registry: dict,
    pile_id: str = "pile", n_points: int = 120, self_frac: float = 0.0,
    signer=None,
) -> RoundOutcome:
    out = RoundOutcome()

    # 1. intake — fail-closed, cheapest-first; never builds a runner before gates pass
    subs = []
    for c in committed:
        d = intake(c.ckpt_dir, tier_budgets[c.tier], ledger, c.hotkey, c.coldkey, c.bond_posted,
                   revealed_hash=c.revealed_hash, salt=c.salt, committed_value=c.committed_value)
        if not d.accepted:
            out.rejected.append((c.hotkey, d.reasons))
            continue
        runner = c.make_runner()
        # model_id = CONTENT HASH (not the hotkey): binds the scored artifact to the
        # committed one, makes koth's copy-guard content-based, and stops two subs from one
        # hotkey colliding in the registry.
        sub = Submission(miner=c.hotkey, tier=c.tier, model_id=d.content_hash,
                         params=d.inspection.params, compute_h100h=c.declared_compute_h100h)
        subs.append((sub, runner))
        out.accepted.append(c.hotkey)

    # 2-3. score (accepted + reigning king, same fresh points) + crown + weights.
    # The seed BINDS the nonce: points do not exist until commitments are locked and the
    # chain nonce is drawn, so no miner can pre-fit to this round's points (seeds.py).
    commit_seed = derive_seed(commit_root, round_nonce, "points")
    res = run_round(round_no, commit_seed, experience, glm, base, judge, tiers, tournament,
                    subs, registry=registry, n_points=n_points, self_frac=self_frac)
    out.weights = res.weights

    # 4. settle bonds — refund on self-improvement, forfeit otherwise
    for mid, s in res.scored.items():
        refund = ledger.settle(s.sub.miner, s.sub.coldkey, s.retention)
        if refund:
            out.refunds[s.sub.miner] = refund

    # 5. signed round record (per-point verdicts -> reproducible + auditable)
    points = sample_points(experience, n_points, self_frac, seed=commit_seed)  # same seed => same points
    out.record = build_round_record(round_no, commit_root, round_nonce, glm.name, judge.name,
                                    base.name, pile_id, points, res.scored, res.events, res.weights)
    if signer is not None:      # attributable, not merely self-consistent (eval/signing.py)
        out.record.sign(signer)
    return out
