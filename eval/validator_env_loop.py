"""One full validator round on the ENV substrate — the production assembly.

Mirrors validator_loop.run_validator_round but scores on the multi-turn env round
(env_round.py, the validated substrate) instead of the CoT text round. The front door
(intake gates, anti-grind economics) and the auditable round record are substrate-
agnostic and reused unchanged; only the scoring core and the runner->agent wrapping
differ. Chain I/O wraps this exactly like run_v2_epoch wraps validator_loop.

  committed submissions
    -> intake (economics + safety + tier)            [intake.py]
    -> SafeStudentRunner -> ModelAgent for each       [runners.py + multiturn.py]
    -> score accepted + reigning king on env round    [env_round.py]
    -> settle bonds                                    [economics.py]
    -> signed round record                             [round_record.py]
    -> weights                                         [koth.py]
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .economics import RegistrationLedger
from .env_round import env_round
from .gates import TierBudget
from .intake import intake
from .koth import Submission, Tier, Tournament
from .multiturn import Agent, MTRollout, sample_env_points
from .round_record import build_round_record, RoundRecord
from .seeds import derive_seed


@dataclass
class CommittedSubmission:
    hotkey: str
    coldkey: str
    tier: str
    ckpt_dir: str
    declared_compute_h100h: float
    bond_posted: float = 0.0
    make_agent: object = None   # callable() -> Agent (ModelAgent(SafeStudentRunner) in prod)


@dataclass
class RoundOutcome:
    accepted: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    weights: dict = field(default_factory=dict)
    record: RoundRecord | None = None
    refunds: dict = field(default_factory=dict)


def run_env_round(
    round_no: int, commit_root: str, round_nonce: str,
    committed: list[CommittedSubmission],
    pile: list[MTRollout], base_agent: Agent, teacher: Agent,
    tiers: list[Tier], tier_budgets: dict[str, TierBudget],
    tournament: Tournament, ledger: RegistrationLedger, registry: dict,
    teacher_id: str = "glm", base_id: str = "base", pile_id: str = "env-pile",
    n_points: int = 300, self_frac: float = 0.4,
) -> RoundOutcome:
    out = RoundOutcome()

    subs = []
    for c in committed:
        d = intake(c.ckpt_dir, tier_budgets[c.tier], ledger, c.hotkey, c.coldkey, c.bond_posted)
        if not d.accepted:
            out.rejected.append((c.hotkey, d.reasons))
            continue
        sub = Submission(miner=c.hotkey, tier=c.tier, model_id=c.hotkey,
                         params=d.inspection.params, compute_h100h=c.declared_compute_h100h)
        subs.append((sub, c.make_agent()))
        out.accepted.append(c.hotkey)

    commit_seed = derive_seed(commit_root, round_nonce, "env-points")
    res = env_round(round_no, commit_seed, pile, base_agent, tiers, tournament, subs,
                    registry=registry, n_points=n_points, self_frac=self_frac, teacher=teacher)
    out.weights = res.weights

    for mid, s in res.scored.items():
        refund = ledger.settle(s.sub.miner, s.retention)
        if refund:
            out.refunds[s.sub.miner] = refund

    # same seed -> same points; render as record-friendly dicts (env oracle IS the judge)
    pts = sample_env_points(pile, n_points, self_frac, seed=commit_seed)
    pt_dicts = [{"rollout_id": p.env_idx, "k": p.k, "mode": p.mode} for p in pts]
    out.record = build_round_record(round_no, commit_root, round_nonce, teacher_id,
                                    "env-oracle(deterministic)", base_id, pile_id,
                                    pt_dicts, res.scored, res.events, res.weights)
    return out
