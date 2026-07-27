"""One full validator round on the AXIS substrate — the GLM-COVER production assembly.

This is the loop the crown substrate was missing. `shadow_epoch` drives validator_loop ->
round_engine, which is the DEPRECATED CoT/LLM-judge path; everything the cover work now
hangs off (axis_round, the deterministic checkers, the worst-axis dethrone, the per-axis
liveness gate) had no operator-runnable end-to-end path with the real front door attached.

  committed submissions
    -> intake: economics -> safety inspect -> tier fit -> content hash + commit-reveal
                                                          [intake.py + identity.py]
    -> SafeStudentRunner for each accepted checkpoint      [runners.py]
    -> axis_round: fresh nonce-seeded items per axis, GLM authors the reference,
       per-axis retention -> worst-domain soft-min -> crown gates -> KOTH   [axis_round.py]
    -> optional genre-overfit crown precondition            [overfit_gate.py]
    -> settle bonds                                          [economics.py]
    -> SIGNED reproducible round record                      [round_record.py + signing.py]
    -> weights                                               [koth.py]

Pure: models + specs in, decisions out. Chain I/O wraps it exactly like run_v2_epoch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .axis_round import AxisSpec, axis_round
from .economics import RegistrationLedger
from .gates import TierBudget
from .intake import intake
from .koth import Submission, Tier, Tournament
from .round_record import RoundRecord, build_round_record
from .seeds import derive_seed


@dataclass
class CommittedSubmission:
    hotkey: str
    coldkey: str
    tier: str
    ckpt_dir: str
    declared_compute_h100h: float
    bond_posted: float = 0.0
    make_runner: object = None      # callable() -> ModelRunner (SafeStudentRunner in prod)
    revealed_hash: str = ""
    salt: str = ""
    committed_value: str = ""


@dataclass
class RoundOutcome:
    accepted: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    weights: dict = field(default_factory=dict)
    record: RoundRecord | None = None
    refunds: dict = field(default_factory=dict)


def run_axis_round(
    round_no: int, commit_root: str, round_nonce: str,
    committed: list[CommittedSubmission],
    specs: list[AxisSpec], glm, base,
    tiers: list[Tier], tier_budgets: dict[str, TierBudget],
    tournament: Tournament, ledger: RegistrationLedger, registry: dict,
    teacher_id: str = "glm", base_id: str = "base",
    items_per_axis: int = 150, max_new_tokens: int = 512,
    overfit_check=None, signer=None, surprise_k: int | None = None,
) -> RoundOutcome:
    out = RoundOutcome()

    # 1. front door — economics, safety, tier fit, identity + commit-reveal. No untrusted
    #    weights are loaded until a submission has passed all of it.
    subs = []
    for c in committed:
        d = intake(c.ckpt_dir, tier_budgets[c.tier], ledger, c.hotkey, c.coldkey,
                   c.bond_posted, revealed_hash=c.revealed_hash, salt=c.salt,
                   committed_value=c.committed_value)
        if not d.accepted:
            out.rejected.append((c.hotkey, d.reasons))
            continue
        # model_id = CONTENT HASH: binds the scored artifact to the committed one and makes
        # koth's copy-guard content-based rather than hotkey-based.
        sub = Submission(miner=c.hotkey, tier=c.tier, model_id=d.content_hash,
                         params=d.inspection.params, compute_h100h=c.declared_compute_h100h,
                         coldkey=c.coldkey)
        subs.append((sub, c.make_runner()))
        out.accepted.append(c.hotkey)

    # 2. score — items are minted from the commit nonce, so they do not exist until after
    #    checkpoints lock (nothing to memorize offline).
    commit_seed = derive_seed(commit_root, round_nonce, "axis-items")
    res = axis_round(round_no, commit_seed, specs, glm, base, tiers, tournament, subs,
                     registry=registry, items_per_axis=items_per_axis,
                     max_new_tokens=max_new_tokens, overfit_check=overfit_check,
                     surprise_k=surprise_k, commit_root=commit_root, round_nonce=round_nonce)
    out.weights = res.weights

    # 3. settle refundable anti-grind bonds (refunded iff the miner improved its own best)
    for mid, s in res.scored.items():
        refund = ledger.settle(s.sub.miner, s.sub.coldkey, s.retention)
        if refund:
            out.refunds[s.sub.miner] = refund

    # 4. auditable record: per-axis points, per-submission verdicts, crown decisions —
    #    SIGNED so a published verdict is attributable, not merely self-consistent.
    # one record entry per axis: (axis name, teacher-passed count) — enough for an auditor
    # to re-mint the same items from the nonce and re-derive every verdict.
    pts = [{"rollout_id": ax, "k": n, "mode": "axis"}
           for ax, n in sorted({a.axis: a.n for s in res.scored.values() for a in s.axes}.items())]
    out.record = build_round_record(round_no, commit_root, round_nonce, teacher_id,
                                    "deterministic-checkers", base_id,
                                    "axes:" + ",".join(s.name for s in specs),
                                    pts, res.scored, res.events, res.weights)
    if signer is not None:
        out.record.sign(signer)
    return out
