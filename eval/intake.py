"""Validator front door — accept or reject a submission before it is ever loaded/run.

Order matters: cheapest + safety-critical checks first, so a hostile or malformed
checkpoint is rejected without spending GPU or (crucially) without ever executing miner
code. Fail-closed: any check that errors is a reject.

    decision = intake(ckpt_dir, tier, ledger, hotkey, coldkey, bond_posted)
    if decision.accepted:
        runner = SafeStudentRunner(ckpt_dir)   # only now is the model touched
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .economics import RegistrationLedger
from .gates import Inspection, TierBudget, inspect_checkpoint, tier_gate


@dataclass
class IntakeDecision:
    accepted: bool
    inspection: Inspection | None = None
    bond_required: float = 0.0
    reasons: list = field(default_factory=list)


def intake(ckpt_dir: str | Path, tier: TierBudget, ledger: RegistrationLedger | None = None,
           hotkey: str = "", coldkey: str = "", bond_posted: float = 0.0) -> IntakeDecision:
    reasons: list[str] = []

    # 1. economics first — cheapest, and stops spam before any file work
    if ledger is not None:
        d = ledger.can_submit(hotkey, coldkey, bond_posted)
        if not d.ok:
            return IntakeDecision(False, bond_required=d.bond_required, reasons=[d.reason])

    # 2. safety + integrity: never loads weights, reads only file list + safetensors header
    try:
        insp = inspect_checkpoint(ckpt_dir)
    except Exception as e:  # fail-closed
        return IntakeDecision(False, reasons=[f"inspection error: {e}"])
    if not insp.ok:
        return IntakeDecision(False, insp, reasons=insp.reasons)

    # 3. tier fit — computed params/bits vs the tier the miner submitted to
    ok, tier_reasons = tier_gate(insp, tier)
    if not ok:
        return IntakeDecision(False, insp, reasons=tier_reasons)

    # accepted — record the submission (degeneracy + pass@k gates run later, on outputs)
    if ledger is not None:
        ledger.record(hotkey, coldkey, bond_posted)
    return IntakeDecision(True, insp, reasons=[])
