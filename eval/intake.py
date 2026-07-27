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
    # content hash of the accepted artifact — the model_id used downstream. Binds the
    # scored artifact to the committed one (bait-and-switch) and makes koth's copy-guard
    # content-based rather than hotkey-based.
    content_hash: str = ""
    # measured bit budget (bitrate.BitReport) when the tier declares one. None = not measured.
    bits: object = None


def intake(ckpt_dir: str | Path, tier: TierBudget, ledger: RegistrationLedger | None = None,
           hotkey: str = "", coldkey: str = "", bond_posted: float = 0.0,
           revealed_hash: str = "", salt: str = "", committed_value: str = "") -> IntakeDecision:
    """`revealed_hash`/`salt`/`committed_value`: when all three are supplied, the artifact
    is fetch-verified against the on-chain commit-reveal before it is ever loaded."""
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

    # 3b. BIT BUDGET, measured from the actual tensor data. `tier_gate` above works off dtype
    #     HEADERS, which is the right check for "is a bigger model being smuggled in" and the
    #     wrong one for "how compressed is this": a genuinely 1.125-bit model stored in a BF16
    #     container (PrismML ship exactly that as `-unpacked`) reads as 16.0 from headers. For a
    #     subnet whose product is "a runnable checkpoint at a stated bit budget", the budget has
    #     to bind on the bytes, so this is where it binds.
    if getattr(tier, "bit_tier", None) is not None:
        from .bitrate import bit_tier_gate, measure_checkpoint
        try:
            rep = measure_checkpoint(ckpt_dir)
        except Exception as e:
            return IntakeDecision(False, insp, reasons=[f"bit measurement failed: {e}"])
        bok, breasons = bit_tier_gate(rep, tier.bit_tier)
        if not bok:
            return IntakeDecision(False, insp, reasons=breasons, bits=rep)
        bits_report = rep
    else:
        bits_report = None

    # 4. identity: content-hash the artifact, and fetch-verify it against the commitment
    #    when one was supplied. Done before any runner is built (no untrusted load yet).
    from .identity import content_hash, verify_reveal
    if revealed_hash and salt and committed_value:
        ok, res = verify_reveal(ckpt_dir, revealed_hash, salt, committed_value)
        if not ok:
            return IntakeDecision(False, insp, reasons=[f"commit-reveal: {res}"])
        chash = res
    else:
        try:
            chash = content_hash(ckpt_dir)
        except Exception as e:
            return IntakeDecision(False, insp, reasons=[f"identity: {e}"])

    # accepted — record the submission (degeneracy + pass@k gates run later, on outputs)
    if ledger is not None:
        ledger.record(hotkey, coldkey, bond_posted)
    return IntakeDecision(True, insp, reasons=[], content_hash=chash, bits=bits_report)
