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
    # parent.ParentCheck when the tier pins a parent. None = no parent declared.
    parent: object = None


def intake(ckpt_dir: str | Path, tier: TierBudget, ledger: RegistrationLedger | None = None,
           hotkey: str = "", coldkey: str = "", bond_posted: float = 0.0,
           revealed_hash: str = "", salt: str = "", committed_value: str = "") -> IntakeDecision:
    """`revealed_hash`/`salt`/`committed_value`: the artifact is fetch-verified against the
    on-chain commit-reveal before it is ever loaded.

    A `committed_value` with no reveal is REFUSED, not skipped. The check used to run only when all
    three arrived together, so an unrevealed submission bypassed the one gate that binds the scored
    bytes to what was sealed before the round nonce existed — and since nothing populated the
    reveal in production, that was every submission."""
    reasons: list[str] = []

    # 1. economics first — cheapest, and stops spam before any file work
    if ledger is not None:
        d = ledger.can_submit(hotkey, coldkey, bond_posted, tier=getattr(tier, "name", ""))
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
    # LOUD ABOUT WHAT IS NOT ENFORCED. bit_tier and parent are optional, and when they are None
    # the two gates that define this subnet — the measured bit budget and the pinned parent — are
    # skipped without a word. A validator built from a bare TierBudget therefore runs four gates
    # while believing it runs six. RALPH_STRICT_TIERS=1 refuses to accept anything in that state;
    # otherwise the omission is at least recorded on the decision.
    import os as _os
    unenforced = [n for n in ("bit_tier", "parent") if getattr(tier, n, None) is None]
    if unenforced and _os.environ.get("RALPH_STRICT_TIERS", "").strip() not in ("", "0", "false"):
        return IntakeDecision(False, insp, reasons=[
            f"tier {tier.name!r} does not configure {', '.join(unenforced)} — refusing to accept a "
            f"submission whose bit budget or pinned parent would go unchecked "
            f"(RALPH_STRICT_TIERS=1)"])

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

    # 3c. PINNED PARENT. The task is "compress this specific model, architecture unchanged", so
    #     an artifact that is not shape-compatible with the tier's parent is not a submission to
    #     this tier at all. Header-only. Not a provenance proof — no behavioural test can be —
    #     but it stops a foreign artifact being scored against a denominator it never had.
    parent_check = None
    if getattr(tier, "parent", None) is not None:
        from .parent import parent_compat
        parent_check = parent_compat(ckpt_dir, tier.parent, inspection=insp)
        if not parent_check.ok:
            return IntakeDecision(False, insp, reasons=parent_check.reasons, bits=bits_report)

    # 4. identity: content-hash the artifact, and fetch-verify it against the commitment
    #    when one was supplied. Done before any runner is built (no untrusted load yet).
    from .identity import content_hash, verify_reveal
    # FAIL CLOSED. This used to be `if all three are supplied`, which meant a submission with no
    # reveal skipped the seal check ENTIRELY and was scored as if it had passed — the one gate that
    # binds the scored bytes to what was committed before the nonce existed, silently absent. A
    # missing reveal is a miner who has not finished submitting, not a miner who passed.
    if committed_value and not (revealed_hash and salt):
        reasons.append("committed but not revealed: the sealed value cannot be checked against the "
                       "fetched bytes, so nothing binds this artifact to the commitment. Run "
                       "`python -m miner.submit reveal`.")
        return IntakeDecision(False, reasons=reasons)
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
        ledger.record(hotkey, coldkey, bond_posted, tier=getattr(tier, "name", ""))
    return IntakeDecision(True, insp, reasons=[], content_hash=chash, bits=bits_report,
                          parent=parent_check)
