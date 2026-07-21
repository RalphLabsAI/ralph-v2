"""Anti-grind economics + compute-aware ranking.

The prior subnet's single most durable defense was economic, not heuristic: one eval
per registration. Detection-based Sybil/spam defenses were all evaded or weaponized.
So the rules here are deliberately economic:

  * ONE FREE EVAL per registration (hotkey). Extra submissions cost a bond.
  * PER-COLDKEY round cap — bounds how much of a round's eval budget one operator can
    consume, without an identity-ban (bans hit legit multi-entry and were reverted).
  * RESUBMISSION BOND, REFUNDED when the submission improves the miner's OWN best —
    the fee taxes noise/best-of-N grinding, not honest iteration.
  * Compute is metered (declared, reconciled elsewhere) and only enters ranking as a
    tie-break in v0; capability-per-compute is the documented v1 ranking mode.

Pure logic — no chain. The validator wraps it: the ledger is rebuilt from on-chain
commitments + the bond extrinsic each epoch.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SubmitDecision:
    ok: bool
    bond_required: float = 0.0
    reason: str = ""


@dataclass
class RegistrationLedger:
    """Per-epoch submission accounting. Rebuildable from chain state (idempotent)."""

    per_coldkey_round_cap: int = 2
    base_bond: float = 0.0                # bond for a 2nd+ submission by the same hotkey
    # state
    hotkey_submissions: dict = field(default_factory=dict)      # hotkey -> count this epoch
    coldkey_submissions: dict = field(default_factory=dict)     # coldkey -> count this epoch
    best_score: dict = field(default_factory=dict)             # hotkey -> best retention so far
    bonds_held: dict = field(default_factory=dict)             # hotkey -> bond posted, pending refund

    def can_submit(self, hotkey: str, coldkey: str, bond_posted: float = 0.0) -> SubmitDecision:
        if self.coldkey_submissions.get(coldkey, 0) >= self.per_coldkey_round_cap:
            return SubmitDecision(False, reason=f"coldkey round cap {self.per_coldkey_round_cap} reached")
        n = self.hotkey_submissions.get(hotkey, 0)
        if n == 0:
            return SubmitDecision(True, bond_required=0.0, reason="free eval")
        # resubmission: a bond is required (refunded on self-improvement)
        need = self.base_bond * n
        if bond_posted + 1e-9 < need:
            return SubmitDecision(False, bond_required=need,
                                  reason=f"resubmission #{n+1} requires bond {need}")
        return SubmitDecision(True, bond_required=need, reason="bonded resubmission")

    def record(self, hotkey: str, coldkey: str, bond_posted: float = 0.0) -> None:
        self.hotkey_submissions[hotkey] = self.hotkey_submissions.get(hotkey, 0) + 1
        self.coldkey_submissions[coldkey] = self.coldkey_submissions.get(coldkey, 0) + 1
        if bond_posted > 0:
            self.bonds_held[hotkey] = self.bonds_held.get(hotkey, 0.0) + bond_posted

    def settle(self, hotkey: str, new_score: float) -> float:
        """Called after scoring. Refund the held bond iff the submission improved the
        miner's own best; otherwise the bond is forfeit (taxes noise, not work).
        Returns the refunded amount."""
        prev = self.best_score.get(hotkey, float("-inf"))
        improved = new_score > prev + 1e-9
        if improved:
            self.best_score[hotkey] = new_score
        held = self.bonds_held.pop(hotkey, 0.0)
        return held if improved else 0.0   # forfeit if not improved


def capability_per_compute(retention: float, compute_h100h: float, floor: float = 1e-6) -> float:
    """v1 ranking option: retention per normalized H100-hour. Kept out of the v0
    dethrone test (the paired bootstrap is on retention); available as a tie-break or
    an efficiency-tier ranking."""
    return retention / max(compute_h100h, floor)


def rank(subs: list, *, mode: str = "retention") -> list:
    """Order scored submissions. `mode`:
      - "retention": clean v0 crown metric (paired-LCB dethrone lives in koth).
      - "per_compute": efficiency ranking (v1), retention / compute.
    `subs` items need `.retention` and `.compute` (koth.Scored satisfies both).
    """
    if mode == "per_compute":
        return sorted(subs, key=lambda s: -capability_per_compute(s.retention, s.compute))
    return sorted(subs, key=lambda s: -s.retention)
