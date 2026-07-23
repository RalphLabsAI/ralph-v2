"""King-of-the-hill tournament — per-tier kings, dethrone on margin.

The rigorous core: a challenger dethrones the tier king only if it is STRICTLY better
past a noise-floor margin, measured on the SAME fresh points as the king that round
(the reigning king is re-scored every round, like SN3). The test is a bootstrap lower
bound on the paired per-point agreement difference — so a copy of the king ties, a tie
clears no margin, and micro-shaving is below the noise. No detector to evade; copying
is simply unprofitable.

Compute is metered and reported; in v0 the crown metric is retention (the statistically
clean quantity). Ranking by capability-per-compute is a documented v1 extension
(rank_metric="per_compute") — kept out of the dethrone test for now because folding a
scalar compute term into a paired bootstrap muddies the noise-floor guarantee.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Tier:
    name: str
    max_params: int          # inclusive upper bound on student size
    weight: float            # share of total emission for this tier


@dataclass
class Submission:
    miner: str               # hotkey
    tier: str
    model_id: str            # content hash of the checkpoint
    params: int
    compute_h100h: float     # declared, normalized; reconciled elsewhere


# minimum aggregate lower-bound retention to be crownable AT ALL (open or contested).
# soft_min floors a failed axis to eps=1e-3 to avoid a zero-division in the power mean,
# so `retention_lb > 0` was a no-op (always true). This real floor gives it teeth: a
# student whose worst live axis sits at/below base (aggregate LB ~eps) is not crownable,
# so it cannot grab an open throne at a near-zero score.
MIN_CROWN_LB = 0.02


@dataclass
class Scored:
    sub: Submission
    retention: float               # aggregate (soft-min over state-coverage modes)
    retention_lb: float
    per_point: list[float]         # aligned to the round's fixed points (for pairing)
    gates_ok: bool
    reasons: list[str] = field(default_factory=list)
    # per-axis paired vectors (axis_name -> agreements on that axis's points). The
    # dethrone test uses the WORST axis, not the pooled `per_point`, so worst-domain
    # governs dethroning as well as crown selection. Empty -> legacy pooled fallback.
    per_axis: dict = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.gates_ok and self.retention_lb > MIN_CROWN_LB

    def per_compute(self) -> float:
        return self.retention / max(self.compute, 1e-6)

    @property
    def compute(self) -> float:
        return self.sub.compute_h100h


@dataclass
class King:
    miner: str
    model_id: str
    retention: float
    crowned_round: int
    reign: int = 0


def bootstrap_lcb_diff(a: list[float], b: list[float], z_reps: int = 2000,
                       alpha: float = 0.05, seed: int = 0) -> float:
    """Lower bound on mean(a_i - b_i) over paired points. a beats b iff this > margin."""
    n = len(a)
    if n == 0 or len(b) != n:
        return -1.0
    diffs = [a[i] - b[i] for i in range(n)]
    rng = random.Random(seed)
    means = []
    for _ in range(z_reps):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    return means[int(alpha * z_reps)]


def worst_axis_lcb_diff(a: "Scored", b: "Scored", z_reps: int = 2000,
                        alpha: float = 0.05, seed: int = 0) -> float:
    """Dethrone margin = the WORST per-axis paired lower bound.

    Crown SELECTION already uses worst-domain retention (soft-min); the dethrone test
    must too, or a challenger that wins a large axis while LOSING a small one clears the
    pooled-mean margin and dethrones a king it is worse than on its weakest domain — the
    exact "sell one capability to buy another" worst-domain aggregation is meant to
    forbid. We take the min over shared axes of the paired bootstrap lower bound. An exact
    copy ties every axis (diff 0) -> min 0 -> never clears the margin, so the anti-copy
    property is preserved. Falls back to the pooled vector only when per-axis vectors are
    absent (legacy callers)."""
    shared = [ax for ax in a.per_axis
              if ax in b.per_axis and a.per_axis[ax]
              and len(a.per_axis[ax]) == len(b.per_axis[ax])]
    if not shared:
        return bootstrap_lcb_diff(a.per_point, b.per_point, z_reps, alpha, seed)
    return min(bootstrap_lcb_diff(a.per_axis[ax], b.per_axis[ax], z_reps, alpha, seed)
               for ax in shared)


class Tournament:
    """Holds the reigning king per tier and applies dethrone-on-margin each round."""

    def __init__(self, tiers: list[Tier], margin: float = 0.05):
        # 0.05 production floor: at the small end of eval sizes the paired bootstrap
        # can't resolve a dethrone below ~this margin (adversarial-review Q4). Raise
        # toward 0.07 for tiny piles, lower only with n >= ~1500 points/round.
        self.tiers = {t.name: t for t in tiers}
        self.margin = margin
        self.kings: dict[str, King] = {}
        self.round = 0

    def consider(self, tier: str, scored: list[Scored], king_scored: Scored | None,
                 seed: int = 0) -> dict:
        """Resolve one tier for the current round.

        `scored` are challenger submissions; `king_scored` is the reigning king
        re-scored on THIS round's points (None if no king yet). Returns an event dict.
        """
        valid = [s for s in scored if s.valid]
        event = {"tier": tier, "round": self.round, "action": "none",
                 "king": self.kings.get(tier).model_id if tier in self.kings else None}

        if not valid:
            return event

        # best challenger by retention (clean crown metric)
        best = max(valid, key=lambda s: s.retention)

        if tier not in self.kings or king_scored is None:
            # open throne: crown the best valid challenger outright
            self.kings[tier] = King(best.sub.miner, best.sub.model_id, best.retention, self.round)
            event.update(action="crown", king=best.sub.model_id, miner=best.sub.miner,
                         retention=round(best.retention, 4))
            return event

        # contested throne: dethrone only past the noise-floor margin, on the WORST axis
        # (paired on the same points as the re-scored king). Worst-axis, not pooled mean,
        # so a challenger strong on one axis but weak on another cannot buy the crown.
        lcb = worst_axis_lcb_diff(best, king_scored, seed=seed)
        if lcb > self.margin and best.sub.model_id != king_scored.sub.model_id:
            self.kings[tier] = King(best.sub.miner, best.sub.model_id, best.retention, self.round)
            event.update(action="dethrone", king=best.sub.model_id, miner=best.sub.miner,
                         retention=round(best.retention, 4), margin_lcb=round(lcb, 4),
                         beaten=king_scored.sub.model_id)
        else:
            self.kings[tier].reign += 1
            event.update(action="hold", margin_lcb=round(lcb, 4),
                         best_challenger=best.sub.model_id)
        return event

    def weights(self) -> dict[str, float]:
        """Emission per miner: each tier's weight goes to its king. Normalized to sum 1
        over tiers that have a king."""
        live = {t: k for t, k in self.kings.items() if t in self.tiers}
        total_w = sum(self.tiers[t].weight for t in live)
        if total_w <= 0:
            return {}
        out: dict[str, float] = {}
        for t, k in live.items():
            out[k.miner] = out.get(k.miner, 0.0) + self.tiers[t].weight / total_w
        return out
