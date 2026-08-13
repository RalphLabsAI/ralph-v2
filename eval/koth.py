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


def kings_from_events(events) -> dict:
    """{tier: model_id} for every throne a round leaves occupied.

    `hold` COUNTS. A round where the incumbent survives emits `hold`, not `crown` — the crown did
    not change hands, so nothing is re-announced — and code that scans only for `crown`/`dethrone`
    concludes the tier is empty. That reading has already broken the emission audit once and it is
    the same mistake twice over, so the rule lives in one place: a tier has a king if the last event
    that named one said `crown`, `dethrone` or `hold`."""
    kings: dict = {}
    for e in events or []:
        tier, king = e.get("tier"), e.get("king")
        if not tier:
            continue
        if e.get("action") in ("crown", "dethrone", "hold") and king:
            kings[tier] = king
    return kings


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
    coldkey: str = ""        # operator identity — the unit anti-grind economics key on


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
    # per-axis AxisScore objects (retention/n/live) for the record + diagnosis. Optional.
    axes: list = field(default_factory=list)
    # frozen per-sample generations + effect triples, carried into the signed round record so an
    # auditor can recompute the score without re-running generation.
    steps: list = field(default_factory=list)
    effects: list = field(default_factory=list)
    role: str = "challenger"
    artifact_uri: str = ""
    manifest_root: str = ""
    code_bits: float = 0.0
    container_bits: float = 0.0

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
    # resolvable locator for the crowned bytes + the manifest root that pins the file set
    artifact_uri: str = ""
    manifest_root: str = ""
    code_bits: float = 0.0
    container_bits: float = 0.0


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


SOFTMIN_P = -3.0        # matches scoring.SOFTMIN_P; worst-domain pressure without annihilation


def _soft_min(rates: list[float], p: float = SOFTMIN_P) -> float:
    """Power-mean over per-axis rates — the same worst-domain aggregate the crown score uses,
    on plain rates so the bootstrap can recompute it per replicate."""
    if not rates:
        return 0.0
    eps = 1e-3
    acc = sum(max(r, eps) ** p for r in rates) / len(rates)
    return acc ** (1.0 / p)


def _shared_axes(a: "Scored", b: "Scored") -> list[str]:
    return [ax for ax in sorted(a.per_axis)
            if ax in b.per_axis and a.per_axis[ax]
            and len(a.per_axis[ax]) == len(b.per_axis[ax])]


def softmin_lcb_diff(a: "Scored", b: "Scored", z_reps: int = 2000, alpha: float = 0.05,
                     seed: int = 0, p: float = SOFTMIN_P) -> float:
    """Dethrone statistic: bootstrap lower bound on the difference of WORST-DOMAIN AGGREGATES.

    Replaces the old min-over-per-axis-LCBs, which was negatively biased to the point of being
    unusable. Measured on the production item count (n=64/axis, 3 axes): two models of IDENTICAL
    skill with independent error patterns scored -0.156, because each per-axis paired LCB is the
    5th percentile of a noisy difference and taking the MIN over axes selects the unluckiest one.
    A challenger therefore had to beat the king by ~+0.25 on EVERY axis to dethrone, and the case
    this subnet exists to reward — tied on English, far better on the axis a cloned model loses
    (multilingual) — scored -0.109 and HELD the throne. The mechanism could not pay for its own
    strategy.

    Here each replicate resamples the items ONCE per axis (paired: both models are scored on the
    same items), recomputes each model's soft-min aggregate on that resample, and differences
    them. The 5th percentile of that distribution is the bound.

    The anti-copy guarantee is preserved exactly and for the same reason as before: an exact copy
    produces identical per-item results, so in EVERY replicate both aggregates coincide, the
    difference is exactly 0 with zero variance, and the bound is 0 — never clearing the margin.
    Improving your WORST axis moves the soft-min, which is precisely the improvement we want to
    buy; improving an already-strong axis moves it barely at all.
    """
    shared = _shared_axes(a, b)
    if not shared:
        return bootstrap_lcb_diff(a.per_point, b.per_point, z_reps, alpha, seed)
    rng = random.Random(seed)
    vecs = [(a.per_axis[ax], b.per_axis[ax]) for ax in shared]
    diffs = []
    for _ in range(z_reps):
        ar, br = [], []
        for av, bv in vecs:
            n = len(av)
            idx = [rng.randrange(n) for _ in range(n)]
            ar.append(sum(av[i] for i in idx) / n)
            br.append(sum(bv[i] for i in idx) / n)
        diffs.append(_soft_min(ar, p) - _soft_min(br, p))
    diffs.sort()
    return diffs[int(alpha * z_reps)]


def axis_regression(a: "Scored", b: "Scored", z_reps: int = 1000, alpha: float = 0.05,
                    seed: int = 0) -> str | None:
    """Name of an axis where `a` is CONFIDENTLY worse than `b`, else None.

    The soft-min difference already punishes selling one capability to buy another (the
    aggregate is dominated by the weakest axis), but this keeps the old worst-axis guarantee
    explicit and cheap: a challenger that genuinely regresses on any shared axis cannot take the
    crown no matter how much it gained elsewhere. Confidence-based, so sampling noise on a tied
    axis does not block an honest challenger."""
    for ax in _shared_axes(a, b):
        av, bv = a.per_axis[ax], b.per_axis[ax]
        n = len(av)
        rng = random.Random(f"{seed}|{ax}")
        ds = []
        for _ in range(z_reps):
            idx = [rng.randrange(n) for _ in range(n)]
            ds.append(sum(av[i] - bv[i] for i in idx) / n)
        ds.sort()
        if ds[int((1 - alpha) * len(ds))] < 0.0:      # upper bound still below zero
            return ax
    return None


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

        # contested throne: dethrone only past the noise-floor margin, on the WORST-DOMAIN
        # AGGREGATE (paired on the same points as the re-scored king), plus an explicit
        # no-regression rule. The aggregate difference is what makes "improve your weakest
        # axis" the paying strategy; the regression check keeps the old guarantee that you
        # cannot sell one capability to buy another.
        lcb = softmin_lcb_diff(best, king_scored, seed=seed)
        regressed = axis_regression(best, king_scored, seed=seed) if lcb > self.margin else None
        if regressed:
            self.kings[tier].reign += 1
            event.update(action="hold", margin_lcb=round(lcb, 4),
                         best_challenger=best.sub.model_id, regressed_axis=regressed)
            return event
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
        """Emission per miner: each tier's weight goes to its king.

        NORMALIZED OVER EVERY TIER, NOT ONLY THE OCCUPIED ONES, so the result may sum to less than
        1 — see `unclaimed()`. This used to divide by the weight of tiers that had a king, which
        meant an empty tier handed its share to whoever had shown up elsewhere. With binary and
        sub2 empty, the ternary and sub4 kings split the whole emission, and sub4 — the tier a
        miner can enter with one `llama-quantize` invocation — collected half of everything.

        That is precisely backwards. The hard tiers are the ones worth paying for, and under the old
        rule the surest way to raise your income was for nobody to attempt them. An unclaimed tier
        now pays nobody, so the reward for entering an empty hard tier is the whole of its share
        rather than a slice of it."""
        total_w = sum(t.weight for t in self.tiers.values())
        if total_w <= 0:
            return {}
        out: dict[str, float] = {}
        for t, k in self.kings.items():
            if t in self.tiers:
                out[k.miner] = out.get(k.miner, 0.0) + self.tiers[t].weight / total_w
        return out

    def unclaimed(self) -> float:
        """The share of emission belonging to tiers with no king.

        Returned rather than silently redistributed, because what to do with it is a protocol
        decision — burn it, or hold it — and not one the tournament should make by accident. The
        caller must handle it explicitly; `weights()` summing to less than 1 is the signal."""
        total_w = sum(t.weight for t in self.tiers.values())
        if total_w <= 0:
            return 0.0
        held = sum(self.tiers[t].weight for t in self.kings if t in self.tiers)
        return max(0.0, (total_w - held) / total_w)
