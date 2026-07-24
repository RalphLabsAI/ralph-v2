"""GLM-COVER crown round — verifiable-outcome retention of GLM across capability axes.

This replaces the gridworld as the crown substrate. Each axis is a capability GLM has
(math, code, logic, instruction-following, extraction, long-context) with a DETERMINISTIC
checker — no LLM judge, no logits/KL (the SN97 trap), only checker pass/fail, so a CPU
auditor re-runs it bit-for-bit. Per round:

  1. each axis generates FRESH items, seed = derive_seed(commit_root, round_nonce, axis)
     (items do not exist until checkpoints lock -> nothing to memorize offline);
  2. GLM answers them -> the teacher-passed subset (const's design: score retention on
     what the teacher can actually do);
  3. every student AND the reigning king answer the SAME items -> per-item pass/fail;
  4. per-axis normalized retention (student vs base on the teacher-passed subset,
     scoring.axis_retention) -> worst-DOMAIN soft-min (can't sell one capability to buy
     another) -> verdict() crown gates -> KOTH dethrone-on-margin.

The crown is thus "matches GLM's verifiable behavior across a broad fresh distribution",
not "reproduces GLM's token distribution" — style-matching earns nothing because a wrong
answer fails the checker regardless of how GLM-like it reads. Binding to *GLM specifically*
(vs any capable model) is a measured property: run non-GLM siblings through this same
pipeline (a control panel) and confirm they score at base — see docs/ship-status.md.

Reuses scoring/koth/verdict/seeds unchanged; validator + chain wrap it like env_round.
"""
from __future__ import annotations

from dataclasses import dataclass

from .core import ModelRunner
from .gates import degeneracy_flags
from .koth import Scored, Submission, Tier, Tournament
from .round_engine import RoundResult
from .scoring import AxisScore, axis_retention, soft_min


@dataclass
class AxisSpec:
    """A capability axis: fresh items + a deterministic checker. `axis` is any object with
    generate(seed, n, difficulty) -> list[Item] and check(item, output) -> bool (the
    axes/ modules already implement this)."""
    axis: object
    name: str
    weight: float = 1.0
    difficulty: int = 1
    # Optional per-axis item count. Liveness needs >= MIN_AXIS_N teacher-passed ITEMS, not a
    # minimum pass RATE — so an axis the teacher only clears at (say) 31% just needs more
    # items sampled, rather than being made easier. Keeping a hard axis and sampling more
    # preserves its discriminating power; dumbing it down would not.
    items: int | None = None
    # role: "crown" axes SET the crown score + the dethrone margin; "floor" axes only gate
    # (liveness / degeneracy / no-negative) but do NOT set the score. This is const's SN97
    # fix: the crown is decided by real∩verifiable axes (extractive QA over fresh real docs,
    # real-repo tests) that carry information a miner can't pre-distill, while the SYNTHETIC
    # generators — which a specialist can ace offline without retaining GLM — are demoted to
    # a liveness/calibration floor and can no longer buy the crown. Default "crown" keeps the
    # legacy all-axes-count behavior for callers that don't split roles.
    role: str = "crown"


@dataclass
class RoundItems:
    """Items minted for one round + GLM's pass mask per axis (authored once, amortized)."""
    by_axis: dict           # axis_name -> list[Item]
    teacher_pass: dict      # axis_name -> list[bool] (GLM's own pass on each item)


def mint_and_author(specs: list[AxisSpec], glm: ModelRunner, seed_fn,
                    items_per_axis: int, max_new_tokens: int = 512) -> RoundItems:
    """Generate fresh items per axis (nonce-seeded) and have GLM author the reference by
    answering them. `seed_fn(axis_name) -> int` binds each axis to the commit nonce."""
    by_axis, teacher_pass = {}, {}
    for s in specs:
        items = s.axis.generate(seed_fn(s.name), s.items or items_per_axis, s.difficulty)
        outs = glm.generate([it.prompt for it in items], max_new_tokens)
        by_axis[s.name] = items
        teacher_pass[s.name] = [s.axis.check(it, o) for it, o in zip(items, outs)]
    return RoundItems(by_axis, teacher_pass)


def _score_student(specs: list[AxisSpec], items: RoundItems, base_pass: dict,
                   runner: ModelRunner, max_new_tokens: int, crown_names: set[str]
                   ) -> tuple[float, float, list[float], dict, list[AxisScore], list[str]]:
    """Per-axis retention (conditioned on GLM-passed items) + per-axis paired vectors +
    the raw outputs (for the degeneracy gate). The crown score and the dethrone vectors are
    built from CROWN axes only; floor axes are still scored (returned in `axes` for the
    gate/record) but do not set the score."""
    axes: list[AxisScore] = []
    per_point: list[float] = []
    per_axis: dict[str, list[float]] = {}
    all_outs: list[str] = []
    spec_by_name = {s.name: s for s in specs}
    for name, its in items.by_axis.items():
        s = spec_by_name[name]
        outs = runner.generate([it.prompt for it in its], max_new_tokens)
        all_outs += outs
        stud = [s.axis.check(it, o) for it, o in zip(its, outs)]
        tp, bp = items.teacher_pass[name], base_pass[name]
        axes.append(axis_retention(name, s.weight, tp, stud, bp))
        # paired dethrone vector uses only the GLM-passed items (the retention subset), kept
        # PER-AXIS so koth dethrones on the worst axis. CROWN axes only, so a copy of the
        # king on the real axes ties even if it drifts on a floor axis.
        if name in crown_names:
            vec = [1.0 if stud[i] else 0.0 for i in range(len(its)) if tp[i]]
            per_axis[name] = vec
            per_point += vec
    crown_axes = [a for a in axes if a.axis in crown_names]
    return (soft_min(crown_axes, use_lower_bound=False), soft_min(crown_axes, use_lower_bound=True),
            per_point, per_axis, axes, all_outs)


def axis_round(round_no: int, commit_seed: int, specs: list[AxisSpec],
               glm: ModelRunner, base: ModelRunner, tiers: list[Tier], tournament: Tournament,
               submissions: list[tuple[Submission, ModelRunner]],
               registry: dict[str, ModelRunner] | None = None,
               items_per_axis: int = 64, max_new_tokens: int = 512,
               items: RoundItems | None = None, base_pass: dict | None = None,
               overfit_check=None) -> RoundResult:
    """`overfit_check(submission, runner) -> (ok, info)`: optional crown PRECONDITION — the
    stale-vs-fresh diff-in-diff genre-overfit gate (overfit_gate.diff_in_diff_over_corpus,
    closed over the round's timestamped corpus + pinned teacher/base). A flagged student is
    NOT crownable regardless of its capability retention, so acing the (pre-distillable)
    covered distribution can't buy the crown if the edge collapses on genuinely-fresh
    same-genre documents. Skipped when None."""
    from .seeds import derive_seed
    tournament.round = round_no
    registry = registry if registry is not None else {}
    for sub, runner in submissions:
        registry[sub.model_id] = runner

    # 1-2. fresh items + GLM reference (once per round, amortized across all miners + king).
    # Each axis seeded independently off the round seed so `derive_surprise_axes` can later
    # hide which axes count until after commit.
    if items is None:
        seed_fn = lambda name: derive_seed(str(commit_seed), name, "items")
        items = mint_and_author(specs, glm, seed_fn, items_per_axis, max_new_tokens)
    if base_pass is None:
        base_pass = {}
        for s in specs:
            outs = base.generate([it.prompt for it in items.by_axis[s.name]], max_new_tokens)
            base_pass[s.name] = [s.axis.check(it, o) for it, o in zip(items.by_axis[s.name], outs)]

    crown_names = {s.name for s in specs if s.role == "crown"}

    def gate(sub: Submission, axes: list[AxisScore], outputs: list[str]) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if sub.params > tournament.tiers[sub.tier].max_params:
            reasons.append("over tier param budget")
        # looping / runaway free-text is a DoS on the 512-token axis path — disqualify.
        deg_ok, dflags = degeneracy_flags(outputs)
        if not deg_ok or dflags:
            reasons.append(f"degenerate output: {', '.join(dflags)}")
        # negative retention on ANY axis (crown OR floor) is a genuine capability sacrifice
        # vs the base — disqualify. A floor axis is still a real capability, so acing the
        # real axes while regressing below base on a floor axis is not crownable.
        neg = [a.axis for a in axes if a.negative]
        if neg:
            reasons.append(f"negative retention on: {', '.join(neg)}")
        # every CROWN (real∩verifiable) axis must be LIVE — a student whose real-capability
        # retention couldn't be measured this round (thin fresh corpus) is not crownable.
        # Floor (synthetic) axes are always-available calibration; they don't gate liveness,
        # so acing them cannot substitute for real signal.
        live_crown = sum(1 for a in axes if a.axis in crown_names and a.live)
        if live_crown < len(crown_names):
            reasons.append(f"only {live_crown}/{len(crown_names)} crown axes live — "
                           "cannot measure real-capability retention this round")
        return (not reasons), reasons

    # 3-4. score every submission
    scored: dict[str, Scored] = {}
    by_tier: dict[str, list[Scored]] = {t.name: [] for t in tiers}
    for sub, runner in submissions:
        ret, ret_lb, per_pt, per_ax, axes, outs = _score_student(specs, items, base_pass, runner, max_new_tokens, crown_names)
        ok, reasons = gate(sub, axes, outs)
        if overfit_check is not None and ok:   # only run the corpus gate on a crownable student
            of_ok, of_info = overfit_check(sub, runner)
            if not of_ok:
                ok = False
                reasons = reasons + [f"genre-overfit ({of_info.get('verdict')}, diff_lb {of_info.get('diff_lb')})"]
        s = Scored(sub=sub, retention=ret, retention_lb=ret_lb, per_point=per_pt,
                   gates_ok=ok, reasons=reasons, per_axis=per_ax, axes=axes)
        scored[sub.model_id] = s
        by_tier.setdefault(sub.tier, []).append(s)

    # 5. tournament per tier — re-score the king on the SAME items; FAIL CLOSED if it
    # can't be re-scored (never fall through to an open-throne no-margin crown).
    events = []
    for t in tiers:
        king = tournament.kings.get(t.name)
        if king is not None and registry.get(king.model_id) is None:
            events.append({"tier": t.name, "round": round_no, "action": "hold",
                           "king": king.model_id, "reason": "king unavailable for re-score"})
            tournament.kings[t.name].reign += 1
            continue
        king_scored = None
        if king is not None:
            ret, ret_lb, per_pt, per_ax, _, _ = _score_student(specs, items, base_pass, registry[king.model_id], max_new_tokens, crown_names)
            king_scored = Scored(sub=Submission(king.miner, t.name, king.model_id, 0, 0.0),
                                 retention=ret, retention_lb=ret_lb, per_point=per_pt,
                                 gates_ok=True, per_axis=per_ax)
        events.append(tournament.consider(t.name, by_tier.get(t.name, []), king_scored, seed=commit_seed))

    return RoundResult(round_no, sum(len(v) for v in items.by_axis.values()), scored, events, tournament.weights())
