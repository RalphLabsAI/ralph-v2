"""Round engine — one full scoring round of the subnet.

Ties the pieces together:

  1. sample fresh points from the experience pile (seed from the round's commit)
  2. compute GLM's reference behavior per point ONCE (amortized across all miners)
  3. score every submission AND the reigning king on the SAME points (exact pairing)
  4. aggregate per-point agreement into normalized retention (scoring.py)
  5. run the KOTH tournament (koth.py): crown / dethrone-on-margin
  6. emit per-miner weights

Model access is via the ModelRunner interface, so this runs on CPU with sim models and
against a real pinned GLM on a GPU box unchanged. Chain I/O (read commitments, set
weights, publish the signed round record) is deliberately NOT here — that layer wraps
this and reuses Ralph's existing validator.
"""
from __future__ import annotations

from dataclasses import dataclass

from .core import ModelRunner
from .koth import Scored, Submission, Tier, Tournament
from .scoring import AxisScore, axis_retention, soft_min
from .trajectory import (EvalPoint, Rollout, StepJudge, prepare_refs, sample_points,
                         score_on_points)


@dataclass
class RoundResult:
    round: int
    n_points: int
    scored: dict[str, Scored]        # by model_id
    events: list[dict]               # one per tier
    weights: dict[str, float]        # by miner


def _retention_from_points(agrees: list[float], teacher_agrees: list[float],
                           base_agrees: list[float], modes: list[str]) -> tuple[float, float, list[float]]:
    """Aggregate per-point agreement into normalized retention.

    Two axes = the two state-coverage modes (teacher_state, self_state), each
    normalized (student-base)/(teacher-base) and combined by worst-domain soft-min so a
    student strong from good states but drifting from its own is pulled to its weaker
    mode. Per-point agreements (student) are returned aligned for the paired dethrone
    test.
    """
    axes: list[AxisScore] = []
    for mode in ("teacher_state", "self_state"):
        idx = [i for i, m in enumerate(modes) if m == mode]
        if not idx:
            continue
        # teacher "passes" every point by construction (it defines the reference); we
        # treat agreement as a rate and reuse the retention normalizer against the base.
        t = [True] * len(idx)
        s = [agrees[i] >= 0.5 for i in idx]
        b = [base_agrees[i] >= 0.5 for i in idx]
        axes.append(axis_retention(mode, 0.5, t, s, b))
    if not axes:
        return 0.0, 0.0, agrees
    return soft_min(axes, use_lower_bound=False), soft_min(axes, use_lower_bound=True), agrees


def run_round(round_no: int, commit_seed: int, experience: list[Rollout],
              glm: ModelRunner, base: ModelRunner, judge: StepJudge,
              tiers: list[Tier], tournament: Tournament,
              submissions: list[tuple[Submission, ModelRunner]],
              registry: dict[str, ModelRunner] | None = None,
              n_points: int = 120, self_frac: float = 0.25,
              max_new_tokens: int = 256) -> RoundResult:
    tournament.round = round_no
    # the validator PERSISTS every king's checkpoint and re-scores it each round,
    # whether or not the miner resubmits. `registry` is that persistent store; this
    # round's submissions are added so any future king stays re-scorable.
    registry = registry if registry is not None else {}
    for sub, runner in submissions:
        registry[sub.model_id] = runner

    # 1-2. fixed points + cached GLM refs (once per round)
    points = sample_points(experience, n_points, self_frac, seed=commit_seed)
    refs = prepare_refs(points, experience, glm, judge, max_new_tokens)
    modes = [p.mode for p in points]

    # base is scored once — it is the shared denominator for retention
    base_pts = [sp.agreement for sp in score_on_points(points, refs, experience, glm, base, judge, max_new_tokens)]

    def score_one(runner: ModelRunner) -> tuple[float, float, list[float]]:
        ag = [sp.agreement for sp in score_on_points(points, refs, experience, glm, runner, judge, max_new_tokens)]
        return _retention_from_points(ag, [1.0] * len(ag), base_pts, modes)

    # 3-4. score every submission
    scored: dict[str, Scored] = {}
    by_tier: dict[str, list[Scored]] = {t.name: [] for t in tiers}
    for sub, runner in submissions:
        ret, ret_lb, per_pt = score_one(runner)
        s = Scored(sub=sub, retention=ret, retention_lb=ret_lb, per_point=per_pt,
                   gates_ok=(sub.params <= tournament.tiers[sub.tier].max_params),
                   reasons=[] if sub.params <= tournament.tiers[sub.tier].max_params else ["over tier param budget"])
        scored[sub.model_id] = s
        by_tier.setdefault(sub.tier, []).append(s)

    # 5. tournament per tier — re-score the reigning king on the same points
    events = []
    for t in tiers:
        king = tournament.kings.get(t.name)
        king_scored = None
        if king is not None:
            king_runner = registry.get(king.model_id)
            if king_runner is not None:
                ret, ret_lb, per_pt = score_one(king_runner)
                king_scored = Scored(
                    sub=Submission(king.miner, t.name, king.model_id, 0, 0.0),
                    retention=ret, retention_lb=ret_lb, per_point=per_pt, gates_ok=True)
        events.append(tournament.consider(t.name, by_tier.get(t.name, []), king_scored, seed=commit_seed))

    # 6. weights
    return RoundResult(round_no, len(points), scored, events, tournament.weights())
