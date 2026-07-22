"""Production scoring round on the multi-turn ENV substrate.

This is the round the live protocol runs — on the substrate that was actually validated
(eval/experiments/multiturn_exposure.py + the real-Qwen run), not the CoT text path
(round_engine.py) which restarts on re-prompt and can't score self_state. Here each point
is an (env, depth, mode); the deterministic env oracle is the reference (no LLM judge in
the crown path); teacher_state and self_state are the two axes, aggregated worst-first so
a fluent drifter (high teacher_state, low self_state) is pulled down to its weaker mode.

Same machinery as round_engine but on env agents:
  fresh points (commit-seed) -> score every submission + the king on the SAME points ->
  normalized retention per axis (vs the pinned base agent) -> crown gates (verdict:
  negative-axis interval + min-live-axis) -> KOTH dethrone-on-margin -> weights.

Model access is via the Agent interface (multiturn.ModelAgent for real models,
Sim/DistShift/Oracle for CPU tests), so this runs on CPU with sims and against real GLM
unchanged. Chain I/O wraps this exactly like validator_loop wraps round_engine.
"""
from __future__ import annotations

from .koth import Scored, Submission, Tier, Tournament
from .multiturn import Agent, MTRollout, EnvPoint, sample_env_points, score_points
from .round_engine import RoundResult
from .scoring import AxisScore, axis_retention, soft_min, verdict


def _axes_from_points(points: list[EnvPoint], agree: list[float],
                      base_agree: list[float]) -> list[AxisScore]:
    """One axis per mode (teacher_state / self_state), normalized (student-base)/(teacher-
    base). The env oracle is the reference so the 'teacher' passes every point by
    construction; agreement is treated as a pass-rate against the base agent."""
    axes: list[AxisScore] = []
    for mode in ("teacher_state", "self_state"):
        idx = [i for i, p in enumerate(points) if p.mode == mode]
        if not idx:
            continue
        t = [True] * len(idx)
        s = [agree[i] >= 0.5 for i in idx]
        b = [base_agree[i] >= 0.5 for i in idx]
        axes.append(axis_retention(mode, 1.0, t, s, b))
    return axes


def env_round(round_no: int, commit_seed: int, pile: list[MTRollout],
              base_agent: Agent, tiers: list[Tier], tournament: Tournament,
              submissions: list[tuple[Submission, Agent]],
              registry: dict[str, Agent] | None = None,
              n_points: int = 300, self_frac: float = 0.4,
              teacher: Agent | None = None) -> RoundResult:
    """`teacher` (production = the pinned GLM agent that authored the pile) is the
    reference: agreement = 'the student took GLM's action here' — const's design, checked
    deterministically on discrete env actions. teacher=None scores against the env oracle
    (validation)."""
    tournament.round = round_no
    registry = registry if registry is not None else {}
    for sub, agent in submissions:
        registry[sub.model_id] = agent

    points = sample_env_points(pile, n_points, self_frac, seed=commit_seed)
    base_agree = score_points(pile, points, base_agent, reference=teacher)

    def score_one(agent: Agent):
        ag = score_points(pile, points, agent, reference=teacher)
        axes = _axes_from_points(points, ag, base_agree)
        return soft_min(axes, use_lower_bound=False), soft_min(axes, use_lower_bound=True), ag, axes

    def gate(sub: Submission, axes: list[AxisScore]) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if sub.params > tournament.tiers[sub.tier].max_params:
            reasons.append("over tier param budget")
        # degeneracy on the env substrate = the agent that never advances (its action
        # from a state never reduces oracle distance) is caught by low retention, not a
        # separate output gate; a real-model ModelAgent's raw text is degeneracy-checked
        # upstream at generate time. Here the crown gates are verdict()'s.
        reasons += verdict(axes, passk_ok=True).reasons
        return (not reasons), reasons

    scored: dict[str, Scored] = {}
    by_tier: dict[str, list[Scored]] = {t.name: [] for t in tiers}
    for sub, agent in submissions:
        ret, ret_lb, per_pt, axes = score_one(agent)
        ok, reasons = gate(sub, axes)
        s = Scored(sub=sub, retention=ret, retention_lb=ret_lb, per_point=per_pt,
                   gates_ok=ok, reasons=reasons)
        scored[sub.model_id] = s
        by_tier.setdefault(sub.tier, []).append(s)

    events = []
    for t in tiers:
        king = tournament.kings.get(t.name)
        if king is not None and registry.get(king.model_id) is None:
            # FAIL CLOSED: a reigning king exists but its agent can't be re-scored (a
            # validator restart / registry eviction). Do NOT fall through to the
            # open-throne branch, which would crown the best challenger with no margin
            # test and could hand the crown to a model strictly WORSE than the king. Hold
            # the king this round. (Durable fix: reconstruct the king agent from its
            # on-chain checkpoint hash — TODO, needs the checkpoint store.)
            events.append({"tier": t.name, "round": round_no, "action": "hold",
                           "king": king.model_id, "reason": "king unavailable for re-score"})
            tournament.kings[t.name].reign += 1
            continue
        king_scored = None
        if king is not None:
            ret, ret_lb, per_pt, _ = score_one(registry[king.model_id])
            king_scored = Scored(sub=Submission(king.miner, t.name, king.model_id, 0, 0.0),
                                 retention=ret, retention_lb=ret_lb, per_point=per_pt, gates_ok=True)
        events.append(tournament.consider(t.name, by_tier.get(t.name, []), king_scored, seed=commit_seed))

    return RoundResult(round_no, len(points), scored, events, tournament.weights())
