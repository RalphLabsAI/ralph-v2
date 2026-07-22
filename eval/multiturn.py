"""Multi-turn exposure-bias scoring on the gridworld env.

This is the experiment the CoT pile could not run. Because the env observation is Markov
(position + inventory + neighbours fully describe the state), each turn conditions on the
current observation — so there is NO re-prompt "restart" artifact. teacher_state and
self_state then differ ONLY by whose trajectory produced the state being scored:

  teacher_state(r, k): score the student's action from the state the TEACHER reached at
    step k (in-distribution for a distilled student — clean states).
  self_state(r, k):    roll the STUDENT k steps from the start on its OWN actions, then
    score its action from the drifted state it reached (out-of-distribution — this is
    where a fluent drifter fails and a robust compression holds).

Agreement per point = is the student's action in the deterministic optimal set at that
state (`oracle_optimal_actions`) — the "did it do what the teacher would do here" check,
tie-tolerant, no LLM judge in the crown path. The signature of exposure bias is:
self_state agreement DROPS with k for a drifter while teacher_state stays flat.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

from .env import grid as G


class Agent(Protocol):
    def act(self, state: "G.GridState") -> str: ...


class OracleAgent:
    """Perfect teacher: always an optimal action."""
    name = "oracle"

    def act(self, state):
        a = G.oracle_action(state)
        return a or "north"


class SimAgent:
    """Student stand-in with a fixed per-step DRIFT rate: acts optimally with prob
    (1-drift), else takes a random (often wrong) action. Higher drift = worse student.
    Deterministic given (seed, state) so scoring is reproducible."""

    def __init__(self, drift: float, seed: int = 0, name: str | None = None):
        self.drift, self.seed = drift, seed
        self.name = name or f"sim(drift={drift})"

    def act(self, state):
        r = random.Random(f"{self.seed}|{state.pos}|{sorted(state.keys)}|{state.steps}")
        opt = G.oracle_optimal_actions(state)
        if r.random() >= self.drift and opt:
            return sorted(opt)[0]
        return r.choice(G.ACTIONS)


class DistShiftAgent:
    """A realistic distilled student: competent ON the teacher's distribution, worse OFF
    it. SFT on teacher trajectories makes a model good at states LIKE the ones it trained
    on and shaky elsewhere — exactly the failure the self_state slice must catch. Acts
    optimally with prob (1-base_drift) in a FAMILIAR state (one the teacher visited) and
    only (1-ood_drift) in an unfamiliar state. A uniform-competence SimAgent is the
    control: it has no exposure bias and should show ~0 self_state gap."""

    def __init__(self, familiar: set, base_drift: float = 0.05, ood_drift: float = 0.6,
                 seed: int = 0, name: str | None = None):
        self.familiar = familiar
        self.base_drift, self.ood_drift, self.seed = base_drift, ood_drift, seed
        self.name = name or f"distshift(base={base_drift},ood={ood_drift})"

    def act(self, state):
        r = random.Random(f"{self.seed}|{state.grid}|{state.pos}|{sorted(state.keys)}|{state.steps}")
        drift = self.base_drift if (state.grid, state.pos, state.keys) in self.familiar else self.ood_drift
        opt = G.oracle_optimal_actions(state)
        if r.random() >= drift and opt:
            return sorted(opt)[0]
        return r.choice(G.ACTIONS)


def familiar_states(pile: "list[MTRollout]") -> set:
    """The states the teacher visited — keyed on (grid, pos, keys) so familiarity is
    PER-GRID (a state is in-distribution only if the teacher visited it in THIS grid, not
    a coincidentally-same coordinate in another). Keying on (pos, keys) alone pools across
    grids and makes the whole space look familiar."""
    fam = set()
    for r in pile:
        for s in r.states:
            fam.add((s.grid, s.pos, s.keys))
    return fam


@dataclass
class MTRollout:
    id: str
    seed: int
    difficulty: int
    task: str
    states: list       # GridState at each step (states[0] = start), the teacher's path
    actions: list      # teacher action taken from states[i]
    success: bool
    domain: str = "grid"

    def __len__(self):
        return len(self.actions)


def author_rollout(seed: int, difficulty: int, teacher: Agent, horizon: int = 30) -> MTRollout:
    """Teacher plays the env → a trajectory of (state, action). States are the native
    step boundaries; the teacher's actions define the on-policy-for-teacher path."""
    state, task = G.make_task(seed, difficulty)
    states, actions = [state], []
    for _ in range(horizon):
        a = teacher.act(state)
        actions.append(a)
        state, _, done = G.step(state, G.parse_action(a) if len(a) > 8 else a)
        states.append(state)
        if done:
            break
    return MTRollout(id=f"grid{seed}", seed=seed, difficulty=difficulty, task=task,
                     states=states, actions=actions, success=states[-1].solved)


def author_pile(n: int, difficulty: int = 2, teacher: Agent | None = None,
                horizon: int = 30, seed0: int = 1000) -> list[MTRollout]:
    teacher = teacher or OracleAgent()
    return [author_rollout(seed0 + i, difficulty, teacher, horizon) for i in range(n)]


@dataclass
class ExposureResult:
    by_k: dict = field(default_factory=dict)   # mode -> {k: mean_agreement}
    overall: dict = field(default_factory=dict)


def _agree(state, student: Agent) -> float:
    """1.0 if the student's action from `state` is optimal (in the teacher/oracle set)."""
    opt = G.oracle_optimal_actions(state)
    if not opt:
        return 1.0   # already at goal / no move improves — vacuous, skip upstream
    return 1.0 if student.act(state) in opt else 0.0


def score_exposure(rollouts: list[MTRollout], student: Agent,
                   k_buckets: list[int], self_state: bool = True) -> ExposureResult:
    """Agreement at each depth k, for teacher_state and (optionally) self_state.

    teacher_state@k: student acts from the teacher's state at step k.
    self_state@k:    student is rolled k steps on ITS OWN actions, then scored at the
                     state it reached — the exposure-bias measurement.
    """
    res = ExposureResult(by_k={"teacher_state": {}, "self_state": {}})
    for k in k_buckets:
        ts, ss = [], []
        for r in rollouts:
            if k < len(r.states) and not r.states[k].solved:
                ts.append(_agree(r.states[k], student))
            if self_state:
                s = r.states[0]
                ok = True
                for _ in range(k):
                    if s.solved:
                        ok = False
                        break
                    s, _, _ = G.step(s, student.act(s))
                if ok and not s.solved:
                    ss.append(_agree(s, student))
        if ts:
            res.by_k["teacher_state"][k] = round(sum(ts) / len(ts), 4)
        if self_state and ss:
            res.by_k["self_state"][k] = round(sum(ss) / len(ss), 4)
    for mode in ("teacher_state", "self_state"):
        vals = list(res.by_k[mode].values())
        res.overall[mode] = round(sum(vals) / len(vals), 4) if vals else None
    return res
