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


class ModelAgent:
    """Wraps an HFRunner as a gridworld agent. The observation is Markov, so one turn =
    act on the current observation. Used for real (teacher/student) models on a GPU."""

    def __init__(self, runner, name: str | None = None):
        self.runner = runner
        self.name = name or getattr(runner, "name", "model")

    def _prompt(self, state) -> list:
        return [{"role": "user",
                 "content": (f"You are in a gridworld. Reach 'G'. Doors 'D<c>' need the "
                             f"matching key 'K<c>' (stand on it and 'take'). Actions: "
                             f"{', '.join(G.ACTIONS)}. Reply with ONLY the next action.\n"
                             f"{G.observe(state)}")}]

    def act(self, state) -> str:
        out = self.runner.generate_chat([self._prompt(state)], max_new_tokens=8)[0]
        return G.parse_action(out) or "north"

    def act_batch(self, states) -> list:
        outs = self.runner.generate_chat([self._prompt(s) for s in states], max_new_tokens=8)
        return [G.parse_action(o) or "north" for o in outs]


def sft_examples_from_pile(pile: "list[MTRollout]", max_k: int | None = None) -> list:
    """(observation -> optimal action) SFT pairs for distilling a grid agent. max_k caps
    the horizon (the exposure-bias drifter trains only on early states)."""
    from train.distill import SFTExample
    ex = []
    for r in pile:
        hi = len(r.actions) if max_k is None else min(len(r.actions), max_k)
        for i in range(hi):
            s = r.states[i]
            opt = G.oracle_optimal_actions(s)
            if not opt:
                continue
            prompt = ModelAgent(None)._prompt(s)[0]["content"]
            ex.append(SFTExample(prompt=prompt, target=sorted(opt)[0], domain="grid", k=i))
    return ex


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


def _acts(student: Agent, states: list) -> list:
    """Batch the student's actions over many states (uses act_batch if available)."""
    if not states:
        return []
    if hasattr(student, "act_batch"):
        return student.act_batch(states)
    return [student.act(s) for s in states]


def _agree_batch(states: list, student: Agent) -> list[float]:
    scored = [(i, s) for i, s in enumerate(states) if G.oracle_optimal_actions(s)]
    acts = _acts(student, [s for _, s in scored])
    out = [1.0] * len(states)   # vacuous (at goal / no improving move) counts as agree
    for (i, s), a in zip(scored, acts):
        out[i] = 1.0 if a in G.oracle_optimal_actions(s) else 0.0
    return out


def score_exposure(rollouts: list[MTRollout], student: Agent,
                   k_buckets: list[int], self_state: bool = True) -> ExposureResult:
    """Agreement at each depth k, for teacher_state and (optionally) self_state.

    teacher_state@k: student acts from the teacher's state at step k (batched across
                     rollouts). In-distribution states — measures raw capability.
    self_state@k:    all rollouts are rolled forward IN LOCKSTEP on the student's own
                     actions (one batched act() per step across every rollout), and the
                     student is scored at the drifted state it reached — the exposure-bias
                     measurement. Lockstep => ~max(k_buckets) batched calls, not k*n.
    """
    res = ExposureResult(by_k={"teacher_state": {}, "self_state": {}})
    kset = set(k_buckets)

    # teacher_state: batch every rollout's step-k state
    for k in k_buckets:
        states = [r.states[k] for r in rollouts if k < len(r.states) and not r.states[k].solved]
        if states:
            ag = _agree_batch(states, student)
            res.by_k["teacher_state"][k] = round(sum(ag) / len(ag), 4)

    # self_state: lockstep rollout across all rollouts, scoring at each k in kset
    if self_state:
        cur = [r.states[0] for r in rollouts]
        alive = [True] * len(rollouts)
        maxk = max(k_buckets)
        for k in range(maxk + 1):
            if k in kset:
                idx = [i for i in range(len(cur)) if alive[i] and not cur[i].solved]
                if idx:
                    ag = _agree_batch([cur[i] for i in idx], student)
                    res.by_k["self_state"][k] = round(sum(ag) / len(ag), 4)
            if k == maxk:
                break
            # advance every alive rollout one step on the student's own action
            live_idx = [i for i in range(len(cur)) if alive[i] and not cur[i].solved]
            acts = _acts(student, [cur[i] for i in live_idx])
            for i, a in zip(live_idx, acts):
                cur[i], _, _ = G.step(cur[i], a)
            for i in range(len(cur)):
                if alive[i] and cur[i].solved:
                    alive[i] = False

    for mode in ("teacher_state", "self_state"):
        vals = list(res.by_k[mode].values())
        res.overall[mode] = round(sum(vals) / len(vals), 4) if vals else None
    return res
