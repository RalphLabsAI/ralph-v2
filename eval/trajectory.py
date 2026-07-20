"""Trajectory step-agreement — const's mechanism, real models.

Scores a student by how much of what GLM does at a step the student also does,
sampled over (rollout, step) pairs. No fixed test set: the rollouts are the eval,
they are infinite and self-labeling, and the label is GLM's own step. This is the
substrate; the retention/KOTH/margin scoring in scoring.py sits on top unchanged.

Two evaluation modes, and the whole point of running both:

  OFF-POLICY (const's original): prefix 0->K is GLM's real trajectory. Both models
    generate K->K+1 from GLM's context. Dense and cheap — GLM's step and the judge's
    questions cache once per round and amortize across every miner.

  ON-POLICY (the addition): the STUDENT generates the prefix 0->K itself, then we ask
    whether, in the state the student actually reached, it did what GLM would do here.
    Catches the exposure-bias drift that teacher-forcing cannot see — the failure mode
    that collapsed the previous distillation subnet (see experiments/exposure_bias.py).

Judge is grounded local comparison, never open-ended quality: "GLM did X here; did the
student also do X?" anchored to a fresh reference GLM produced one step ago. Keep the
judge model != GLM (no self-preference) and rotate it.

Needs a GPU for a real GLM teacher. Runs on CPU against SimRunner/SimJudge for tests.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from .core import ModelRunner


@dataclass(frozen=True)
class Rollout:
    """A recorded trajectory. `steps` are the successive action/turn texts; `context`
    is any leading system/task text. Albedo-style agentic-coding rollouts are the
    canonical source. `success` lets us prefer distilling from winning trajectories."""
    id: str
    context: str
    steps: list[str]
    success: bool = True

    def prefix(self, k: int) -> str:
        return self.context + "\n" + "\n".join(self.steps[:k])


@runtime_checkable
class StepJudge(Protocol):
    """Grounded local comparison -> agreement in [0, 1]."""
    name: str
    def agreement(self, prefix: str, reference_step: str, candidate_step: str) -> float: ...


# --------------------------------------------------------------------------
# Judges
# --------------------------------------------------------------------------

class RubricJudge:
    """const variant 1: auto-generate N y/n checks FROM GLM's step, grade the student.

    Decomposing the fuzzy "are these similar" into concrete atoms is what keeps the
    judge in its reliable regime. The checks are generated once per (rollout, step)
    and cached — they do not depend on the student.
    """
    name = "rubric-judge"

    def __init__(self, judge: ModelRunner, n_checks: int = 8):
        self.judge = judge
        self.n_checks = n_checks

    def make_checks(self, prefix: str, reference_step: str) -> list[str]:
        prompt = (
            "Given the context and the reference step, list up to "
            f"{self.n_checks} concrete, checkable things the reference did, one per line, "
            "each phrased as a yes/no question starting with 'Did the response '.\n\n"
            f"CONTEXT:\n{prefix[-2000:]}\n\nREFERENCE STEP:\n{reference_step}\n\nQUESTIONS:"
        )
        out = self.judge.generate([prompt], max_new_tokens=400)[0]
        qs = [ln.strip(" -*\t") for ln in out.splitlines() if "did the response" in ln.lower()]
        return qs[: self.n_checks]

    def grade(self, prefix: str, checks: list[str], candidate_step: str) -> float:
        if not checks:
            return 0.0
        block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(checks))
        prompt = (
            "Answer each question yes or no about the candidate step. Reply with only "
            "the number and yes/no, one per line.\n\n"
            f"CANDIDATE STEP:\n{candidate_step}\n\nQUESTIONS:\n{block}\n\nANSWERS:"
        )
        out = self.judge.generate([prompt], max_new_tokens=300)[0].lower()
        yes = len(re.findall(r"\byes\b", out))
        return min(yes / len(checks), 1.0)

    def agreement(self, prefix: str, reference_step: str, candidate_step: str) -> float:
        return self.grade(prefix, self.make_checks(prefix, reference_step), candidate_step)


class SimJudge:
    """Token-overlap stand-in so the harness runs and is testable without a judge model.
    NOT for production — real grading uses RubricJudge."""
    name = "sim-judge"

    def agreement(self, prefix: str, reference_step: str, candidate_step: str) -> float:
        a, b = set(reference_step.lower().split()), set(candidate_step.lower().split())
        return len(a & b) / len(a) if a else 0.0


# --------------------------------------------------------------------------
# Eval points
# --------------------------------------------------------------------------

@dataclass
class StepPoint:
    rollout_id: str
    k: int
    mode: str            # "off_policy" | "on_policy"
    agreement: float
    meta: dict = field(default_factory=dict)


def off_policy_point(rollout: Rollout, k: int, glm: ModelRunner, student: ModelRunner,
                     judge: StepJudge, max_new_tokens: int = 512,
                     cached_glm_step: str | None = None) -> StepPoint:
    """Both models continue GLM's real prefix. `cached_glm_step` reuses GLM's step
    across miners in a round."""
    prefix = rollout.prefix(k)
    glm_step = cached_glm_step or glm.generate([prefix], max_new_tokens)[0]
    student_step = student.generate([prefix], max_new_tokens)[0]
    return StepPoint(rollout.id, k, "off_policy",
                     judge.agreement(prefix, glm_step, student_step),
                     {"cached": cached_glm_step is not None})


def on_policy_point(rollout: Rollout, k: int, glm: ModelRunner, student: ModelRunner,
                    judge: StepJudge, max_new_tokens: int = 512) -> StepPoint:
    """The STUDENT builds the prefix 0->K itself; then we ask whether, in the state it
    reached, it did what GLM would do there. This is the drift-catching slice."""
    base = rollout.context
    student_prefix = base
    for _ in range(k):
        nxt = student.generate([student_prefix], max_new_tokens)[0]
        student_prefix = student_prefix + "\n" + nxt
    glm_here = glm.generate([student_prefix], max_new_tokens)[0]      # what GLM would do in the student's state
    student_here = student.generate([student_prefix], max_new_tokens)[0]
    return StepPoint(rollout.id, k, "on_policy",
                     judge.agreement(student_prefix, glm_here, student_here), {})


def score_student(rollouts: Sequence[Rollout], glm: ModelRunner, student: ModelRunner,
                  judge: StepJudge, points_per_rollout: int = 3, on_policy_frac: float = 0.3,
                  seed: int = 0) -> dict:
    """Aggregate step-agreement into off/on-policy scores for one student.

    Feeds scoring.axis_retention as two axes ('step_off', 'step_on') so the existing
    worst-domain soft-min + margin + KOTH machinery applies unchanged: a fluent drifter
    with high off / low on is pulled down to its weakest axis exactly like a laundering
    student is.
    """
    import random
    rng = random.Random(seed)
    off, on = [], []
    for r in rollouts:
        if len(r.steps) < 2:
            continue
        for _ in range(points_per_rollout):
            k = rng.randint(1, len(r.steps) - 1)
            if rng.random() < on_policy_frac:
                on.append(on_policy_point(r, k, glm, student, judge))
            else:
                off.append(off_policy_point(r, k, glm, student, judge))
    mean = lambda xs: sum(p.agreement for p in xs) / len(xs) if xs else 0.0
    return {
        "off_policy": mean(off), "n_off": len(off),
        "on_policy": mean(on), "n_on": len(on),
        "points": off + on,
    }
