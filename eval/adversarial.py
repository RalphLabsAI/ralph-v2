"""Phase 2 — the adversarial proof.

Runs a cast of simulated students through the real scoring pipeline and asserts the
property the whole mechanism rests on:

    THE HONEST STUDENT WINS, AND EVERY CHEATER LOSES.

Multi-axis (math + code), because the central anti-laundering claim is that
worst-domain aggregation punishes a student that buys one capability by selling
another. A single-axis test cannot demonstrate that.

If this fails we have found the hole before miners did, which is the entire reason
it runs before anything is claimed publicly.

    python -m eval.adversarial
"""
from __future__ import annotations

import random

from .axes.code_exec import CodeExec
from .axes.math_gsm import MathGSM
from .core import run_axis
from .scoring import axis_retention, verdict
from .seeds import derive_seed

TEACHER_ACC = 0.85


class Competence:
    """A simulated model defined by per-axis accuracy.

    Given a prompt it either emits the correct answer for that item or a plausible
    wrong one. This models CAPABILITY, which is all the harness can observe — the
    point being that no amount of fluent-sounding output substitutes for it.
    """

    def __init__(self, name: str, acc_by_axis: dict[str, float], truth: dict[str, str],
                 axis_of: dict[str, str], seed: int = 0, style_filler: bool = False,
                 template_of: dict[str, str] | None = None,
                 template_boost: tuple[str, float] | None = None,
                 noop_of: dict[str, bool] | None = None, noop_acc: float | None = None):
        self.name = name
        self.acc, self.truth, self.axis_of = acc_by_axis, truth, axis_of
        self.template_of, self.template_boost = template_of or {}, template_boost
        self.noop_of, self.noop_acc = noop_of or {}, noop_acc
        self.style_filler = style_filler
        self._rng = random.Random(seed)

    def _accuracy_for(self, prompt: str) -> float:
        a = self.acc.get(self.axis_of.get(prompt, ""), 0.0)
        if self.template_boost and self.template_of.get(prompt) == self.template_boost[0]:
            a = self.template_boost[1]
        if self.noop_acc is not None and self.noop_of.get(prompt):
            a = self.noop_acc
        return a

    def generate(self, prompts, max_new_tokens: int = 512) -> list[str]:
        out = []
        filler = ("Let me reconsider. Wait — I should double-check that step. "
                  "Breaking it down carefully. Hmm, let me verify.\n") if self.style_filler else ""
        for p in prompts:
            correct = self._rng.random() < self._accuracy_for(p)
            truth = self.truth.get(p)
            if correct and truth is not None:
                out.append(filler + truth)
            else:
                if self.axis_of.get(p) == "code":
                    out.append(filler + "```python\ndef f(*a, **k):\n    return None\n```")
                else:
                    out.append(filler + f"Answer: {self._rng.randint(1, 400)}")
        return out


def build_round(commit_root: str, round_nonce: str, n_per_axis: int = 180):
    math_axis, code_axis = MathGSM(), CodeExec()
    axes = [math_axis, code_axis]

    items, truth, axis_of, template_of, noop_of = [], {}, {}, {}, {}
    for ax in axes:
        seed = derive_seed(commit_root, round_nonce, ax.name)
        generated = ax.generate(seed, n=n_per_axis, difficulty=2)
        for it in generated:
            truth[it.prompt] = (f"Answer: {it.answer}" if ax.name == "math"
                                else code_axis.reference_solution(it))
            axis_of[it.prompt] = ax.name
            template_of[it.prompt] = it.meta["template"]
            noop_of[it.prompt] = bool(it.meta.get("noop"))
        items.append((ax, generated))
    return items, truth, axis_of, template_of, noop_of


def main() -> int:
    commit_root, round_nonce = "0xCOMMITROOT_demo", "0xBLOCKHASH_demo"
    per_axis, truth, axis_of, template_of, noop_of = build_round(commit_root, round_nonce)
    common = dict(truth=truth, axis_of=axis_of, template_of=template_of, noop_of=noop_of)

    teacher = Competence("teacher(sim)", {"math": TEACHER_ACC, "code": TEACHER_ACC}, seed=0, **common)
    base = Competence("pinned-base(sim)", {"math": 0.40, "code": 0.38}, seed=99, **common)

    contenders = [
        # honest: uniformly ~85% of the teacher on BOTH axes
        Competence("student-honest", {"math": 0.72, "code": 0.72}, seed=1, **common),
        # the free baseline — must earn ~nothing
        Competence("naive-quant-control", {"math": 0.47, "code": 0.45}, seed=5, **common),
        # style: fluent teacher-shaped prose, no capability
        Competence("student-style-only", {"math": 0.02, "code": 0.02}, seed=2, style_filler=True, **common),
        # LAUNDERING: bought math, sold code. Average looks fine; worst-domain must kill it.
        Competence("student-narrow-math", {"math": 0.95, "code": 0.12}, seed=3, **common),
        # pattern-matcher: collapses when a NoOp distractor is present
        Competence("student-noop-brittle", {"math": 0.80, "code": 0.70}, seed=4, noop_acc=0.08, **common),
    ]

    t_res, b_res = {}, {}
    for ax, items in per_axis:
        t_res[ax.name] = [r.passed for r in run_axis(ax, teacher, items)]
        b_res[ax.name] = [r.passed for r in run_axis(ax, base, items)]
    print("fresh round from commit-then-generate seed:")
    for ax, items in per_axis:
        print(f"  {ax.name:<6} {len(items):>4} items | teacher {sum(t_res[ax.name]):>3} "
              f"| base {sum(b_res[ax.name]):>3}")
    print()

    rows = []
    for runner in contenders:
        axis_scores = []
        for ax, items in per_axis:
            s_res = [r.passed for r in run_axis(ax, runner, items)]
            axis_scores.append(axis_retention(ax.name, ax.weight, t_res[ax.name], s_res, b_res[ax.name]))
        rows.append((runner.name, axis_scores, verdict(axis_scores)))

    hdr = f"{'student':<24}" + "".join(f"{a.axis+' ret':>12}" for a in rows[0][1]) + f"{'SCORE':>10}  gates"
    print(hdr); print("-" * len(hdr))
    for name, scores, v in rows:
        line = f"{name:<24}" + "".join(f"{a.retention:>12.3f}" for a in scores) + f"{v.score:>10.3f}"
        print(line + "  " + ("ok" if v.gates_passed else "; ".join(v.reasons)))

    honest = next(r for r in rows if r[0] == "student-honest")
    cheats = [r for r in rows if r[0] != "student-honest"]
    print()
    failures = [(n, v.score) for n, _, v in cheats if v.score >= honest[2].score]
    for n, s in failures:
        print(f"  FAIL: {n} scored >= honest ({s:.3f} vs {honest[2].score:.3f})")
    if not failures:
        best = max(v.score for _, _, v in cheats)
        print(f"  PASS: honest student {honest[2].score:.3f} beats every adversary "
              f"(best cheater {best:.3f}, margin {honest[2].score - best:.3f})")
        narrow = next(v for n, _, v in rows if n == "student-narrow-math")
        print(f"  laundering check: narrow-math averaged well on math but scored "
              f"{narrow.score:.3f} — worst-domain aggregation held.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
