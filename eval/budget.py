"""Score-at-budget — separating "doesn't know" from "didn't finish".

THE FINDING THIS IMPLEMENTS. The sharpest independent result in the Bonsai corpus
(github.com/Astezelex/bonsai-27b-16gb-bench, AIME26 at a 60k-token budget) is that a compressed
model and its conventional-quant competitor failed in two DIFFERENT ways:

    Ternary Bonsai 27B   score 0.867   capped 10%   accuracy-if-converged 96%
    Qwen3.6 IQ2_XXS      score 0.633   capped 37%   accuracy-if-converged 100%

The competitor was MORE accurate whenever it finished, and lost on cap rate. Their conclusion,
which is the one line to carry into any compression crown:

    "Thinking-mode models fail benchmarks two different ways: they don't know, or they don't
     CONVERGE within the token budget."

A single accuracy scalar silently merges these, so it pays for TERSENESS and calls it
intelligence. The predecessor subnet learned the same lesson from the other end: their KL-crowned
king looped a 6-word phrase 102x on the prompt "Hi" and never emitted an answer — a pure
convergence failure that their scalar could not see.

WHAT THIS GIVES THE CROWN.
  * `score` already counts non-convergence as failure (a truncated answer fails the checker), so
    the crown is honest by construction — but the REASON is invisible, which is what makes it
    un-diagnosable and lets a pathology hide inside a plausible number.
  * `cap_rate` surfaces it, and is gated: a student capping far more often than the TEACHER on
    the same items is not compressed, it is broken (the UID-107 signature). This is a
    compression-specific failure mode that no accuracy metric detects.
  * Reporting the triple keeps the metric interpretable and stops "we paid for verbosity
    reduction" from ever being the explanation of a crown.

Token counting is injectable. The default is a deterministic character proxy — honest about
being a proxy, and adequate for CAP DETECTION (which only asks "did this run to the wall"),
though a real tokenizer should be passed in production for the mean-token figure.
"""
from __future__ import annotations

from dataclasses import dataclass

# a generation is treated as CAPPED when its estimated length reaches this fraction of the
# budget: a model that stops on its own essentially never lands within a hair of the wall.
CAP_FRACTION = 0.97
CHARS_PER_TOKEN = 4.0          # crude but deterministic; override with a real tokenizer


def approx_tokens(text: str) -> int:
    """Deterministic length proxy. Max of a word count and a character estimate, so neither
    dense code nor whitespace-heavy prose is badly mis-measured."""
    if not text:
        return 0
    return max(len(text.split()), int(len(text) / CHARS_PER_TOKEN))


@dataclass
class BudgetReport:
    n: int = 0
    score: float = 0.0             # pass rate at this budget (non-convergence counts as fail)
    cap_rate: float = 0.0          # fraction that ran to the wall
    mean_tokens: float = 0.0
    acc_if_converged: float = 0.0  # pass rate among generations that finished on their own
    budget: int = 0

    def as_dict(self) -> dict:
        return {"n": self.n, "score": round(self.score, 4), "cap_rate": round(self.cap_rate, 4),
                "mean_tokens": round(self.mean_tokens, 1),
                "acc_if_converged": round(self.acc_if_converged, 4), "budget": self.budget}


def score_at_budget(passes: list[bool], outputs: list[str], budget: int,
                    token_fn=approx_tokens) -> BudgetReport:
    """The (score, cap_rate, tokens) triple for one model on one axis at a FIXED budget."""
    n = len(outputs)
    if n == 0 or n != len(passes):
        return BudgetReport(budget=budget)
    toks = [token_fn(o) for o in outputs]
    capped = [t >= CAP_FRACTION * budget for t in toks]
    conv = [i for i in range(n) if not capped[i]]
    return BudgetReport(
        n=n,
        score=sum(1 for p in passes if p) / n,
        cap_rate=sum(1 for c in capped if c) / n,
        mean_tokens=sum(toks) / n,
        acc_if_converged=(sum(1 for i in conv if passes[i]) / len(conv)) if conv else 0.0,
        budget=budget,
    )


def convergence_gate(student: BudgetReport, teacher: BudgetReport,
                     max_excess: float = 0.25, floor: float = 0.5) -> tuple[bool, str]:
    """Fail a student that cannot FINISH, independent of whether it is right.

    Two ways to fail, both compression pathologies rather than knowledge gaps:
      * capping far more often than the teacher on the SAME items (`max_excess`) — the model
        has lost the ability to terminate, the UID-107 loop signature;
      * capping on more than `floor` of items outright, even if the teacher is also struggling.
    Judged against the teacher on identical items, so a genuinely hard axis does not fire it."""
    if student.n == 0:
        return False, "no generations"
    if student.cap_rate > floor:
        return False, f"cap rate {student.cap_rate:.2f} over {floor:.2f} — does not terminate"
    excess = student.cap_rate - teacher.cap_rate
    if excess > max_excess:
        return False, (f"cap rate {student.cap_rate:.2f} exceeds teacher "
                       f"{teacher.cap_rate:.2f} by {excess:.2f} — convergence failure")
    return True, ""
