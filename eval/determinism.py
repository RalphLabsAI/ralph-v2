"""Determinism harness — measure the noise floor, then refuse to crown inside it.

WHY THIS EXISTS. Observer-KL is derived from logits, and logit arithmetic is not bit-stable:
batch composition, attention backend, kernel selection and dtype all move the low-order bits, and
a KL is sensitive enough to turn that into a score difference that looks real. This is not a
theoretical worry. The predecessor subnet measured **2-5 percentage points** of run-to-run
variance on IFEval before pinning their stack — and then published 0.42-1.2 pp margins as
findings. A crown decided inside the noise band is a coin flip a miner can bias by resubmitting.

So the rule this module enforces is simple and unusual: **the validator measures its own noise
floor every round, publishes it, and will not award a crown on a margin smaller than it.** That
is the check SN97 specified in their hardening doc and never shipped.

Three things to pin BEFORE trusting any number here (they belong in the operator runbook, and
are the difference between a 5pp noise floor and a 0.1pp one):
  * greedy decode only, temperature fixed, seed fixed — already enforced validator-side by
    runners._gen_config, which also stops a miner's generation_config from leaking in;
  * a pinned model revision and a pinned attention backend, with batch-invariant kernels;
  * fixed batch composition — the same sample scored alongside different neighbours can produce
    different logits, so batches must not depend on which miners happened to submit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .observer_kl import kl


@dataclass
class NoiseReport:
    n_repeats: int = 0
    n_positions: int = 0
    max_kl: float = 0.0          # worst same-input disagreement seen
    mean_kl: float = 0.0
    max_prob_delta: float = 0.0  # largest single-token probability drift
    identical: bool = False      # bit-identical across repeats
    reasons: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"n_repeats": self.n_repeats, "n_positions": self.n_positions,
                "max_kl": round(self.max_kl, 8), "mean_kl": round(self.mean_kl, 8),
                "max_prob_delta": round(self.max_prob_delta, 8), "identical": self.identical,
                "reasons": self.reasons}


def measure_noise(observer, prefix: str, continuation: str, repeats: int = 3,
                  batch_variations=()) -> NoiseReport:
    """Score the SAME (prefix, continuation) several times and report the disagreement.

    `batch_variations` optionally supplies decoy prefixes to be scored alongside, so batch
    composition changes between repeats. That is the sneakiest source of drift: a sample scored
    next to different neighbours can come back with different logits, which means a miner's score
    could depend on who else submitted that round."""
    rep = NoiseReport(n_repeats=repeats)
    runs: list[list[dict]] = []
    for i in range(repeats):
        if batch_variations:
            for decoy in batch_variations[: i + 1]:
                observer.distributions(decoy, continuation)      # perturb batch state
        runs.append(observer.distributions(prefix, continuation))
    if not runs or not runs[0]:
        rep.reasons.append("observer returned no distributions")
        return rep
    n = min(len(r) for r in runs)
    rep.n_positions = n
    if any(len(r) != len(runs[0]) for r in runs):
        rep.reasons.append("position count differs between identical runs — not reproducible")
    kls, deltas = [], []
    for t in range(n):
        base = runs[0][t]
        for other in runs[1:]:
            kls.append(kl(base, other[t]))
            for tok, p in base.items():
                deltas.append(abs(p - other[t].get(tok, 0.0)))
    rep.max_kl = max(kls) if kls else 0.0
    rep.mean_kl = (sum(kls) / len(kls)) if kls else 0.0
    rep.max_prob_delta = max(deltas) if deltas else 0.0
    rep.identical = rep.max_kl == 0.0 and rep.max_prob_delta == 0.0
    if not rep.identical:
        rep.reasons.append(f"non-deterministic: max KL {rep.max_kl:.2e} over identical inputs")
    return rep


def margin_floor(noise: NoiseReport, safety: float = 3.0, floor: float = 1e-4) -> float:
    """The smallest score margin worth acting on, given the measured noise.

    `safety` multiples of the observed same-input KL, never below `floor`. A margin under this
    is indistinguishable from the validator's own jitter, and awarding a crown on it is a lottery
    a miner wins by resubmitting."""
    return max(floor, safety * noise.max_kl)


def crownable(margin: float, noise: NoiseReport, safety: float = 3.0) -> tuple[bool, str]:
    """Gate a crown decision on the measured noise floor. Publish both numbers either way —
    a subnet that reports its own reproducibility is checkable in a way that a leaderboard
    quoting three decimal places is not."""
    need = margin_floor(noise, safety)
    if margin < need:
        return False, (f"margin {margin:.6f} is inside the measured noise floor {need:.6f} "
                       f"(max same-input KL {noise.max_kl:.2e} x{safety:g}) — not crownable")
    return True, f"margin {margin:.6f} clears the noise floor {need:.6f}"


# ---------------------------------------------------------------------------------------------
# IDENTITY CANARY
# ---------------------------------------------------------------------------------------------

IDENTITY_TOLERANCE = 1e-6


def identity_check(shared, parent, observer, max_step_tokens: int = 256) -> tuple[bool, dict]:
    """Score the PINNED PARENT as if it were a submission. It must come back at exactly 1.0.

    This is the cheapest and strictest determinism test available, and it is strictly better than
    the noise probe it complements. By construction, when the miner's step IS the parent's step,
    s = KL(P_G || P_A) = 0 and |d_A - d_G| = 0, so sample_score = exp(0)exp(0) = 1 for every
    sample and the worst-slice aggregate is 1. There is no modelling assumption to argue about:
    any shortfall is the pipeline disagreeing with itself.

    MEASURED VALUE OF THIS GATE. On an H100 PCIe the noise probe reported a clean floor while this
    check returned 0.8754 — a 12-point self-inconsistency, caused by generate() being
    nondeterministic above ~64 new tokens on that box (see runners.HFRunner.ATTN_IMPLEMENTATION).
    An A100 and an L40S returned exactly 1.0 on the same code. So determinism is a property of the
    hardware and the kernel selection, and a validator can be silently unfit to award crowns while
    every other gate passes. Run this at startup and refuse to score if it fails: a box that cannot
    reproduce the identity has no business ranking anyone else.
    """
    from .observer_round import score_submission
    ms = score_submission(shared, parent, observer, max_step_tokens=max_step_tokens)
    info = {"score": ms.score, "n_scored": ms.n_scored, "reasons": list(ms.reasons),
            "shortfall": round(1.0 - ms.score, 6)}
    ok = abs(ms.score - 1.0) <= IDENTITY_TOLERANCE and not ms.reasons
    if not ok:
        thin = any("slice" in r for r in ms.reasons)
        info["verdict"] = (
            f"parent scored {ms.score:.6f} against itself, not 1.0 — the pipeline disagrees with "
            f"itself by {1.0 - ms.score:.4f}. "
            + ("EVERY SCORING SLICE WAS DROPPED for having too few samples, so this is a round "
               "SIZE problem, not a hardware one: raise n_items until each (language x depth) "
               "slice clears the per-slice floor."
               if thin else
               "Almost always nondeterministic generation: check attn_implementation and try a "
               "shorter max_step_tokens."))
    return ok, info
