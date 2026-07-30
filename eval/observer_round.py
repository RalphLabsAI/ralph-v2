"""One observer-KL round — shared rollouts, then per-miner scoring.

THE COST STRUCTURE IS THE WHOLE DESIGN. Of the five generations const's spec describes, four are
MINER-INDEPENDENT and are computed exactly once per round no matter how many miners submitted:

    per round, once            per miner
    -----------------------    ---------------------------
    RollGLM  (parent step)     RollA   (miner's step)      1 generation
    C        (observer cont.)  P_A     (observer forward)  1 forward pass
    P_G, P_0 (observer fwd)

So the marginal cost of an extra miner is one generation plus one forward pass over a short
continuation — cheaper than the six-axis capability round, which ran every miner over every axis.
The parent and observer are loaded once and amortized.

THE OBSERVER IS DRAWN FROM THE ROUND NONCE. const's step 14 suggests rotating observers; drawing
the observer post-commit makes it un-pre-fittable rather than merely varied. A miner that tuned
its steps to move Qwen the way the parent does cannot know Qwen is the observer this round.

WHAT THE PARENT'S STEP IS. Not the dataset's recorded step — the PINNED PARENT generates it, on
the same prefix, under the same frozen decode. The miner's job is to reproduce THIS model's
contribution, which is what makes the number "retention of the parent" rather than "agreement
with whoever wrote the dataset".

Everything here is protocol-typed, so the round is CPU-testable against stubs and the real
models drop in unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .observer_kl import MIN_TEACHER_EFFECT, StepEffect, score_miner, step_effect


class Stepper(Protocol):
    """Produces one step K -> K+1. The parent and every miner satisfy this."""
    def generate(self, prompts: Sequence[str], max_new_tokens: int = 512) -> list[str]: ...


class Observer(Protocol):
    """Returns the per-position next-token distributions the observer assigns to `continuation`
    when conditioned on `prefix`. Sparse top-k maps; the KL is taken over the union support."""
    def distributions(self, prefix: str, continuation: str) -> list[dict]: ...


@dataclass
class SharedSample:
    """The miner-independent half of one sample. Computed once, reused by every submission."""
    traj_id: str = ""
    slice_key: str = ""            # observer x language x step-length — the aggregation slice
    prefix: str = ""               # K
    parent_step: str = ""          # the pinned parent's K -> K+1
    continuation: str = ""         # C, the observer's continuation after K + parent_step
    p_parent: list = field(default_factory=list)   # P_G over C
    p_base: list = field(default_factory=list)     # P_0 over C
    d_parent: float = 0.0
    usable: bool = True
    reason: str = ""


def build_shared(trajectories, parent: Stepper, observer: Observer, observer_name: str,
                 max_step_tokens: int = 256, max_cont_tokens: int = 128,
                 min_parent_effect: float = MIN_TEACHER_EFFECT) -> list[SharedSample]:
    """The once-per-round work. Batches the parent's steps in one call, then one observer pass
    per sample for C, P_G and P_0.

    Samples whose PARENT step barely moves the observer are marked unusable here, before any
    miner is involved — the discard decision must never depend on a miner's output, or a miner
    could bury its hard samples by emitting bland steps."""
    trajectories = list(trajectories)
    if not trajectories:
        return []
    steps = parent.generate([t.prefix for t in trajectories], max_step_tokens)
    out: list[SharedSample] = []
    for t, parent_step in zip(trajectories, steps):
        s = SharedSample(traj_id=t.id, prefix=t.prefix, parent_step=parent_step,
                         slice_key=_slice_key(t, observer_name))
        if not parent_step.strip():
            s.usable, s.reason = False, "parent produced an empty step"
            out.append(s)
            continue
        # C: what the observer expects to follow, GIVEN the parent's step. Using the parent's
        # own continuation as the common sequence is what makes P_G and P_A comparable at all —
        # they are then distributions over the SAME tokens at the SAME positions.
        s.continuation = _continue(observer, s.prefix + "\n" + parent_step, max_cont_tokens)
        if not s.continuation:
            s.usable, s.reason = False, "observer produced no continuation"
            out.append(s)
            continue
        s.p_parent = observer.distributions(s.prefix + "\n" + parent_step, s.continuation)
        s.p_base = observer.distributions(s.prefix, s.continuation)
        eff = step_effect(s.p_parent, s.p_parent, s.p_base, min_parent_effect)
        s.d_parent = eff.d_teacher
        if eff.discarded:
            s.usable, s.reason = False, eff.reason
        out.append(s)
    return out


def _continue(observer: Observer, prefix: str, max_tokens: int) -> str:
    gen = getattr(observer, "generate", None)
    if gen is None:
        return ""
    outs = gen([prefix], max_tokens)
    return outs[0] if outs else ""


def _slice_key(traj, observer_name: str) -> str:
    """Aggregation slice. Worst-slice over (observer x language x step-length) means a miner
    cannot buy a weak language or a weak step depth with a strong one."""
    lang = (traj.meta or {}).get("lang") or _lang_of(traj.source)
    depth = "deep" if traj.index >= 3 else "shallow"
    return f"obs={observer_name}|lang={lang}|depth={depth}"


def _lang_of(source: str) -> str:
    if source.startswith("samvaad"):
        return "hi"
    if source.startswith("zh_"):
        return "zh"
    return "en"


def score_submission(shared: Sequence[SharedSample], miner: Stepper, observer: Observer,
                     max_step_tokens: int = 256, alpha: float = 1.0, beta: float = 1.0):
    """Per-miner half: one batched generation, then one observer pass per usable sample."""
    usable = [s for s in shared if s.usable]
    if not usable:
        from .observer_kl import MinerScore
        ms = MinerScore()
        ms.reasons.append("no usable samples this round (parent effect too small everywhere)")
        return ms
    miner_steps = miner.generate([s.prefix for s in usable], max_step_tokens)
    samples: list[tuple[str, StepEffect]] = []
    for s, a_step in zip(usable, miner_steps):
        if not a_step.strip():
            # an empty step is inert by definition — score it, never discard it, or "say
            # nothing" becomes a way to dodge hard samples
            samples.append((s.slice_key, StepEffect(s=s.d_parent, d_teacher=s.d_parent,
                                                    d_miner=0.0,
                                                    n_positions=len(s.p_parent))))
            continue
        p_miner = observer.distributions(s.prefix + "\n" + a_step, s.continuation)
        eff = step_effect(s.p_parent, p_miner, s.p_base, min_teacher_effect=0.0)
        eff.d_teacher = s.d_parent          # the shared, miner-independent value
        samples.append((s.slice_key, eff))
    return score_miner(samples, alpha=alpha, beta=beta)


def select_trajectories(pool, commit_root: str, round_nonce: str, n: int,
                        tag: str = "observer-items"):
    """Draw WHICH trajectories are scored from POST-COMMIT entropy, and record the choice.

    This closes the largest hole in the crown path. `run_observer_round` used to accept a
    trajectory LIST from its caller, which meant the operator chose the exam — the same trust
    the single-validator subnets have, except invisible, because no record showed it. With the
    selection derived from `commit_root ‖ round_nonce`:

      * the operator cannot pick favourable items (the nonce comes from a block drawn after the
        commitment window closed);
      * an auditor re-derives the identical selection and re-runs the round;
      * the chosen indices go in the signed record, so the claim is checkable rather than
        asserted.

    Returns (selected, indices). Indices are into the pool AS PASSED, so the record must also
    pin the corpus spec that produced the pool — an index is only meaningful against a known,
    revision-pinned ordering."""
    import random as _r
    total = len(pool)
    if total == 0:
        return [], []
    k = min(n, total)
    rng = _r.Random(derive_seed(commit_root, round_nonce, tag))
    idx = sorted(rng.sample(range(total), k))
    return [pool[i] for i in idx], idx


def derive_seed(commit_root: str, round_nonce: str, tag: str) -> int:
    from .seeds import derive_seed as _d
    return _d(commit_root, round_nonce, tag)


def pick_observer(commit_root: str, round_nonce: str, pool: Sequence[str]) -> str:
    """Draw the round's observer from POST-COMMIT entropy.

    Rotating observers (const's step 14) guards against a miner fitting one observer; drawing
    the observer after weights are sealed makes that impossible rather than merely harder. Same
    derivation as the item seeds, so an auditor re-derives the choice and can re-run the round."""
    from .seeds import derive_seed
    if not pool:
        raise ValueError("empty observer pool")
    return sorted(pool)[derive_seed(commit_root, round_nonce, "__observer__") % len(pool)]
