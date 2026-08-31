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
from .runners import continuation



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
    from .progress import tick

    trajectories = list(trajectories)
    if not trajectories:
        return []
    # chat=False ON BOTH LEGS — see score_submission. A prefix is text to be continued, and the
    # parent's step is the reference every miner is measured against, so this is the leg that
    # defines what the question even is.
    steps = continuation(parent, [t.prefix for t in trajectories], max_step_tokens)
    out: list[SharedSample] = []
    for n, (t, parent_step) in enumerate(zip(trajectories, steps), 1):
        # Two observer passes and a continuation per sample, none of them batched. This is the
        # miner-independent four-fifths of the round and the second-longest silent block in it.
        tick("shared", f"{n}/{len(trajectories)}")
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
    from .progress import tick

    usable = [s for s in shared if s.usable]
    if not usable:
        from .observer_kl import MinerScore
        ms = MinerScore()
        ms.reasons.append("no usable samples this round (parent effect too small everywhere)")
        return ms
    # chat=False, EXPLICITLY AND ON BOTH LEGS. `HFRunner.generate` defaulted to wrapping the prompt
    # as a user turn and `GGUFStudentRunner.generate` has never had a chat path, so a safetensors
    # submission and a GGUF submission were handed different prompts and then ranked against each
    # other — two miners with numerically identical models scored differently by file extension.
    # The parent was prompted the safetensors way, so GGUF submissions were also being compared to
    # a reference step produced under a prompt they never saw.
    miner_steps = continuation(miner, [s.prefix for s in usable], max_step_tokens)
    samples: list[tuple[str, StepEffect]] = []
    for n, (s, a_step) in enumerate(zip(usable, miner_steps), 1):
        tick("score", f"{getattr(miner, 'name', '?')} {n}/{len(usable)}")
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
    ms = score_miner(samples, alpha=alpha, beta=beta)
    # HAND THE STEPS BACK. `_freeze` used to call generate() again over the identical prompts just
    # to record this text — half the round's most expensive leg, recomputing what we already had,
    # and a second opportunity for greedy decode to disagree with the scored run.
    ms.steps = list(miner_steps)
    return ms


def select_trajectories(pool, commit_root: str, round_nonce: str, n: int,
                        tag: str = "observer-items", min_per_lang: int = 10):
    """Draw WHICH trajectories are scored from POST-COMMIT entropy, STRATIFIED BY LANGUAGE.

    This closes the largest hole in the crown path. `run_observer_round` used to accept a trajectory
    LIST from its caller, which meant the operator chose the exam — the same trust the
    single-validator subnets have, except invisible, because no record showed it. With the selection
    derived from `commit_root ‖ round_nonce`:

      * the operator cannot pick favourable items (the nonce comes from a block drawn after the
        commitment window closed);
      * an auditor re-derives the identical selection and re-runs the round;
      * the chosen indices go in the signed record, so the claim is checkable rather than asserted.

    STRATIFICATION IS NOT COSMETIC — a uniform draw silently disarms the anti-clone axis. Scores
    aggregate worst-slice over (observer x language x depth), and `score_miner` DROPS any slice with
    fewer than its per-slice floor of samples. On a pool that is 67% English, a uniform draw of 24
    items yields ~4 Hindi and ~5 Chinese: both under the floor, both dropped, and the crown is
    decided on English alone. That is precisely where a cloned artifact is strong, so the one axis
    that separates honest work from `git clone` stops binding — invisibly, with the record still
    showing a multilingual pool.

    So each language present gets at least `min_per_lang` slots before the remainder is shared out
    in proportion. Still one RNG seeded from the same post-commit value, so the draw is exactly as
    unpredictable and exactly as re-derivable as before; `eval/rerun.py` re-runs this same function,
    so the auditor and the round cannot disagree about what stratification means.

    Returns (selected, indices). Indices are into the pool AS PASSED, so the record must also pin
    the corpus spec that produced the pool — an index is only meaningful against a known,
    revision-pinned ordering."""
    import random as _r

    # THE EXAM IS CLAMPED AT POOL-BUILD TIME (eval/pool.py), not here. It used to be filtered here
    # by CHARACTER length, which was wrong twice over: measured on the real pool it deleted 267 of
    # 900 items and the ENTIRE en/deep slice — the anti-clone axis is multilingual x depth, so that
    # pointed the one axis a cloned artifact cannot fake in exactly the wrong direction. And a
    # character bound is non-monotonic in tokens: it kept a 2,183-char/2,010-token Hindi item and
    # excluded a 3,198-char/975-token English one, because it binds hardest where characters are
    # least dense. The justification for characters — "an auditor must re-derive selection without
    # a tokenizer" — was answering a problem that does not exist: `dump_pool` publishes the prefix
    # TEXT and `pool_sha256` binds it, so an auditor gets the same bytes and never tokenizes.
    #
    # THE EXAM MUST STILL BE ANSWERABLE BY EVERY LEGAL SUBMISSION FORMAT. GGUF is the only format that can
    # pass the bit tiers, it runs under llama.cpp, and llama.cpp has a HARD context window
    # (`GGUFStudentRunner.N_CTX`). A trajectory whose prefix exceeds it does not score a miner
    # badly — it raises, and the round dies for everyone:
    #
    #     ValueError: Requested tokens (5990) exceed context window of 4096
    #
    # That is a validator bug wearing a miner's clothes. Setting an exam that no admissible
    # submission can read is not a hard exam, it is an invalid one, so oversized items are excluded
    # from SELECTION rather than left to fail. Filtering here (not in the pool) keeps `pool_sha256`
    # and the published indices meaningful: the pool is unchanged and an auditor re-deriving the
    # selection applies the identical, pinned bound.
    #
    # BOUNDED IN CHARACTERS, not tokens, and deliberately. Selection must be reproducible by an
    # auditor without loading a tokenizer, and the bound has to hold for the worst tokeniser ratio
    # in the corpus — CJK approaches one token per character, where English is nearer a quarter of
    # that. So the character bound IS the token bound in the worst case, and it is conservative
    # everywhere else. Observed real contexts this round were 68-398 tokens, so this excludes
    # almost nothing.
    total = len(pool)
    if total == 0:
        return [], []
    k = min(n, total)
    rng = _r.Random(derive_seed(commit_root, round_nonce, tag))

    # Stratify by the FULL slice key the scorer uses, not just by language. score_miner drops any
    # slice under its per-slice floor, and the key is (observer x language x DEPTH) — so balancing
    # languages alone still leaves 3 languages x 2 depths = 6 slices from 24 items, ~4 each, every
    # one of them dropped. That is not a subtle degradation: with every slice dropped the score is
    # 0.0, the identity check fails, and the round ABORTS. It aborted the first real publish.
    #
    # The observer is constant within a round, so grouping on (language, depth) is the same
    # partition the aggregation will see.
    by_lang: dict = {}
    for i, t in enumerate(pool):
        lang = (getattr(t, "meta", None) or {}).get("lang") or _lang_of(t.source)
        depth = "deep" if getattr(t, "index", 0) >= 3 else "shallow"
        by_lang.setdefault((lang, depth), []).append(i)

    if len(by_lang) < 2:
        idx = sorted(rng.sample(range(total), k))
        return [pool[i] for i in idx], idx

    # ARITHMETIC PRECONDITION. With S slices and a per-slice floor F, a draw of fewer than S*F
    # items cannot fill them however it is stratified — every slice lands under the floor, all of
    # them are dropped, the score is 0.0 and the round aborts. It is worth failing here with the
    # numbers rather than there: the identity check's own diagnosis is "nondeterministic
    # generation, check attn_implementation", which would send an operator hunting a GPU bug that
    # does not exist.
    import inspect as _i
    from .observer_kl import score_miner as _sm
    floor = _i.signature(_sm).parameters["min_per_slice"].default
    need = len(by_lang) * floor
    if k < need:
        raise ValueError(
            f"n_items={k} cannot fill {len(by_lang)} scoring slices at {floor} samples each; "
            f"need at least {need}. Slices are (language x depth) — raise n_items, or narrow the "
            f"pool's language mix.")

    langs = sorted(by_lang)
    # floor first, capped by what the pool actually holds and by the total budget
    alloc = {}
    for lang in langs:
        alloc[lang] = min(min_per_lang, len(by_lang[lang]), max(1, k // len(langs)))
    # then share the remainder to the SMALLEST slice first, never exceeding a stratum's size.
    # The aggregate is a soft-MIN over slices, so precision bought on the largest slice never
    # reaches the decision. Sharing in proportion to availability spent the remainder on English
    # (~67% of the pool): a 72-item draw put 22 points in en/shallow while the slice that
    # actually decided the crown sat at its floor of 11. Levelling up instead puts every spare
    # item where the dethrone test is resolved, at no extra scoring cost, and makes the deciding
    # slice grow with n_items as intended rather than pinned near the floor. Still one RNG seeded
    # from the same post-commit value, ties broken by draw over a sorted list, so the selection is
    # exactly as unpredictable and exactly as re-derivable as before.
    remaining = k - sum(alloc.values())
    while remaining > 0:
        room = [l for l in langs if alloc[l] < len(by_lang[l])]
        if not room:
            break
        low = min(alloc[l] for l in room)
        pick = rng.choice([l for l in room if alloc[l] == low])
        alloc[pick] += 1
        remaining -= 1

    idx: list = []
    for lang in langs:
        idx.extend(rng.sample(by_lang[lang], alloc[lang]))
    idx = sorted(idx)
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
