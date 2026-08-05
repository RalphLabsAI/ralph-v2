"""One full validator round on the OBSERVER-KL substrate — the production assembly.

This is the crown path. It replaces the capability-axis round rather than sitting beside it:
having two substrates is how the predecessor ended up with fixes landing in one while the other
shipped, and how our own overfit gate and surprise-axis selection ended up as dead code.

    committed submissions
      -> intake: economics -> safety inspect -> bit budget -> pinned parent -> commit-reveal
                                        [intake.py + bitrate.py + parent.py + identity.py]
      -> observer drawn from the ROUND NONCE                        [observer_round.py]
      -> shared rollouts: parent steps, observer continuation C, P_G and P_0   ONCE per round
      -> per submission: one generation + one observer pass          [observer_round.py]
      -> worst-slice score, per-sample vectors kept for the paired bootstrap  [observer_kl.py]
      -> KOTH: king re-scored and re-gated, dethrone on softmin LCB  [koth.py]
      -> NOISE GATE: refuse to crown inside the measured floor       [determinism.py]
      -> settle bonds -> SIGNED reproducible record -> weights

The capability axes are NOT deleted — they run as a cheap CANARY, because observer-KL is
structurally blind to exactly one failure: a student that moves the observer correctly while
being unusable. That is the fidelity paradox, and it is what produced the predecessor's
102x-repeated-phrase king. A canary that costs a fraction of a round is worth keeping for it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .determinism import crownable, identity_check, measure_noise
from .economics import RegistrationLedger
from .gates import TierBudget, degeneracy_flags
from .intake import intake
from .koth import MIN_CROWN_LB, Scored, Submission, Tier, Tournament
from .observer_round import build_shared, pick_observer, score_submission, select_trajectories
from .round_record import RoundRecord, build_round_record


def _sha(text: str) -> str:
    import hashlib
    return hashlib.sha256((text or "").encode()).hexdigest()


def _pool_sha(trajectory_pool) -> str:
    """Best-effort: a pool of plain Trajectory objects digests; anything else records "" rather
    than raising, because a round must not die over an audit convenience."""
    try:
        from .pool import pool_digest
        return pool_digest(trajectory_pool)
    except Exception:
        return ""


def _freeze(runner, usable, max_step_tokens: int) -> list:
    """Re-emit this model's steps as literal text for the record, so an audit is a pure forward
    pass over recorded strings instead of a reproduction of batched greedy generation — the
    noisiest part of the round. The manifest's artifact_uri lets an auditor optionally re-generate
    and spot-check that these strings are what the model really emits."""
    try:
        return list(runner.generate([x.prefix for x in usable], max_step_tokens))
    except Exception:
        return []


def _effects(ms) -> list:
    """[slice_key, s, d_parent, d_miner] per scored sample — the raw measurements the score is a
    pure function of, so an auditor recomputes the aggregate with no GPU."""
    return [[e_key, round(e.s, 6), round(e.d_teacher, 6), round(e.d_miner, 6)]
            for e_key, e in getattr(ms, "keyed_effects", []) or []]


def _versions() -> dict:
    """Pin the stack a re-run must match — INCLUDING THE GPU.

    Measured 2026-07-30 on three architectures scoring byte-identical items with identical code:

        H100 PCIe   (cap 9.0)   4-bit student 0.6240   foreign control 0.5529
        A100 SXM4   (cap 8.0)                 0.6407                   0.6240
        L40S        (cap 8.9)                 0.6114                   0.4508

    Within a single box the round is EXACTLY reproducible — 0.0 score spread over repeats, 0.0
    per-sample delta, byte-identical generations. Across boxes it is not: ~0.03 retention on a
    genuine compression and ~0.17 on the control. So `reproduction_tolerance`, derived from
    within-box noise, is roughly two orders of magnitude tighter than the cross-hardware spread,
    and an honest auditor on a different GPU would report DIVERGED on a perfectly good record.
    Recording the device and compute capability is what lets them tell "this crown was rigged"
    apart from "I am on Ada and you were on Hopper"."""
    out = {}
    try:
        import torch
        out["torch"] = torch.__version__
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            out["gpu_capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
            out["cuda"] = torch.version.cuda
    except Exception:
        pass
    try:
        import transformers
        out["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        # A GGUF student's output depends on llama.cpp's thread count and batch exactly as an HF
        # student's depends on torch's batch size — reduction order either way. An auditor
        # re-running at a different thread count gets a different answer through no fault of the
        # record, so it is pinned here rather than left to whatever the box defaulted to.
        from .runners import GGUFStudentRunner
        out["gguf_n_threads"] = GGUFStudentRunner.N_THREADS
        out["gguf_n_batch"] = GGUFStudentRunner.N_BATCH
        import llama_cpp
        out["llama_cpp"] = llama_cpp.__version__
    except Exception:
        pass
    try:
        from .runners import HFRunner
        out["topk"] = HFRunner.TOPK
        out["prob_decimals"] = HFRunner.PROB_DECIMALS
        out["attn_implementation"] = HFRunner.ATTN_IMPLEMENTATION
        # BATCH SIZE CHANGES THE ANSWER, so it has to be pinned like the GPU. Measured across
        # batch_size in {1,2,3,5,8} on fixed items: score spread 0.050 (H100), 0.178 (A100),
        # 0.051 (L40S), and the generated STEP TEXT moved in every case. Left padding is the
        # mechanism — a prompt gets a different number of pad tokens depending on its batch, which
        # shifts accumulation order enough to flip greedy near-ties. That is inherent to batched
        # inference and is NOT fixable by kernel selection.
        #
        # It is a REPRODUCIBILITY requirement, not a fairness one: `usable` is computed once per
        # round and every miner is generated over the identical list at the identical batch_size,
        # so no miner's score depends on how many others submitted. But an auditor who re-runs at
        # a different batch_size gets a materially different number, which is exactly the false
        # DIVERGED the audit tool must not produce.
        # Read the default BY NAME. Indexing __defaults__ positionally is precisely the bug that
        # silently disabled the audit's per-slice sample floor earlier (it read min_live_effect
        # where it wanted min_per_slice), and it fails the same way here: silently, with a
        # plausible number.
        import inspect
        out["batch_size"] = inspect.signature(HFRunner.__init__).parameters["batch_size"].default
    except Exception:
        pass
    return out


@dataclass
class CommittedSubmission:
    hotkey: str
    coldkey: str
    tier: str
    ckpt_dir: str
    declared_compute_h100h: float = 0.0
    bond_posted: float = 0.0
    make_runner: object = None
    revealed_hash: str = ""
    salt: str = ""
    committed_value: str = ""
    artifact_uri: str = ""


@dataclass
class ObserverRoundOutcome:
    accepted: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    weights: dict = field(default_factory=dict)
    record: RoundRecord | None = None
    refunds: dict = field(default_factory=dict)
    observer: str = ""
    noise: dict = field(default_factory=dict)
    item_indices: list = field(default_factory=list)
    corpus_spec: str = ""
    identity: dict = field(default_factory=dict)
    # the tier->King map as it stood BEFORE this round scored anything. tournament.consider()
    # mutates in place, so without a snapshot a withheld round's crown survives in memory and the
    # next round writes it on chain — the gate would withhold payment and then pay anyway.
    kings_before: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)


def run_observer_round(
    round_no: int, commit_root: str, round_nonce: str,
    committed: list[CommittedSubmission],
    trajectory_pool, parent, observers: dict,
    tiers: list[Tier], tier_budgets: dict[str, TierBudget],
    tournament: Tournament, ledger: RegistrationLedger, registry: dict,
    parent_id: str = "parent", signer=None,
    max_step_tokens: int = 256, max_cont_tokens: int = 128,
    noise_safety: float = 3.0, canary=None,
    n_items: int = 64, corpus_spec: str = "", prev_anchor: str = "",
) -> ObserverRoundOutcome:
    """`trajectory_pool` is a LARGE pool; which items are scored is drawn from the nonce.

    That argument used to be the scored list itself, which meant the OPERATOR CHOSE THE EXAM —
    the same trust a single-validator subnet has, but invisible because no record showed it. Now
    the selection derives from commit_root+round_nonce and the chosen indices are signed into the
    record, so an auditor re-derives the identical exam.

    `corpus_spec` must identify the pool's ORDERING (source + revision + dedup rule): an index is
    only meaningful against a pinned ordering. `observers`: name -> model; the round's observer is
    also drawn from the nonce. `canary(sub, runner) -> (ok, info)` is the capability tripwire."""
    out = ObserverRoundOutcome()
    out.kings_before = dict(tournament.kings)

    # 1. front door. Nothing untrusted is loaded until economics, safety, the bit budget, the
    #    pinned parent and commit-reveal have all passed.
    subs = []
    d_root: dict = {}
    d_bits: dict = {}
    for c in committed:
        # THE TIER IS MINER-CONTROLLED. `tier_budgets[c.tier]` was a bare lookup, so a submission
        # naming a tier this round does not run — `binary` is a REAL tier, just not one in this
        # round's config — raised KeyError and took the whole round down for everyone. Same shape
        # as the unguarded runner: a miner-supplied value reaching a lookup with no guard is a
        # denial of service that costs the attacker one commitment.
        if c.tier not in tier_budgets:
            out.rejected.append((c.hotkey, [f"tier {c.tier!r} is not being scored this round "
                                            f"(running: {sorted(tier_budgets)})"]))
            continue
        d = intake(c.ckpt_dir, tier_budgets[c.tier], ledger, c.hotkey, c.coldkey,
                   c.bond_posted, revealed_hash=c.revealed_hash, salt=c.salt,
                   committed_value=c.committed_value)
        if not d.accepted:
            out.rejected.append((c.hotkey, d.reasons))
            continue
        d_root[d.content_hash] = getattr(d, "manifest_root", "")
        # carry the MEASURED bit budget forward so intelligence density is derivable from the
        # published record. d.bits is None when the tier declares no bit_tier — in which case the
        # density is genuinely unknown and must read as 0 rather than be guessed from the header.
        d_bits[d.content_hash] = d.bits
        sub = Submission(miner=c.hotkey, tier=c.tier, model_id=d.content_hash,
                         params=d.inspection.params, compute_h100h=c.declared_compute_h100h,
                         coldkey=c.coldkey)
        # ONE MINER'S ARTIFACT MUST NEVER TAKE THE ROUND DOWN. This constructor was called bare,
        # and it is the first code that touches miner-controlled bytes as a MODEL rather than as a
        # file — SafeStudentRunner raises on anything it cannot open. So a single artifact that
        # cleared all six gates and then failed to load unwound the whole round: the GPU was rented,
        # the parent downloaded, and nothing was scored for anybody. That is a denial of service any
        # registered miner can mount for the cost of one commitment, indefinitely.
        #
        # It is also not hypothetical for honest miners: a GGUF passes every gate (the format is
        # explicitly allowlisted and measured) and SafeStudentRunner has no GGUF path at all.
        try:
            runner = c.make_runner()
        except Exception as e:
            out.rejected.append((c.hotkey, [f"artifact passed the gates but could not be loaded: "
                                            f"{type(e).__name__}: {e}"]))
            continue
        subs.append((sub, runner, c.artifact_uri))
        out.accepted.append(c.hotkey)

    # 2. the observer is post-commit entropy, so it cannot be pre-fitted
    obs_name = pick_observer(commit_root, round_nonce, sorted(observers))
    observer = observers[obs_name]
    out.observer = obs_name

    # 2b. WHICH items are scored is post-commit entropy, not an operator choice
    trajectories, item_idx = select_trajectories(trajectory_pool, commit_root, round_nonce,
                                                n_items)
    out.item_indices = list(item_idx)
    out.corpus_spec = corpus_spec

    # 3. shared rollouts — the miner-independent four-fifths of the work, once per round
    shared = build_shared(trajectories, parent, observer, obs_name,
                          max_step_tokens=max_step_tokens, max_cont_tokens=max_cont_tokens)
    usable = [s for s in shared if s.usable]
    if not usable:
        out.events.append({"round": round_no, "action": "abort",
                           "reason": "no usable trajectories (parent moved the observer nowhere)"})
        return out

    # 4. measure the validator's OWN noise floor on this round's inputs. A crown decided inside
    #    it is a lottery a miner wins by resubmitting, so it is measured before any scoring and
    #    published either way.
    probe = usable[0]
    noise = measure_noise(observer, probe.prefix + "\n" + probe.parent_step, probe.continuation,
                          repeats=3)
    out.noise = noise.as_dict()

    # 4b. IDENTITY CANARY — strictly stronger than the noise probe, and it caught a box the noise
    #     probe passed. The parent scored against itself must be exactly 1.0 by construction; an
    #     H100 PCIe returned 0.8754 because generate() is nondeterministic above ~64 new tokens
    #     there, while an A100 and an L40S returned exactly 1.0 on the same code. A validator that
    #     cannot reproduce the identity is unfit to rank anyone, so this ABORTS the round rather
    #     than annotating it.
    id_ok, id_info = identity_check(shared, parent, observer, max_step_tokens=max_step_tokens)
    out.identity = id_info
    if not id_ok:
        out.events.append({"round": round_no, "action": "abort",
                           "reason": "identity check failed: " + id_info.get("verdict", ""),
                           "identity": id_info})
        return out

    # 5. score each submission: one generation + one observer pass per usable sample
    scored: dict[str, Scored] = {}
    by_tier: dict[str, list[Scored]] = {t.name: [] for t in tiers}
    for sub, runner, art_uri in subs:
        registry[sub.model_id] = runner
        ms = score_submission(shared, runner, observer, max_step_tokens=max_step_tokens)
        ok, reasons = True, list(ms.reasons)
        if canary is not None and ms.score > 0:
            c_ok, c_info = canary(sub, runner)
            if not c_ok:
                ok = False
                reasons.append(f"capability canary: {c_info}")
        s = Scored(sub=sub, retention=ms.score, retention_lb=ms.score,
                   per_point=[v for vs in ms.slice_samples.values() for v in vs],
                   gates_ok=ok and ms.score > 0 and not ms.reasons,
                   reasons=reasons, per_axis=dict(ms.slice_samples))
        s.artifact_uri = art_uri
        s.manifest_root = d_root.get(sub.model_id, "")
        _b = d_bits.get(sub.model_id)
        s.code_bits = float(getattr(_b, "code_bits", 0.0) or 0.0)
        s.container_bits = float(getattr(_b, "container_bits", 0.0) or 0.0)
        s.steps = _freeze(runner, usable, max_step_tokens)
        s.effects = _effects(ms)
        scored[sub.model_id] = s
        by_tier.setdefault(sub.tier, []).append(s)
        out.scores[sub.model_id] = ms.as_dict()

    # 6. crown per tier, re-scoring AND re-gating the incumbent on the same samples
    for t in tiers:
        king = tournament.kings.get(t.name)
        king_scored = None
        if king is not None:
            kr = registry.get(king.model_id)
            if kr is None:
                out.events.append({"tier": t.name, "round": round_no, "action": "hold",
                                   "king": king.model_id, "reason": "king unavailable"})
                tournament.kings[t.name].reign += 1
                continue
            kms = score_submission(shared, kr, observer, max_step_tokens=max_step_tokens)
            if kms.score <= MIN_CROWN_LB or kms.reasons:
                out.events.append({"tier": t.name, "round": round_no, "action": "vacate",
                                   "king": king.model_id, "score": round(kms.score, 5),
                                   "reasons": kms.reasons or [f"score <= {MIN_CROWN_LB}"]})
                del tournament.kings[t.name]
            else:
                king_scored = Scored(
                    sub=Submission(king.miner, t.name, king.model_id, 0, 0.0),
                    retention=kms.score, retention_lb=kms.score,
                    per_point=[v for vs in kms.slice_samples.values() for v in vs],
                    gates_ok=True, per_axis=dict(kms.slice_samples))
                # PUBLISH the incumbent's re-score. The dethrone margin is a PAIRED statistic; a
                # record holding only challengers describes one side of a comparison and leaves
                # margin_lcb unrecomputable. Registered under a distinct key so it never collides
                # with the same model submitted as a challenger.
                king_scored.role = "incumbent"
                # carry the crowned artifact's locator onto the re-score, or L3 cannot fetch the
                # incumbent and the paired half of every dethrone stays unverifiable
                king_scored.artifact_uri = getattr(king, "artifact_uri", "")
                king_scored.manifest_root = getattr(king, "manifest_root", "")
                king_scored.effects = _effects(kms)
                # the incumbent's steps must be frozen too, or L2 can re-derive only one side of
                # a paired comparison — and the dethrone margin is the number that moves emission
                king_scored.steps = _freeze(kr, usable, max_step_tokens)
                scored[f"{king.model_id}#incumbent"] = king_scored
        ev = tournament.consider(t.name, by_tier.get(t.name, []), king_scored)
        # A crown has to remember WHERE ITS BYTES ARE. Tournament.consider() mints a King from the
        # score alone, so the locator was dropped at exactly the moment it started mattering —
        # leaving "every crown ships a downloadable model" with no path from king to bytes.
        nk = tournament.kings.get(t.name)
        if nk is not None and ev.get("action") in ("crown", "dethrone"):
            src = scored.get(nk.model_id)
            if src is not None:
                nk.artifact_uri = getattr(src, "artifact_uri", "")
                nk.manifest_root = getattr(src, "manifest_root", "")
        # NOISE GATE: a dethrone whose margin sits inside the measured floor is not a result.
        if ev.get("action") == "dethrone" and king_scored is not None:
            ok_margin, why = crownable(ev.get("margin_lcb", 0.0), noise, noise_safety)
            if not ok_margin:
                tournament.kings[t.name] = king          # roll the dethrone back
                ev = {"tier": t.name, "round": round_no, "action": "hold",
                      "king": king.model_id, "reason": why}
        out.events.append(ev)

    # 7. bonds, signed record, weights
    for mid, s in scored.items():
        refund = ledger.settle(s.sub.miner, s.sub.coldkey, s.retention)
        if refund:
            out.refunds[s.sub.miner] = refund
    out.weights = tournament.weights()
    # POINTS: every usable sample, not a truncated 64, and each carries the FROZEN parent step
    # and continuation C. That is what turns a re-run into a pure forward pass: the auditor does
    # not have to reproduce batched greedy generation (the noisiest part of the round, and the
    # part the zero-noise measurement never covered) — only the observer's distributions over
    # text the record hands it.
    pts = [{"rollout_id": s.traj_id, "k": 0, "mode": "observer_kl",
            "prefix_sha256": _sha(s.prefix), "parent_step": s.parent_step,
            "continuation": s.continuation, "d_parent": round(s.d_parent, 6)}
           for s in usable]
    manifest = {
        "corpus_spec": corpus_spec,
        # The pool's digest lives in the SIGNED body. corpus_spec names the sources; this pins the
        # exact bytes, so an operator cannot score against one pool and hand an auditor another in
        # which the drawn indices happen to land on easier items.
        "pool_sha256": _pool_sha(trajectory_pool),
        "item_indices": list(out.item_indices),
        "n_items_requested": n_items,
        "observer": obs_name,
        "observer_pool": sorted(observers),
        "parent": parent_id,
        "max_step_tokens": max_step_tokens,
        "max_cont_tokens": max_cont_tokens,
        "noise_safety": noise_safety,
        "identity": out.identity,
        "frozen_rollouts": True,
        "versions": _versions(),
    }
    out.record = build_round_record(round_no, commit_root, round_nonce, parent_id,
                                    f"observer:{obs_name}", "unconditioned",
                                    f"trajectories:{len(usable)}", pts, scored, out.events,
                                    out.weights, manifest=manifest, noise=out.noise,
                                    safety=noise_safety, prev_anchor=prev_anchor)
    if signer is not None:
        out.record.sign(signer)
    return out
