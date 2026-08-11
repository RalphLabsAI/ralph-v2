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
from .progress import tick as _tick
from .round_record import RoundRecord, build_round_record
from .runners import continuation


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


def _determinism_drift(a, b) -> float:
    """Largest per-slice difference between two scorings of the SAME bytes. 0.0 means reproduced.

    Compares SLICES, not just the headline score. Two runs can agree on the aggregate while
    disagreeing underneath — and since the aggregate is a worst-slice minimum, a slice that moved
    without changing the minimum this time can change it next time. The point is to catch the
    instability, not only the occasion it happened to matter."""
    sa, sb = dict(getattr(a, "slice_samples", {}) or {}), dict(getattr(b, "slice_samples", {}) or {})
    worst = abs(float(getattr(a, "score", 0.0)) - float(getattr(b, "score", 0.0)))
    for k in set(sa) | set(sb):
        va, vb = sa.get(k) or [], sb.get(k) or []
        if len(va) != len(vb):
            return float("inf")          # a different sample count is not a drift, it is a break
        for x, y in zip(va, vb):
            worst = max(worst, abs(float(x) - float(y)))
    return worst


def _close_runner(runner) -> None:
    """Release a student's backend if it has one. Never raises: cleanup that throws would convert
    a scored submission into a rejected one, which is the failure this exists to prevent."""
    try:
        close = getattr(runner, "close", None)
        if callable(close):
            close()
    except Exception:
        pass


def _freeze(ms, runner, usable, max_step_tokens: int) -> list:
    """The model's steps as literal text for the record, so an audit is a pure forward pass over
    recorded strings rather than a reproduction of greedy generation — the noisiest part of the
    round.

    TAKEN FROM THE SCORED RUN, NOT REGENERATED. This used to call `generate()` a second time over
    the identical prompts, which doubled the most expensive leg of the round: measured in flight, a
    ternary submission spent ~17 s per prompt and did all 72 twice. Worse than the cost, the record
    then held text from a DIFFERENT generation than the one that produced the score — and greedy
    decode is only reproducible while nothing shifts. The scored steps are the honest ones.

    The fallback regenerates only when a runner predates the change and returns no steps."""
    steps = list(getattr(ms, "steps", None) or [])
    if steps:
        return steps
    try:
        # chat=False to match observer_round's scoring legs. A freeze recorded under a different
        # prompt than the score would be a record of something that never happened.
        return continuation(runner, [x.prefix for x in usable], max_step_tokens)
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
        # THE EXAM'S CONTEXT CEILING AND THE RUNTIME THAT SETS IT. Both change which items are
        # scored, so both are part of the measurement. Omitting them would make an auditor who
        # re-derives the selection pick a different 72 items and report a false DIVERGED.
        from .pool import MAX_PREFIX_TOKENS, PREFIX_TOKENIZER, PREFIX_TRUNCATION_SIDE
        from .runners import GGUFStudentRunner
        out["max_prefix_tokens"] = MAX_PREFIX_TOKENS
        out["prefix_tokenizer"] = PREFIX_TOKENIZER
        out["prefix_truncation_side"] = PREFIX_TRUNCATION_SIDE
        # The student's window is part of the measurement too: llama.cpp silently trades prompt
        # length against `max_tokens`, so this decides how many step tokens a miner actually got.
        out["student_n_ctx"] = GGUFStudentRunner.N_CTX
        out["student_n_threads"] = GGUFStudentRunner.N_THREADS
        out["student_n_gpu_layers"] = GGUFStudentRunner.N_GPU_LAYERS
        # WHERE THE SUBMISSIONS RAN. The parent's device is pinned and an absent one is fatal;
        # the students' was invisible. llama.cpp on CPU and on CUDA do not emit identical tokens,
        # so this is part of the measurement, and it is read from llama.cpp's capability flag
        # rather than from config — the CPU-only wheel accepts `n_gpu_layers` and ignores it.
        out["student_backend"] = GGUFStudentRunner.backend()
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
    # share of emission with no king this round; burned rather than given to the
    # kings that exist, so an unentered hard tier never subsidises an easy one
    unclaimed: float = 0.0
    record: RoundRecord | None = None
    refunds: dict = field(default_factory=dict)
    observer: str = ""
    noise: dict = field(default_factory=dict)
    item_indices: list = field(default_factory=list)
    corpus_spec: str = ""
    identity: dict = field(default_factory=dict)
    # What re-scoring ONE submission twice on this box produced. `identity` checks the PARENT
    # through the HF/torch path and says nothing about llama.cpp, which is the path that actually
    # decides crowns — round 2 scored one file at 0.3030 and 0.2875 with `identity` reporting a
    # perfect 1.000 the whole time.
    determinism: dict = field(default_factory=dict)
    # the tier->King map as it stood BEFORE this round scored anything. tournament.consider()
    # mutates in place, so without a snapshot a withheld round's crown survives in memory and the
    # next round writes it on chain — the gate would withhold payment and then pay anyway.
    kings_before: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)


def c_hotkey_of(sub) -> str:
    """The miner behind a Submission, for rejection records."""
    return getattr(sub, "miner", "") or ""


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
    references: list | None = None,
) -> ObserverRoundOutcome:
    """`trajectory_pool` is a LARGE pool; which items are scored is drawn from the nonce.

    That argument used to be the scored list itself, which meant the OPERATOR CHOSE THE EXAM —
    the same trust a single-validator subnet has, but invisible because no record showed it. Now
    the selection derives from commit_root+round_nonce and the chosen indices are signed into the
    record, so an auditor re-derives the identical exam.

    `corpus_spec` must identify the pool's ORDERING (source + revision + dedup rule): an index is
    only meaningful against a pinned ordering. `observers`: name -> model; the round's observer is
    also drawn from the nonce. `canary(sub, runner) -> (ok, info)` is the capability tripwire.

    `references` are `(name, tier, runner, artifact_uri)` — artifacts scored on this round's exam
    that CANNOT WIN ANYTHING. A stock `llama-quantize` of the parent, or a competitor's published
    model. They exist because a retention of 0.30 is meaningless without a fixed point: nobody,
    including us, could say whether the crowns beat what `llama-quantize` gives away for free.
    Putting the answer IN THE SIGNED RECORD makes it a property an auditor re-derives rather than a
    claim we assert, and re-measuring it every round turns scoring drift into something visible.

    They are not miners: no hotkey, no commit-reveal, no bond. So they are scored and recorded and
    then kept out of `by_tier`, which is the single place a crown can come from — a reference can
    never be crowned, never enter the weight vector, and never dethrone anyone."""
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
            _tick("rejected", f"{c.hotkey[:12]}… tier {c.tier!r} not scored this round", force=True)
            continue
        # intake re-hashes the whole artifact to check commit-reveal; on a multi-GB checkpoint
        # that is minutes of pure I/O with nothing else happening.
        _tick("intake", f"{c.hotkey[:12]}… tier={c.tier}", force=True)
        d = intake(c.ckpt_dir, tier_budgets[c.tier], ledger, c.hotkey, c.coldkey,
                   c.bond_posted, revealed_hash=c.revealed_hash, salt=c.salt,
                   committed_value=c.committed_value)
        if not d.accepted:
            out.rejected.append((c.hotkey, d.reasons))
            _tick("rejected", f"{c.hotkey[:12]}… {'; '.join(d.reasons)[:120]}", force=True)
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
            _tick("rejected", f"{c.hotkey[:12]}… could not load: {type(e).__name__}", force=True)
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
    _tick("shared rollouts", f"{len(trajectories)} items, observer {obs_name}", force=True)
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
    _tick("noise floor", f"{len(usable)} usable samples", force=True)
    noise = measure_noise(observer, probe.prefix + "\n" + probe.parent_step, probe.continuation,
                          repeats=3)
    out.noise = noise.as_dict()

    # 4b. IDENTITY CANARY — strictly stronger than the noise probe, and it caught a box the noise
    #     probe passed. The parent scored against itself must be exactly 1.0 by construction; an
    #     H100 PCIe returned 0.8754 because generate() is nondeterministic above ~64 new tokens
    #     there, while an A100 and an L40S returned exactly 1.0 on the same code. A validator that
    #     cannot reproduce the identity is unfit to rank anyone, so this ABORTS the round rather
    #     than annotating it.
    _tick("identity canary", force=True)
    id_ok, id_info = identity_check(shared, parent, observer, max_step_tokens=max_step_tokens)
    out.identity = id_info
    if not id_ok:
        out.events.append({"round": round_no, "action": "abort",
                           "reason": "identity check failed: " + id_info.get("verdict", ""),
                           "identity": id_info})
        return out

    # 5. score each submission: one generation + one observer pass per usable sample
    #
    # THE COUNT GAP MUST BE VISIBLE. Intake ticked 11 accepted artifacts and the scoring loop then
    # announced `[1/10]`, and nothing in between said where the eleventh went. The reason WAS
    # recorded against that miner in `out.rejected`, so the record is honest — but the record only
    # exists after publish, which means for the whole expensive leg the operator is watching a
    # number they cannot account for. Every rejection now ticks, and the two totals are stated
    # together here so the arithmetic closes in the log itself.
    _tick("scoring", f"{len(subs)} of {len(committed)} committed "
                     f"({len(out.rejected)} rejected)", force=True)
    scored: dict[str, Scored] = {}
    by_tier: dict[str, list[Scored]] = {t.name: [] for t in tiers}
    # RELEASE EACH CHALLENGER'S WEIGHTS AS SOON AS NOTHING NEEDS THEM. Reported by a miner on
    # 2026-08-07 and confirmed: every challenger's runner stayed resident for the whole round, so
    # by the tenth submission nine llama.cpp contexts were holding ~45 GB of weights, KV and
    # compute buffer on an 80 GB card, beside the bf16 parent and observer. The tenth load failed
    # with a bare `ValueError: Failed to load model from file`, which reads exactly like a corrupt
    # artifact and was not one — the miner reproduced a clean load of the same bytes on CPU.
    #
    # `registry` STILL RECEIVES EVERY CHALLENGER. It is a cross-round cache: a long-lived caller
    # runs several rounds against one dict, and step 6 looks the incumbent up in it. Evicting
    # entries here would turn a re-score into a silent `king unavailable` hold. What changes is
    # that the runner's WEIGHTS are dropped — `close()` clears the cached handle and `_load()` is
    # lazy, so the object stays valid and transparently reloads if a later round needs it. Cache
    # eviction, not destruction.
    keep_loaded = {k.model_id for k in tournament.kings.values()}

    # THE STUDENT DETERMINISM CANARY, and which submission it runs on is the whole design.
    #
    # Round 2 scored ONE file twice — as a challenger and as the re-scored incumbent — and got
    # 0.3030 and 0.2875. Four of five slices were bit-identical; the fifth differed, and because
    # the score is the WORST slice, that one unstable slice WAS the score. Worst-slice aggregation
    # systematically selects the slice most exposed to nondeterminism: the worst slice is where the
    # model is struggling, and struggling generations sit on near-tied token probabilities where a
    # floating-point wobble flips a token. The 0.0155 spread is about a third of the 0.05 dethrone
    # margin. Nothing caught it — `identity_check` scores the PARENT against itself through the
    # HF/torch path and never touches the llama.cpp student path at all.
    #
    # It runs on the LARGEST submission, not the cheapest. The observed failure was on the biggest
    # model in the round, which is the one nearest the point where llama.cpp stops offloading every
    # layer; a canary on a small model would pass while the thing it exists to catch goes by. A
    # canary that tests the easy case is decoration.
    #
    # It costs ONE extra pass, not two: the first result is kept as that submission's real score.
    probe_id = ""
    if subs:
        def _bulk(t):
            b = d_bits.get(t[0].model_id)
            return float(getattr(b, "container_bits", 0.0) or 0.0) * (t[0].params or 1)
        probe_id = max(subs, key=_bulk)[0].model_id

    for n, (sub, runner, art_uri) in enumerate(subs, 1):
        _tick("submission", f"[{n}/{len(subs)}] {sub.miner[:12]}… tier={sub.tier}", force=True)
        registry[sub.model_id] = runner
        # ONE MINER'S MODEL MUST NEVER TAKE THE ROUND DOWN — the third time this shape has bitten:
        # a miner-controlled tier hit a bare dict lookup, an artifact that cleared every gate failed
        # to load, and now a runtime raised mid-generation. Whatever the cause, the blast radius has
        # to be one submission. The reason is recorded against that miner rather than swallowed.
        try:
            ms = score_submission(shared, runner, observer, max_step_tokens=max_step_tokens)
            if sub.model_id == probe_id:
                _tick("determinism canary", f"re-scoring {sub.miner[:12]}… to check the student "
                                            f"path reproduces", force=True)
                again = score_submission(shared, runner, observer,
                                         max_step_tokens=max_step_tokens)
                drift = _determinism_drift(ms, again)
                out.determinism = {"model_id": sub.model_id, "first": round(ms.score, 6),
                                   "second": round(again.score, 6),
                                   "max_slice_drift": round(drift, 6), "reproduced": drift == 0.0}
                if drift != 0.0:
                    out.events.append({
                        "round": round_no, "action": "abort",
                        "reason": f"the student path is not reproducible: scoring "
                                  f"{sub.model_id[:12]}… twice on this box gave "
                                  f"{ms.score:.6f} and {again.score:.6f} (worst slice drifted "
                                  f"{drift:.6f}). A crown decided inside that drift is a lottery.",
                        "determinism": out.determinism})
                    return out
        except Exception as e:
            out.rejected.append((c_hotkey_of(sub), [f"scoring raised {type(e).__name__}: "
                                                    f"{str(e)[:200]}"]))
            # A submission that FAILED may still have loaded a model before it raised, and the
            # failure path is exactly when the card is already under pressure.
            if sub.model_id not in keep_loaded:
                _close_runner(runner)
            continue
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
        s.steps = _freeze(ms, runner, usable, max_step_tokens)
        # IS THIS OUTPUT LANGUAGE AT ALL. `degeneracy_flags` was imported by this module and never
        # called — `round_engine` and `axis_round` both gate on it, the live v2 path did not. So
        # round 2's ternary tier took four entries, three of them token soup, and only one miner
        # uploading a working model kept `retention_lb > 0.02` from crowning a broken quantiser.
        # Retention cannot refuse that on its own: soup still shifts an observer's distribution a
        # little, and a little is all the floor ever asked for.
        deg_ok, dflags = degeneracy_flags(s.steps)
        if not deg_ok:
            s.gates_ok = False
            s.reasons.extend(dflags)
        s.effects = _effects(ms)
        # PROVISIONAL, and it says so. The signed record is still the only authority — this number
        # has not been audited, signed or anchored, and the round can still be WITHHELD after it is
        # emitted. But withholding it from the operator for the two hours the round takes meant the
        # dashboard read as "nothing is happening" while the GPU worked, which is its own kind of
        # dishonesty. The consumer is required to label it; see status.py.
        _tick("retention", f"{sub.miner[:12]}… tier={sub.tier} retention={ms.score:.4f} "
                           f"gates={'ok' if s.gates_ok else 'REJECTED'}", force=True)
        scored[sub.model_id] = s
        by_tier.setdefault(sub.tier, []).append(s)
        out.scores[sub.model_id] = ms.as_dict()
        # EVERYTHING THAT NEEDED THIS RUNNER HAS RUN: scoring, the capability canary, and the
        # freeze. Release it unless it is a reigning king's, and never let cleanup raise — a
        # failed close would turn a scored submission into a rejected one.
        if sub.model_id not in keep_loaded:
            _close_runner(runner)

    # 5b. THE FIXED POINTS. Scored on the same exam, recorded beside the miners, and structurally
    #     unable to win: they never enter `by_tier`, which is the only list `Tournament.consider`
    #     reads, so there is no path from a reference to a crown or to a weight. A reference that
    #     fails to load is a NOTE, never a round failure — it is our artifact, not a miner's, and
    #     losing our yardstick must not cost eleven people their round.
    for ref_name, ref_tier, ref_runner, ref_uri in (references or []):
        _tick("reference", f"{ref_name} tier={ref_tier}", force=True)
        try:
            rms = score_submission(shared, ref_runner, observer, max_step_tokens=max_step_tokens)
        except Exception as e:
            out.events.append({"round": round_no, "action": "reference_failed", "tier": ref_tier,
                               "name": ref_name,
                               "reason": f"{type(e).__name__}: {str(e)[:200]}"})
            _close_runner(ref_runner)
            continue
        rid = f"reference:{ref_name}"
        rs = Scored(sub=Submission(rid, ref_tier, rid, 0, 0.0),
                    retention=rms.score, retention_lb=rms.score,
                    per_point=[v for vs in rms.slice_samples.values() for v in vs],
                    gates_ok=True, per_axis=dict(rms.slice_samples))
        rs.role = "reference"
        rs.artifact_uri = ref_uri
        rs.steps = _freeze(rms, ref_runner, usable, max_step_tokens)
        rs.effects = _effects(rms)
        scored[rid] = rs          # into the RECORD...
        # ...and deliberately NOT into by_tier. That omission is the whole safety property.
        out.scores[rid] = rms.as_dict()
        _tick("reference", f"{ref_name} retention={rms.score:.4f}", force=True)
        _close_runner(ref_runner)

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
            _tick("incumbent re-score", f"tier {t.name} king {king.model_id[:12]}…", force=True)
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
                king_scored.steps = _freeze(kms, kr, usable, max_step_tokens)
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
    out.unclaimed = tournament.unclaimed()
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
        "determinism": out.determinism,
        "frozen_rollouts": True,
        "versions": _versions(),
    }
    out.record = build_round_record(round_no, commit_root, round_nonce, parent_id,
                                    f"observer:{obs_name}", "unconditioned",
                                    f"trajectories:{len(usable)}", pts, scored, out.events,
                                    out.weights, manifest=manifest, noise=out.noise,
                                    safety=noise_safety, prev_anchor=prev_anchor,
                                    unclaimed=out.unclaimed,
                                    # so a dropped miner learns why from the SIGNED record rather
                                    # than from a file on a box they cannot see
                                    rejected=out.rejected)
    if signer is not None:
        out.record.sign(signer)
    return out
