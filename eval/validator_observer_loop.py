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

from .determinism import crownable, measure_noise
from .economics import RegistrationLedger
from .gates import TierBudget, degeneracy_flags
from .intake import intake
from .koth import MIN_CROWN_LB, Scored, Submission, Tier, Tournament
from .observer_round import build_shared, pick_observer, score_submission
from .round_record import RoundRecord, build_round_record


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


@dataclass
class ObserverRoundOutcome:
    accepted: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    weights: dict = field(default_factory=dict)
    record: RoundRecord | None = None
    refunds: dict = field(default_factory=dict)
    observer: str = ""
    noise: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)


def run_observer_round(
    round_no: int, commit_root: str, round_nonce: str,
    committed: list[CommittedSubmission],
    trajectories, parent, observers: dict,
    tiers: list[Tier], tier_budgets: dict[str, TierBudget],
    tournament: Tournament, ledger: RegistrationLedger, registry: dict,
    parent_id: str = "parent", signer=None,
    max_step_tokens: int = 256, max_cont_tokens: int = 128,
    noise_safety: float = 3.0, canary=None,
) -> ObserverRoundOutcome:
    """`observers`: name -> observer model. The round's observer is DRAWN FROM THE NONCE, so a
    miner cannot know which one it will face; `canary(sub, runner) -> (ok, info)` is the optional
    capability tripwire."""
    out = ObserverRoundOutcome()

    # 1. front door. Nothing untrusted is loaded until economics, safety, the bit budget, the
    #    pinned parent and commit-reveal have all passed.
    subs = []
    for c in committed:
        d = intake(c.ckpt_dir, tier_budgets[c.tier], ledger, c.hotkey, c.coldkey,
                   c.bond_posted, revealed_hash=c.revealed_hash, salt=c.salt,
                   committed_value=c.committed_value)
        if not d.accepted:
            out.rejected.append((c.hotkey, d.reasons))
            continue
        sub = Submission(miner=c.hotkey, tier=c.tier, model_id=d.content_hash,
                         params=d.inspection.params, compute_h100h=c.declared_compute_h100h,
                         coldkey=c.coldkey)
        subs.append((sub, c.make_runner()))
        out.accepted.append(c.hotkey)

    # 2. the observer is post-commit entropy, so it cannot be pre-fitted
    obs_name = pick_observer(commit_root, round_nonce, sorted(observers))
    observer = observers[obs_name]
    out.observer = obs_name

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

    # 5. score each submission: one generation + one observer pass per usable sample
    scored: dict[str, Scored] = {}
    by_tier: dict[str, list[Scored]] = {t.name: [] for t in tiers}
    for sub, runner in subs:
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
        ev = tournament.consider(t.name, by_tier.get(t.name, []), king_scored)
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
    pts = [{"rollout_id": s.traj_id, "k": s.index if hasattr(s, "index") else 0,
            "mode": "observer_kl"} for s in usable[:64]]
    out.record = build_round_record(round_no, commit_root, round_nonce, parent_id,
                                    f"observer:{obs_name}", "unconditioned",
                                    f"trajectories:{len(usable)}", pts, scored, out.events,
                                    out.weights)
    if signer is not None:
        out.record.sign(signer)
    return out
