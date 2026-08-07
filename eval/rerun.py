"""Re-run a signed round record and say, out loud, whether it holds up.

WHY THIS EXISTS. Publishing a record makes a round TRANSPARENT; being able to re-run it makes the
round REPRODUCIBLE, and only the second one constrains the operator. The single-validator subnet
we studied publishes a great deal — every prompt, every judge verdict, every score — and it is
still unfalsifiable where it counts, because the verdicts come from an unpinned LLM with
`allow_fallbacks` and no seed. An outsider can prove they added the numbers up wrong. Nobody can
ever prove they GRADED wrong. That asymmetry is the thing to avoid, not the single validator.

So this tool audits in three levels, and it is explicit about which ones it actually ran:

  L0  ARITHMETIC — free, no models, no corpus. Verifies the signature, then recomputes the whole
      score from the raw measurements the record publishes: sample scores from (s, d_G, d_A),
      per-slice means, the worst-slice aggregate, the paired dethrone margin, the noise gate, and
      the emission weights. Catches a validator that publishes honest KLs and a fabricated
      aggregate, or a crown whose margin never cleared its own noise floor.

  L1  SELECTION — needs the corpus. Re-derives WHICH items were scored from commit_root‖nonce and
      checks they match what the record claims, then checks each recorded prefix hash against the
      pinned pool. This is what makes "the operator did not choose the exam" checkable.

  L2  JUDGMENT — needs the corpus and an observer model. Recomputes the observer's distributions
      over the frozen text and re-derives (s, d_G, d_A) from scratch. This is the level a
      judge-based subnet cannot have at all.

      BUT NOTE EXACTLY WHAT IT PROVES, because the first version of this file overclaimed it. L2
      verifies steps -> effects. It does NOT verify steps -> model. The miner's steps are FROZEN
      into the record by the same operator who signs it, so an operator can write whatever steps
      they like — ideal ones, or a competitor's — and L2 will faithfully confirm that those strings
      produce the recorded numbers. A perfect forged score reproduces at L0, L1 and L2 together.
      Freezing the steps removes generation nondeterminism from the audit; it does not bind the
      steps to the checkpoint.

  L3  PROVENANCE — needs the artifacts. Loads each scored checkpoint and RE-GENERATES its steps on
      the recorded prefixes, comparing them to the frozen text. This is the level that binds the
      record to the models, and it is the only one that makes a crown non-forgeable by the
      operator. It is expensive (a checkpoint download and a generation pass per submission),
      which is why the cheaper levels exist — not a reason to skip it and call the round verified.

FAILING LOUDLY IS A FEATURE, not politeness. Ralph v1's CPU auditor ran for weeks reporting
success while diverging from the validator on 37 of 40 reports, because it wrote through a logging
handler that a library's init had already removed, and because it exited 0 regardless. So: this
writes to stdout directly, never through `logging`; it exits non-zero on failure; and an audit that
SKIPPED the science exits 2 (INCOMPLETE) rather than 0. You cannot get a green light here by
running the cheap half.

    python -m eval.rerun record.json                          # L0 only -> exits 2, INCOMPLETE
    python -m eval.rerun record.json --pool items.jsonl       # L0+L1
    python -m eval.rerun record.json --pool items.jsonl --observer Qwen/Qwen2.5-0.5B-Instruct
    python -m eval.rerun record.json --pool i.jsonl --observer <hf> --artifacts ./ckpts   # L0-L3
    python -m eval.rerun --history ./published --head <anchor committed on chain>
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass
class Check:
    level: str
    name: str
    status: str
    detail: str = ""
    # a SKIPPED check that is not required (e.g. no dethrone happened this round) must not make
    # the audit incomplete; a skipped check that WAS required must.
    required: bool = True


@dataclass
class Audit:
    checks: list = field(default_factory=list)

    def add(self, level, name, status, detail="", required=True):
        self.checks.append(Check(level, name, status, detail, required))
        return status

    def ok(self, level, name, cond, fail="", info="", required=True):
        """`fail` explains a FAILURE; `info` is shown either way. Keeping them separate matters
        for readability: a passing check printed alongside the reason it would have failed reads
        exactly like a failure, and an audit report nobody can skim is an audit nobody reads."""
        return self.add(level, name, PASS if cond else FAIL,
                        info if cond else (fail or info), required)

    @property
    def failed(self) -> list:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def skipped(self) -> list:
        return [c for c in self.checks if c.status == SKIP and c.required]

    @property
    def exit_code(self) -> int:
        if self.failed:
            return 1
        return 2 if self.skipped else 0

    @property
    def verdict(self) -> str:
        return {0: "REPRODUCED", 1: "DIVERGED", 2: "INCOMPLETE"}[self.exit_code]


def record_from_blob(blob: bytes):
    """Bytes -> RoundRecord. The auditor daemon fetches records from a sink and never touches the
    filesystem, so parsing lives here rather than inside the file loader."""
    from .round_record import RoundRecord, SubmissionRecord
    raw = json.loads(blob.decode() if isinstance(blob, bytes) else blob)
    subs = [SubmissionRecord(**s) for s in raw.pop("submissions", [])]
    return RoundRecord(**{**raw, "submissions": subs})


def load_record(path: str):
    with open(path, "rb") as fh:
        return record_from_blob(fh.read())


def _obs_key(name: str) -> str:
    """`HuggingFaceTB/SmolLM2-1.7B-Instruct` and `SmolLM2-1.7B-Instruct` are the same observer.

    The round's registry may be keyed either way; an auditor must name the full repo id to download
    it. Comparing the raw strings made the check fail for every third party and pass only for the
    operator whose registry produced the manifest."""
    return (name or "").strip().split("/")[-1].lower()


# ---------------------------------------------------------------- L0: arithmetic

def audit_arithmetic(rec, a: Audit) -> None:
    """Everything recomputable from the record alone. No models, no corpus, no network."""
    from .koth import MIN_CROWN_LB, Scored, Submission, softmin_lcb_diff
    from .observer_kl import StepEffect, sample_score

    # The signature is what makes the record ATTRIBUTABLE. An unsigned record is a text file:
    # self-consistent, and mintable by anyone with an editor.
    if not (rec.signature and rec.signer):
        a.add("L0", "signature present", FAIL, "record is unsigned — attributable to nobody")
    else:
        try:
            a.ok("L0", "signature valid", rec.verify_signature(),
                 fail="signature does not verify over the canonical payload",
                 info=f"{rec.sig_scheme} by {rec.signer[:16]}… (attributes the record; a "
                      f"valid signature does NOT make its contents true)")
        except Exception as e:
            a.add("L0", "signature valid", FAIL, f"verify raised {type(e).__name__}: {e}")

    a.ok("L0", "manifest present", bool(rec.manifest),
         fail="no re-run manifest — corpus/observer/versions unpinned")
    a.ok("L0", "noise floor published", bool(rec.noise),
         fail="the crown was gated on a number the record does not contain",
         info=f"max_kl={rec.noise.get('max_kl')} tol={rec.reproduction_tolerance:.2e}"
              if rec.noise else "")

    n_pts = len(rec.points)
    a.ok("L0", "points present", n_pts > 0, info=f"{n_pts} points")
    a.ok("L0", "rollouts frozen",
         bool(rec.points) and all(p.get("parent_step") and p.get("continuation")
                                  for p in rec.points),
         fail="some points lack frozen parent_step/continuation — a re-run would have to "
              "reproduce batched greedy generation, which is not reproducible")

    scored_by_id = {}
    for s in rec.submissions:
        tag = f"{s.model_id[:12]}…/{s.role}"
        if not s.effects:
            a.add("L0", f"recompute score {tag}", SKIP, "no per-sample effects recorded")
            continue
        if len(s.effects) != n_pts or (s.steps and len(s.steps) != n_pts):
            a.add("L0", f"alignment {tag}", FAIL,
                  f"effects={len(s.effects)} steps={len(s.steps)} vs points={n_pts}")
        # THE ARITHMETIC CHECK: rebuild the samples and call THE PRODUCTION SCORER.
        #
        # This used to re-implement score_miner inline — group by slice, mean, take the min — and
        # got it wrong in three ways at once. It read min_per_slice as score_miner.__defaults__[2],
        # which is min_live_effect (0.02), so the per-slice sample floor became `len(v) >= 0.02` and
        # was disabled entirely. It skipped the LIVENESS gate, so a model whose steps move the
        # observer nowhere recomputed to a healthy score while production would have returned 0 and
        # refused to crown it. And it skipped every other early-return, so any future guard added to
        # score_miner would be silently absent from the audit.
        #
        # Reimplementing the scorer in the auditor is the mistake, not the details: two copies of a
        # rule diverge, and the copy that decides "did this reproduce" is the one nobody runs in
        # anger. So the audit now calls the same function the round did.
        from .observer_kl import StepEffect, score_miner
        samples = []
        for row in s.effects:
            key, sv, d_g, d_m = row[0], float(row[1]), float(row[2]), float(row[3])
            samples.append((key, StepEffect(s=sv, d_teacher=d_g, d_miner=d_m, n_positions=1)))
        ms = score_miner(samples)
        tol = max(float(rec.reproduction_tolerance or 0.0), 1e-6)
        a.ok("L0", f"recompute score {tag}", abs(ms.score - s.retention) <= tol,
             fail=f"recomputed {ms.score:.6f} vs recorded {s.retention:.6f} "
                  f"(tol {tol:.2e}){'; ' + '; '.join(ms.reasons) if ms.reasons else ''}",
             info=f"recomputed {ms.score:.6f} vs recorded {s.retention:.6f} (tol {tol:.2e})")
        # retention_lb is the field MIN_CROWN_LB is read from, i.e. the crownability floor, and the
        # audit never recomputed it at all — so a record could carry an honest `retention` and an
        # inflated `retention_lb` and be crowned on a number nobody checked.
        a.ok("L0", f"recompute crown floor {tag}", abs(ms.score - s.retention_lb) <= tol,
             fail=f"retention_lb {s.retention_lb:.6f} does not follow from the measurements "
                  f"({ms.score:.6f}) — this is the field the crown floor is read from")
        kept = {k: v for k, v in ms.slice_samples.items()}
        if not kept:
            a.add("L0", f"paired vectors match effects {tag}", SKIP,
                  "; ".join(ms.reasons) or "no slice reached the sample floor")
            continue
        # the published paired vectors must be the ones the aggregate came from
        if s.slices:
            same = set(s.slices) == set(kept) and all(
                len(s.slices[k]) == len(kept[k]) and
                all(abs(x - y) <= 1e-4 for x, y in zip(s.slices[k], kept[k]))
                for k in kept)
            a.ok("L0", f"paired vectors match effects {tag}", same,
                 fail="recorded slices disagree with the effects they should derive from")
        scored_by_id[f"{s.model_id}|{s.role}"] = s

    # DETHRONE MARGINS. The dethrone statistic is paired, so it is recomputable only because the
    # incumbent's re-score is published too. This is the check that says a crown was earned.
    dethrones = [e for e in rec.events if e.get("action") == "dethrone"]
    if not dethrones:
        a.add("L0", "dethrone margin", SKIP, "no dethrone this round", required=False)
    for e in dethrones:
        ch = scored_by_id.get(f"{e.get('king')}|challenger")
        inc = scored_by_id.get(f"{e.get('beaten')}|incumbent")
        if not (ch and inc and ch.slices and inc.slices):
            a.add("L0", f"dethrone margin {str(e.get('king'))[:12]}…", FAIL,
                  "cannot recompute: challenger or incumbent paired vectors absent")
            continue
        mk = lambda s: Scored(sub=Submission(s.miner, s.tier, s.model_id, 0, 0.0),
                              retention=s.retention, retention_lb=s.retention_lb,
                              per_point=list(s.per_point), gates_ok=True,
                              per_axis={k: list(v) for k, v in s.slices.items()})
        got = softmin_lcb_diff(mk(ch), mk(inc), seed=0)
        claimed = float(e.get("margin_lcb", 0.0))
        a.ok("L0", f"dethrone margin {str(e.get('king'))[:12]}…",
             abs(got - claimed) <= 5e-3,
             info=f"recomputed {got:+.4f} vs claimed {claimed:+.4f}")
        # NOISE GATE: a margin inside the validator's own measured floor is a coin flip the miner
        # wins by resubmitting, not a result.
        if rec.noise:
            from .determinism import NoiseReport, crownable
            nr = NoiseReport()
            nr.max_kl = float(rec.noise.get("max_kl", 0.0) or 0.0)
            safety = float((rec.manifest or {}).get("noise_safety", 3.0))
            ok_m, why = crownable(claimed, nr, safety)
            a.ok("L0", f"margin clears noise floor {str(e.get('king'))[:12]}…", ok_m,
                 fail=why, info=f"margin {claimed:+.4f} vs floor "
                                f"{safety:g}x{nr.max_kl:.2e}")

    # crowning is only legal above the crownability floor
    bad = [s.model_id for s in rec.submissions
           if s.role == "challenger" and s.model_id in
           {e.get("king") for e in rec.events if e.get("action") in ("crown", "dethrone")}
           and s.retention_lb <= MIN_CROWN_LB]
    a.ok("L0", "crowns above MIN_CROWN_LB", not bad, fail=f"below floor: {bad}")

    audit_emission(rec, a)


def audit_emission(rec, a: Audit) -> None:
    """WEIGHTS. Emission is the only thing that actually moves value, and it was barely checked.

    THE HOLE THIS CLOSES, because it is the one that mattered most. The old version keyed the king
    map off crown/dethrone events only, then guarded the whole block with `if rec.weights and
    kings:` — so in a HOLD round, which is the steady state of a working subnet, `kings` was empty
    and the entire check degraded to a non-required SKIP. The operator could name any hotkey in
    `weights`, in any amount, and every level of the audit passed: L1, L2 and L3 never look at
    `weights` or `events` at all, so running the expensive levels added nothing. Verified against
    the real code with three signed synthetic records — a hold round paying an arbitrary hotkey, a
    crown naming a model that was never submitted, and a round with no events and arbitrary
    weights — all three reported zero failures and zero required skips.

    That is worse than an unchecked field. Auditors that copy `weights` after an ACCEPT would have
    been signing verdicts attesting to a number nobody verified.

    So this runs UNCONDITIONALLY and every part of it is required. The king map is rebuilt from
    every event that names a king (a hold event carries the reigning king's model_id, which is what
    made the carried-over case recoverable at all), the kings are bound to miners through the
    record's own submissions rather than through a field the operator writes beside them, and the
    weight vector must name exactly that set — in both directions, so dropping a legitimate king is
    caught as well as inventing one."""
    from .koth import MIN_CROWN_LB

    # 1. WHO IS KING — through eval/lineage, the SAME walk the scorer seeds its Tournament from.
    #    Deriving it separately here is how the audit and the round come to disagree about who holds
    #    the throne, at which point every honest round fails its own emission check. One walk.
    from .lineage import apply_events
    kings = {t: r.model_id for t, r in
             apply_events({}, rec.events, round_no=rec.round, submissions=rec.submissions).items()}
    # A HOLD ROUND IS THE STEADY STATE OF A WORKING SUBNET, AND IT HAD NO KINGS HERE.
    # `apply_events` is seeded from `{}` because one record is all an auditor has, and a `hold`
    # deliberately does not MINT a throne — it maintains one established in an earlier round, which
    # is right for `replay()` walking the whole chain and wrong here. Round 2 held both tiers, so
    # this map came back empty, the weight vector named two kings the record "did not have", and a
    # correct round was refused after 7 hours of scoring.
    #
    # A hold event names the tier and the model. That is enough to say WHICH throne is claimed;
    # WHO gets paid still comes from the scored submission below, never from the event's own
    # `miner` field, so the anti-circularity that makes this check worth running is untouched. A
    # held king with no re-scored incumbent in the record falls through to the `unbound` SKIP,
    # which already says the inheritance cannot be checked from one record alone.
    for e in rec.events or []:
        if e.get("action") == "hold" and e.get("tier") and e.get("king"):
            kings.setdefault(e["tier"], e["king"])

    # 2. BIND MODEL -> MINER THROUGH THE SCORED SUBMISSIONS, not through the event's own `miner`
    #    field. The event is written by the operator next to the weights; using it to validate the
    #    weights is circular. A submission is bound to bytes by content hash and to a hotkey by
    #    commit-reveal, so it is the thing an auditor can actually stand on.
    by_model: dict = {}
    for s in rec.submissions:
        by_model.setdefault(s.model_id, s)
    king_miner, unbound = {}, []
    for tier, mid in kings.items():
        s = by_model.get(mid)
        if s is None:
            unbound.append(f"{tier}:{str(mid)[:12]}…")
        else:
            king_miner[tier] = s.miner

    # 3. A CROWN MUST BE A SCORED CHALLENGER, AND THE BEST ONE. koth.Tournament.consider picks
    #    `max(valid, key=retention)` over challengers that cleared the gates and the floor, so the
    #    winner is recomputable from the record — and an operator crowning a model that never
    #    competed, or the second-best one, is now a failure rather than an assertion.
    crowned = [e for e in rec.events if e.get("action") in ("crown", "dethrone")]
    for e in crowned:
        tier, mid = e.get("tier"), e.get("king")
        s = by_model.get(mid)
        tag = f"{str(mid)[:12]}…"
        if s is None or s.role != "challenger":
            a.ok("L0", f"crowned model was scored this round {tag}", False,
                 fail=f"tier {tier} crowned {tag} but no challenger with that model_id appears in "
                      f"the record — the crown names bytes nobody scored")
            continue
        field_ok = e.get("miner") in (None, s.miner)
        a.ok("L0", f"crown event agrees with the submission {tag}", field_ok,
             fail=f"event credits {str(e.get('miner'))[:12]}… but the scored submission belongs "
                  f"to {str(s.miner)[:12]}… — emission would go to the wrong hotkey")
        rivals = [x for x in rec.submissions
                  if x.role == "challenger" and x.tier == tier
                  and x.gates_ok and x.retention_lb > MIN_CROWN_LB]
        if rivals:
            best = max(rivals, key=lambda x: x.retention)
            a.ok("L0", f"crown went to the best valid challenger {tag}",
                 abs(best.retention - s.retention) <= 1e-9,
                 fail=f"tier {tier} crowned {tag} at {s.retention:.4f} but "
                      f"{best.model_id[:12]}… scored {best.retention:.4f} — the crown is not the "
                      f"argmax the tournament claims to compute",
                 info=f"{s.retention:.4f}, best of {len(rivals)} valid challengers")
    for e in (x for x in rec.events if x.get("action") == "dethrone"):
        inc = [s for s in rec.submissions if s.role == "incumbent" and s.tier == e.get("tier")]
        a.ok("L0", f"dethrone names the re-scored incumbent {str(e.get('king'))[:12]}…",
             bool(inc) and e.get("beaten") == inc[0].model_id,
             fail=f"claims to have beaten {str(e.get('beaten'))[:12]}… but the record's incumbent "
                  f"for tier {e.get('tier')} is "
                  f"{(inc[0].model_id[:12] + '…') if inc else '(absent)'}")

    # 4. THE VECTOR ITSELF. Unconditional and required — this is the number that moves TAO.
    tot = sum(rec.weights.values()) if rec.weights else 0.0
    a.ok("L0", "weights normalised", abs(tot - 1.0) <= 1e-6 or tot == 0.0,
         fail=f"sum={tot:.6f}, which is neither 1.0 nor 0.0",
         info=f"sum={tot:.6f} over {len(rec.weights or {})} miner(s)")
    a.ok("L0", "weights are non-negative", all(w >= 0 for w in (rec.weights or {}).values()),
         fail="a negative weight is not representable on chain and means the vector was not "
              "produced by Tournament.weights()")
    if unbound:
        # An absence of evidence, reported as one. A tier whose king was not re-scored this round
        # (koth's "king unavailable" hold) leaves the operator's own weights unverifiable from this
        # record alone — it would take walking the anchor chain back to the crowning round.
        a.add("L0", "every king is bound to a miner by the record", SKIP,
              f"kings with no scored submission here: {unbound} — their share of emission cannot "
              f"be checked against this record alone", required=True)
    else:
        want = set(king_miner.values())
        got = {m for m, w in (rec.weights or {}).items() if w > 0}
        a.ok("L0", "weights name exactly the kings", got == want,
             fail=f"weighted {sorted(got)} but the record's own kings are {sorted(want)} "
                  f"(extra: {sorted(got - want)}, missing: {sorted(want - got)})",
             info=f"{len(want)} king(s) across {len(kings)} tier(s)")


# ---------------------------------------------------------------- L1: selection

def load_pool(path: str):
    """JSONL pool, one item per line: {id, prefix, source, index, meta}. Order IS the identity —
    the record's item indices are meaningless against a differently-ordered pool, which is why
    corpus_spec is checked before the indices are."""
    from .steps import Trajectory
    fields = {"id", "source", "prefix", "step", "index", "meta"}
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(Trajectory(**{k: v for k, v in row.items() if k in fields}))
    return out


def audit_selection(rec, pool, a: Audit) -> list:
    """Re-derive the exam from post-commit entropy. Returns the reconstructed items."""
    from .observer_round import select_trajectories
    man = rec.manifest or {}
    claimed = list(man.get("item_indices") or [])
    if not claimed:
        a.add("L1", "item selection", FAIL, "record does not say which items were scored")
        return []
    n_req = int(man.get("n_items_requested") or len(claimed))
    items, idx = select_trajectories(pool, rec.commit_root, rec.round_nonce, n_req)
    same_sel = list(idx) == claimed
    a.ok("L1", "item selection derives from the nonce", same_sel,
         fail=f"re-derived {list(idx)[:6]}… but the record claims {claimed[:6]}… — "
              f"the operator, not the nonce, chose the exam",
         info=f"re-derived all {len(idx)} indices from commit_root‖nonce")

    # NO PRUNING. Checking only that every scored point is IN the drawn selection is one-sided: the
    # operator can simply drop the items their model did badly on and every remaining point still
    # passes. So the selection must be fully COVERED by the points, not merely contain them. This is
    # the same asymmetry the nonce exists to remove — drawing the exam fairly buys nothing if you
    # can hand back a subset of it.
    scored_ids = [p.get("rollout_id") for p in rec.points]
    drawn_ids = [t.id for t in items]
    missing = [i for i in drawn_ids if i not in set(scored_ids)]
    a.ok("L1", "every drawn item was scored", not missing,
         fail=f"{len(missing)} of {len(drawn_ids)} drawn items are absent from the record "
              f"({missing[:5]}) — the exam was pruned after it was drawn",
         info=f"all {len(drawn_ids)} drawn items appear in the record")
    dupes = len(scored_ids) - len(set(scored_ids))
    a.ok("L1", "no duplicated items", dupes == 0,
         fail=f"{dupes} duplicated rollout_ids — repeating an easy item inflates its slice")
    # Bind the recorded text to the pinned corpus. Without this the frozen prefixes are just
    # strings the validator chose, and freezing them would launder rather than pin the exam.
    import hashlib
    by_id = {t.id: t for t in items}
    mism = []
    for p in rec.points:
        t = by_id.get(p.get("rollout_id"))
        if t is None:
            mism.append(f"{p.get('rollout_id')}: not in the derived selection")
            continue
        h = hashlib.sha256((t.prefix or "").encode()).hexdigest()
        if p.get("prefix_sha256") and h != p["prefix_sha256"]:
            mism.append(f"{t.id}: prefix hash differs")
    a.ok("L1", "recorded prefixes match the pinned corpus", not mism,
         fail="; ".join(mism[:3]) + (f" (+{len(mism) - 3} more)" if len(mism) > 3 else ""),
         info=f"{len(rec.points)} prefix hashes checked")

    # SLICE KEYS ARE DERIVED, NOT DECLARED. The worst-slice aggregate is a min over slices, so the
    # slice assignment IS the score. The keys lived only inside `effects` and were bound to
    # nothing, which let an operator pool every sample into one favourable slice (raising the min)
    # or shard the samples so no slice reached the sample floor. The key is a pure function of the
    # trajectory and the observer name, so it is recomputable — and now recomputed.
    from .observer_round import _slice_key
    obs = man.get("observer") or ""
    if obs:
        want = {}
        for p in rec.points:
            t = by_id.get(p.get("rollout_id"))
            if t is not None:
                want[p["rollout_id"]] = _slice_key(t, obs)
        bad = []
        for sub in rec.submissions:
            for p, row in zip(rec.points, sub.effects or []):
                exp = want.get(p.get("rollout_id"))
                if exp is not None and row and row[0] != exp:
                    bad.append(f"{p.get('rollout_id')}: {row[0]} != {exp}")
        a.ok("L1", "slice keys derive from the items", not bad,
             fail="; ".join(bad[:3]) + (f" (+{len(bad) - 3} more)" if len(bad) > 3 else ""),
             info=f"{len(want)} slice keys recomputed from (trajectory, observer)")
    return items


# ---------------------------------------------------------------- L2: judgment

def audit_judgment(rec, items, observer, a: Audit, sub_ids=None, observer_name: str = "") -> None:
    """Recompute the observer's distributions over the FROZEN text and re-derive (s, d_G, d_A).

    This is the level that makes a rigged SCORE detectable rather than merely a rigged sum. It is
    a pure forward pass — no sampling, no generation — because the record froze the parent step,
    the continuation and every miner step."""
    from .observer_kl import step_effect
    # WHICH OBSERVER. The auditor supplies the model on the command line, and nothing checked it
    # against the one the round actually used — so a green L2 proved only "some model reproduces
    # these numbers", which is not the claim. The observer is drawn from the nonce, so it is also
    # re-derivable: check both that the manifest names the model we were handed AND that the
    # manifest's choice is the one the nonce implies.
    # HARDWARE. Measured across three architectures on byte-identical items: within a box the
    # round is exactly reproducible (0.0 spread), across boxes it moves ~0.03 retention on a genuine
    # compression and ~0.17 on a foreign control. reproduction_tolerance comes from within-box
    # noise, so an honest auditor on different hardware would report DIVERGED on a good record.
    # Saying so up front is the difference between a useful audit and one people learn to ignore.
    vers = (rec.manifest or {}).get("versions") or {}
    rec_gpu = vers.get("gpu") or ""
    my_gpu = ""
    try:
        import torch
        if torch.cuda.is_available():
            my_gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    # BATCH SIZE. Measured spread across batch_size {1,2,3,5,8} on fixed items: 0.050 / 0.178 /
    # 0.051 on H100 / A100 / L40S, with the generated step text moving every time. An auditor at a
    # different batch_size gets a materially different number through no fault of the record.
    rec_bs = vers.get("batch_size")
    bs_match = True
    if rec_bs is not None:
        try:
            from .runners import HFRunner
            import inspect as _i
            my_bs = _i.signature(HFRunner.__init__).parameters["batch_size"].default
        except Exception:
            my_bs = None
        if my_bs is not None:
            bs_match = int(rec_bs) == int(my_bs)
            a.add("L2", "same batch size as the round", PASS if bs_match else SKIP,
                  f"batch_size={rec_bs}" if bs_match else
                  f"round scored at batch_size={rec_bs}, this audit uses {my_bs} — measured spread "
                  f"up to ~0.18 retention. The numbers below cannot be compared.")
    else:
        a.add("L2", "same batch size as the round", SKIP,
              "record does not pin batch_size", required=False)

    # HARDWARE MISMATCH IS NOT EVIDENCE OF ANYTHING, so it must not be reported as a failure.
    #
    # This check used to be a FAIL whose own message said "expect differences that are HARDWARE,
    # not fraud" — i.e. it told the reader the result was inadmissible and then recorded it as a
    # divergence anyway. For a lone human running `eval.rerun` that is merely confusing. For the
    # auditor daemon it is much worse: any FAIL becomes a REJECT, so an honest third party on an
    # A100 would publish a signed verdict accusing an honest operator on an H100, and hold its
    # weights against them. The README recommends running L2 on "a laptop", which is precisely the
    # configuration that would have fired.
    #
    # The correct verdict on non-matching hardware is INCOMPLETE — the check did not run — so the
    # mismatch is a REQUIRED SKIP and the numeric comparison below is skipped with it.
    hw_match = bool(rec_gpu) and bool(my_gpu) and rec_gpu == my_gpu
    if not rec_gpu:
        a.add("L2", "same GPU as the round", SKIP,
              "record does not pin the GPU, so a divergence cannot be attributed", required=False)
    elif hw_match:
        a.add("L2", "same GPU as the round", PASS, my_gpu)
    else:
        a.add("L2", "same GPU as the round", SKIP,
              f"round ran on {rec_gpu}, this audit is on {my_gpu or 'CPU'} — measured cross-box "
              f"spread is ~0.03 retention on a genuine compression and ~0.17 on a control, well "
              f"above reproduction_tolerance. Re-run on matching hardware; a difference here is "
              f"NOT evidence of fraud.")
    comparable = (rec_gpu == "" or hw_match) and bs_match

    want = (rec.manifest or {}).get("observer") or ""
    pool = (rec.manifest or {}).get("observer_pool") or []
    # Compare on the bare model name. The round registers its observers by whatever key the
    # operator built the registry with, and an auditor has to name a full HuggingFace repo id to
    # download the thing — so a literal string comparison made this check fail for every honest
    # third party while passing for the operator, which is the wrong way round.
    same_obs = bool(want) and _obs_key(observer_name) == _obs_key(want)
    a.ok("L2", "audited with the round's observer", same_obs,
         fail=f"re-running with {observer_name or '(unnamed)'} but the round used "
              f"{want or '(unrecorded)'} — a match here would prove nothing about this round",
         info=f"observer {want}")
    if pool:
        from .observer_round import pick_observer
        drawn = pick_observer(rec.commit_root, rec.round_nonce, pool)
        a.ok("L2", "observer derives from the nonce", drawn == want,
             fail=f"nonce implies {drawn} but the record used {want} — the operator picked the "
                  f"grader",
             info=f"{drawn} drawn from commit_root‖nonce over {len(pool)} candidates")
    by_id = {t.id: t for t in items}
    tol = max(float(rec.reproduction_tolerance or 0.0), 1e-4)
    for s in rec.submissions:
        if sub_ids and s.model_id not in sub_ids:
            continue
        tag = f"{s.model_id[:12]}…/{s.role}"
        if not (s.steps and s.effects):
            a.add("L2", f"recompute effects {tag}", SKIP, "no frozen steps recorded")
            continue
        worst, n = 0.0, 0
        for p, step, row in zip(rec.points, s.steps, s.effects):
            t = by_id.get(p.get("rollout_id"))
            if t is None or not p.get("continuation"):
                continue
            pref, cont = t.prefix, p["continuation"]
            p_g = observer.distributions(pref + "\n" + p["parent_step"], cont)
            p_0 = observer.distributions(pref, cont)
            p_a = observer.distributions(pref + "\n" + step, cont) if step.strip() else p_0
            eff = step_effect(p_g, p_a, p_0, min_teacher_effect=0.0)
            worst = max(worst, abs(eff.s - float(row[1])),
                        abs(eff.d_teacher - float(row[2])), abs(eff.d_miner - float(row[3])))
            n += 1
        if not n:
            a.add("L2", f"recompute effects {tag}", SKIP, "no reconstructable samples")
            continue
        if not comparable:
            # Measured, not assumed: the numbers below are real, they just cannot be attributed.
            # Reporting them as a divergence would make every honest auditor on other hardware an
            # accuser, so the observed delta is recorded and the check reads as not-run.
            a.add("L2", f"recompute effects {tag}", SKIP,
                  f"worst |Δ| over {n} samples = {worst:.3e}, but this box does not match the "
                  f"round's (gpu/batch_size) — the difference is not attributable to the record")
            continue
        a.ok("L2", f"recompute effects {tag}", worst <= tol,
             fail=f"worst |Δ| over {n} samples = {worst:.3e} EXCEEDS tol {tol:.2e} — the "
                  f"published measurements are not what this observer produces",
             info=f"worst |Δ| over {n} samples = {worst:.3e} (tol {tol:.2e})")


# ---------------------------------------------------------------- L3: provenance

def audit_provenance(rec, items, make_runner, a: Audit, max_items: int = 0) -> None:
    """Re-generate each submission's steps FROM ITS CHECKPOINT and compare to the frozen text.

    This is the level that stops the operator forging a crown. Everything below it takes the miner's
    steps from the record, and the record is written and signed by the operator, so a score computed
    honestly from fabricated steps reproduces perfectly at L0-L2. Only running the model closes it.

    `make_runner(model_id, artifact_uri) -> Stepper | None`. Returning None means the bytes were not
    available to this auditor, which is reported as a SKIP for that submission rather than a pass —
    an artifact you could not fetch has not been checked.
    """
    by_id = {t.id: t for t in items}
    budget = int((rec.manifest or {}).get("max_step_tokens") or 256)
    for s in rec.submissions:
        tag = f"{s.model_id[:12]}…/{s.role}"
        if not s.steps:
            a.add("L3", f"steps come from the model {tag}", SKIP, "no frozen steps recorded")
            continue
        if not s.artifact_uri:
            a.ok("L3", f"artifact locator present {tag}", False,
                 fail="record carries no artifact_uri, so nobody can fetch the bytes this score "
                      "claims to describe — the frozen steps are unfalsifiable")
            continue
        try:
            runner = make_runner(s.model_id, s.artifact_uri)
        except Exception as e:
            runner = None
            a.add("L3", f"steps come from the model {tag}", SKIP,
                  f"could not load the artifact: {type(e).__name__}: {e}")
            continue
        if runner is None:
            a.add("L3", f"steps come from the model {tag}", SKIP,
                  f"artifact not available to this auditor ({s.artifact_uri})")
            continue
        pts = list(zip(rec.points, s.steps))
        if max_items:
            pts = pts[:max_items]
        prefixes, frozen = [], []
        for p, step in pts:
            t = by_id.get(p.get("rollout_id"))
            if t is None:
                continue
            prefixes.append(t.prefix)
            frozen.append(step)
        if not prefixes:
            a.add("L3", f"steps come from the model {tag}", SKIP, "no reconstructable prefixes")
            continue
        got = runner.generate(prefixes, budget)
        diffs = [i for i, (g, f) in enumerate(zip(got, frozen)) if g != f]
        a.ok("L3", f"steps come from the model {tag}", not diffs,
             fail=f"{len(diffs)}/{len(prefixes)} regenerated steps differ from the frozen text "
                  f"(first at item {diffs[0] if diffs else '-'}) — the record's steps are not what "
                  f"this checkpoint produces",
             info=f"{len(prefixes)} steps regenerated and byte-identical"
                  + (f" (sampled {len(prefixes)} of {len(s.steps)})" if max_items else ""))


# ---------------------------------------------------------------- driver

def audit_loaded(rec, pool=None, observer=None, observer_name: str = "",
                 make_runner=None, max_l3_items: int = 0) -> Audit:
    """The audit itself, over an already-parsed record and an already-loaded pool.

    Split out from `audit` so a long-running auditor can hold ONE pool in memory across hundreds of
    rounds instead of re-reading and re-parsing a corpus file per round — and so a record fetched
    from a sink never has to be written to disk just to be checked."""
    a = Audit()
    audit_arithmetic(rec, a)

    items = []
    if pool is None:
        a.add("L1", "item selection derives from the nonce", SKIP, "no --pool given")
        a.add("L1", "recorded prefixes match the pinned corpus", SKIP, "no --pool given")
    else:
        spec = (rec.manifest or {}).get("corpus_spec") or ""
        # Checked BEFORE the indices: an index into the wrong ordering would compare two unrelated
        # things and could pass or fail for no reason connected to the round.
        a.ok("L1", "corpus spec pinned", bool(spec),
             fail="record does not pin the pool's ordering — indices are unanchored",
             info=spec)
        # THE POOL IS PART OF THE EXAM. corpus_spec names sources and revisions in prose; it does
        # not bind bytes, so an auditor handed "a pool matching the spec" was checking indices
        # against an ordering the record never committed to. The digest is inside the signed body,
        # so this is the check that makes L1 about the round rather than about some pool.
        want_pool = (rec.manifest or {}).get("pool_sha256") or ""
        if want_pool:
            from .pool import pool_digest
            got_pool = pool_digest(pool)
            a.ok("L1", "pool matches the one the record signed", got_pool == want_pool,
                 fail=f"loaded pool digests to {got_pool[:16]}… but the record pins "
                      f"{want_pool[:16]}… — this is a different corpus, so every index below "
                      f"points somewhere else",
                 info=f"{len(pool)} trajectories, sha256 {got_pool[:16]}…")
        else:
            a.add("L1", "pool matches the one the record signed", SKIP,
                  "record pins no pool_sha256, so the corpus is named but not bound")
        items = audit_selection(rec, pool, a)

    if observer is None:
        a.add("L2", "recompute effects", SKIP, "no --observer given")
    elif not items:
        a.add("L2", "recompute effects", SKIP, "needs L1 (the corpus) to reconstruct prefixes")
    else:
        audit_judgment(rec, items, observer, a, observer_name=observer_name)

    if make_runner is None:
        a.add("L3", "steps come from the model", SKIP,
              "no --artifacts given; the miner steps in this record are taken on the operator's "
              "word, so a forged score would reproduce at every level above")
    elif not items:
        a.add("L3", "steps come from the model", SKIP,
              "needs L1 (the corpus) to reconstruct prefixes")
    else:
        audit_provenance(rec, items, make_runner, a, max_items=max_l3_items)
    return a


def audit(record_path: str, pool_path: str = "", observer=None, observer_name: str = "",
          make_runner=None, max_l3_items: int = 0) -> Audit:
    return audit_loaded(load_record(record_path),
                        pool=load_pool(pool_path) if pool_path else None,
                        observer=observer, observer_name=observer_name,
                        make_runner=make_runner, max_l3_items=max_l3_items)


def report(a: Audit, out=sys.stdout) -> int:
    """Written straight to a stream. Never through `logging` — v1's auditor reported success for
    weeks into a handler a library's init had already detached."""
    w = out.write
    w("\n")
    for c in a.checks:
        mark = {PASS: "ok  ", FAIL: "FAIL", SKIP: "skip"}[c.status]
        w(f"  {mark}  [{c.level}] {c.name}\n")
        if c.detail:
            w(f"          {c.detail}\n")
    n = len(a.checks)
    w(f"\n  {a.verdict}: {n - len(a.failed) - len([c for c in a.checks if c.status == SKIP])}"
      f"/{n} passed, {len(a.failed)} failed, "
      f"{len([c for c in a.checks if c.status == SKIP])} skipped\n")
    if a.failed:
        w("  the record does NOT reproduce — do not treat this round as verified\n")
    elif a.skipped:
        w("  nothing contradicted the record, but the checks that matter were not run:\n")
        for c in a.skipped:
            w(f"    - [{c.level}] {c.name}: {c.detail}\n")
        w("  pass --pool (L1) and --observer (L2) for a verdict that means something\n")
    w("\n")
    return a.exit_code


def audit_history(sink_root: str, head_anchor: str = "", out=sys.stdout) -> int:
    """`--history <dir> [--head <on-chain anchor>]`: is the published TRAIL complete?

    A different question from auditing one record: a round that was scored, paid out and never
    published has no record to audit, so only walking the index finds it.

    The on-chain head is what makes this more than a self-consistency check. Without it, the walk
    compares the operator's records against the operator's index — so the first version of this
    command, which passed no anchor at all, reported COMPLETE on a fully unanchored trail. Pass
    --head with the anchor committed on chain (any block explorer shows it)."""
    from .publish import LocalSink, RecordPublisher, verify_history
    pub = RecordPublisher(LocalSink(sink_root), allow_no_state=True)
    h = verify_history(pub, head_anchor_fn=(lambda: head_anchor) if head_anchor else None)
    w = out.write
    w(f"\n  published trail: {h.n_rounds} rounds, head round={h.head}\n")
    w(f"  recomputed anchor head: {h.head_anchor or '(none)'}\n")
    for label, items in (("GAPS (scored but never published)", h.gaps),
                         ("BROKEN (no longer fetches or verifies)", h.broken),
                         ("CHAIN BREAKS (history altered or reordered)", h.chain_breaks),
                         ("ANCHOR MISMATCH", h.mismatched),
                         ("UNANCHORED (no chain link at all)", h.unanchored)):
        if items:
            w(f"    {label}: {items}\n")
    broke = bool(h.gaps or h.broken or h.chain_breaks or h.mismatched or h.unanchored)
    if not h.head_checked:
        w("    NOT CHECKED AGAINST THE CHAIN: pass --head <on-chain anchor>. Without it this "
          "compares the operator's records to the operator's index.\n")
    w(f"  {'COMPLETE' if h.ok else ('TRAIL BROKEN' if broke else 'INCOMPLETE')}\n\n")
    # Same convention as the record audit: 1 means the trail is DEMONSTRABLY broken, 2 means
    # nothing contradicted it but the check that matters was not run. Returning 1 for a missing
    # --head would read as "tampering detected" when it only means "you did not give me the chain
    # head", and a CI that cannot tell those apart will learn to ignore both.
    return 0 if h.ok else (1 if broke else 2)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    if argv[0] == "--history":
        head = argv[argv.index("--head") + 1] if "--head" in argv else ""
        return audit_history(argv[1], head)
    path, pool, obs_name, art, n_l3 = argv[0], "", "", "", 0
    i = 1
    while i < len(argv):
        if argv[i] == "--pool":
            pool = argv[i + 1]; i += 2
        elif argv[i] == "--observer":
            obs_name = argv[i + 1]; i += 2
        elif argv[i] == "--artifacts":
            art = argv[i + 1]; i += 2
        elif argv[i] == "--l3-items":
            n_l3 = int(argv[i + 1]); i += 2
        else:
            i += 1
    observer = None
    if obs_name:
        from .runners import HFRunner
        observer = HFRunner(obs_name)

    make_runner = None
    if art:
        import os

        def make_runner(model_id, artifact_uri, _root=art):
            """Checkpoints laid out as <root>/<model_id>/. Loaded through SafeStudentRunner, the
            same locked-down loader the validator uses — an auditor should not need to trust a
            miner's checkpoint any more than the validator does."""
            d = os.path.join(_root, model_id)
            if not os.path.isdir(d):
                return None
            from .runners import SafeStudentRunner
            return SafeStudentRunner(d)

    return report(audit(path, pool, observer, observer_name=obs_name,
                        make_runner=make_runner, max_l3_items=n_l3))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
