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

FAILING LOUDLY IS A FEATURE, not politeness. Ralph v1's CPU auditor ran for weeks reporting
success while diverging from the validator on 37 of 40 reports, because it wrote through a logging
handler that a library's init had already removed, and because it exited 0 regardless. So: this
writes to stdout directly, never through `logging`; it exits non-zero on failure; and an audit that
SKIPPED the science exits 2 (INCOMPLETE) rather than 0. You cannot get a green light here by
running the cheap half.

    python -m eval.rerun record.json                          # L0 only -> exits 2, INCOMPLETE
    python -m eval.rerun record.json --pool items.jsonl       # L0+L1
    python -m eval.rerun record.json --pool items.jsonl --observer Qwen/Qwen2.5-0.5B-Instruct
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


def load_record(path: str):
    from .round_record import RoundRecord, SubmissionRecord
    raw = json.loads(open(path).read())
    subs = [SubmissionRecord(**s) for s in raw.pop("submissions", [])]
    rec = RoundRecord(**{**raw, "submissions": subs})
    return rec


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
        # THE ARITHMETIC CHECK. score = worst slice mean of sample_score(s, d_G, d_A). Recomputed
        # from the published raw measurements, so a fabricated aggregate over honest KLs shows up.
        by_slice: dict = {}
        for row in s.effects:
            key, sv, d_g, d_m = row[0], float(row[1]), float(row[2]), float(row[3])
            eff = StepEffect(s=sv, d_teacher=d_g, d_miner=d_m, n_positions=1)
            by_slice.setdefault(key, []).append(sample_score(eff))
        # mirrors score_miner: slices below the sample floor are excluded, not averaged in
        from .observer_kl import score_miner as _sm
        min_per_slice = _sm.__defaults__[2]
        kept = {k: v for k, v in by_slice.items() if len(v) >= min_per_slice}
        if not kept:
            a.add("L0", f"recompute score {tag}", SKIP,
                  f"no slice reached {min_per_slice} samples")
            continue
        got = min(sum(v) / len(v) for v in kept.values())
        tol = max(float(rec.reproduction_tolerance or 0.0), 1e-6)
        a.ok("L0", f"recompute score {tag}", abs(got - s.retention) <= tol,
             info=f"recomputed {got:.6f} vs recorded {s.retention:.6f} (tol {tol:.2e})")
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

    # WEIGHTS. Emission is the only thing that actually moves value, so the record's weights must
    # follow from the record's own crown decisions rather than being asserted alongside them.
    kings = {}
    for e in rec.events:
        if e.get("action") in ("crown", "dethrone") and e.get("miner"):
            kings[e["tier"]] = e["miner"]
        elif e.get("action") == "vacate":
            kings.pop(e.get("tier"), None)
    if rec.weights and kings:
        claimed_miners = {m for m, w in rec.weights.items() if w > 0}
        a.ok("L0", "weights follow the crowns", claimed_miners <= set(kings.values()),
             fail=f"weighted {sorted(claimed_miners)} but the kings are "
                  f"{sorted(set(kings.values()))}")
        tot = sum(rec.weights.values())
        a.ok("L0", "weights normalised", abs(tot - 1.0) <= 1e-6 or tot == 0.0,
             fail=f"sum={tot:.6f}, not 1.0")
    else:
        a.add("L0", "weights follow the crowns", SKIP, "no weights or no crowns this round",
              required=False)


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
    return items


# ---------------------------------------------------------------- L2: judgment

def audit_judgment(rec, items, observer, a: Audit, sub_ids=None) -> None:
    """Recompute the observer's distributions over the FROZEN text and re-derive (s, d_G, d_A).

    This is the level that makes a rigged SCORE detectable rather than merely a rigged sum. It is
    a pure forward pass — no sampling, no generation — because the record froze the parent step,
    the continuation and every miner step."""
    from .observer_kl import step_effect
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
        a.ok("L2", f"recompute effects {tag}", worst <= tol,
             fail=f"worst |Δ| over {n} samples = {worst:.3e} EXCEEDS tol {tol:.2e} — the "
                  f"published measurements are not what this observer produces",
             info=f"worst |Δ| over {n} samples = {worst:.3e} (tol {tol:.2e})")


# ---------------------------------------------------------------- driver

def audit(record_path: str, pool_path: str = "", observer=None) -> Audit:
    a = Audit()
    rec = load_record(record_path)
    audit_arithmetic(rec, a)

    items = []
    if not pool_path:
        a.add("L1", "item selection derives from the nonce", SKIP, "no --pool given")
        a.add("L1", "recorded prefixes match the pinned corpus", SKIP, "no --pool given")
    else:
        pool = load_pool(pool_path)
        spec = (rec.manifest or {}).get("corpus_spec") or ""
        # Checked BEFORE the indices: an index into the wrong ordering would compare two unrelated
        # things and could pass or fail for no reason connected to the round.
        a.ok("L1", "corpus spec pinned", bool(spec),
             fail="record does not pin the pool's ordering — indices are unanchored",
             info=spec)
        items = audit_selection(rec, pool, a)

    if observer is None:
        a.add("L2", "recompute effects", SKIP, "no --observer given")
    elif not items:
        a.add("L2", "recompute effects", SKIP, "needs L1 (the corpus) to reconstruct prefixes")
    else:
        audit_judgment(rec, items, observer, a)
    return a


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


def audit_history(sink_root: str, out=sys.stdout) -> int:
    """`--history <dir>`: is the published TRAIL complete? Answers a different question from
    auditing one record — a round that was paid out and never published has no record to audit,
    so only walking the index can find it."""
    from .publish import LocalSink, RecordPublisher, verify_history
    h = verify_history(RecordPublisher(LocalSink(sink_root)))
    w = out.write
    w(f"\n  published trail: {h.n_rounds} rounds, head={h.head}\n")
    for label, items in (("GAPS (scored but never published)", h.gaps),
                         ("BROKEN (no longer fetches or verifies)", h.broken),
                         ("ANCHOR MISMATCH", h.mismatched),
                         ("unanchored", h.unanchored)):
        if items:
            w(f"    {label}: {items}\n")
    w(f"  {'COMPLETE' if h.ok else 'INCOMPLETE TRAIL'}\n\n")
    return 0 if h.ok else 1


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    if argv[0] == "--history":
        return audit_history(argv[1])
    path, pool, obs_name = argv[0], "", ""
    i = 1
    while i < len(argv):
        if argv[i] == "--pool":
            pool = argv[i + 1]; i += 2
        elif argv[i] == "--observer":
            obs_name = argv[i + 1]; i += 2
        else:
            i += 1
    observer = None
    if obs_name:
        from .runners import HFRunner
        observer = HFRunner(obs_name)
    return report(audit(path, pool, observer))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
