"""Stale-vs-fresh diff-in-diff — the genre-overfit detector (const's SN97 gate).

Freshness alone does NOT defeat distribution-overfit: a miner distills the pinned teacher
over the ANNOUNCED genre offline, and every fresh item lands in an already-distilled
region (the covering assessment). What catches it: score the student on BOTH
  * STALE items — pre-commit, same genre (what a miner could have pre-distilled on), and
  * FRESH items — post-commit, same genre (genuinely new information),
and gate on the DIFFERENCE, not the level. An honest broad compression GENERALIZES, so its
retention on fresh ~ retention on stale. A genre-overfitter memorized the pre-commit
distribution, so retention_stale >> retention_fresh.

Two properties make this a sound gate rather than a difficulty artifact:
  1. Retention is normalized (student-base)/(teacher-base), so a natural difficulty gap
     between stale and fresh (teacher AND base move together) cancels — the diff isolates
     OVERFIT, not difficulty.
  2. Gating on the DIFFERENCE means a uniformly weak-but-honest student (low retention on
     BOTH, diff ~ 0) is NOT flagged, and neither is a uniformly strong one. Only a student
     whose edge over base COLLAPSES on fresh same-genre items is flagged.

Same pattern as the 2026-07-06 held-out-memorization detector. This is a crown PRECONDITION
(down-ranks a flagged student); it needs a real post-commit corpus to source fresh vs stale
sets in production — here is the detector logic + its control, testable without a corpus.
"""
from __future__ import annotations

import random


def _retention(tp: list, sp: list, bp: list, idx: list | None = None) -> float | None:
    """Normalized retention on the teacher-passed subset (optionally a resampled index
    list, for the bootstrap). None when the axis carries no headroom this resample."""
    tpi = [i for i, ok in enumerate(tp) if ok] if idx is None else idx
    if not tpi:
        return None
    s = sum(1 for i in tpi if sp[i]) / len(tpi)
    b = sum(1 for i in tpi if bp[i]) / len(tpi)
    denom = 1.0 - b
    if denom <= 1e-9:            # base already saturates -> no headroom to measure
        return None
    return (s - b) / denom


def _tp_idx(tp: list) -> list:
    return [i for i, ok in enumerate(tp) if ok]


def diff_in_diff_gate(stale: dict, fresh: dict, threshold: float = 0.15,
                      z_reps: int = 2000, alpha: float = 0.05, min_n: int = 30,
                      seed: int = 0) -> tuple[bool, dict]:
    """`stale`/`fresh`: each a dict(teacher_pass, student_pass, base_pass) of aligned bool
    lists for one same-genre document set. Returns (ok, info).

    ok = False (FLAGGED as genre-overfit) iff the paired-bootstrap LOWER bound of
    (retention_stale - retention_fresh) exceeds `threshold` — the student's edge over the
    base COLLAPSES on genuinely-fresh same-genre items with statistical confidence.

    Fail-open on insufficient signal (too few teacher-passed items, or base saturated):
    this gate DOWN-ranks a positive detection; 'no signal' is not a detection, and a
    separate live-axis / crown gate handles the can't-measure case.
    """
    si, fi = _tp_idx(stale["teacher_pass"]), _tp_idx(fresh["teacher_pass"])
    r_stale = _retention(stale["teacher_pass"], stale["student_pass"], stale["base_pass"])
    r_fresh = _retention(fresh["teacher_pass"], fresh["student_pass"], fresh["base_pass"])
    info = {"retention_stale": None if r_stale is None else round(r_stale, 4),
            "retention_fresh": None if r_fresh is None else round(r_fresh, 4),
            "n_stale": len(si), "n_fresh": len(fi)}
    if r_stale is None or r_fresh is None or len(si) < min_n or len(fi) < min_n:
        info["verdict"] = "inconclusive"
        return True, info

    rng = random.Random(seed)
    diffs = []
    for _ in range(z_reps):
        bs = [si[rng.randrange(len(si))] for _ in range(len(si))]
        bf = [fi[rng.randrange(len(fi))] for _ in range(len(fi))]
        rs = _retention(stale["teacher_pass"], stale["student_pass"], stale["base_pass"], bs)
        rf = _retention(fresh["teacher_pass"], fresh["student_pass"], fresh["base_pass"], bf)
        if rs is not None and rf is not None:
            diffs.append(rs - rf)
    diffs.sort()
    lb = diffs[int(alpha * len(diffs))] if diffs else -1.0
    flagged = lb > threshold
    info.update(diff=round(r_stale - r_fresh, 4), diff_lb=round(lb, 4),
                verdict="genre-overfit" if flagged else "ok")
    return (not flagged), info


# ---- wiring: run the gate over a timestamped document corpus ---------------------------

def score_docs(docs: list, runner, seed: int, max_new_tokens: int = 64):
    """One deterministic reading probe per doc (aligned across models by `seed`), exact
    numeric check. Docs without a usable probe are skipped consistently. Returns
    (pass_mask, kept_docs)."""
    from .corpus import check_probe, make_probe
    probes = [(d, make_probe(d, seed)) for d in docs]
    probes = [(d, p) for d, p in probes if p is not None]
    if not probes:
        return [], []
    outs = runner.generate([p[0] for _, p in probes], max_new_tokens)
    return [check_probe(p[1], o) for (_, p), o in zip(probes, outs)], [d for d, _ in probes]


def diff_in_diff_over_corpus(docs: list, commit_ts: int, teacher, base, student,
                             seed: int = 0, threshold: float = 0.15, min_n: int = 30):
    """Wire the gate to a timestamped corpus: split at `commit_ts` into stale (pre-commit)
    vs fresh (post-commit), one reading probe per doc, score teacher/base/student on the
    SAME probes, run diff-in-diff. `teacher` (pinned GLM in prod) defines the retention
    subset. Returns (ok, info) — ok=False flags a genre-overfitter (aced pre-distillable
    stale docs, collapses on genuinely-new fresh docs)."""
    from .corpus import split_by_commit
    stale, fresh = split_by_commit(docs, commit_ts)

    def masks(dset):
        tp, kept = score_docs(dset, teacher, seed)          # teacher defines the subset
        bp, _ = score_docs(kept, base, seed)                # base + student on the same docs
        sp, _ = score_docs(kept, student, seed)
        return {"teacher_pass": tp, "student_pass": sp, "base_pass": bp}

    return diff_in_diff_gate(masks(stale), masks(fresh), threshold=threshold,
                             min_n=min_n, seed=seed)
