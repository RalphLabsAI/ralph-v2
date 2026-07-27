"""Corpus-swap invariance + calibration-sensitivity dispersion.

THE ATTACK THIS CLOSES. Compressing a model to 1-2 bits needs data-driven recovery, and that
recovery data is the miner's private choice. The literature is unambiguous that this is where
the cheating lives:

  * QDrop (ICLR'22) measured the inversion directly: across W2A2 variants, the method with the
    LOWEST calibration accuracy (66.11) had the HIGHEST test accuracy (51.14), and the one that
    looked best on its own calibration data (69.54) was worst in the wild (45.77). Optimizing
    the reconstruction objective and preserving capability pull in opposite directions.
  * Williams & Aletras (ACL'24) held everything fixed and changed only the random draw of 128
    calibration sequences: perplexity moved sigma=0.18 while BoolQ swung 57.0% -> 71.6%. A
    14-point downstream swing that a fidelity metric cannot see.
  * arXiv:2406.12928 found "cross-dataset optimal calibration data for specific tasks" — i.e. a
    searchable degree of freedom, and the winning corpus need not resemble the eval data. A
    miner will grid-search it against whatever axes we publish.

PrismML themselves know the shape of the defense — and apply it to the wrong artifact. Their
`kv-mean-center` README argues calibration-invariance properly: swap the corpus, show the result
is unchanged ("cosine similarity above 0.95 on every layer"), validate with "logit KL divergence
... over 12x512-token held-out chunks". That is a few-MB KV bias vector. Nothing equivalent is
published for the weights, and when asked directly whether recovery training was English-only
they did not answer — while independent testing found Persian collapsing 79.8% -> 45.2%.

THE MEASUREMENT. We cannot re-derive a miner's weights, so we test the behavioral analog: score
the student on N probe sets drawn from DIFFERENT sources, and look at the shape of the result.

    honest compression   -> retention is roughly FLAT across sources
    calibration-overfit  -> retention is high where the recovery data lived, low elsewhere
    uniformly weak       -> retention is flat and LOW  (must NOT be flagged as cheating)

Two numbers come out, and they do different jobs — keeping them separate is what makes this
adversarially sound:

  * WORST-SOURCE retention is the SCORE. Scoring the level (not the spread) means a miner cannot
    win by being uniformly mediocre; this is worst-domain aggregation applied to the corpus axis,
    consistent with how the crown already treats capability axes.
  * DISPERSION is the GATE. Flagged only when a paired bootstrap says the spread is real, so
    sampling noise on a thin probe set cannot demote an honest miner.

Retention is normalized (student - base) / (teacher - base) per source, so a source that is
simply harder moves the teacher and base together and cancels — the dispersion isolates
SOURCE-DEPENDENCE, not difficulty. Deterministic, CPU-auditable, no judge, no teacher-agreement.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .overfit_gate import _retention, _tp_idx, score_docs

# Flag only a spread this large; below it the differences are not worth acting on even if real.
DISPERSION_THRESHOLD = 0.20
MIN_SOURCE_N = 20          # teacher-passed probes required before a source may vote


@dataclass
class InvarianceReport:
    per_source: dict = field(default_factory=dict)     # source -> retention (None = no signal)
    worst_source: str = ""
    worst_retention: float = 0.0                       # THE SCORE
    dispersion: float = 0.0                            # max - min across live sources
    dispersion_lb: float = 0.0                         # bootstrap lower bound on the spread
    n_live: int = 0
    verdict: str = "inconclusive"

    def as_dict(self) -> dict:
        return {"per_source": {k: (None if v is None else round(v, 4))
                               for k, v in self.per_source.items()},
                "worst_source": self.worst_source,
                "worst_retention": round(self.worst_retention, 4),
                "dispersion": round(self.dispersion, 4),
                "dispersion_lb": round(self.dispersion_lb, 4),
                "n_live": self.n_live, "verdict": self.verdict}


def _boot_dispersion_lb(masks: dict, z_reps: int, alpha: float, seed: int) -> float:
    """Lower bound on (max - min) retention across sources, by resampling each source's
    teacher-passed items independently. A spread that survives its own noise floor is real."""
    rng = random.Random(seed)
    idx = {s: _tp_idx(m["teacher_pass"]) for s, m in masks.items()}
    spreads = []
    for _ in range(z_reps):
        vals = []
        for s, m in masks.items():
            ii = idx[s]
            if len(ii) < MIN_SOURCE_N:
                continue
            bs = [ii[rng.randrange(len(ii))] for _ in range(len(ii))]
            r = _retention(m["teacher_pass"], m["student_pass"], m["base_pass"], bs)
            if r is not None:
                vals.append(r)
        if len(vals) >= 2:
            spreads.append(max(vals) - min(vals))
    if not spreads:
        return -1.0
    spreads.sort()
    return spreads[int(alpha * len(spreads))]


def corpus_swap_invariance(masks_by_source: dict, threshold: float = DISPERSION_THRESHOLD,
                           z_reps: int = 1000, alpha: float = 0.05,
                           seed: int = 0) -> InvarianceReport:
    """`masks_by_source`: source -> dict(teacher_pass, student_pass, base_pass), aligned bool
    lists from scoring teacher/base/student on the SAME probes of that source."""
    rep = InvarianceReport()
    live = {}
    for s, m in sorted(masks_by_source.items()):
        n = len(_tp_idx(m["teacher_pass"]))
        r = _retention(m["teacher_pass"], m["student_pass"], m["base_pass"])
        rep.per_source[s] = r
        if r is not None and n >= MIN_SOURCE_N:
            live[s] = r
    rep.n_live = len(live)
    if len(live) < 2:
        rep.verdict = "inconclusive"        # cannot compare sources; the caller decides
        return rep

    rep.worst_source = min(live, key=lambda k: live[k])
    rep.worst_retention = live[rep.worst_source]
    rep.dispersion = max(live.values()) - min(live.values())
    rep.dispersion_lb = _boot_dispersion_lb(
        {s: m for s, m in masks_by_source.items() if s in live}, z_reps, alpha, seed)
    rep.verdict = "calibration-overfit" if rep.dispersion_lb > threshold else "ok"
    return rep


def invariance_over_sources(sources: dict, teacher, base, student, seed: int = 0,
                            threshold: float = DISPERSION_THRESHOLD,
                            max_new_tokens: int = 64) -> InvarianceReport:
    """Score teacher/base/student on one probe per doc for EACH source, then compare.

    `sources`: name -> list[Doc]. Use genuinely different origins (different crawl snapshots,
    different publishers, different genres) — swapping between two slices of one distribution
    tests nothing, which is the same reason a synthetic generator's stale/fresh split is blind."""
    masks = {}
    for name, docs in sources.items():
        tp, kept = score_docs(docs, teacher, seed, max_new_tokens)   # teacher sets the subset
        if not kept:
            continue
        bp, _ = score_docs(kept, base, seed, max_new_tokens)
        sp, _ = score_docs(kept, student, seed, max_new_tokens)
        masks[name] = {"teacher_pass": tp, "student_pass": sp, "base_pass": bp}
    return corpus_swap_invariance(masks, threshold=threshold, seed=seed)


def make_invariance_check(sources: dict, teacher, base, seed: int = 0,
                          threshold: float = DISPERSION_THRESHOLD,
                          require_sources: int = 2):
    """Build an `(ok, info)` crown precondition, same shape as the genre-overfit gate.

    FAIL-CLOSED on insufficient signal when `require_sources` sources were expected: unlike the
    diff-in-diff gate (whose 'inconclusive' means the validator's corpus was thin), an operator
    who configured N sources and got fewer has a broken feed, and a silently-skipped gate is how
    a defense becomes decorative."""
    def check(sub, runner):
        rep = invariance_over_sources(sources, teacher, base, runner, seed=seed,
                                      threshold=threshold)
        info = rep.as_dict()
        if rep.n_live < require_sources:
            return False, {**info, "verdict": "inconclusive-failclosed"}
        return (rep.verdict != "calibration-overfit"), info
    return check
