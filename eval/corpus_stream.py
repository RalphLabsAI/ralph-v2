"""Large-pool seeded corpus sampling — the anti-enumeration defense.

THE POINT. Every eval whose source is small, fixed, or enumerable is eventually overfit. The
predecessor subnet's own post-mortem says it twice: static benches got memorized, and even
after moving to procedural generators they conceded "a miner can still overfit their model on
the distribution the procedural generator samples from... any rule a miner can read becomes a
training target." Their successor's answer was not a cleverer metric — it was CORPUS SCALE
(they draw ~100 items per round from 12.4M real trajectories).

So the bar is not "impossible to enumerate", it is "UNECONOMICAL to enumerate": make the pool
so large that pre-fitting it IS training a broadly capable model, which is the thing we want
to pay for anyway. This module supplies that pool.

    docs = load_stream_slice(seed, n=400)      # 400 docs drawn from a ~10^8-10^10 row pool

THREE PROPERTIES, all load-bearing:

  1. SCALE. Sources declare a real `pool_rows` (verified against the HF datasets-server, see
     SOURCES). fineweb alone is 2.6e10 rows across 114 dated CC-MAIN snapshots; one snapshot
     is ~2.8e8. Drawing 400 from that is a ~1e-6..1e-8 sampling rate.
  2. SEEDED DISPERSION. The round seed picks the snapshot, the shard order, and the offset —
     so the draw lands anywhere in the pool and is unknowable before the seed exists. Selecting
     post-commit entropy (chain nonce) as that seed is what makes the slice un-pre-fittable;
     `derive_seed(commit_root, round_nonce, ...)` already produces it.
  3. AUDITABILITY. Selection is a pure function of the seed: same seed -> same docs. Publish
     `corpus_fingerprint(docs)` in the signed round record and a CPU auditor re-derives the
     identical slice and re-checks every verdict. No hidden validator state.

HONEST LIMITS. Scale raises the cost of pre-fitting; it does not make it impossible, and a
crawl corpus is public, so a determined miner can train on the same distribution (that is
partly the point — doing so broadly IS compression). What scale does NOT solve is provenance
of the FRESH half: for the diff-in-diff gate to be sighted, "fresh" must genuinely post-date
the commit, which needs a live post-commit feed with attested publication times. That remains
the open item; this module is the pool, not the clock.
"""
from __future__ import annotations

import hashlib
import random

from .corpus import Doc
from .corpus_hf import _parse_ts

# VERIFIED against https://datasets-server.huggingface.co (size + first-rows) on 2026-07-25.
# Never guess field names — a wrong mapping silently yields an empty or dateless corpus.
#   `configs`: None means the single default config; a list means the seed picks one (each
#   fineweb config is one dated Common Crawl snapshot, so this also disperses across TIME).
#   `pool_rows`: rows in ONE config — the denominator of the sampling rate we claim.
SOURCES = {
    # 25,886,364,489 rows total across 114 CC-MAIN-YYYY-WW snapshots; ~2.76e8 per snapshot.
    "fineweb": {
        "dataset": "HuggingFaceFW/fineweb", "split": "train",
        "text": "text", "date": "date",
        "configs": ["CC-MAIN-2024-10", "CC-MAIN-2024-18", "CC-MAIN-2024-22",
                    "CC-MAIN-2024-26", "CC-MAIN-2024-30", "CC-MAIN-2024-33",
                    "CC-MAIN-2024-38", "CC-MAIN-2024-42", "CC-MAIN-2024-46",
                    "CC-MAIN-2024-51"],
        "pool_rows": 275_924_241,
    },
    # NON-LATIN real text. This is the incumbent's blind spot and our anti-clone lever in one:
    # PrismML's 15-benchmark suite contains ZERO multilingual coverage, and independent testing
    # measured Persian collapsing 79.8% -> 45.2% at 1-bit while English/coding held at 97-100%.
    # A cloned Bonsai therefore LOSES here, which is the only kind of axis a cloner cannot
    # supply. Digits are script-independent, so a numeric probe stays exactly checkable while
    # the anchor forces genuine reading of Cyrillic/Han/Arabic context.
    "fineweb2_rus": {
        "dataset": "HuggingFaceFW/fineweb-2", "split": "train",
        "text": "text", "date": "date",
        "configs": ["rus_Cyrl"], "pool_rows": 699_118_293,
    },
    "fineweb2_cmn": {
        "dataset": "HuggingFaceFW/fineweb-2", "split": "train",
        "text": "text", "date": "date",
        "configs": ["cmn_Hani"], "pool_rows": 636_092_280,
    },
    "fineweb2_arb": {
        "dataset": "HuggingFaceFW/fineweb-2", "split": "train",
        "text": "text", "date": "date",
        "configs": ["arb_Arab"], "pool_rows": 62_019_605,
    },
    # real journalism with true publication dates, one config per month — the closest thing
    # to a post-commit feed available as a dataset. Small per config; used as a MINORITY
    # source for genre diversity (a second source dilutes any single planted document).
    "bbc_news": {
        "dataset": "RealTimeData/bbc_news_alltime", "split": "train",
        "text": "content", "date": "published_date",
        "configs": ["2024-01", "2024-03", "2024-05", "2024-07", "2024-09", "2024-11"],
        "pool_rows": 1_562,
    },
}

# weighted manifest: no single source is the whole eval (dilutes planting + genre-overfit).
DEFAULT_MIX = (("fineweb", 0.7), ("bbc_news", 0.3))

# Multilingual mix for the non-Latin crown axis. Kept SEPARATE from DEFAULT_MIX rather than
# blended in: mixing a little Cyrillic into a mostly-English draw lets a strong English model
# average over it, which is precisely how the incumbent's suite failed to see a 34-point
# collapse. Scored as its own axis, it cannot be averaged away.
MULTILINGUAL_MIX = (("fineweb2_rus", 0.4), ("fineweb2_cmn", 0.35), ("fineweb2_arb", 0.25))


def _sub_seed(seed: int | str, tag: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{tag}".encode()).digest()[:8], "big")


def _load_one(name: str, seed: int | str, n: int, *, min_chars: int, max_chars: int,
              shuffle_buffer: int, scan_limit: int) -> list[Doc]:
    """Seeded slice from ONE source. Streaming: never downloads the pool."""
    from datasets import load_dataset

    spec = SOURCES[name]
    rng = random.Random(_sub_seed(seed, name))
    configs = spec.get("configs")
    config = rng.choice(configs) if configs else None

    ds = load_dataset(spec["dataset"], name=config, split=spec["split"], streaming=True)
    # shard-order shuffle + reservoir: disperses the draw across the WHOLE config cheaply
    # (skipping to a random absolute offset in a 2.8e8-row stream would mean iterating to it).
    ds = ds.shuffle(seed=_sub_seed(seed, name + "/shuffle") % (2 ** 31), buffer_size=shuffle_buffer)

    docs: list[Doc] = []
    for i, row in enumerate(ds):
        if len(docs) >= n or i >= scan_limit:
            break
        text = (row.get(spec["text"]) or "").strip()
        ts = _parse_ts(row.get(spec["date"]))
        if ts is None or len(text) < min_chars:
            continue
        docs.append(Doc(id=f"{name}:{config or 'default'}:{i}", text=text[:max_chars], ts=ts))
    return docs


def load_stream_slice(seed: int | str, n: int = 400, mix=DEFAULT_MIX, *,
                      min_chars: int = 250, max_chars: int = 2000,
                      shuffle_buffer: int = 10_000, scan_limit: int = 20_000,
                      strict: bool = False) -> list[Doc]:
    """`n` docs drawn from the large multi-source pool, deterministically in `seed`.

    Sources are tried in mix order; a source that fails (offline, gated, renamed) is skipped
    and the remainder is taken from the others, so one dead dataset cannot stall a round.
    `strict=True` re-raises instead — use it in production, where a silently thin corpus would
    quietly weaken the crown. Returns [] if everything fails (callers must fail closed)."""
    docs: list[Doc] = []
    errors: list[str] = []
    for name, w in mix:
        want = max(1, int(round(n * w)))
        try:
            docs += _load_one(name, seed, want, min_chars=min_chars, max_chars=max_chars,
                              shuffle_buffer=shuffle_buffer, scan_limit=scan_limit)
        except Exception as e:                       # offline / gated / renamed
            errors.append(f"{name}: {type(e).__name__}: {e}")
            if strict:
                raise
    if errors and not docs and strict:
        raise RuntimeError("; ".join(errors))
    # deterministic order regardless of which source returned first
    docs.sort(key=lambda d: d.id)
    return docs[:n]


def pool_size(mix=DEFAULT_MIX) -> int:
    """Total rows reachable by this mix — the enumeration denominator we can claim honestly.
    Counts every config of every source (the seed can land in any of them)."""
    total = 0
    for name, _ in mix:
        spec = SOURCES[name]
        total += spec["pool_rows"] * max(1, len(spec.get("configs") or []))
    return total


def corpus_fingerprint(docs: list[Doc]) -> str:
    """Content-addressed id of the drawn slice. Goes in the signed round record so an auditor
    can prove they re-derived the SAME corpus before re-checking any verdict."""
    h = hashlib.sha256()
    for d in sorted(docs, key=lambda d: d.id):
        h.update(d.id.encode())
        h.update(hashlib.sha256(d.text.encode()).digest())
    return h.hexdigest()
