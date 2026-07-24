"""Real timestamped-document loaders for the diff-in-diff genre-overfit gate.

Makes the gate LIVE: real document text instead of the synthetic source. A source is a
VERIFIED (dataset, text_field, date_field) mapping (cf. pile.py's SOURCES — add only
mappings actually checked against the dataset, never guessed).

In PRODUCTION the fresh half comes from a live post-commit feed (a news / arXiv firehose
filtered by publication date > the commit block time — that is what makes fresh docs
un-pre-distillable). A static timestamped dataset is enough to exercise the split + the
reading probes on genuinely real text, which is what this loader provides; the live feed
is the same Doc(id, text, ts) shape behind an operational fetch.
"""
from __future__ import annotations

import calendar
import re
import time

from .corpus import Doc

# verified: field names confirmed against the dataset card / a real row
SOURCES = {
    "cc_news": {"dataset": "vblagoje/cc_news", "split": "train", "text": "text", "date": "date"},
}


def _parse_ts(v) -> int | None:
    """Best-effort unix timestamp from a date string/number. None if unparseable."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(v))
    if not m:
        return None
    return calendar.timegm((int(m[1]), int(m[2]), int(m[3]), 0, 0, 0, 0, 0, 0))


def load_hf_timestamped(source: str = "cc_news", n: int = 400, min_chars: int = 250,
                        max_chars: int = 2000, scan_limit: int = 20000) -> list[Doc]:
    """Stream a verified timestamped HF dataset into Docs. Skips rows that are too short or
    have no parseable date. Streaming so it never downloads the whole corpus."""
    from datasets import load_dataset

    spec = SOURCES[source]
    ds = load_dataset(spec["dataset"], split=spec["split"], streaming=True)
    docs: list[Doc] = []
    for i, row in enumerate(ds):
        if len(docs) >= n or i >= scan_limit:
            break
        text = (row.get(spec["text"]) or "").strip()
        ts = _parse_ts(row.get(spec["date"]))
        if ts is None or len(text) < min_chars:
            continue
        docs.append(Doc(id=f"{source}-{i}", text=text[:max_chars], ts=ts))
    return docs


def median_commit_ts(docs: list[Doc]) -> int:
    """A commit cutoff that splits the corpus into non-empty stale/fresh halves — the
    production value is the commit block's timestamp; this is for offline/shadow use."""
    ts = sorted(d.ts for d in docs)
    return ts[len(ts) // 2] if ts else int(time.time())
