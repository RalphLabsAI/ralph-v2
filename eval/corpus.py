"""Timestamped document corpus for the diff-in-diff genre-overfit gate (overfit_gate.py).

The gate needs STALE (pre-commit) vs FRESH (post-commit) same-genre document sets. The
source must be TIMESTAMPED so 'fresh' is provably post-commit — the un-pre-distillability
property: a miner cannot have distilled the teacher over documents that did not exist when
it committed. This module defines the corpus interface + a deterministic reading probe over
document text + a synthetic source for tests.

The real source is a drop-in loader (cf. pile.py): any timestamped stream — CC-News /
arXiv-new / Wikipedia recent-changes, filtered by publication date around the commit block
time — yields the same `Doc(id, text, ts)` shape. Keep only VERIFIED loaders.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

_NUM = re.compile(r"(?<![\w.])\d{2,}(?![\w.])")


@dataclass
class Doc:
    id: str
    text: str
    ts: int          # publication timestamp (unix, or a round index in tests)


def split_by_commit(docs: list[Doc], commit_ts: int) -> tuple[list[Doc], list[Doc]]:
    """STALE = published before the commit (pre-distillable); FRESH = at/after (novel)."""
    stale = [d for d in docs if d.ts < commit_ts]
    fresh = [d for d in docs if d.ts >= commit_ts]
    return stale, fresh


# ---- deterministic reading probe: locate a number by its textual anchor ----------------

def make_probe(doc: Doc, seed: int) -> tuple[str, str] | None:
    """A deterministic, content-dependent reading probe: pick a number in the passage and
    ask what number follows its preceding phrase. Answer is a verbatim span (exact-match
    checkable, no LLM judge). Content-dependent so a FRESH doc tests genuine reading, not
    a memorized stale answer key. Returns (prompt, answer) or None (no usable number)."""
    nums = list(_NUM.finditer(doc.text))
    if not nums:
        return None
    r = random.Random(f"{seed}|{doc.id}")
    m = r.choice(nums)
    before = doc.text[:m.start()].split()[-3:]
    if len(before) < 2:
        return None
    anchor = " ".join(before)
    prompt = (f"[doc {doc.id}] Passage:\n{doc.text}\n\n"
              f"In the passage, what number appears immediately after \"{anchor}\"? "
              f"Reply with only the number, as `Answer: <number>`.")
    return prompt, m.group()


def extract_num(output: str) -> str | None:
    """Pull the model's numeric answer (first number after the first answer marker)."""
    if not output:
        return None
    tail = re.split(r"(?i)(?:final answer|answer)\s*[:=]", output)
    seg = tail[1] if len(tail) > 1 else output
    nums = _NUM.findall(seg)
    return nums[0] if nums else None


def check_probe(answer: str, output: str) -> bool:
    got = extract_num(output)
    return got is not None and got == answer


# ---- synthetic source (tests / until a real timestamped loader is wired) ---------------

_SUBJECTS = ["The depot", "The registry", "The archive", "The ledger", "The survey",
             "The bulletin", "The inventory", "The report", "The manifest", "The audit"]
_VERBS = ["recorded", "logged", "listed", "reported", "noted", "tallied", "counted"]
_NOUNS = ["units", "entries", "shipments", "credits", "cases", "items", "readings", "tickets"]


def synth_corpus(n: int, seed: int, commit_ts: int, span: int = 20) -> list[Doc]:
    """n synthetic docs with numbers + prose, timestamps spread around `commit_ts` (so
    split_by_commit yields both stale and fresh). Deterministic in `seed`."""
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        sents = []
        for _ in range(rng.randint(4, 7)):
            sents.append(f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} "
                         f"{rng.randint(10, 9999)} {rng.choice(_NOUNS)} this period.")
        ts = commit_ts - span + rng.randint(0, 2 * span)   # spread across the boundary
        docs.append(Doc(id=f"d{seed}_{i}", text=" ".join(sents), ts=ts))
    return docs
