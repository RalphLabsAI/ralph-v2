"""Extractive-QA axis — verifiable-outcome capability retention over REAL text.

This is the CROWN-bearing axis, and the answer to const's SN97 objection. The synthetic
generators (math_gsm, code_exec's parameterized specs, long_context, multihop) emit nothing
a miner lacks: the generator is public and reproducible offline, so a specialist trained on
it aces every "fresh" instance WITHOUT retaining the teacher's capability (Goodhart — and
documented on SN97/Distil's own static->procedural post-mortem, where held-out GSM8K/
HumanEval/MMLU-Pro stayed flat while the eval score climbed). Fresh INSTANCES stop
memorization; they do nothing about the GENERATOR being known.

Real text is different on exactly the axis that matters: a document published AFTER the
commit carries exogenous information the miner could not have distilled at seal time, yet
the OUTCOME stays mechanical — the answer is a verbatim span present IN the document, so the
checker is exact-match with NO judge and NO logits/KL. That is the one corner where
deterministic-CPU-auditable AND real-distribution-validity both hold; the freshness keeps
that real distribution OPEN instead of collapsing into a memorizable benchmark.

Probe families (deterministic, gold span present in the passage, anchor unique so the answer
is unambiguous — no false negatives from aliased spans):
  * number — the number immediately after a short textual anchor (the proven
    corpus.make_probe primitive), and
  * entity — the capitalized token immediately after a short textual anchor.
Difficulty lengthens the anchor. Reading a long real passage to locate the span IS the
long-context/retrieval capability, measured on real content rather than a synthetic haystack.

The axis holds a document set: FRESH post-commit docs in production, synthetic/timestamped in
tests. Items are minted deterministically from the round seed, so a CPU auditor re-derives
every probe and every verdict.
"""
from __future__ import annotations

import random
import re

from ..core import Item

_NUM = re.compile(r"(?<![\w.])\d{2,}(?![\w.])")
_CAP = re.compile(r"\b[A-Z][a-z]{2,}\b")


def _mint(doc, seed: int, kind: str, anchor_len: int) -> tuple[str, str] | None:
    """One deterministic probe over `doc`: locate a number/entity by a UNIQUE preceding
    anchor. Returns (prompt, gold_span) or None if no unambiguous target exists."""
    text = doc.text
    pat = _NUM if kind == "number" else _CAP
    r = random.Random(f"{seed}|{doc.id}|{kind}")
    cands = [m for m in pat.finditer(text) if len(text[:m.start()].split()) >= anchor_len]
    r.shuffle(cands)
    for m in cands:
        anchor = " ".join(text[:m.start()].split()[-anchor_len:])
        # the anchor must occur exactly once, so the gold span is unambiguous (fair check).
        if text.count(anchor) != 1:
            continue
        word = "number" if kind == "number" else "word"
        prompt = (f"[doc {doc.id}] Passage:\n{text}\n\n"
                  f'In the passage, what {word} appears immediately after "{anchor}"? '
                  f"Reply with only the {word}, as `Answer: <{word}>`.")
        return prompt, m.group()
    return None


class ExtractiveQA:
    name = "extractive"
    weight = 1.0

    def __init__(self, docs, kinds: tuple[str, ...] = ("number", "entity")):
        # docs: list of corpus.Doc (needs .id and .text). In production these are FRESH
        # post-commit documents; the freshness is what makes the score un-pre-distillable.
        self.docs = list(docs)
        self.kinds = kinds

    def generate(self, seed: int, n: int, difficulty: int = 1) -> list[Item]:
        rng = random.Random(f"{seed}|extractive")
        docs = list(self.docs)
        rng.shuffle(docs)
        anchor_len = 2 if difficulty <= 1 else 3
        items: list[Item] = []
        for i, d in enumerate(docs):
            if len(items) >= n:
                break
            for kind in (self.kinds[i % len(self.kinds)], *self.kinds):  # preferred kind, then any
                p = _mint(d, seed, kind, anchor_len)
                if p is not None:
                    prompt, ans = p
                    items.append(Item(axis=self.name, prompt=prompt, answer=ans,
                                      meta={"kind": kind, "doc": d.id, "difficulty": difficulty}))
                    break
        return items

    def check(self, item: Item, output: str) -> bool:
        """Exact-span match against the gold the document itself pins down. First token after
        the answer marker (mirrors the other axes) so trailing chatter can't shift it."""
        if not output:
            return False
        tail = re.split(r"(?i)(?:final answer|answer)\s*[:=]", output)
        seg = tail[1] if len(tail) > 1 else output
        gold = str(item.answer)
        if item.meta.get("kind") == "number":
            m = re.search(r"-?\d[\d,]*", seg)
            return m is not None and m.group().replace(",", "") == gold.replace(",", "")
        m = re.search(r"[A-Za-z][A-Za-z0-9'\-]+", seg)
        return m is not None and m.group().lower() == gold.lower()
