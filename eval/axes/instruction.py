"""Instruction-following axis — IFEval-style verifiable constraints.

A capability axis with a fully DETERMINISTIC checker: each item asks for content under a
machine-checkable constraint (exact word count, required/forbidden token, JSON shape,
prefix/suffix, uppercase, N bullet points). Grading is a parser, never a model — so this
axis contributes broad "does the student follow GLM-level instructions" coverage to the
GLM-COVER crown with no LLM judge. Fresh per round from the seed; the constraint params are
drawn from the seed so there is no fixed set to memorize.
"""
from __future__ import annotations

import json
import random
import re

from ..core import Item

_TOPICS = ["a lighthouse", "the water cycle", "a chess opening", "sourdough bread",
           "tidal energy", "the printing press", "a coral reef", "compilers",
           "migratory birds", "the bronze age", "photosynthesis", "a suspension bridge"]
_WORDS = ["azure", "lattice", "quorum", " member", "beacon", "granite", "cipher", "harbor"]


def _c_exact_words(rng):
    n = rng.choice([12, 18, 24, 30])
    return (f"Write exactly {n} words about {rng.choice(_TOPICS)}. Output only those words.",
            {"kind": "exact_words", "n": n})


def _c_include(rng):
    w = rng.choice(_WORDS)
    return (f"Write two sentences about {rng.choice(_TOPICS)}. You MUST include the word "
            f"'{w}'.", {"kind": "include", "word": w})


def _c_forbid(rng):
    w = rng.choice(_WORDS)
    return (f"Write two sentences about {rng.choice(_TOPICS)}. Do NOT use the word "
            f"'{w}'.", {"kind": "forbid", "word": w})


def _c_json(rng):
    keys = rng.sample(["title", "year", "count", "author", "tag"], 3)
    return (f"Output a single JSON object (and nothing else) with exactly these keys: "
            f"{', '.join(keys)}. Any values.", {"kind": "json", "keys": sorted(keys)})


def _c_bullets(rng):
    n = rng.choice([3, 4, 5])
    return (f"List exactly {n} facts about {rng.choice(_TOPICS)}, each on its own line "
            f"starting with '- '.", {"kind": "bullets", "n": n})


def _c_affix(rng):
    tok = rng.choice(["END", "DONE", "OK"])
    return (f"Write one sentence about {rng.choice(_TOPICS)} and finish with the exact "
            f"token {tok} on its own line.", {"kind": "suffix", "tok": tok})


_CONSTRAINTS = [_c_exact_words, _c_include, _c_forbid, _c_json, _c_bullets, _c_affix]


class InstructionFollowing:
    name = "instruction"
    weight = 1.0

    def generate(self, seed: int, n: int, difficulty: int = 1) -> list[Item]:
        rng = random.Random(seed)
        items = []
        for i in range(n):
            make = rng.choice(_CONSTRAINTS)
            prompt, ans = make(rng)
            items.append(Item(axis=self.name, prompt=prompt, answer=ans,
                              meta={"template": ans["kind"], "idx": i}))
        return items

    def check(self, item: Item, output: str) -> bool:
        a = item.answer
        k = a["kind"]
        text = (output or "").strip()
        if k == "exact_words":
            # count word tokens in the whole answer
            return len(re.findall(r"[A-Za-z0-9']+", text)) == a["n"]
        if k == "include":
            return re.search(rf"\b{re.escape(a['word'])}\b", text, re.I) is not None
        if k == "forbid":
            return re.search(rf"\b{re.escape(a['word'])}\b", text, re.I) is None
        if k == "json":
            obj = _first_json(text)
            return isinstance(obj, dict) and sorted(obj.keys()) == a["keys"]
        if k == "bullets":
            lines = [ln for ln in text.splitlines() if ln.strip()]
            return len(lines) == a["n"] and all(ln.lstrip().startswith("- ") for ln in lines)
        if k == "suffix":
            lines = [ln for ln in text.splitlines() if ln.strip()]
            return bool(lines) and lines[-1].strip() == a["tok"]
        return False


def _first_json(text: str):
    """Parse the first {...} JSON object in the text, tolerant of surrounding prose."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None
