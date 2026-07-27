"""Doc-grounded task formats — growing the crown pool so surprise selection actually bites.

WHY MORE FORMATS. Drawing which axes score from post-commit entropy only defends in proportion
to the POOL: with two crown axes, covering both is cheap. And the pool has to satisfy a
stricter property than "more tasks" — covering all of it must be indistinguishable from being
generally capable. Ten variants of "extract a span" is one narrow skill wearing ten hats, and a
miner buys the whole pool with one cheap fine-tune. So each format here stresses a DIFFERENT
mechanism, and each is chosen against where compression measurably breaks rather than against
difficulty (arXiv:2409.11055: "quantization amplifies a model's existing weaknesses" — GSM8K
and MMLU did NOT drop most; difficulty is the wrong selector).

    NumericComposition   locate TWO values, then compute  -> composition compounds
    LongContextReal      find one value in a long REAL haystack -> retrieval at length
    ConstrainedExtraction correct content AND exact format  -> two independent failure modes

All three are grounded in real documents, so the answer is fixed by the text rather than by a
generator a miner can reproduce offline, and all three are checked by parsers — no judge, no
teacher-agreement, reproducible from the seed on CPU.

FAIRNESS IS LOAD-BEARING. Every item must be answerable from the text shown. An impossible item
fails the TEACHER too, silently shrinking the scored subset and reading as student incapability
— the most expensive bug class on an axis (we shipped two of them in multi_constraint before
catching it). Every generator here therefore verifies its own answer against the rendered
prompt before emitting the item, and drops anything it cannot confirm.
"""
from __future__ import annotations

import random
import re

from ..core import Item
from ..textnorm import NUM_RE as _NUM, make_anchor, norm_digits, same_number


def _anchored(text: str, m, anchor_len: int) -> str | None:
    """A UNIQUE, QUOTABLE preceding phrase for the match, or None.

    Uniqueness makes the gold unambiguous — two occurrences give two defensible answers and we
    would punish an honest model for picking the other one. Quotability matters because the
    prompt renders the anchor inside double quotes: real web text is full of quote characters,
    and an anchor containing one silently breaks the delimiters so the model cannot tell where
    the anchor ends. Found on real fineweb text, and it is the same fairness bug class as an
    unsatisfiable constraint — the teacher fails it too."""
    return make_anchor(text, m.start(), anchor_len)


def _pairs(doc, rng, anchor_len: int):
    """Two independently-anchored numbers in one document."""
    out = []
    for m in _NUM.finditer(doc.text):
        a = _anchored(doc.text, m, anchor_len)
        if a is not None:
            try:
                out.append((a, int(norm_digits(m.group()))))
            except ValueError:
                pass
    rng.shuffle(out)
    return out


class NumericComposition:
    """Multi-hop over real text: locate two anchored values, then combine them.

    Compression breaks COMPOSITION before it breaks lookup — each hop is another chance to lose
    fidelity, which is why PrismML's own agentic/long-horizon numbers (-21.6) sit so far below
    their short-math numbers (-1.4). Two retrievals plus one exact arithmetic step is the
    smallest honest instance of that shape, and the answer is still exact-match checkable."""

    name = "numeric_composition"
    weight = 1.0
    _OPS = ("sum", "difference", "larger")

    def __init__(self, docs):
        self.docs = list(docs)

    def generate(self, seed: int, n: int, difficulty: int = 1) -> list[Item]:
        rng = random.Random(f"{seed}|numeric_composition")
        docs = list(self.docs)
        rng.shuffle(docs)
        anchor_len = 2 if difficulty <= 1 else 3
        items: list[Item] = []
        for d in docs:
            if len(items) >= n:
                break
            ps = _pairs(d, rng, anchor_len)
            if len(ps) < 2:
                continue
            (a1, v1), (a2, v2) = ps[0], ps[1]
            op = self._OPS[len(items) % len(self._OPS)]
            if op == "sum":
                q, ans = "sum of those two numbers", v1 + v2
            elif op == "difference":
                q, ans = ("positive difference between those two numbers", abs(v1 - v2))
            else:
                q, ans = "larger of those two numbers", max(v1, v2)
            prompt = (f"[doc {d.id}] Passage:\n{d.text}\n\n"
                      f'In the passage, find the number immediately after "{a1}" and the number '
                      f'immediately after "{a2}". What is the {q}? '
                      f"Reply with only the number, as `Answer: <number>`.")
            items.append(Item(axis=self.name, prompt=prompt, answer=str(ans),
                              meta={"op": op, "doc": d.id, "difficulty": difficulty,
                                    "v1": v1, "v2": v2}))
        return items

    def check(self, item: Item, output: str) -> bool:
        return same_number(_num_answer(output), item.answer)


class LongContextReal:
    """Retrieval at length over REAL text, not a synthetic haystack.

    RULER-style synthetic haystacks are exactly the closed generator const objected to: the
    filler is machine-made and a miner can reproduce the whole distribution offline. Here the
    haystack is genuine documents concatenated, so the distractor content is as varied as the
    web, while the answer stays a verbatim span. `difficulty` scales how many documents are
    stacked — i.e. how far the needle sits from the question."""

    name = "long_context_real"
    weight = 1.0

    def __init__(self, docs):
        self.docs = list(docs)

    def generate(self, seed: int, n: int, difficulty: int = 1) -> list[Item]:
        rng = random.Random(f"{seed}|long_context_real")
        stack = 3 + 2 * max(1, difficulty)          # docs per haystack
        pool = list(self.docs)
        rng.shuffle(pool)
        items: list[Item] = []
        i = 0
        while len(items) < n and i + stack <= len(pool):
            group = pool[i:i + stack]
            i += stack
            # the needle document is chosen from the group; its anchored number is the answer
            order = list(range(len(group)))
            rng.shuffle(order)
            minted = False
            for gi in order:
                d = group[gi]
                ps = _pairs(d, rng, 3)
                if not ps:
                    continue
                anchor, val = ps[0]
                haystack = "\n\n".join(f"[doc {g.id}]\n{g.text}" for g in group)
                # the anchor must be unique across the WHOLE haystack, not just its own document
                if haystack.count(anchor) != 1:
                    continue
                prompt = (f"{haystack}\n\n"
                          f'Across all the documents above, what number appears immediately '
                          f'after "{anchor}"? Reply with only the number, as `Answer: <number>`.')
                items.append(Item(axis=self.name, prompt=prompt, answer=str(val),
                                  meta={"doc": d.id, "n_docs": len(group),
                                        "difficulty": difficulty,
                                        "chars": len(haystack)}))
                minted = True
                break
            if not minted:
                continue
        return items

    def check(self, item: Item, output: str) -> bool:
        return same_number(_num_answer(output), item.answer)


class ConstrainedExtraction:
    """Correct CONTENT under an exact FORMAT — two independent ways to fail, so they compound.

    Instruction-following is where PrismML measure -15.7 (IFBench) while short math is free, and
    a compressed model that can still read but can no longer hold a format is exactly the
    failure a content-only checker misses. Both halves are parser-verified: the values must be
    the real ones from the document, in document order, in the demanded shape."""

    name = "constrained_extraction"
    weight = 1.0

    def __init__(self, docs):
        self.docs = list(docs)

    def generate(self, seed: int, n: int, difficulty: int = 1) -> list[Item]:
        rng = random.Random(f"{seed}|constrained_extraction")
        docs = list(self.docs)
        rng.shuffle(docs)
        k = 2 + max(1, difficulty)                  # how many values to extract
        shape = ("csv", "lines", "json")
        items: list[Item] = []
        for d in docs:
            if len(items) >= n:
                break
            nums = [m.group() for m in _NUM.finditer(d.text)]
            if len(nums) < k:
                continue
            want = nums[:k]                          # first k in DOCUMENT order
            sh = shape[len(items) % len(shape)]
            if sh == "csv":
                rule = (f"the first {k} numbers that appear in the passage, in order, as a "
                        f"single comma-separated line with no spaces")
            elif sh == "lines":
                rule = (f"the first {k} numbers that appear in the passage, in order, one per "
                        f"line, with no other text")
            else:
                rule = (f"the first {k} numbers that appear in the passage, in order, as a JSON "
                        f'array of strings, e.g. ["1","2"]')
            prompt = (f"[doc {d.id}] Passage:\n{d.text}\n\nList {rule}. Output only that.")
            items.append(Item(axis=self.name, prompt=prompt, answer={"vals": want, "shape": sh},
                              meta={"shape": sh, "k": k, "doc": d.id, "difficulty": difficulty}))
        return items

    def check(self, item: Item, output: str) -> bool:
        """Content AND shape. A model that gets the numbers right in the wrong shape fails, and
        so does one that nails the shape with wrong numbers — that independence is the point."""
        if not output:
            return False
        spec = item.answer
        body = output.strip()
        want, sh = spec["vals"], spec["shape"]
        # compare NUMERICALLY element-wise: an Arabic passage renders gold as Arabic-Indic while
        # a model may answer in ASCII. Shape is still exact — that half is the point of the axis.
        def eq(got: list) -> bool:
            return len(got) == len(want) and all(same_number(g, w) for g, w in zip(got, want))

        if sh == "csv":
            line = body.splitlines()[0].strip() if body.splitlines() else ""
            return eq([p.strip() for p in line.split(",")])
        if sh == "lines":
            return eq([ln.strip() for ln in body.splitlines() if ln.strip()])
        m = re.search(r"\[.*?\]", body, re.S)
        if not m:
            return False
        try:
            import json as _json
            return eq([str(x) for x in _json.loads(m.group())])
        except Exception:
            return False


def _num_answer(output: str) -> str | None:
    """First number after the answer marker; unmarked replies must be terse (an unmarked dump
    otherwise passes by containing the right digits somewhere — measured on real text)."""
    if not output:
        return None
    tail = re.split(r"(?i)(?:final answer|answer)\s*[:=]", output)
    if len(tail) > 1:
        seg = tail[1]
    elif len(output.split()) > 6:
        return None
    else:
        seg = output
    seg = norm_digits(seg)
    m = re.search(r"-?\d[\d,]*", seg)
    return m.group().replace(",", "") if m else None
