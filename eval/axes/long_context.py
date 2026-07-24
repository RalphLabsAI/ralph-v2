"""Long-context axis — retrieval + aggregation over a synthetic haystack.

Long-context retrieval/aggregation is the capability compression breaks FIRST (RULER-style
low-bit degradation), and the math/code/instruction axes don't touch it at all — so a
compressed student that quietly lost it sails through the current cover (the SN97 shape).
This axis probes it with a DETERMINISTIC checker (no LLM judge): a context of
entity->value facts buried in distractor filler, and a query whose answer is computed
from the raw text:

  * retrieve  — the value of one named entity (needle-in-haystack)
  * count_gt  — how many entities exceed a threshold (aggregation)
  * argmax    — which entity has the greatest value (aggregation + identity)
  * sum       — the total over all entities (full-context aggregation)

difficulty scales the context length (number of facts + filler), so a student with a
degraded effective context window fails the deeper items. Fresh per round from the seed.

NOTE: this is a SYNTHETIC RULER-style probe (self-contained, deterministic), not real
documents — the post-commit real-corpus stream (docs/ship-status) is separate future work.
Here the point is a fragile-axis the current cover lacks, checkable bit-for-bit on CPU.
"""
from __future__ import annotations

import random
import re

from ..core import Item

_ALPHA = "ABCDEFGHJKLMNPQRSTUVWXYZ"
# (attribute, unit) pairs — all integer-valued so the checker is exact
_ATTRS = [("mass", "grams"), ("length", "millimeters"), ("count", "units"),
          ("price", "credits"), ("age", "days"), ("power", "watts")]
_FILLERS = [
    "The depot logged a routine inspection with no anomalies recorded.",
    "Maintenance notes indicate the shelving was reorganized last quarter.",
    "A shipment manifest was archived pending further review.",
    "The calibration log shows nominal readings across the board.",
    "An unrelated memo referenced the annual audit schedule.",
    "The storage bay temperature remained within tolerance overnight.",
]
_NUM = re.compile(r"-?\d+(?:,\d{3})*")


def _extract_int(output: str) -> str | None:
    if not output:
        return None
    # number after the FIRST answer marker (a model that answers then degenerates into
    # repeated "Answer:" lines otherwise has the wrong one pulled from the tail).
    tail = re.split(r"(?i)(?:final answer|answer)\s*[:=]", output)
    if len(tail) > 1:
        nums = _NUM.findall(tail[1])
        if nums:
            return nums[0].replace(",", "")
    nums = _NUM.findall(output)
    return nums[-1].replace(",", "") if nums else None


_UNIT = re.compile(r"Unit-[A-Z]\d{2}")


def _extract_unit(output: str) -> str | None:
    """First unit id AFTER the answer marker (mirrors _extract_int): naming several ids or
    echoing the context — which always contains the answer id — cannot shotgun a pass."""
    if not output:
        return None
    tail = re.split(r"(?i)(?:final answer|answer)\s*[:=]", output)
    seg = tail[1] if len(tail) > 1 else output
    m = _UNIT.search(seg)
    return m.group(0) if m else None


class LongContext:
    name = "long_context"
    weight = 1.0

    def __init__(self, base_facts: int = 24):
        self.base_facts = base_facts
        self._qtypes = ["retrieve", "count_gt", "argmax", "sum"]

    def generate(self, seed: int, n: int, difficulty: int = 1) -> list[Item]:
        rng = random.Random(seed)
        n_lines = self.base_facts * max(1, difficulty)
        items: list[Item] = []
        for i in range(n):
            n_ent = max(6, n_lines // 2)
            ents: list[str] = []
            while len(ents) < n_ent:
                e = f"Unit-{rng.choice(_ALPHA)}{rng.randint(10, 99)}"
                if e not in ents:
                    ents.append(e)
            attr, unit = rng.choice(_ATTRS)
            vals = {e: rng.randint(10, 999) for e in ents}
            # resolve a UNIQUE maximum BEFORE rendering the haystack, so the argmax answer is
            # always recoverable from the printed facts. (Previously the tie-break bumped a
            # value inside _query AFTER this text was built, so on a max-tie the answer
            # disagreed with the context and penalized a model that read it correctly.)
            mx = max(vals.values())
            top = [e for e, v in vals.items() if v == mx]
            if len(top) > 1:
                vals[top[0]] = mx + rng.randint(1, 50)
            facts = [f"The {attr} of {e} is {vals[e]} {unit}." for e in ents]
            fillers = [rng.choice(_FILLERS) for _ in range(max(0, n_lines - len(ents)))]
            lines = facts + fillers
            rng.shuffle(lines)
            context = "\n".join(lines)

            qtype = self._qtypes[i % len(self._qtypes)]
            q, ans, numeric = self._query(rng, qtype, ents, vals, attr, unit)
            kind = "number" if numeric else "identifier (e.g. Unit-X00)"
            prompt = (f"{context}\n\nQuestion: {q}\n"
                      f"Answer with only the {kind}, on its own line as `Answer: <value>`.")
            items.append(Item(axis=self.name, prompt=prompt, answer=str(ans),
                              meta={"qtype": qtype, "numeric": numeric, "idx": i,
                                    "difficulty": difficulty, "n_lines": n_lines}))
        return items

    def _query(self, rng, qtype, ents, vals, attr, unit):
        if qtype == "retrieve":
            e = rng.choice(ents)
            return f"What is the {attr} of {e}?", vals[e], True
        if qtype == "count_gt":
            thr = sorted(vals.values())[len(vals) // 2]  # median -> non-trivial count
            cnt = sum(1 for v in vals.values() if v > thr)
            return (f"How many units have a {attr} greater than {thr} {unit}?", cnt, True)
        if qtype == "argmax":
            # the max is already unique (resolved before the haystack was rendered), so the
            # answer is well-defined and printed in the context.
            best = max(vals, key=lambda e: vals[e])
            return f"Which unit has the greatest {attr}?", best, False
        # sum
        return (f"What is the total {attr} of all units, in {unit}?", sum(vals.values()), True)

    def check(self, item: Item, output: str) -> bool:
        if item.meta.get("numeric", True):
            got = _extract_int(output)
            return got is not None and got == str(item.answer)
        # identifier: take the FIRST unit id AFTER the answer marker and require an exact
        # match. A substring-anywhere match let a model shotgun several ids (or echo the
        # context, which always names the answer) and pass without doing the retrieval.
        got = _extract_unit(output)
        return got is not None and got == str(item.answer)
