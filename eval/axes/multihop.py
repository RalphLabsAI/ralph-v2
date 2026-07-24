"""Multi-hop axis — compose N stated facts to reach an answer.

Multi-hop composition is compression-fragile in a way retrieval is not: each hop compounds,
so a student that lost a little reasoning fidelity fails 3-hop chains while still acing
1-hop lookups. The long_context axis probes RETRIEVAL over a haystack; this probes
COMPOSITION, a distinct failure mode.

Every fact needed is stated IN the prompt over synthetic entities, so this measures chained
reasoning rather than memorized world knowledge (no real-corpus dependency, and nothing a
miner can pre-distill as trivia).

The SHORTCUT DISTRACTOR is what makes it genuinely multi-hop: alongside
    The mentor of Ann is Bob.   The city of Bob is Prague.
the prompt also states
    The city of Ann is Vienna.
so a model that pattern-matches "city of Ann" instead of following mentor->Bob->city
answers Vienna and is caught. Checker is exact entity match — no LLM judge.
"""
from __future__ import annotations

import random
import re

from ..core import Item

_SYL_A = ["Kal", "Mer", "Tor", "Vin", "Bra", "Sel", "Dor", "Fen", "Gar", "Hal",
          "Jor", "Lum", "Nar", "Pel", "Quin", "Rho", "Sav", "Tal", "Ver", "Wyn"]
_SYL_B = ["dris", "vane", "ric", "mont", "wick", "thorn", "ford", "sby", "lyn", "grave",
          "mere", "stad", "holm", "bury", "crest", "dale", "field", "gate", "haven", "port"]
# relation, phrasing — all one-to-one so the chain is unambiguous
_RELATIONS = ["mentor", "deputy", "archivist", "successor", "supervisor",
              "district", "depot", "registry", "custodian", "sponsor"]


def _names(rng: random.Random, k: int) -> list[str]:
    """k distinct synthetic proper nouns."""
    out: list[str] = []
    while len(out) < k:
        n = rng.choice(_SYL_A) + rng.choice(_SYL_B)
        if n not in out:
            out.append(n)
    return out


class MultiHop:
    name = "multihop"
    weight = 1.0

    def generate(self, seed: int, n: int, difficulty: int = 1) -> list[Item]:
        rng = random.Random(seed)
        hops = max(2, min(4, difficulty + 1))     # d1->2 hops, d2->3, d3->4
        items: list[Item] = []
        for i in range(n):
            # chain e0 -r1-> e1 -r2-> ... -r_hops-> e_hops
            ents = _names(rng, hops + 1 + 3 * hops)      # chain + distractor entities
            chain, pool = ents[:hops + 1], ents[hops + 1:]
            rels = rng.sample(_RELATIONS, hops)

            facts = [f"The {rels[j]} of {chain[j]} is {chain[j + 1]}." for j in range(hops)]

            # SHORTCUT distractor: apply the FINAL relation directly to the start entity,
            # so pattern-matching the question's outer relation onto e0 gives a wrong answer.
            trap = pool[0]
            facts.append(f"The {rels[-1]} of {chain[0]} is {trap}.")
            # plain distractors over unrelated entities, reusing the same relations.
            # Density scales with difficulty: measured at 2 hops with 2*hops distractors the
            # pinned teacher only reached ~53% (borderline for MIN_AXIS_N liveness), and an
            # axis the TEACHER cannot clear measures nothing.
            n_distract = hops if difficulty <= 1 else 2 * hops
            for j in range(1, min(len(pool) - 1, n_distract)):
                facts.append(f"The {rng.choice(rels)} of {pool[j]} is {pool[j + 1]}.")
            rng.shuffle(facts)

            # nested question: the r_k of the r_{k-1} of ... the r_1 of e0
            q = f"the {rels[0]} of {chain[0]}"
            for r in rels[1:]:
                q = f"the {r} of {q}"
            prompt = ("Facts:\n" + "\n".join(facts) +
                      f"\n\nUsing only the facts above, what is {q}? "
                      f"Reply with only the name, as `Answer: <name>`.")
            items.append(Item(axis=self.name, prompt=prompt, answer=chain[-1],
                              meta={"hops": hops, "idx": i, "difficulty": difficulty,
                                    "trap": trap, "entities": ents}))
        return items

    def check(self, item: Item, output: str) -> bool:
        """Exact entity match. Takes the FIRST known entity named after the answer marker,
        so listing several candidates (or naming the trap first) cannot shotgun a pass."""
        if not output:
            return False
        tail = re.split(r"(?i)(?:final answer|answer)\s*[:=]", output)
        seg = tail[1] if len(tail) > 1 else output
        known = set(item.meta.get("entities", []))
        for tok in re.findall(r"[A-Za-z]+", seg):
            if tok in known:
                return tok == item.answer
        return False
