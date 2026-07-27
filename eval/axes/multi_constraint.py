"""Multi-constraint axis — COMPOUNDING verifiable constraints, where compression really breaks.

WHY THIS AXIS EXISTS. PrismML's own 15-benchmark table says exactly where 14x compression costs
capability, and it is not where our synthetic axes were looking:

    tau^2-Bench (agentic, long-horizon)  -21.56     MATH-500   -1.40
    MMMU-Pro (vision)                    -19.46     GSM8K      -2.50  (ternary BEATS fp16)
    IFBench (multi-constraint)           -15.67
    LiveCodeBench                        -11.37

Short-form math is FREE under compression — an axis where the 54 GB teacher and the 3.8 GB
student are within 1.4 points is not an axis, it is a saturated ritual. The damage lives in
LONG-HORIZON and MULTI-CONSTRAINT work, because those COMPOUND: hold k independent constraints
at per-constraint reliability p and the joint pass rate is ~p^k, so a small per-step fidelity
loss that is invisible at k=1 is stark at k=4. That is precisely the difference between IFEval
(single constraint, -9.5) and IFBench (multi-constraint, -15.7) in their own numbers.

WHY IT IS ALSO THE ANTI-OVERFIT FIX — the same property, seen from the other side. An axis with
no discriminative power is exactly an axis you can saturate and then Goodhart: the score stays
beautiful while the substrate degrades (the predecessor subnet watched gsm8k fall 5pp on
held-out while their composite held flat). Measuring where compression actually bites means the
metric MOVES when capability moves, which is what makes it hard to fake. And the search space
here is combinatorial: k constraints drawn from a parameterized family, all of which must hold
simultaneously, so there is no small set of answers to pre-fit — you have to actually follow
instructions.

Every constraint is checked by a PARSER. No judge, no teacher-agreement, CPU-auditable,
reproducible from the seed.
"""
from __future__ import annotations

import random
import re

from ..core import Item

_TOPICS = ["a lighthouse", "the water cycle", "a chess opening", "sourdough bread",
           "tidal energy", "the printing press", "a coral reef", "compilers",
           "migratory birds", "the bronze age", "photosynthesis", "a suspension bridge",
           "monsoon season", "glassblowing", "the postal service", "kelp forests"]
_WORDS = ["azure", "lattice", "quorum", "beacon", "granite", "cipher", "harbor",
          "meridian", "ember", "trellis", "fathom", "cobalt"]

# ---- SHAPE constraints: mutually exclusive (they all govern overall form) ---------------

def _s_exact_words(rng):
    n = rng.choice([12, 16, 20, 24, 28])
    return (f"be exactly {n} words long", {"kind": "exact_words", "n": n})


def _s_bullets(rng):
    n = rng.choice([3, 4, 5])
    return (f"be exactly {n} bullet points, each line starting with '- '",
            {"kind": "bullets", "n": n})


def _s_sentences(rng):
    n = rng.choice([2, 3, 4])
    return (f"be exactly {n} sentences, each ending with a period",
            {"kind": "sentences", "n": n})


# ---- MODIFIER constraints: freely composable with a shape and with each other ----------

def _m_include(rng):
    w = rng.choice(_WORDS)
    return f"include the word '{w}'", {"kind": "include", "word": w}


def _m_forbid(rng):
    w = rng.choice(_WORDS)
    return f"never use the word '{w}'", {"kind": "forbid", "word": w}


def _m_end_with(rng):
    t = rng.choice(["END", "DONE", "FINIS", "STOP"])
    return f"end with the exact token {t}", {"kind": "end_with", "token": t}


def _m_no_commas(rng):
    return "contain no commas", {"kind": "no_commas"}


def _m_lowercase(rng):
    return "be entirely lowercase", {"kind": "lowercase"}


def _m_start_with(rng):
    w = rng.choice(["Notably", "Broadly", "Curiously", "Plainly"])
    return f"start with the word '{w}'", {"kind": "start_with", "word": w}


_SHAPES = [_s_exact_words, _s_bullets, _s_sentences]
_MODS = [_m_include, _m_forbid, _m_end_with, _m_no_commas, _m_lowercase, _m_start_with]

# Pairs that CANNOT both hold. An unsatisfiable item is worse than useless: the teacher fails it
# too, so it silently shrinks the teacher-passed subset and reads as student incapability —
# "brittleness masquerading as incapability", the single most expensive class of bug on an axis.
# `end_with` demands an exact UPPERCASE token; `lowercase` demands the whole body be lowercased.
_INCOMPATIBLE = {
    frozenset({"end_with", "lowercase"}),    # exact UPPERCASE token vs all-lowercase body
    frozenset({"bullets", "start_with"}),    # every line must open with "- ", so the body cannot
                                             # also open with a given word
}


def _compatible(spec: dict, chosen: list[dict]) -> bool:
    k = spec["kind"]
    for p in chosen:
        if frozenset({k, p["kind"]}) in _INCOMPATIBLE:
            return False
        # forbid/include drawn from the same word pool must not contradict
        if {k, p["kind"]} == {"include", "forbid"} and spec.get("word") == p.get("word"):
            return False
    return True


def _n_constraints(difficulty: int) -> int:
    """The compounding knob AND the calibration knob. k must sit where the TEACHER still
    clears the axis (an axis the teacher fails measures nothing) while the base does not —
    raise k for a stronger pair, lower it for a weaker one."""
    return max(2, min(5, difficulty + 1))


class MultiConstraint:
    name = "multi_constraint"
    weight = 1.0

    def generate(self, seed: int, n: int, difficulty: int = 1) -> list[Item]:
        rng = random.Random(f"{seed}|multi_constraint")
        k = _n_constraints(difficulty)
        items: list[Item] = []
        for i in range(n):
            topic = rng.choice(_TOPICS)
            shape_txt, shape_spec = rng.choice(_SHAPES)(rng)
            mods = rng.sample(_MODS, min(k - 1, len(_MODS)))
            specs = [shape_spec]
            texts = [shape_txt]
            for m in mods:
                t, s = m(rng)
                if not _compatible(s, specs):
                    continue
                specs.append(s)
                texts.append(t)
            rules = "\n".join(f"{j+1}. It must {t}." for j, t in enumerate(texts))
            prompt = (f"Write a response about {topic}.\n\n"
                      f"Your response must satisfy ALL of the following rules:\n{rules}\n\n"
                      f"Output only the response itself, with no preamble and no explanation.")
            items.append(Item(axis=self.name, prompt=prompt, answer=specs,
                              meta={"k": len(specs), "idx": i, "difficulty": difficulty,
                                    "kinds": [s["kind"] for s in specs]}))
        return items

    # ---- deterministic per-constraint parsers ------------------------------------------

    @staticmethod
    def _one(spec: dict, out: str) -> bool:
        kind = spec["kind"]
        body = out.strip()
        if kind == "exact_words":
            return len(body.split()) == spec["n"]
        if kind == "bullets":
            lines = [ln for ln in body.splitlines() if ln.strip()]
            return len(lines) == spec["n"] and all(ln.lstrip().startswith("- ") for ln in lines)
        if kind == "sentences":
            # count terminal periods that end a sentence (ignore a trailing end-token line)
            core = re.sub(r"\s*(END|DONE|FINIS|STOP)\s*$", "", body)
            return len(re.findall(r"[.!?](?:\s|$)", core)) == spec["n"]
        if kind == "include":
            return re.search(rf"\b{re.escape(spec['word'])}\b", body, re.I) is not None
        if kind == "forbid":
            return re.search(rf"\b{re.escape(spec['word'])}\b", body, re.I) is None
        if kind == "end_with":
            return body.rstrip().endswith(spec["token"])
        if kind == "no_commas":
            return "," not in body
        if kind == "lowercase":
            return body == body.lower()
        if kind == "start_with":
            return body.lstrip().lower().startswith(spec["word"].lower())
        return False

    def check(self, item: Item, output: str) -> bool:
        """ALL constraints must hold. This is the compounding property that makes the axis
        discriminative — partial credit would wash the signal straight back out."""
        if not output or not output.strip():
            return False
        return all(self._one(s, output) for s in item.answer)

    def per_constraint(self, item: Item, output: str) -> dict:
        """Diagnostic only (never scored): which constraint kinds a student drops first is the
        readout an operator needs when calibrating k."""
        return {s["kind"]: self._one(s, output or "") for s in item.answer}
