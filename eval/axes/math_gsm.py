"""Math axis — templated word problems with a deterministic numeric checker.

Follows GSM-Symbolic (arXiv:2410.05229): each template fixes a solution PROCEDURE
and parameterizes the names/quantities, so resampling yields unlimited fresh
instances that are genuinely new strings but test the same reasoning. Two knobs
matter for discrimination:

  * difficulty — adds reasoning steps (Plus-1 / Plus-2 in the paper's terms)
  * NoOp distractors — an extra clause that is plausible but irrelevant. Models that
    pattern-match rather than reason get dragged down by these, which is exactly the
    failure mode a compressed student exhibits first.

The checker is exact numeric match on the final answer, extracted from the tail of
the response. No LLM judge, no partial credit, no ambiguity.
"""
from __future__ import annotations

import random
import re
from fractions import Fraction

from ..core import Item

NAMES = ["Maya", "Diego", "Priya", "Tomas", "Anika", "Rafael", "Yuki", "Omar",
         "Lena", "Kwame", "Sofia", "Idris", "Mei", "Noor", "Ravi", "Elena"]
OBJECTS = ["apples", "notebooks", "marbles", "tickets", "seedlings", "batteries",
           "candles", "stamps", "bolts", "cards", "beads", "tiles"]
PLACES = ["the market", "the workshop", "the fair", "the depot", "the library"]

NOOPS = [
    "Last year the price of {obj} was different, but that does not affect the count.",
    "{other} owns a bicycle that {other_pron} rarely uses.",
    "The shop at {place} closes at six on weekends.",
    "Two of the {obj} are slightly larger than the rest.",
]


def _fmt(n: Fraction) -> str:
    """Canonical answer string: integers bare, else a reduced decimal."""
    if n.denominator == 1:
        return str(n.numerator)
    val = float(n)
    return f"{val:.4f}".rstrip("0").rstrip(".")


def _noop(rng: random.Random, obj: str, subject: str) -> str:
    other = rng.choice([x for x in NAMES if x != subject])
    return rng.choice(NOOPS).format(
        obj=obj, other=other, other_pron=rng.choice(["he", "she", "they"]), place=rng.choice(PLACES)
    )


# ---- templates: each returns (question_text, exact answer) ------------------
# Every template must be solvable exactly; answers are Fractions so there is no
# floating-point ambiguity in grading.

def _t_rate_total(rng: random.Random, difficulty: int) -> tuple[str, Fraction]:
    who, obj, place = rng.choice(NAMES), rng.choice(OBJECTS), rng.choice(PLACES)
    per_day, days = rng.randint(3, 19), rng.randint(3, 12)
    total = Fraction(per_day * days)
    text = (f"{who} collects {per_day} {obj} each day at {place}. "
            f"After {days} days, how many {obj} does {who} have?")
    if difficulty >= 2:
        given_away = rng.randint(2, max(3, per_day))
        total -= given_away
        text = (f"{who} collects {per_day} {obj} each day at {place}. "
                f"After {days} days, {who} gives away {given_away} {obj}. "
                f"How many {obj} does {who} have left?")
    if difficulty >= 3:
        found = rng.randint(2, 15)
        total += found
        text = text.rstrip("?").rsplit("How many", 1)[0].rstrip() + \
            f" Then {who} finds {found} more {obj}. How many {obj} does {who} have?"
    return text, total


def _t_split_share(rng: random.Random, difficulty: int) -> tuple[str, Fraction]:
    who, obj = rng.choice(NAMES), rng.choice(OBJECTS)
    friends = rng.randint(2, 8)
    each = rng.randint(2, 15)
    total = friends * each
    text = (f"{who} divides {total} {obj} equally among {friends} friends. "
            f"How many {obj} does each friend receive?")
    ans = Fraction(total, friends)
    if difficulty >= 2:
        kept = rng.randint(1, 9)
        total_all = total + kept
        text = (f"{who} has {total_all} {obj}, keeps {kept}, and divides the rest equally "
                f"among {friends} friends. How many {obj} does each friend receive?")
        ans = Fraction(total, friends)
    if difficulty >= 3:
        extra = rng.randint(1, 6)
        text = text.rstrip("?") + f", if each friend then buys {extra} more {obj}?"
        ans = ans + extra
    return text, ans


def _t_price_change(rng: random.Random, difficulty: int) -> tuple[str, Fraction]:
    who, obj = rng.choice(NAMES), rng.choice(OBJECTS)
    unit, count = rng.randint(2, 25), rng.randint(2, 14)
    cost = Fraction(unit * count)
    text = (f"{obj.capitalize()} cost {unit} coins each. {who} buys {count}. "
            f"How many coins does {who} spend?")
    if difficulty >= 2:
        disc = rng.choice([2, 4, 5, 10])
        cost = cost - cost / disc
        text = (f"{obj.capitalize()} cost {unit} coins each. {who} buys {count} and receives "
                f"a discount of one {disc}th off the total. How many coins does {who} spend?")
    if difficulty >= 3:
        fee = rng.randint(1, 9)
        cost += fee
        text = text.rstrip("?") + f", including a delivery fee of {fee} coins?"
    return text, cost


TEMPLATES = {
    "rate_total": _t_rate_total,
    "split_share": _t_split_share,
    "price_change": _t_price_change,
}

_NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_answer(output: str) -> str | None:
    """Pull the model's final numeric answer.

    Prefers an explicit \\boxed{} or 'answer:' marker; otherwise takes the LAST
    number in the response, which is the near-universal convention for CoT output.
    """
    if not output:
        return None
    boxed = re.findall(r"\\boxed\{([^}]*)\}", output)
    if boxed:
        nums = _NUM.findall(boxed[-1])
        if nums:
            return nums[-1].replace(",", "")
    # take the number after the FIRST answer marker, not the last: a model that states its
    # answer then keeps writing (or degenerates into repeated "Answer:" lines) otherwise
    # gets the wrong number pulled from the tail.
    tail = re.split(r"(?i)(?:final answer|answer)\s*[:=]", output)
    if len(tail) > 1:
        nums = _NUM.findall(tail[1])
        if nums:
            return nums[0].replace(",", "")
    nums = _NUM.findall(output)
    return nums[-1].replace(",", "") if nums else None


class MathGSM:
    """GSM-Symbolic-style math axis."""

    name = "math"
    weight = 0.25

    def __init__(self, noop_rate: float = 0.35):
        self.noop_rate = noop_rate

    def generate(self, seed: int, n: int, difficulty: int = 1) -> list[Item]:
        rng = random.Random(seed)
        keys = sorted(TEMPLATES)
        items: list[Item] = []
        for i in range(n):
            key = keys[i % len(keys)]
            text, ans = TEMPLATES[key](rng, difficulty)
            has_noop = rng.random() < self.noop_rate
            if has_noop:
                subject = text.split()[0]
                obj = next((o for o in OBJECTS if o in text), "items")
                # insert the distractor mid-problem, where it is most tempting
                parts = text.split(". ")
                parts.insert(max(1, len(parts) - 1), _noop(rng, obj, subject))
                text = ". ".join(parts)
            prompt = (
                f"{text}\n\n"
                "Solve step by step, then give the final numeric answer on its own line "
                "in the form: Answer: <number>"
            )
            items.append(Item(
                axis=self.name, prompt=prompt, answer=_fmt(ans),
                meta={"template": key, "idx": i, "difficulty": difficulty, "noop": has_noop},
            ))
        return items

    def check(self, item: Item, output: str) -> bool:
        got = extract_answer(output)
        if got is None:
            return False
        try:
            return abs(Fraction(got) - Fraction(str(item.answer))) < Fraction(1, 10_000)
        except (ValueError, ZeroDivisionError):
            return str(got).strip() == str(item.answer).strip()
