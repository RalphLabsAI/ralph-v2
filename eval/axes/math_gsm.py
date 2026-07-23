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


_FRACTION_WORD = {2: "half", 3: "third", 4: "quarter", 5: "fifth", 10: "tenth"}


def _t_price_change(rng: random.Random, difficulty: int) -> tuple[str, Fraction]:
    who, obj = rng.choice(NAMES), rng.choice(OBJECTS)
    unit = rng.randint(2, 25)
    disc = rng.choice([2, 4, 5, 10])
    # count is a multiple of the discount denominator so the discounted total is an EXACT
    # integer — a decimal answer invites the model to write a fraction the numeric
    # extractor would mis-read.
    count = disc * rng.randint(1, 4) if difficulty >= 2 else rng.randint(2, 14)
    cost = Fraction(unit * count)
    text = (f"{obj.capitalize()} cost {unit} coins each. {who} buys {count}. "
            f"How many coins does {who} spend?")
    if difficulty >= 2:
        cost = cost - cost / disc
        # "one 2th off" was ambiguous English and depressed pass rates for the wrong reason
        text = (f"{obj.capitalize()} cost {unit} coins each. {who} buys {count} and receives "
                f"a discount of one {_FRACTION_WORD[disc]} off the total. "
                f"How many coins does {who} spend?")
    if difficulty >= 3:
        fee = rng.randint(1, 9)
        cost += fee
        text = text.rstrip("?") + f", including a delivery fee of {fee} coins?"
    return text, cost


# --- harder families -------------------------------------------------------------------
# The three grade-school templates above are compression-ROBUST: the pinned 0.5B base
# passed 22/25 of them, which collapses the retention denominator (teacher-base) and makes
# the axis measure almost nothing. These require genuinely composed multi-step reasoning
# (work-rate, mixture, interest, ratio splits, back-solving) while keeping answers EXACT
# Fractions so the checker stays deterministic.

def _t_work_rate(rng: random.Random, difficulty: int) -> tuple[str, Fraction]:
    """Combined work rates — reciprocal reasoning, not a single arithmetic chain.
    Constructed so the combined time is an EXACT INTEGER: for 1/a + 1/b = 1/t the integer
    solutions are a = t+x, b = t + t^2/x over divisors x of t^2. (Answers stay integral so
    a model answering a fraction like "14/9" can't be mis-graded by the numeric extractor.)"""
    t = rng.randint(2, 10)
    x = rng.choice([d for d in range(1, t * t + 1) if (t * t) % d == 0])
    a, b = t + x, t + (t * t) // x
    who, obj = rng.choice(NAMES), rng.choice(OBJECTS)
    other = rng.choice([x for x in NAMES if x != who])   # distinct parties
    text = (f"{who} can sort a crate of {obj} alone in {a} hours, and {other} can sort the "
            f"same crate alone in {b} hours. Working together at those rates, how many "
            f"hours does it take them to sort one crate?")
    ans = Fraction(t)
    if difficulty >= 3:
        crates = rng.randint(2, 6)
        ans = Fraction(t * crates)
        text = (f"{who} can sort a crate of {obj} alone in {a} hours, and {other} in {b} "
                f"hours. Working together, how many hours do {crates} crates take?")
    return text, ans


def _t_mixture(rng: random.Random, difficulty: int) -> tuple[str, Fraction]:
    """Weighted average — a classic trap for shallow pattern-matching (averaging the two
    prices instead of weighting them). Prices are constructed around a chosen integer
    average so the answer is exact: p1 = avg + q2*d, p2 = avg - q1*d."""
    q1, q2 = rng.randint(2, 12), rng.randint(2, 12)
    d = rng.randint(1, 3)
    avg = rng.randint(q1 * d + 2, q1 * d + 30)
    p1, p2 = avg + q2 * d, avg - q1 * d
    place = rng.choice(PLACES)
    text = (f"A trader at {place} mixes {q1} kilograms costing {p1} coins per kilogram with "
            f"{q2} kilograms costing {p2} coins per kilogram. What is the cost per kilogram "
            f"of the mixture?")
    ans = Fraction(avg)
    if difficulty >= 3:
        ans = Fraction(avg * (q1 + q2))
        text = text.rstrip("?").replace("What is the cost per kilogram of the mixture",
                                        "What is the total cost of the whole mixture")
    return text, ans


def _t_back_solve(rng: random.Random, difficulty: int) -> tuple[str, Fraction]:
    """Invert a described forward process — must run the steps BACKWARD."""
    who, obj = rng.choice(NAMES), rng.choice(OBJECTS)
    spent = rng.randint(2, 6)
    gave = rng.randint(3, 15)
    left = rng.randint(4, 30)
    start = Fraction((left + gave) * spent)
    text = (f"{who} split a pile of {obj} into {spent} equal groups and kept one group. "
            f"{who} then gave away {gave} {obj} from that group and has {left} left. "
            f"How many {obj} were in the original pile?")
    if difficulty >= 3:
        found = rng.randint(2, 12)
        start = Fraction((left - found + gave) * spent)
        text = (f"{who} split a pile of {obj} into {spent} equal groups and kept one group. "
                f"{who} gave away {gave} {obj}, then found {found} more, and now has {left}. "
                f"How many {obj} were in the original pile?")
    return text, start


def _t_ratio_split(rng: random.Random, difficulty: int) -> tuple[str, Fraction]:
    """Split a total in a ratio — needs part-of-whole reasoning."""
    r1, r2 = rng.randint(1, 7), rng.randint(1, 7)
    per_part = rng.randint(3, 20)
    total = (r1 + r2) * per_part
    who, obj = rng.choice(NAMES), rng.choice(OBJECTS)
    other = rng.choice([x for x in NAMES if x != who])   # distinct parties
    ans = Fraction(total * r1, r1 + r2)
    text = (f"{who} and {other} divide {total} {obj} in the ratio {r1}:{r2}. "
            f"How many {obj} does {who} receive?")
    if difficulty >= 3:
        r3 = rng.randint(1, 6)
        total = (r1 + r2 + r3) * per_part
        ans = Fraction(total * r1, r1 + r2 + r3)
        text = (f"{who}, {other} and a third partner divide {total} {obj} in the ratio "
                f"{r1}:{r2}:{r3}. How many {obj} does {who} receive?")
    return text, ans


def _t_compound(rng: random.Random, difficulty: int) -> tuple[str, Fraction]:
    """Repeated proportional change — compounding, not a single percentage step. The base
    is a multiple of denom^years so the compounded value is an exact integer."""
    denom = rng.choice([2, 4, 5, 10])
    years = 2 if difficulty < 3 else 3
    m = rng.randint(1, 20)
    base = m * denom ** years
    ans = Fraction(m * (denom + 1) ** years)
    place = rng.choice(PLACES)
    text = (f"A fund at {place} holds {base} coins and grows by one {_FRACTION_WORD[denom]} "
            f"of its value each year. What is its value after {years} years?")
    return text, ans


TEMPLATES = {
    "rate_total": _t_rate_total,
    "split_share": _t_split_share,
    "price_change": _t_price_change,
    "work_rate": _t_work_rate,
    "mixture": _t_mixture,
    "back_solve": _t_back_solve,
    "ratio_split": _t_ratio_split,
    "compound": _t_compound,
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
