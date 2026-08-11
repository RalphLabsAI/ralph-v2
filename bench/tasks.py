"""The task suite. Four axes, because one benchmark answered one question badly.

GSM8K alone told us the ternary crown scores 22.5% on English arithmetic. It did not tell us whether
that model is broken everywhere or merely bad at maths, and it could not see the axis the subnet's
own metric is built on — a model that holds English and loses every other language scores fine on
GSM8K and is exactly what worst-slice exists to catch.

    gsm8k      English arithmetic, generative      does it still compute
    mgsm       the SAME arithmetic in 6 languages  does it hold across languages
    mmlu       knowledge, multiple choice          does it still know things
    hellaswag  commonsense, multiple choice        does it still make sense

THE DESIGN DECISION THAT MATTERS: a multiple-choice score conflates two failures. A model that
answers "B" wrongly and a model that ignores the format and rambles both score zero, and only the
first one has actually lost knowledge. The 22.5% result came from a harness that measured the prompt
rather than the model, so every task here reports `format_fail` separately from `correct`. A low
score with high format-failure is an instruction-following collapse; a low score with clean
formatting is genuine capability loss. Those need different fixes and must not be one number.
"""
from __future__ import annotations

import re

# --- answer extraction --------------------------------------------------------------------------

NUM = re.compile(r"(-?[\d,]*\.?\d+)")
LETTER = re.compile(r"\b([A-D])\b")


def _num(text: str) -> str:
    """Prefer what follows the '####' marker we asked for; fall back to the last number."""
    if "####" in text:
        tail = text.split("####")[-1]
        hit = NUM.search(tail.replace(",", ""))
        if hit:
            return hit.group(1).strip(".")
    hits = NUM.findall(text.replace(",", ""))
    return hits[-1].strip(".") if hits else ""


def _letter(text: str) -> str:
    """First standalone A-D. Answers arrive as 'B', 'B.', '**B**', 'Answer: B'."""
    t = text.strip()
    for pat in (r"^\s*\**\s*([A-D])\b", r"[Aa]nswer\s*[:\-]?\s*\**\s*([A-D])\b", r"\b([A-D])\b"):
        m = re.search(pat, t)
        if m:
            return m.group(1).upper()
    return ""


# --- tasks --------------------------------------------------------------------------------------

MC_INSTRUCTION = ("{q}\n\n"
                  "A. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
                  "Reply with only the letter of the correct answer.")

MATH_INSTRUCTION = ("Solve the problem. Reason briefly, then give the final numeric answer on its "
                    "own line after '####'.\n\nProblem: {q}")

MGSM_LANGS = ("es", "fr", "de", "ru", "zh", "sw")   # sw = Swahili: the low-resource stress case


def load_gsm8k(limit):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for i in range(min(limit, len(ds))):
        gold = ds[i]["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"prompt": MATH_INSTRUCTION.format(q=ds[i]["question"]), "gold": gold,
                    "kind": "num", "slice": "en"})
    return out


def load_mgsm(limit):
    """The same arithmetic in six languages. This is the axis GSM8K is blind to, and the one the
    subnet's worst-slice aggregation claims to protect."""
    from datasets import load_dataset
    per = max(1, limit // len(MGSM_LANGS))
    out = []
    for lang in MGSM_LANGS:
        try:
            ds = load_dataset("juletxara/mgsm", lang, split="test")
        except Exception:
            continue
        for i in range(min(per, len(ds))):
            gold = str(ds[i]["answer_number"]).strip()
            out.append({"prompt": MATH_INSTRUCTION.format(q=ds[i]["question"]), "gold": gold,
                        "kind": "num", "slice": lang})
    return out


def load_mmlu(limit):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    step = max(1, len(ds) // limit)          # spread across subjects rather than taking one block
    out = []
    for i in range(0, len(ds), step):
        if len(out) >= limit:
            break
        r = ds[i]
        ch = r["choices"]
        if len(ch) != 4:
            continue
        out.append({"prompt": MC_INSTRUCTION.format(q=r["question"], a=ch[0], b=ch[1],
                                                    c=ch[2], d=ch[3]),
                    "gold": "ABCD"[int(r["answer"])], "kind": "letter",
                    "slice": str(r.get("subject", "mmlu"))[:24]})
    return out


def load_hellaswag(limit):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    out = []
    for i in range(min(limit, len(ds))):
        r = ds[i]
        e = r["endings"]
        if len(e) != 4 or not str(r["label"]).isdigit():
            continue
        out.append({"prompt": MC_INSTRUCTION.format(q=r["ctx"], a=e[0], b=e[1], c=e[2], d=e[3]),
                    "gold": "ABCD"[int(r["label"])], "kind": "letter", "slice": "hellaswag"})
    return out


TASKS = {
    "gsm8k":     {"load": load_gsm8k,     "max_new": 1024, "default_n": 400},
    "mgsm":      {"load": load_mgsm,      "max_new": 1024, "default_n": 600},
    "mmlu":      {"load": load_mmlu,      "max_new": 16,   "default_n": 800},
    "hellaswag": {"load": load_hellaswag, "max_new": 16,   "default_n": 800},
}

DATASET_IDS = ("openai/gsm8k", "juletxara/mgsm", "cais/mmlu", "Rowan/hellaswag")


def score(item, text):
    """(correct, format_fail). A format failure is NOT counted as a wrong answer — it is reported
    beside it, because 'ignored the instruction' and 'got it wrong' are different diagnoses."""
    if item["kind"] == "letter":
        got = _letter(text)
        return (got == item["gold"], got == "")
    got = _num(text)
    if got == "":
        return False, True
    try:
        return abs(float(got) - float(item["gold"])) < 1e-6, False
    except ValueError:
        return got == item["gold"], False
