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

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def _strip_think(text):
    """(answer_region, truncated) — everything after the LAST `</think>`.

    Qwen3 is a hybrid-reasoning model, and llama.cpp hands back the whole `<think>…</think>` trace
    inside the message content. Scoring that trace is not a cosmetic problem: `_num` falls back to
    the last number ANYWHERE in the text, so a page of arithmetic scratch work becomes an answer,
    and `_letter` matches any standalone A-D, so reasoning prose becomes a multiple-choice pick.
    Both convert reasoning into a confident answer that was never given.

    An UNCLOSED `<think>` is the third outcome this function exists for: generation hit its token
    budget inside the reasoning block, so the model never reached an answer at all. That is not a
    wrong answer and not an ignored instruction — it is a measurement that did not finish, and
    counting it as either one understates the model. It is reported separately."""
    if THINK_CLOSE in text:
        return text.rsplit(THINK_CLOSE, 1)[1], False
    if THINK_OPEN in text:
        return "", True
    return text, False


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
    """First standalone A-D. Answers arrive as 'B', 'B.', '**B**', 'Answer: B'.

    The leading pattern REQUIRES A TERMINATOR after the letter. Without one it fired on the English
    article "A" — so a model that opened with "A student is asked to…" scored as having answered A,
    silently, and was never counted as a format failure. Chance-level accuracy manufactured out of
    prose is worse than a zero, because a zero is visibly wrong.

    Call this on `_strip_think(text)[0]`, never on a raw completion."""
    t = text.strip()
    # The last pattern's `(?!\s+[a-z])` is the same guard as the first one's terminator: a capital
    # A-D followed by a lowercase word is English, not an answer. Without it the fallback re-admits
    # every article "A" the strict pattern just excluded. It costs a real answer only in the shape
    # "I pick C since…", which the prompt explicitly asks models not to produce — and that lands as
    # a visible format-failure rather than as a silently wrong letter, which is the trade to want.
    for pat in (r"^\s*\**\s*([A-Da-d])\s*[.):,\-]?\s*(?:$|\n)",
                r"[Aa]nswer\s*(?:is)?\s*[:\-]?\s*\**\s*\(?([A-Da-d])\b",
                r"\b([A-D])\b(?!\s+[a-z])"):
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

# MGSM's own split of its eleven languages. The first run used the high-resource five plus Swahili,
# and Swahili alone carried the finding: both competitor models kept 80-95% of the parent on
# well-resourced languages and 41% on Swahili. One language is an anecdote; these three make it a
# class. (`en` is omitted because it duplicates GSM8K, `ja` because it adds a fourth well-resourced
# language to a group that already agrees with itself.)
MGSM_HIGH = ("es", "fr", "de", "ru", "zh")
MGSM_LOW = ("sw", "bn", "te", "th")
MGSM_LANGS = MGSM_HIGH + MGSM_LOW

# THE SLICE SIZE IS THE THING TO HOLD FIXED, so the budget is derived from it rather than the other
# way round. `default_n` used to be a hard-coded 600 that `load_mgsm` divided among the languages;
# adding three languages to that would have taken every slice from 100 items to 66 — widening the
# error bars on a finding whose entire weight rests on ONE language's slice, in the very run meant
# to strengthen it. Nothing would have warned us. Adding a language now costs GPU time instead of
# statistical power, and the slices stay comparable with the run already published.
MGSM_PER_LANG = 100

# The languages this process will actually run. `load_mgsm` has no language argument and the runner
# has no way to pass one, so extending MGSM_LANGS from six to nine would have re-run all nine for
# every model — turning "add three languages" into a full re-run at roughly three times the price.
# `select()` narrows this, and keeps `default_n` in step so the per-language count never moves.
MGSM_SELECTED = MGSM_LANGS


def select_mgsm_langs(spec):
    """Narrow the MGSM run to a comma-separated subset, e.g. "bn,te,th". Returns the selection.

    Keeps the two things that must not drift — the language list and the item budget derived from
    it — adjusted together, in one place, so a caller cannot set one and forget the other."""
    global MGSM_SELECTED
    want = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [w for w in want if w not in MGSM_LANGS]
    if unknown:
        raise SystemExit(f"unknown MGSM language(s): {', '.join(unknown)}. "
                         f"known: {', '.join(MGSM_LANGS)}")
    MGSM_SELECTED = tuple(want) or MGSM_LANGS
    TASKS["mgsm"]["default_n"] = MGSM_PER_LANG * len(MGSM_SELECTED)
    return MGSM_SELECTED


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
    """The same arithmetic in nine languages. This is the axis GSM8K is blind to, and the one the
    subnet's worst-slice aggregation claims to protect.

    `limit` is a whole-task budget that `--limit-scale` has already multiplied; dividing it back out
    recovers the per-language count, which is MGSM_PER_LANG at scale 1.0 for any language count."""
    from datasets import load_dataset
    per = max(1, round(limit / len(MGSM_SELECTED)))
    out = []
    missing = []
    for lang in MGSM_SELECTED:
        try:
            ds = load_dataset("juletxara/mgsm", lang, split="test")
        except Exception as e:
            # LOUD. A silent `continue` deletes a language from the comparison and leaves a results
            # file that looks complete — the reader sees eight languages and no reason to ask about
            # the ninth. Every other silent fallback in this harness has already cost a run.
            missing.append(f"{lang} ({type(e).__name__})")
            continue
        for i in range(min(per, len(ds))):
            gold = str(ds[i]["answer_number"]).strip()
            out.append({"prompt": MATH_INSTRUCTION.format(q=ds[i]["question"]), "gold": gold,
                        "kind": "num", "slice": lang})
    if missing:
        import sys
        sys.stderr.write(f"  !! MGSM LANGUAGES ABSENT FROM THIS RUN: {', '.join(missing)}\n")
    return out


def load_mmlu(limit):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    # SPANS THE WHOLE SPLIT. `range(0, len(ds), len(ds)//limit)` with an integer step overshoots and
    # gets cut off by the `break`, silently dropping the tail — on the 14,042-row split that was the
    # last 458 rows, i.e. 3 of MMLU's 57 subjects were never benchmarked and the results said
    # "mmlu". This form lands `limit` indices across the full range by construction.
    out = []
    for i in (j * len(ds) // limit for j in range(limit)):
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
    "mgsm":      {"load": load_mgsm,      "max_new": 1024,
                  "default_n": MGSM_PER_LANG * len(MGSM_LANGS)},
    # max_new WAS 16, AND 16 WAS TOO SMALL EVEN WITHOUT A REASONING BLOCK. It produced 86 of 800
    # format-failures on the parent — concentrated in the quantitative subjects, where a model
    # restates the question before it answers — and 800 of 800 on the models whose runtime left
    # thinking enabled, where the budget was spent opening `<think>`. A model that answers in two
    # tokens stops at EOS and pays nothing for the larger cap; only the verbose ones use it, and
    # those are exactly the ones the old cap was silently scoring as ignorant.
    "mmlu":      {"load": load_mmlu,      "max_new": 256,  "default_n": 800},
    "hellaswag": {"load": load_hellaswag, "max_new": 256,  "default_n": 800},
}

DATASET_IDS = ("openai/gsm8k", "juletxara/mgsm", "cais/mmlu", "Rowan/hellaswag")


def score(item, text):
    """(correct, format_fail, truncated). THREE outcomes, not two.

    A format failure is not counted as a wrong answer, and a truncation is not counted as either,
    because 'got it wrong', 'ignored the instruction' and 'ran out of tokens before answering' are
    three different diagnoses with three different fixes. Collapsing the third into the first is how
    a model that reasons at length gets published as a model that cannot do arithmetic."""
    body, truncated = _strip_think(text or "")
    if truncated:
        return False, False, True
    if item["kind"] == "letter":
        got = _letter(body)
        return got == item["gold"], got == "", False
    got = _num(body)
    if got == "":
        return False, True, False
    try:
        return abs(float(got) - float(item["gold"])) < 1e-6, False, False
    except ValueError:
        return got == item["gold"], False, False
