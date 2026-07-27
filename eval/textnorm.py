"""Script-aware number handling — what makes the multilingual axis actually work.

Two bugs killed the non-Latin crown axis, both invisible on English test data:

1. `(?<![\\w.])\\d{2,}(?![\\w.])` finds NOTHING in unspaced CJK. Python's `\\w` matches Han and
   Kana, so in 公司在2024年 the neighbours 在 and 年 veto the match. Chinese web text is
   predominantly unspaced, so the cmn_Hani source — 35% of the multilingual mix — minted zero
   items. Cyrillic escaped only because Russian is space-separated.
2. Arabic text often writes numerals as Arabic-Indic (٢٠٢٤). Those match `\\d`, so they became
   the GOLD, while any model answering 2024 was scored wrong. The axis punished the correct
   answer.

The fix in both cases is to stop treating "word character" as a Latin-centric proxy:
  * veto a digit run only when it abuts an ASCII alphanumeric (a real token like `abc123`) or a
    decimal point — NOT when it abuts a Han/Cyrillic/Arabic letter, where the digit genuinely is
    a standalone number;
  * normalize every digit family to ASCII on BOTH sides of the comparison, so gold and answer
    are compared as numbers rather than as codepoints.
"""
from __future__ import annotations

import re

# Digit families seen in real web text. Value-preserving map to ASCII.
_DIGIT_MAP = {}
for _base in (0x0660,      # Arabic-Indic          ٠١٢٣٤٥٦٧٨٩
              0x06F0,      # Extended Arabic-Indic ۰۱۲۳۴۵۶۷۸۹ (Persian/Urdu)
              0x0966,      # Devanagari
              0x0BE6,      # Tamil
              0xFF10):     # Fullwidth              ０１２３４５６７８９ (common in CJK text)
    for _d in range(10):
        _DIGIT_MAP[chr(_base + _d)] = chr(48 + _d)
_DIGIT_TABLE = str.maketrans(_DIGIT_MAP)

# A standalone number: not glued to an ASCII alphanumeric and not part of a decimal. Unlike
# `\w`, this permits digits embedded directly in CJK/Cyrillic/Arabic script.
NUM_RE = re.compile(r"(?<![0-9A-Za-z_.])[0-9٠-٩۰-۹०-९"
                    r"௦-௯０-９]{2,}(?![0-9A-Za-z_.])")


def norm_digits(s: str) -> str:
    """Map any supported digit family to ASCII, leaving everything else untouched."""
    return s.translate(_DIGIT_TABLE) if s else s


def find_numbers(text: str):
    """Standalone numbers in `text`, script-agnostic. Yields (match, ascii_value)."""
    for m in NUM_RE.finditer(text or ""):
        yield m, norm_digits(m.group())


_BAD_ANCHOR = '"“”\n\r'


def make_anchor(text: str, start: int, anchor_len: int = 2, char_window: int = 8) -> str | None:
    """A UNIQUE, quotable phrase immediately preceding `text[start]`, or None.

    Word-based for space-separated scripts, CHARACTER-based for unspaced ones. The word form
    alone silently excludes Chinese and Japanese: `"公司在2024年".split()` is a single token, so
    "the N words before the number" can never be built and the whole source mints nothing —
    which is how the cmn_Hani axis came to contribute zero items even after the number regex
    was fixed. Any anchor is acceptable as long as it is unique (unambiguous gold) and contains
    no quote character (the prompt renders it inside double quotes)."""
    if start <= 0:
        return None
    prefix = text[:start]
    words = prefix.split()
    cands = []
    if len(words) >= anchor_len:
        cands.append(" ".join(words[-anchor_len:]))
    # character window: works for any script, and is the only option for unspaced text
    for w in (char_window, char_window + 4):
        seg = prefix[-w:].strip()
        if seg:
            cands.append(seg)
    for anchor in cands:
        if any(c in anchor for c in _BAD_ANCHOR):
            continue
        if text.count(anchor) == 1:
            return anchor
    return None


def same_number(a: str, b: str) -> bool:
    """Numeric equality across digit families and thousands separators, so a model that answers
    2024 for a gold of ٢٠٢٤ (or 2,024) is scored correct — it read the passage right."""
    if a is None or b is None:
        return False
    na = norm_digits(str(a)).replace(",", "").lstrip("+").lstrip("0") or "0"
    nb = norm_digits(str(b)).replace(",", "").lstrip("+").lstrip("0") or "0"
    return na == nb
