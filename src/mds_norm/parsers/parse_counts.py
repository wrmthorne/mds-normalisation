from __future__ import annotations

import re
from dataclasses import dataclass

WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
QUAL_MAP = {
    "c": "approx",
    "c.": "approx",
    "ca": "approx",
    "ca.": "approx",
    "approx": "approx",
    "approx.": "approx",
    "approximately": "approx",
    "about": "approx",
    "est": "approx",
    "est.": "approx",
    "estimated": "approx",
}

SCALE_WORDS = ("hundred", "thousand", "million", "dozen", "score")
"""Multipliers this parser deliberately cannot read, so a value using one is refused"""

_QUAL = "|".join(re.escape(w) for w in sorted(QUAL_MAP, key=len, reverse=True))
# longest-first, or 'seventeen' matches the 'seven' branch
_WORD = "|".join(sorted(WORDS, key=len, reverse=True))
# the noun may not open with a numeral or multiplier
_NOT_NOUN = "|".join((*WORDS, *SCALE_WORDS))
_NOUN = rf"(?!(?:{_NOT_NOUN})\b)[a-z]+(?:\s+[a-z]+){{0,4}}"

_RANGE = re.compile(
    rf"(?i)^(?:(?P<qual>{_QUAL})\s+)?(?P<lo>\d+)\s*(?:-|to)\s*(?P<hi>\d+)"
    rf"\s*(?P<noun>{_NOUN})?$"
)
_SIMPLE = re.compile(rf"(?i)^(?:(?P<qual>{_QUAL})\s+)?(?P<num>\d+)\s*(?P<noun>{_NOUN})?$")
_WORDNUM = re.compile(rf"(?i)^(?:(?P<qual>{_QUAL})\s+)?(?P<word>{_WORD})\s*(?P<noun>{_NOUN})?$")
_PLUS = re.compile(r"\s*\+\s*")


@dataclass(frozen=True)
class Count:
    """One homogeneous count: an integer (with optional noun) or a range"""

    count: int | None = None
    qualifier: str | None = None  # 'approx'
    noun: str | None = None
    range_lo: int | None = None
    range_hi: int | None = None


def _qual(raw: str | None) -> str | None:
    return QUAL_MAP[raw.lower()] if raw else None


def _noun(raw: str | None) -> str | None:
    return raw.strip().lower() if raw else None


def _part(s: str) -> Count | None:
    if m := _RANGE.match(s):
        lo, hi = int(m["lo"]), int(m["hi"])
        if lo > hi:
            return None
        return Count(qualifier=_qual(m["qual"]), noun=_noun(m["noun"]), range_lo=lo, range_hi=hi)
    if m := _SIMPLE.match(s):
        return Count(count=int(m["num"]), qualifier=_qual(m["qual"]), noun=_noun(m["noun"]))
    if m := _WORDNUM.match(s):
        return Count(count=WORDS[m["word"].lower()], qualifier=_qual(m["qual"]), noun=_noun(m["noun"]))
    return None


def parse_count(raw: str) -> list[Count] | None:
    """Parse a recorded object-count string into its homogeneous counts, None if unverifiable"""
    if not raw:
        return None
    s = raw.strip().rstrip(".").strip()
    if not s:
        return None
    parts = [_part(seg.strip()) for seg in _PLUS.split(s)]
    if not parts or any(p is None for p in parts):
        return None
    return parts
