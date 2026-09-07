from __future__ import annotations

import re
from dataclasses import dataclass

MAX_MARKED_LEN = 80
"""Values longer than this never count as uncertainty-marked"""

LEXICAL_MARKERS = ("probably", "possibly", "perhaps", "presumably", "presumed")
"""Lexical uncertainty vocabulary accepted leading, trailing or parenthesised"""

_LEX = "|".join(LEXICAL_MARKERS)

PREFILTER = r"(?:\?|^\s*\[[^\[\]]+\]\s*$|(?i:\b(?:" + _LEX + r")\b)|(?i:\bc(?:irc)?a?\.?\s*\d))"
"""Cheap polars-side prefilter: only values matching this are worth parsing"""

# '?' guards: URLs and query strings dominate
_URLISH = re.compile(r"https?://|www\.|\?[\w.%+-]+=|&[\w.%+-]+=", re.IGNORECASE)

# Whole value in square brackets: cataloguer-supplied/inferred content
_BRACKETED = re.compile(r"^\s*\[\s*(?P<clean>[^\[\]]+?)\s*]\s*$", re.DOTALL)
# Trailing question mark(s), optionally parenthesised/bracketed
_TRAIL_Q = re.compile(r"^(?P<clean>.*?\S)\s*(?P<notation>\(\s*\?+\s*\)|\[\s*\?+\s*]|\?+)\s*$", re.DOTALL)
# Leading question mark(s): '?oak'
_LEAD_Q = re.compile(r"^\s*(?P<notation>\?+)\s*(?P<clean>\S.*?)\s*$", re.DOTALL)
# Parenthesised lexical marker: 'oak (probably)'
_PAREN_LEX = re.compile(rf"^(?P<clean>.*?\S)\s*\(\s*(?P<notation>{_LEX})\s*\)\s*$", re.IGNORECASE | re.DOTALL)
# Trailing lexical marker after a separator: 'oak, probably'
_TRAIL_LEX = re.compile(rf"^(?P<clean>.*?\S)\s*[,;]\s*(?P<notation>{_LEX})\s*$", re.IGNORECASE | re.DOTALL)
# Leading lexical marker: 'probably oak'
_LEAD_LEX = re.compile(rf"^\s*(?P<notation>{_LEX})\s+(?P<clean>\S.*?)\s*$", re.IGNORECASE | re.DOTALL)
# 'circa'/'c.' only before a digit: 'C. Smith' is an initial
_CIRCA = re.compile(r"^\s*(?P<notation>circa|ca?\.)\s*(?P<clean>\d.*?)\s*$", re.IGNORECASE | re.DOTALL)

_RULES = (_BRACKETED, _TRAIL_Q, _LEAD_Q, _PAREN_LEX, _TRAIL_LEX, _LEAD_LEX, _CIRCA)


@dataclass(frozen=True)
class CertaintyMatch:
    clean: str  # The value with the uncertainty notation removed
    notation: str  # The notation as recorded, e.g. '?', '(?)', '[]', 'probably', 'circa'


def _strip_once(value: str) -> tuple[str, str] | None:
    for rule in _RULES:
        if m := rule.match(value):
            notation = "[]" if rule is _BRACKETED else m["notation"]
            return m["clean"], notation
    return None


def parse_certainty(value: str) -> CertaintyMatch | None:
    """Return the clean value and stripped notation, or None if unmarked"""
    if not value or len(value) > MAX_MARKED_LEN or _URLISH.search(value):
        return None
    notations: list[str] = []
    clean = value
    for _ in range(2):
        hit = _strip_once(clean)
        if hit is None:
            break
        clean, notation = hit
        notations.append(notation)
    if not notations:
        return None
    clean = clean.strip()
    # the residue must still say something alphanumeric
    if not re.search(r"[0-9A-Za-z]", clean):
        return None
    return CertaintyMatch(clean=clean, notation="+".join(notations))
