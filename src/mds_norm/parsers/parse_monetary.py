from __future__ import annotations

import re
from dataclasses import dataclass

GUINEA_WORDS = ("guineas", "guinea", "gns", "gn")
"""Guinea = 21 (old) shillings = 252 old pence; never 20 shillings/£1"""

_BRACKETED = re.compile(r"^\s*\[\s*(?P<inner>[^\[\]]+?)\s*\]\s*$")
# a trailing 'for ...' is context, not the amount
_CONTEXT = re.compile(r"(?i)\s+for\s+.+$")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")

_DECIMAL = re.compile(r"^£?\s*(?P<pounds>\d+)(?:\.(?P<pence>\d{1,2}))?$")
# £L/S/D or £L-S-D; a bare '-' slot means zero
_LSD_FULL = re.compile(r"^£\s*(?P<pounds>\d+)\s*[/-]\s*(?P<shillings>\d{1,2}|-)\s*[/-]\s*(?P<pence>\d{1,2}|-)$")
# S/D or S/- — shillings/pence only, no pounds slot
_SD_SLASH = re.compile(r"^(?P<shillings>\d{1,2})\s*/\s*(?P<pence>\d{1,2}|-)$")
# 's'/'d' suffix notation: '17s 6d', '5s', '6d'
_SD_SUFFIX = re.compile(r"^(?:(?P<shillings>\d+)\s*s\b)?\s*(?:(?P<pence>\d+)\s*d\b)?$")
_GUINEA_NUM = re.compile(rf"(?i)^(?P<n>\d+(?:\.\d+)?)\s*(?:{'|'.join(GUINEA_WORDS)})$")
# Spelled-out denominations; 'shillings' and 'pence' are pre-decimal
_WORD_UNIT = re.compile(r"(?i)^(?P<n>\d+(?:\.\d+)?)\s*(?P<unit>pounds?|shillings?|pence|penny)$")
_WORD_SYSTEM = {
    "pound": ("gbp_decimal", 100),
    "shilling": ("gbp_lsd", 12),
    "pence": ("gbp_lsd", 1),
    "penny": ("gbp_lsd", 1),
}


@dataclass(frozen=True)
class Amount:
    currency_system: str
    """'gbp_decimal' | 'gbp_lsd' | 'gbp_guinea'"""
    normalised_pence: int
    """Integer amount in the system's native subdivision: new pence, old pence, or guineas"""
    uncertain: bool = False
    """Whole value was cataloguer-bracketed, e.g. '[7/6]'"""
    context: str | None = None
    """Trailing descriptive text split off the amount, e.g. 'for set of 25'"""


MAX_PENCE = 11  # twelve pence to the shilling


def _lsd(shillings: str, pence: str, *, max_shillings: int | None = 19) -> tuple[int, int] | None:
    sh = 0 if shillings == "-" else int(shillings)
    pe = 0 if pence == "-" else int(pence)
    if (max_shillings is not None and sh > max_shillings) or pe > MAX_PENCE:
        return None
    return sh, pe


def parse_monetary(raw: str) -> Amount | None:
    """Parse a recorded price/amount string, or None if it cannot be verified"""
    if not raw:
        return None
    s = raw.strip()
    uncertain = False
    if m := _BRACKETED.match(s):
        uncertain = True
        s = m["inner"].strip()

    context = None
    if m := _CONTEXT.search(s):
        context = s[m.start() :].strip()
        head = s[: m.start()].strip()
        if not head or re.search(r"(?i)\band\b", head):
            return None  # ambiguous which figure is the price
        s = head

    s = _THOUSANDS.sub("", s).rstrip(".").strip()
    if not s:
        return None

    if m := _GUINEA_NUM.match(s):
        return Amount("gbp_guinea", round(float(m["n"]) * 252), uncertain, context)

    if m := _WORD_UNIT.match(s):
        system, per = _WORD_SYSTEM[m["unit"].lower().rstrip("s")]
        return Amount(system, round(float(m["n"]) * per), uncertain, context)

    if m := _DECIMAL.match(s):
        pounds, pence = int(m["pounds"]), int((m["pence"] or "0").ljust(2, "0"))
        return Amount("gbp_decimal", pounds * 100 + pence, uncertain, context)

    if m := _LSD_FULL.match(s):
        sp = _lsd(m["shillings"], m["pence"])  # carries into pounds: cap at 19
        if sp is None:
            return None
        return Amount("gbp_lsd", int(m["pounds"]) * 240 + sp[0] * 12 + sp[1], uncertain, context)

    if m := _SD_SLASH.match(s):
        # no pounds slot to carry into, so shillings is uncapped
        sp = _lsd(m["shillings"], m["pence"], max_shillings=None)
        if sp is None:
            return None
        return Amount("gbp_lsd", sp[0] * 12 + sp[1], uncertain, context)

    if re.search(r"(?i)\d\s*[sd]\b", s) and (m := _SD_SUFFIX.match(s)) and (m["shillings"] or m["pence"]):
        sp = _lsd(m["shillings"] or "0", m["pence"] or "0", max_shillings=None)
        if sp is None:
            return None
        return Amount("gbp_lsd", sp[0] * 12 + sp[1], uncertain, context)

    return None
