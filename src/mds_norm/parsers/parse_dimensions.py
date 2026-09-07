from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from lark import Lark, Token, Transformer, v_args

# Bumped when readings change, so the dimension cache reparses
PARSER_VERSION = "2026-08-01.1"

UNITS = {
    "mm": "mm",
    "mms": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "cm": "cm",
    "cms": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "m": "m",
    "metre": "m",
    "metres": "m",
    "meter": "m",
    "meters": "m",
    "in": "in",
    "ins": "in",
    "inch": "in",
    "inches": "in",
    "inhes": "in",  # frequent corpus typo (2.6k occurrences)
    "ft": "ft",
    "foot": "ft",
    "feet": "ft",
    "g": "g",
    "gm": "g",
    "gms": "g",
    "gram": "g",
    "grams": "g",
    "gramme": "g",
    "grammes": "g",
    "kg": "kg",
    "kgs": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "lb": "lb",
    "lbs": "lb",
    "pound": "lb",
    "pounds": "lb",
    "oz": "oz",
    "ounce": "oz",
    "ounces": "oz",
    "ton": "ton",
    "tons": "ton",
    "tonne": "ton",
    "tonnes": "ton",
    "ml": "ml",
    "millilitre": "ml",
    "millilitres": "ml",
    "deg": "deg",
    "degree": "deg",
    "degrees": "deg",
    "min": "min",
    "mins": "min",
    "minute": "min",
    "minutes": "min",
    "sec": "sec",
    "secs": "sec",
    "second": "sec",
    "seconds": "sec",
}
UNIT_CLASS = {
    "mm": "linear",
    "cm": "linear",
    "m": "linear",
    "in": "linear",
    "ft": "linear",
    "g": "mass",
    "kg": "mass",
    "lb": "mass",
    "oz": "mass",
    "ton": "mass",
    "ml": "volume",
    "deg": "angle",
    "min": "time",
    "sec": "time",
}
# One unit in its class's base unit; 'ton' is ambiguous
TO_BASE = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "in": 25.4,
    "ft": 304.8,
    "g": 1.0,
    "kg": 1000.0,
    "lb": 453.59237,
    "oz": 28.349523125,
    "ml": 1.0,
    "deg": 1.0,
    "min": 60.0,
    "sec": 1.0,
}
# pairs summed exactly into the smaller stated unit
COMPOUND_PAIRS = {("ft", "in"): 12, ("lb", "oz"): 16, ("kg", "g"): 1000}

KEYWORDS = {
    "height": "height",
    "high": "height",
    "ht": "height",
    "h": "height",
    "width": "width",
    "wide": "width",
    "wd": "width",
    "w": "width",
    "depth": "depth",
    "deep": "depth",
    "dp": "depth",
    "d": "depth",
    "length": "length",
    "long": "length",
    "len": "length",
    "l": "length",
    "diameter": "diameter",
    "diam": "diameter",
    "dia": "diameter",
    "thickness": "thickness",
    "thick": "thickness",
    "breadth": "breadth",
    "weight": "weight",
    "wt": "weight",
    "weighs": "weight",
    "circumference": "circumference",
    "circ": "circumference",
    "radius": "radius",
}
# single letters read via the h/w/d convention
AMBIGUOUS_KEYWORDS = {"d"}

QUALIFIERS = {
    "approx": "approx",
    "approximate": "approx",
    "approximately": "approx",
    "about": "approx",
    "ca": "approx",
    "circa": "approx",
    "est": "approx",
    "estimated": "approx",
    "max": "maximum",
    "maximum": "maximum",
    "min": "minimum",
    "minimum": "minimum",
    "exact": "exact",
    "exactly": "exact",
    "nominal": "nominal",
    "normally": "nominal",
}

VULGAR = {
    "¼": "1/4",
    "½": "1/2",
    "¾": "3/4",
    "⅓": "1/3",
    "⅔": "2/3",
    "⅕": "1/5",
    "⅖": "2/5",
    "⅗": "3/5",
    "⅘": "4/5",
    "⅙": "1/6",
    "⅚": "5/6",
    "⅛": "1/8",
    "⅜": "3/8",
    "⅝": "5/8",
    "⅞": "7/8",
}

PAREN = re.compile(r"\(([^)]*)\)")
_NUMLIKE = re.compile(r"\d")
_UNIT_ALT = "|".join(sorted(UNITS, key=len, reverse=True))


@dataclass
class Flags:
    uncertain: bool = False
    bound: str | None = None  # '<1 g' → maximum; '>2 cm' → minimum


def normalise(raw: str, prime_unit: str = "ft") -> tuple[str, Flags]:
    """Collapse to a canonical token stream, returning (text, flags)"""
    flags = Flags()
    s = raw.strip().lower()
    s = s.replace("×", "x").replace("*", " x ")
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("[", "(").replace("]", ")")
    s = s.replace("=", " : ")  # 'Length = 30' — assignment, not range
    s = s.replace(":", " : ")  # unglue 'height:' so the keyword is seen
    for ch, frac in VULGAR.items():
        s = s.replace(ch, f" {frac}")
    if "?" in s:
        flags.uncertain = True
        s = s.replace("?", " ")
    if "<" in s:
        flags.bound = "maximum"
        s = s.replace("<", " ")
    elif ">" in s:
        flags.bound = "minimum"
        s = s.replace(">", " ")
    s = re.sub(r"(?<=\d)\s*[\"″“”]", " in ", s)
    s = re.sub(r"(?<=\d)\s*['′‘’]", f" {prime_unit} ", s)
    s = s.replace("~", " approx ")
    # thousands separator, then continental decimal comma (29,40 → 29.40)
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    s = re.sub(r"(?<=\d),(?=\d{1,2}(?!\d))", ".", s)
    # 'kw / mm:23' and 'kw (mm):215' → 'kw: 23 mm'
    s = re.sub(rf"\s*[/(]\s*({_UNIT_ALT})\s*\)?\s*:\s*(\d[\d .x/-]*)", r": \2 \1 ", s)
    s = re.sub(rf"\s*[/(]\s*({_UNIT_ALT})\s*\)?\s*:", ":", s)
    s = re.sub(rf"\(\s*({_UNIT_ALT})\s*\)", r" \1 ", s)
    # dot-written mixed numbers: '18.5/8' means 18 5/8
    s = re.sub(r"(?<=\d)\.(?=\d+\s*/\s*\d)", " ", s)
    # abbreviation dots ('in.', 'approx.') — decimals are digit-bounded, safe
    s = re.sub(r"(?<=[a-z])\.", " ", s)
    # unglue: 4insx6ins, 6x5cm, 50mm, length571mm
    s = re.sub(r"(?<=[a-z])x(?=\d)", " x ", s)
    s = re.sub(r"(?<=\d)x(?=\d)", " x ", s)
    s = re.sub(r"(?<=\d)(?=[a-z])", " ", s)
    s = re.sub(r"(?<=[a-z])(?=\d)", " ", s)
    return re.sub(r"[ \t]+", " ", s).strip(), flags


def _alt(words: Iterable[str]) -> str:
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


GRAMMAR = rf"""
?start: chain
chain: item (_X item)*
item: pre* measure post*
pre: QUAL _C?    -> qual
   | KW _C?      -> kw
post: UNIT -> unit
    | KW   -> kw
    | QUAL -> qual
?measure: range | number | compound
range: number _SEP number
compound: NUM UNIT NUM UNIT
?number: NUM      -> num
       | FRAC     -> frac
       | NUM FRAC -> mixed
_SEP: "-" | "to"
_X: "x"
_C: ":"
KW: /{_alt(KEYWORDS)}/
UNIT: /{_alt(UNITS)}/
QUAL: /{_alt(QUALIFIERS)}/
NUM: /\d+(\.\d+)?|\.\d+/
FRAC: /\d+\/\d+/
%import common.WS
%ignore WS
"""


type _Entry = tuple[float, str | None, str | None]


@v_args(inline=True)
class ToItems(Transformer):
    """Each measure becomes a list of (value, qualifier, unit_override)"""

    def num(self, t: Token) -> list[_Entry]:
        return [(float(t), None, None)]

    def frac(self, t: Token) -> list[_Entry]:
        a, b = str(t).split("/")
        if int(b) == 0:
            raise ValueError("zero denominator")
        return [(int(a) / int(b), None, None)]

    def mixed(self, n: Token, f: Token) -> list[_Entry]:
        a, b = str(f).split("/")
        if int(b) == 0:
            raise ValueError("zero denominator")
        return [(float(n) + int(a) / int(b), None, None)]

    def range(self, lo: list[_Entry], hi: list[_Entry]) -> list[_Entry]:
        if lo[0][0] > hi[0][0]:
            raise ValueError("descending range")
        return [(lo[0][0], "minimum", None), (hi[0][0], "maximum", None)]

    def compound(self, n1: Token, u1: Token, n2: Token, u2: Token) -> list[_Entry]:
        c1, c2 = UNITS[str(u1)], UNITS[str(u2)]
        factor = COMPOUND_PAIRS.get((c1, c2))
        if factor is None:
            raise ValueError(f"not a compound pair: {c1} {c2}")
        # one tuple in the smaller stated unit
        return [(float(n1) * factor + float(n2), None, c2)]

    def kw(self, t: Token) -> tuple[str, str, bool]:
        w = str(t)
        return ("kw", KEYWORDS[w], w in AMBIGUOUS_KEYWORDS)

    def qual(self, t: Token) -> tuple[str, str]:
        return ("qual", QUALIFIERS[str(t)])

    def unit(self, t: Token) -> tuple[str, str]:
        return ("unit", UNITS[str(t)])

    def item(self, *parts: list[_Entry] | tuple[str, str] | tuple[str, str, bool]) -> dict:
        out = {"entries": None, "kw": None, "quals": [], "unit": None, "ambiguous": False}
        for p in parts:
            if isinstance(p, list):
                out["entries"] = p
            elif p[0] == "kw":
                out["kw"], out["ambiguous"] = p[1], p[2]
            elif p[0] == "qual":
                out["quals"].append(p[1])
            elif p[0] == "unit":
                out["unit"] = p[1]
        return out

    def chain(self, *items: dict) -> list[dict]:
        return list(items)


_parser = Lark(GRAMMAR, parser="earley", ambiguity="resolve")
_transformer = ToItems()


@dataclass
class Measurement:
    """Field names follow mds_data_model.models.dimension.Dimension"""

    dimension_type: str | None
    dimension_value: float
    dimension_measurement_unit: str | None
    dimension_value_qualifier: str | None = None
    dimension_measured_part: str | None = None
    axis: int | None = None
    type_ambiguous: bool = False


def join_slots(*quals: str | None) -> str | None:
    seen = [q for q in quals if q]
    uniq = list(dict.fromkeys(seen))
    return " ".join(uniq) or None


def _extract_context(seg: str) -> tuple[str, str | None, list[str]]:
    """Pull a 'label:' prefix and leading unknown words out of a segment"""
    part_words, quals = [], []
    head, sep, tail = seg.partition(":")
    if sep and not _NUMLIKE.search(head):
        kws, units = [], []
        for tok in head.split():
            if tok in KEYWORDS:
                kws.append(tok)
            elif tok in QUALIFIERS:
                quals.append(QUALIFIERS[tok])
            elif tok in UNITS:
                units.append(tok)  # 'w (in) plate size: 3 15/16'
            else:
                part_words.append(tok)
        seg = ((" ".join(kws) + " : " + tail) if kws else tail) + (" " + " ".join(units) if units else "")
    toks = seg.split()
    i = 0
    while (
        i < len(toks)
        and toks[i] not in KEYWORDS
        and toks[i] not in QUALIFIERS
        and toks[i] not in UNITS
        and not _NUMLIKE.search(toks[i])
    ):
        part_words.append(toks[i])
        i += 1
    part = " ".join(w for w in part_words if w not in {":", ""}) or None
    return " ".join(toks[i:]), part, quals


# one unit and one value, in either order
_UNIT_VALUE_TOKENS = 2


def _parse_segment(seg: str) -> list[Measurement] | None:
    seg, part, label_quals = _extract_context(seg)
    if not _NUMLIKE.search(seg):
        return None
    # inverted unit-first values ('mm 209') read as value-unit
    toks = seg.split()
    if len(toks) == _UNIT_VALUE_TOKENS and toks[0] in UNITS and _NUMLIKE.search(toks[1]):
        seg = f"{toks[1]} {toks[0]}"
    try:
        items = _transformer.transform(_parser.parse(seg))
    except Exception:
        return None
    # a trailing unit distributes backward, a leading one forward
    fill = None
    for it in reversed(items):
        if it["unit"]:
            fill = it["unit"]
        elif fill:
            it["unit"] = fill
    fill = None
    for it in items:
        if it["unit"]:
            fill = it["unit"]
        elif fill:
            it["unit"] = fill
    # bare 2-3 crosses read as height, width, depth
    convention = None
    spatial = {
        id(it): i for i, it in enumerate(it for it in items if UNIT_CLASS.get(it["unit"], "linear") == "linear")
    }
    if not any(it["kw"] for it in items) and len(spatial) in (2, 3):
        convention = ["height", "width", "depth"]
    out = []
    for axis, it in enumerate(items):
        base_type, ambiguous = it["kw"], it["ambiguous"]
        if base_type is None and convention and id(it) in spatial:
            base_type = convention[spatial[id(it)]]
            ambiguous = True
        for value, range_qual, unit_override in it["entries"]:
            unit = unit_override or it["unit"]
            dim_type = base_type
            if dim_type is None and UNIT_CLASS.get(unit) == "mass":
                dim_type = "weight"
            out.append(
                Measurement(
                    dimension_type=dim_type,
                    dimension_value=value,
                    dimension_measurement_unit=unit,
                    dimension_value_qualifier=join_slots(*label_quals, *it["quals"], range_qual),
                    dimension_measured_part=part,
                    axis=axis if len(items) > 1 else None,
                    type_ambiguous=ambiguous,
                )
            )
    return out


# Rewrites tried only where `_parse_segment` already returned None

_KW_ALT = "|".join(re.escape(w) for w in sorted(KEYWORDS, key=len, reverse=True))
# a keyword starting a new measurement immediately after
_JUXTAPOSED = re.compile(rf"\b({_UNIT_ALT})\s+(?=(?:{_KW_ALT})\b)")
_BY = re.compile(r"\bby\b")
_TRAILING_JUNK = re.compile(r"[\s.;:,]+$")


def _drop_tail(seg: str) -> str | None:
    """Trailing prose the grammar has no rule for ('30 oz capacity' -> '30 oz')"""
    toks = seg.split()
    while (
        toks
        and toks[-1] not in UNITS
        and toks[-1] not in KEYWORDS
        and toks[-1] not in QUALIFIERS
        and not _NUMLIKE.search(toks[-1])
    ):
        toks.pop()
    return " ".join(toks) if any(t in UNITS for t in toks) else None


def _recover_segment(seg: str) -> list[Measurement] | None:
    """Second chance for a segment the grammar rejected. Least invasive first"""
    seen = set()
    for use_by in (False, True):
        for use_jux in (False, True):
            s = seg
            if use_by:
                s = _BY.sub(" x ", s)
            if use_jux:
                s = _JUXTAPOSED.sub(r"\1 x ", s)
            for cand in (_TRAILING_JUNK.sub("", s), _drop_tail(s)):
                if not cand or cand in seen:
                    continue
                seen.add(cand)
                if (parsed := _parse_segment(cand)) is not None:
                    return parsed
    return None


def parse_dimensions(raw: str, prime_unit: str = "ft") -> dict | None:
    """Parse a recorded dimension string"""
    if not raw:
        return None
    s, flags = normalise(raw, prime_unit)
    if not s:
        return None
    default_part = None
    default_quals: list[str] = []
    alt_segments: list[str] = []

    def paren_sub(m: re.Match) -> str:
        nonlocal default_part
        inner = m.group(1).strip()
        toks = inner.split()
        if _NUMLIKE.search(inner):
            alt_segments.append(inner)  # alternate reading: 15 cm (6 in)
        elif toks and all(t in QUALIFIERS for t in toks):
            # '23.5 cms (approx)' — a qualifier, never a measured part
            default_quals.extend(QUALIFIERS[t] for t in toks)
        elif inner and default_part is None:
            default_part = inner  # measured part: 5 x 5 cm (mount)
        return " "

    s = PAREN.sub(paren_sub, s)
    segments = [(t.strip(), False) for t in re.split(r"[;,\n\r]", s) if t.strip()]
    segments += [(t, True) for t in alt_segments]
    measurements: list[Measurement] = []
    residue: list[str] = []
    for seg, is_alt in segments:
        parsed = _parse_segment(seg)
        if parsed is None:
            parsed = _recover_segment(seg)
        if parsed is None:
            residue.append(seg)
            continue
        for m in parsed:
            if is_alt:
                m.dimension_value_qualifier = join_slots(m.dimension_value_qualifier, "parenthetical")
            m.dimension_value_qualifier = join_slots(
                m.dimension_value_qualifier, *default_quals, "uncertain" if flags.uncertain else None, flags.bound
            )
            if m.dimension_measured_part is None:
                m.dimension_measured_part = default_part
        measurements.extend(parsed)
    if not measurements:
        return None
    return {
        "measurements": [asdict(m) for m in measurements],
        "residue": residue,
        "status": "resolved" if not residue else "partial",
    }
