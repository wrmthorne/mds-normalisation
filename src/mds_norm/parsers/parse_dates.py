from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from lark import Lark, Token, Transformer, v_args

# Bumped when readings change, so stale parse caches reparse
PARSER_VERSION = "2026-08-03.2"

MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "sept": 9,
}
NUM_MONTH = {
    i: m
    for m, i in {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }.items()
}
ROMAN_MONTH = {
    "i": "jan",
    "ii": "feb",
    "iii": "mar",
    "iv": "apr",
    "v": "may",
    "vi": "jun",
    "vii": "jul",
    "viii": "aug",
    "ix": "sep",
    "x": "oct",
    "xi": "nov",
    "xii": "dec",
}
NARROWING = {
    "early",
    "late",
    "mid",
    "1st-half",
    "2nd-half",
    "1st-quarter",
    "2nd-quarter",
    "3rd-quarter",
    "4th-quarter",
}
# A leading start verb is dropped so the year reads
_START_VERB = re.compile(
    r"^(?:born|b\.|est\.|established|founded|opened|bapti[sz]ed|published|"
    r"printed|issued|acquired)\b[\s.:-]*|^\*(?=\s*\d)"
)

PAREN = re.compile(r"\(([^)]*)\)")


@dataclass(frozen=True)
class Conventions:
    """Per-institution date conventions, exported by institutional_priors.py"""

    dm_order: str | None = None  # 'DM' | 'MD' for ambiguous numeric dates
    eq_range: bool = False  # '=' read as a range separator
    zero_null: bool = False  # 0/00 day-month slots mean "not recorded"
    # latest year a two-digit year may complete to
    century_ceiling: int | None = None


DEFAULT_CONVENTIONS = Conventions()

CENTURY = 100
MONTHS_IN_YEAR = 12
MAX_DAY = 31
QUARTERS = 4
SHORT_YEAR_DIGITS = 2
MIN_YEAR_DIGITS = 3
YEAR_DIGITS = 4
# an eight-digit date's year must land in this range
MIN_COMPACT_YEAR = 1000
MAX_COMPACT_YEAR = 2099


def complete_year(tok: str, conv: Conventions) -> str | None:
    """A two-digit year completed to the latest century not later than the ceiling; None if it cannot be"""
    if len(tok) != SHORT_YEAR_DIGITS:
        return tok
    if conv.century_ceiling is None:
        return None
    year = (conv.century_ceiling // CENTURY) * CENTURY + int(tok)
    return f"{year - CENTURY if year > conv.century_ceiling else year:04d}"


PERIODS: dict[str, tuple[int | None, int | None]] = {}


@dataclass
class _PeriodIndex:
    """The normalised label lookup, and the alternation that finds a label inside a longer value"""

    norm: dict[str, str] = field(default_factory=dict)
    pattern: re.Pattern[str] | None = None


_INDEX = _PeriodIndex()


def _load_periods(path: Path) -> dict[str, tuple[int | None, int | None]]:
    def year(v: str) -> int | None:
        return int(v) if v.strip() else None

    with path.open(newline="") as f:
        return {row["label"].lower(): (year(row["start"]), year(row["stop"])) for row in csv.DictReader(f)}


def _pnorm(s: str) -> str:
    s = re.sub(r"[-\s]+", " ", s.lower())
    s = re.sub(r"\bthe\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_period_index(periods: dict[str, tuple[int | None, int | None]]) -> None:
    PERIODS.clear()
    PERIODS.update(periods)
    _INDEX.norm = {_pnorm(k): k.title() for k in periods}
    if _INDEX.norm:
        alt = "|".join(re.escape(k) for k in sorted(_INDEX.norm, key=len, reverse=True))
        _INDEX.pattern = re.compile(rf"\b({alt})\b")
    else:
        _INDEX.pattern = None


def load_periods(path: str | Path) -> None:
    build_period_index(_load_periods(Path(path)))


def find_period(text: str) -> str | None:
    n = _pnorm(text)
    if not n:
        return None
    if n in _INDEX.norm:
        return _INDEX.norm[n]
    if _INDEX.pattern and (m := _INDEX.pattern.search(n)):
        return _INDEX.norm[m.group(1)]
    return None


# Circa markers: 'c.', glued 'c1912', parenthesised '(circa.)'
_CIRCA = re.compile(
    r"\(\s*(?:circa|ca?)\.?\s*\)"  # (c) (ca.) (circa.)
    r"|\bcirca\b|\bca?\.|\bapprox\b|\babout\b|~"
    r"|\bca?(?=\s?\d)"  # c1912, ca 1890s
)
_EMPTY_PAREN = re.compile(r"\(\s*[.,;\s]*\)")


@dataclass
class Flags:
    certainty: str | None = None
    bce: bool = False
    era: bool = False  # an explicit era marker (BC/BCE/AD/CE) was present
    dm_ambiguous: bool = False


def _dm(g1: str, g2: str, conv: Conventions, flags: Flags) -> tuple[int, int]:
    """Resolve a numeric day/month pair"""
    a, b = int(g1), int(g2)
    if a <= MONTHS_IN_YEAR < b:
        return b, a
    if a <= MONTHS_IN_YEAR and b <= MONTHS_IN_YEAR:
        if conv.dm_order == "MD":
            return b, a
        if conv.dm_order is None and a != b:
            flags.dm_ambiguous = True
    return a, b


def normalise(raw: str, conv: Conventions = DEFAULT_CONVENTIONS) -> tuple[str, Flags]:
    flags = Flags()
    s = raw.strip().lower()
    s = re.sub(r"^\s*\d+\)\s*", "", s)
    s = re.sub(r"\bdated\b", " ", s)
    s = _START_VERB.sub("", s)
    s = re.sub(r"(\d)['’]s\b", r"\1s", s)  # 1930's — the apostrophe decade
    s = re.sub(r"\[(\d{2})](\d{2})", r"\1\2", s)
    if "[" in s or "]" in s:
        flags.certainty = "inferred"
        s = s.replace("[", "").replace("]", "")
    # the circa family, written four ways besides the dot
    if _CIRCA.search(s):
        flags.certainty = "circa"
        s = _CIRCA.sub(" ", s)
        s = _EMPTY_PAREN.sub(" ", s)  # '1843 ( . )' left behind by the removal
    if "?" in s:
        flags.certainty = flags.certainty or "uncertain"
        s = s.replace("?", " ")
        s = _EMPTY_PAREN.sub(" ", s)  # '1924 (?)' leaves '1924 ( )' behind
    if re.search(r"\bbce?\b", s):
        flags.bce = True
    # an era marker makes a short year unambiguous ('AD 43')
    if re.search(r"\b(bce|bc|ce|ad)\b", s):
        flags.era = True
    s = re.sub(r"\b(bce|bc|ce|ad)\b", " ", s)
    # the grammar reads only a leading 'before'/'after' marker
    s = re.sub(r"<\s*=?", " before ", s)
    s = re.sub(r">\s*=?", " after ", s)
    s = re.sub(r"\b(?:pre|ante)\b[\s-]*", " before ", s)
    s = re.sub(r"\bonwards?\b", " after ", s)
    # '1914<pre' leaves the marker doubled after the rewrites
    s = re.sub(r"\b(before|after)\b(?:\s+\1\b)+", r"\1", s)
    s = re.sub(r"^(.*\d)\s+(before|after)\s*$", r"\2 \1", s)
    if conv.eq_range:
        s = s.replace("=", " to ")
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    if conv.zero_null:  # zero-placeholder day/month slots mean "not recorded"
        # a zero day degrades the value to month precision
        s = re.sub(r"\b(\d{3,4})([./-])(\d{1,2})\2\s*0{1,2}(?!\d)", r"\1\2\3", s)
        s = re.sub(r"(?<!\d)0{1,2}([./-])(\d{1,2})\1(\d{3,4})\b", r"\2\1\3", s)
        s = re.sub(r"\b(\d{3,4})([./-])0{1,2}(?:\2\s*0{1,2})?(?!\d)", r"\1", s)
        s = re.sub(r"(?<!\d)0{1,2}([./-])(?:0{1,2}\1)?(\d{3,4})\b", r"\2", s)

    def iso_sub(m: re.Match) -> str:
        y, mo, d = int(m.group(1)), int(m.group(2)), m.group(3)
        if not 1 <= mo <= MONTHS_IN_YEAR:
            return m.group(0)
        if d:
            return f" {int(d)} {NUM_MONTH[mo]} {y} "
        return f" {NUM_MONTH[mo]} {y} "

    s = re.sub(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?(?:t[\d:.]*z?)?\b", iso_sub, s)
    s = re.sub(r"\b(\d{4})\.(\d{2})(?:\.(\d{2}))?\b", iso_sub, s)
    s = re.sub(r"\bfirst\b", "1st", s)
    s = re.sub(r"\bsecond\b", "2nd", s)
    s = re.sub(r"\bthird\b", "3rd", s)
    s = re.sub(r"\bfourth\b", "4th", s)
    # an ordinal day loses its suffix: '18th' → '18'
    s = re.sub(
        r"\b(\d{1,2})(?:st|nd|rd|th)\b"
        r"(?!\s*(?:century|half|quarter|(?:-|to)\s*(?:early|late|mid)?"
        r"\s*\d{1,2}(?:st|nd|rd|th)\s*century))",
        r"\1",
        s,
    )
    # a trailing 'early'/'late'/'mid' qualifier moves to the front
    if m := re.match(r"^(.*?)(?:,\s*|\s*\(\s*)(early|late|mid)\s*\)?\s*$", s):
        s = f"{m.group(2)} {m.group(1).strip()}"
    s = re.sub(r"\b(early|late|mid)-(?=\d)", r"\1 ", s)

    def roman_sub(m: re.Match) -> str:
        tok = m.group(1)
        return f" {ROMAN_MONTH[tok]} " if tok in ROMAN_MONTH else m.group(0)

    s = re.sub(r"(?<![a-z])([ivx]+)(?![a-z])", roman_sub, s)

    def slash_sub(m: re.Match) -> str:
        d, mo = _dm(m.group(1), m.group(2), conv, flags)
        y = complete_year(m.group(3), conv)
        return f" {d} {NUM_MONTH[mo]} {y} " if 1 <= mo <= MONTHS_IN_YEAR and y else m.group(0)

    s = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", slash_sub, s)

    def dmy_sub(m: re.Match) -> str:
        d, mo = _dm(m.group(1), m.group(2), conv, flags)
        y = complete_year(m.group(3), conv)
        return f" {d} {NUM_MONTH[mo]} {y} " if 1 <= mo <= MONTHS_IN_YEAR and y else m.group(0)

    s = re.sub(r"\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b", dmy_sub, s)

    def my_sub(m: re.Match) -> str:
        mo = int(m.group(1))
        return f" {NUM_MONTH[mo]} {m.group(2)} " if 1 <= mo <= MONTHS_IN_YEAR else m.group(0)

    s = re.sub(r"\b(\d{1,2})\.(\d{4})\b", my_sub, s)
    # slash-written month/year ('1/1983') reads like the dot form
    s = re.sub(r"\b(\d{1,2})/(\d{4})\b", my_sub, s)

    def eight_sub(m: re.Match) -> str:
        s8 = m.group(0)
        for y, mo, d in (
            (int(s8[4:]), int(s8[2:4]), int(s8[:2])),  # ddmmyyyy
            (int(s8[:4]), int(s8[4:6]), int(s8[6:])),
        ):  # yyyymmdd
            if 1 <= mo <= MONTHS_IN_YEAR and 1 <= d <= MAX_DAY and MIN_COMPACT_YEAR <= y <= MAX_COMPACT_YEAR:
                return f" {d} {NUM_MONTH[mo]} {y} "
        return s8

    s = re.sub(r"\b\d{8}\b", eight_sub, s)
    s = s.replace("/", " ").replace(".", " ")
    return re.sub(r"\s+", " ", s).strip(), flags


GRAMMAR = r"""
?start: day_range | ymd_day_range | ym_range | my_range | cent_range | range | single

day_range.3: INT "-" INT MONTH INT     -> day_range
ymd_day_range.3: INT MONTH INT "-" INT -> ymd_day_range
ym_range.3: INT MONTH "-" MONTH        -> ym_range
my_range.3: MONTH "-" MONTH INT        -> my_range
cent_range.3: qualifier? ORD _sep qualifier? ORD "century" -> cent_range
range: bound _sep bound                -> mk_range
single: bound

?bound: qualifier? unit                -> apply_bound_qualifier

qualifier: "early"   -> early
         | "late"    -> late
         | "mid"     -> mid
         | "before"  -> before
         | "after"   -> after
         | "post"    -> after
         | "from"    -> after
         | ORD "half" "of"? "the"?    -> half
         | ORD "quarter" "of"? "the"? -> quarter

?unit: ymd | my | dm | year | decade | century

ymd:    INT MONTH INT   -> dmy
my:     MONTH INT       -> month_year
dm:     INT MONTH       -> day_month
year:   INT             -> year
decade: DECADE          -> decade
century: ORD "century"  -> century

_sep: "to" | "-"

MONTH: "january"|"february"|"march"|"april"|"may"|"june"|"july"|"august"|"september"|"october"|"november"|"december"
     | "jan"|"feb"|"mar"|"apr"|"jun"|"jul"|"aug"|"sep"|"sept"|"oct"|"nov"|"dec"
ORD: /\d+(st|nd|rd|th)/
DECADE: /\d{3,4}s/
INT: /\d+/

%import common.WS
%ignore WS
"""


@dataclass
class Point:
    year: int | None
    month: int | None = None
    day: int | None = None


@dataclass
class Span:
    start: Point | None
    end: Point | None
    start_qual: str | None = None
    end_qual: str | None = None
    kind: str = "point"


def _century_span(tok: str, bce: bool = False) -> tuple[int, int]:
    """Year bounds of the nth century"""
    n = int(re.match(r"\d+", tok).group())
    if bce:
        return (n - 1) * 100 + 1, n * 100
    if n == 1:  # no year 0 CE, so century 1 runs 0001-0100
        return 1, 100
    return (n - 1) * 100, (n - 1) * 100 + 99


def _decade_span(tok: str) -> tuple[int, int]:
    base = int(tok.rstrip("s"))
    return base, base + 9


def _year(tok: Token, short_ok: bool = False) -> int:
    """Only a 3-4 digit run is a year, unless an era marker made a short year unambiguous"""
    s = str(tok)
    if not (MIN_YEAR_DIGITS <= len(s) <= YEAR_DIGITS or (short_ok and 1 <= len(s) <= SHORT_YEAR_DIGITS)):
        raise ValueError(f"implausible year: {s}")
    return int(s)


def _narrow(span: Span, q: str) -> Span:
    s, e = span.start, span.end
    if not (s and e and s.month is None and e.month is None):
        return Span(s, e, q, q, span.kind)
    a, b, n = s.year, e.year, e.year - s.year + 1
    if b <= a:  # one year or less leaves nothing to narrow
        return Span(s, e, q, q, "point")
    third = max(n // 3, 1)
    if q == "early":
        b = a + third - 1
    elif q == "late":
        a = b - third + 1
    elif q == "mid":
        a, b = a + third, b - third
    elif q == "1st-half":
        b = s.year + n // 2 - 1
    elif q == "2nd-half":
        a = s.year + n // 2
    elif q.endswith("-quarter"):
        k = int(q[0])
        a, b = s.year + (k - 1) * n // 4, s.year + k * n // 4 - 1
    return Span(Point(a), Point(b), q, q, "range")


@v_args(inline=True)
class ToSpan(Transformer):
    def __init__(self, short_years: bool = False, bce: bool = False) -> None:
        super().__init__()
        self._short = short_years
        self._bce = bce

    def _yr(self, tok: Token) -> int:
        return _year(tok, short_ok=self._short)

    def year(self, n: Token) -> Span:
        y = self._yr(n)
        return Span(Point(y), Point(y), kind="point")

    def month_year(self, mon: Token, n: Token) -> Span:
        y, m = self._yr(n), MONTHS[str(mon)]
        return Span(Point(y, m), Point(y, m), kind="point")

    def day_month(self, n: Token, mon: Token) -> Span:
        v, m = int(n), MONTHS[str(mon)]
        if v <= MAX_DAY:
            return Span(Point(None, m, v), Point(None, m, v), kind="point")
        y = self._yr(n)
        return Span(Point(y, m), Point(y, m), kind="point")

    def dmy(self, d: Token, mon: Token, n: Token) -> Span:
        # day-first, then year-first: whichever pair is plausible wins
        m = MONTHS[str(mon)]
        for y_tok, d_tok in ((n, d), (d, n)):
            try:
                y, dd = self._yr(y_tok), int(d_tok)
            except ValueError:
                continue
            if 1 <= dd <= MAX_DAY:
                return Span(Point(y, m, dd), Point(y, m, dd), kind="point")
        raise ValueError(f"implausible day/year pair: {d} {n}")

    def decade(self, tok: Token) -> Span:
        a, b = _decade_span(str(tok))
        return Span(Point(a), Point(b), kind="decade")

    def century(self, ordt: Token) -> Span:
        a, b = _century_span(str(ordt), self._bce)
        return Span(Point(a), Point(b), kind="century")

    def early(self) -> str:
        return "early"

    def late(self) -> str:
        return "late"

    def mid(self) -> str:
        return "mid"

    def before(self) -> str:
        return "before"

    def after(self) -> str:
        return "after"

    def half(self, ordt: Token) -> str:
        return "1st-half" if str(ordt).startswith("1") else "2nd-half"

    def quarter(self, ordt: Token) -> str:
        k = int(re.match(r"\d+", str(ordt)).group())
        if not 1 <= k <= QUARTERS:
            raise ValueError(f"not a quarter: {ordt}")
        return f"{['1st', '2nd', '3rd', '4th'][k - 1]}-quarter"

    def apply_bound_qualifier(self, *args: Span | str) -> Span:
        if len(args) == 1:
            return args[0]
        q, span = args
        if q in NARROWING:
            return _narrow(span, q)
        if q == "before":
            return Span(None, span.end, end_qual="before", kind="open_before")
        if q == "after":
            return Span(span.start, None, start_qual="after", kind="open_after")
        return span

    def single(self, span: Span) -> Span:
        return span

    def mk_range(self, a: Span, b: Span) -> Span:
        start, end = a.start, b.end
        # a trailing year governs both sides of the range
        if start and end and start.year is None and end.year is not None:
            start.year = end.year
        return Span(start, end, a.start_qual, b.end_qual, "range")

    def _day_span(self, y: Token, mon: Token, d1: Token, d2: Token) -> Span:
        # both slots must be real days, not a year
        days = [int(d1), int(d2)]
        if not all(1 <= d <= MAX_DAY for d in days):
            raise ValueError(f"implausible day pair: {d1} {d2}")
        m, yy = MONTHS[str(mon)], _year(y)
        return Span(Point(yy, m, days[0]), Point(yy, m, days[1]), kind="range")

    def day_range(self, d1: Token, d2: Token, mon: Token, y: Token) -> Span:
        return self._day_span(y, mon, d1, d2)

    def ymd_day_range(self, y: Token, mon: Token, d1: Token, d2: Token) -> Span:
        return self._day_span(y, mon, d1, d2)

    def ym_range(self, y: Token, m1: Token, m2: Token) -> Span:
        yy = _year(y)
        return Span(Point(yy, MONTHS[str(m1)]), Point(yy, MONTHS[str(m2)]), kind="range")

    def my_range(self, m1: Token, m2: Token, y: Token) -> Span:
        # the year is written once, governing both months
        return self.ym_range(y, m1, m2)

    def cent_range(self, *args: Token | str) -> Span:
        # optional qualifiers vary the arity, so re-read positionally
        sides, pending = [], None
        for a in args:
            s = str(a)
            if re.fullmatch(r"\d+(st|nd|rd|th)", s):
                sides.append((pending, s))
                pending = None
            else:
                pending = s

        def side(q: str | None, ordt: str) -> Span:
            lo, hi = _century_span(ordt, self._bce)
            span = Span(Point(lo), Point(hi), kind="century")
            return _narrow(span, q) if q in NARROWING else span

        (q1, o1), (q2, o2) = sides
        a, b = side(q1, o1), side(q2, o2)
        return Span(a.start, b.end, a.start_qual, b.end_qual, "range")


_parser = Lark(GRAMMAR, parser="earley", ambiguity="resolve")
_TRANSFORMERS = {(s, b): ToSpan(s, b) for s in (False, True) for b in (False, True)}


def parse_to_span(text: str, short_years: bool = False, bce: bool = False) -> Span | None:
    try:
        return _TRANSFORMERS[short_years, bce].transform(_parser.parse(text))
    except Exception:
        return None


# A year followed by a later year's tail: '1914-15', '1992/3'
_SHORT_RANGE = re.compile(r"(\d{4})\s*[-/ ]\s*(\d{1,2})")


def _short_range(text: str) -> Span | None:
    m = _SHORT_RANGE.fullmatch(text.strip())
    if m is None:
        return None
    start, tail = int(m.group(1)), m.group(2)
    end = int(m.group(1)[: 4 - len(tail)] + tail)
    if end <= start:  # '1899-02' rolls into the next century
        end += 10 ** len(tail)
    if not start < end <= start + 99:
        return None
    return Span(Point(start), Point(end), kind="range")


def _negate(span: Span) -> None:
    """BCE years into EDTF's astronomical numbering: n BCE is -(n-1), so 1 BCE writes as 0000"""
    for p in (span.start, span.end):
        if p and p.year is not None:
            p.year = 1 - p.year


def _order(span: Span) -> None:
    """Put a range's endpoints in chronological order"""
    s, e = span.start, span.end
    if span.kind == "point" or not (s and e) or s.year is None or e.year is None:
        return
    if (s.year, s.month or 0, s.day or 0) > (e.year, e.month or 0, e.day or 0):
        span.start, span.end = e, s
        span.start_qual, span.end_qual = span.end_qual, span.start_qual


_SET = re.compile(r"\s*[\[{]\s*(.+?)\s*[\]}]\s*")


def _edtf_set(raw: str) -> Span | None:
    """An EDTF date set collapses to the span from its earliest to its latest member"""
    m = _SET.fullmatch(raw.strip())
    if not m:
        return None
    body = m.group(1)
    members = [x.strip() for x in body.split(",") if x.strip()]
    if len(members) <= 1 and ".." not in body:
        return None
    years = [
        int(tok) for mem in members for part in mem.split("..") if re.fullmatch(r"-?\d{3,4}", tok := part.strip())
    ]
    if not years:
        return None
    lo, hi = min(years), max(years)
    if lo == hi:
        return Span(Point(lo), Point(lo), kind="point")
    return Span(Point(lo), Point(hi), kind="range")


def _multidate(text: str, short_years: bool = False, bce: bool = False) -> Span | None:
    segs = [p.strip() for p in re.split(r",|\band\b", text) if p.strip()]
    spans = [s for p in segs if (s := parse_to_span(p, short_years, bce))]
    if not spans:
        return None
    years = [p.year for s in spans for p in (s.start, s.end) if p and p.year is not None]
    yr = years[0] if years else None
    pts = []
    for s in spans:
        for p in (s.start, s.end):
            if p is None:
                continue
            if p.year is None and yr is not None:
                p.year = yr
            if p.year is not None:
                pts.append(p)
    if not pts:
        return None
    lo = min(pts, key=lambda p: (p.year, p.month or 1, p.day or 1))
    hi = max(pts, key=lambda p: (p.year, p.month or 12, p.day or 31))
    if (lo.year, lo.month, lo.day) == (hi.year, hi.month, hi.day):
        return Span(lo, lo, kind="point")
    return Span(lo, hi, kind="range")


def _iso(p: Point | None) -> str | None:
    # a 0 month/day is a placeholder, never emit '-00'
    if p is None or p.year is None:
        return None
    sign, y = ("-", -p.year) if p.year < 0 else ("", p.year)
    if p.month and p.day:
        return f"{sign}{y:04d}-{p.month:02d}-{p.day:02d}"
    if p.month:
        return f"{sign}{y:04d}-{p.month:02d}"
    return f"{sign}{y:04d}"


def _edtf_point(p: Point | None) -> str:
    # a point with no year is unknown (`..`), never `XXXX`
    if p is None or p.year is None:
        return ".."
    y = f"-{-p.year:04d}" if p.year < 0 else f"{p.year:04d}"
    if p.month and p.day:
        return f"{y}-{p.month:02d}-{p.day:02d}"
    if p.month:
        return f"{y}-{p.month:02d}"
    return y


def _edtf(span: Span, certainty: str | None) -> str:
    q = "~" if certainty == "circa" else ("?" if certainty else "")

    def mark(pt: Point | None) -> str:
        s = _edtf_point(pt)
        return s + q if s != ".." else s  # a certainty marker never rides `..`

    if span.kind == "point":
        return mark(span.start)
    if span.kind == "decade":
        return f"{span.start.year // 10}X" + q
    if span.kind == "open_before":
        return f"../{mark(span.end)}"
    if span.kind == "open_after":
        return f"{mark(span.start)}/.."
    return f"{mark(span.start)}/{mark(span.end)}"


# A lone signed year ('-118') is an ISO-style BCE year
_SIGNED_YEAR = re.compile(r"-\s?(\d{1,4})")


def parse_date(raw: str, conv: Conventions = DEFAULT_CONVENTIONS) -> dict | None:
    span = _edtf_set(raw)
    if span is None and (m := _SIGNED_YEAR.fullmatch(raw.strip())) and int(m.group(1)):
        y = -int(m.group(1))
        span = Span(Point(y), Point(y), kind="point")
    if span is not None:
        flags, period = Flags(), None
    else:
        text, flags = normalise(raw, conv)
        if not text:
            return None
        label_src = text
        if pm := PAREN.search(text):
            span = parse_to_span(pm.group(1), flags.era, flags.bce)
            label_src = (text[: pm.start()] + " " + text[pm.end() :]).strip()
        elif " and " in text or "," in text:
            span = _multidate(text, flags.era, flags.bce)
        else:
            span = parse_to_span(text, flags.era, flags.bce)
        if span is None:
            span = _short_range(text)
        period = find_period(label_src)
    if span is None and period is None:
        return None
    # a span with no year locates nothing without a period
    if (
        span is not None
        and period is None
        and not any(p is not None and p.year is not None for p in (span.start, span.end))
    ):
        return None
    if span and flags.bce:
        _negate(span)
    if span:
        _order(span)
    if span is None:
        return {
            "date_earliest_single": None,
            "date_earliest_single_certainty": None,
            "date_earliest_single_qualifier": None,
            "date_latest": None,
            "date_latest_certainty": None,
            "date_latest_qualifier": None,
            "date_period": period,
            "value_edtf": None,
            "dm_ambiguous": False,
        }
    is_point = span.kind == "point"
    e_iso = _iso(span.start)
    l_iso = None if is_point else _iso(span.end)
    return {
        "date_earliest_single": e_iso,
        "date_earliest_single_certainty": flags.certainty if e_iso else None,
        "date_earliest_single_qualifier": span.start_qual if e_iso else None,
        "date_latest": l_iso,
        "date_latest_certainty": flags.certainty if l_iso else None,
        "date_latest_qualifier": span.end_qual if l_iso else None,
        "date_period": period,
        "value_edtf": _edtf(span, flags.certainty),
        "dm_ambiguous": flags.dm_ambiguous,
    }
