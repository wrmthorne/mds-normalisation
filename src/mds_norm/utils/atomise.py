from __future__ import annotations

import json
import re
from collections.abc import Iterable

import polars as pl


def norm_term(col: pl.Expr) -> pl.Expr:
    """The vocabulary norm every index, atom and decision keys on"""
    return col.str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars(" .,;:")


# Semicolon and pipe always split; other literals are discovered
BASE_SEPARATORS = (";", "|")
# the only word separators; "on" would split "oil on canvas"
WORD_SEPARATORS = ("and", "or")
# Earlier artefacts name separators rather than quoting them
LEGACY_NAMES = {"semicolon": ";", "pipe": "|", "comma": ",", "slash": "/", "ampersand": "&", "plus": "+"}


def separator_literal(sep: str) -> str:
    return LEGACY_NAMES.get(sep, sep)


def separator_regex(sep: str) -> str:
    """The whitespace-tolerant pattern for one separator; digit guards keep 1,200 and 1/2 whole"""
    lit = separator_literal(sep)
    if lit.isalpha():
        return rf"\s+{re.escape(lit)}\s+"
    return rf"(?<!\d)\s*{re.escape(lit)}\s*(?!\d)"


def separators_regex(seps: Iterable[str]) -> str:
    """The union pattern over a separator set, longer literals first so a compound literal wins its prefix"""
    lits = sorted({separator_literal(s) for s in seps}, key=lambda lit: (-len(lit), lit))
    return "|".join(separator_regex(lit) for lit in lits)


# Null markers, including source-system column headers leaked into the data
PLACEHOLDER_MARKERS = {
    "-",
    "--",
    "?",
    "??",
    ".",
    "n/a",
    "na",
    "no",
    "none",
    "nil",
    "null",
    "no data",
    "various",
    "place",
    "period",
    "association details",
    "misc",
    "miscellaneous",
    "other",
    "tbc",
    "see notes",
}
# Knowledge-state markers are dropped inside lists, kept as whole values
SEMANTIC_MARKERS = {"unknown", "not known", "unspecified", "unidentified", "not recorded", "not stated"}
NULL_MARKERS = PLACEHOLDER_MARKERS | SEMANTIC_MARKERS  # the list-context drop set

# Fields whose values are single concepts however long they run
NO_ATOMISE_FIELDS = {"spectrum/associated_concept"}
PROSE_MAX_CHARS = 200  # whole-value guard before splitting
PROSE_MAX_TOKENS = 25
ATOM_PROSE_MAX_CHARS = 100  # per-atom guard after splitting (matches SEMANTIC_MAX_LEN)
ATOM_PROSE_MAX_TOKENS = 15

_pattern_cache: dict[tuple, re.Pattern] = {}


def split_pattern(seps: Iterable[str] | None) -> re.Pattern:
    lits = tuple(sorted({separator_literal(s) for s in seps or ()} | set(BASE_SEPARATORS)))
    if lits not in _pattern_cache:
        _pattern_cache[lits] = re.compile(separators_regex(lits))
    return _pattern_cache[lits]


def atomise(value: str, seps: Iterable[str] | None) -> list[dict]:
    rx = split_pattern(seps)
    parts, last = [], 0
    for m in rx.finditer(value):
        parts.append((value[last : m.start()], last, m.start()))
        last = m.end()
    parts.append((value[last:], last, len(value)))
    out = []
    for part, start, _end in parts:
        lead = len(part) - len(part.lstrip())
        atom = part.strip()
        if atom:
            out.append({"atom": atom, "span_start": start + lead, "span_end": start + lead + len(atom)})
    return out


GB_EXCEPTIONS = {
    "hour",
    "hours",
    "flour",
    "sour",
    "tour",
    "tours",
    "four",
    "pour",
    "pours",
    "our",
    "velour",
    "contour",
    "contours",
    "gourd",
    "gourds",
    "genre",
    "genres",
    "macabre",
    "turquoise",
    "praise",
    "rise",
    "wise",
    "paradise",
    "called",
    "rolled",
    "filled",
    "milled",
    "drilled",
    "spelled",
    "killed",
    "walled",
    "pulled",
    "chilled",
    "skilled",
}
GB_WORD_MAP = {
    "grey": "gray",
    "greys": "grays",
    "aluminium": "aluminum",
    "sulphur": "sulfur",
    "sulphate": "sulfate",
    "sulphide": "sulfide",
    "jewellery": "jewelry",
    "mould": "mold",
    "moulds": "molds",
    "moulded": "molded",
    "moulding": "molding",
    "mouldings": "moldings",
    "plough": "plow",
    "ploughs": "plows",
    "pyjamas": "pajamas",
    "tyre": "tire",
    "tyres": "tires",
    "kerb": "curb",
    "kerbs": "curbs",
    "storey": "story",
    "storeys": "stories",
    "gramme": "gram",
    "programme": "program",
    "programmes": "programs",
    "manoeuvre": "maneuver",
    "woollen": "woolen",
    "chequered": "checkered",
    "draught": "draft",
    "draughts": "drafts",
}
GB_RULES = [
    (re.compile(r"our(s|ed|ing|ings)?$"), lambda m: "or" + (m.group(1) or "")),
    (re.compile(r"(?<=[tbh])re(s)?$"), lambda m: "er" + (m.group(1) or "")),
    (re.compile(r"(?<=[lnrmtvdg])is(e|es|ed|ing|ation|ations)$"), lambda m: "iz" + m.group(1)),
    (re.compile(r"(?<=[a-z])ae"), lambda _m: "e"),
    (re.compile(r"ll(ed|ing|er|ers)$"), lambda m: "l" + m.group(1)),
]


def us_variant(norm: str) -> str | None:
    """The GB to US respelling of a normalised atom, or None if unchanged"""
    out, changed = [], False
    for tok in norm.split(" "):
        if tok in GB_WORD_MAP:
            out.append(GB_WORD_MAP[tok])
            changed = True
            continue
        if tok in GB_EXCEPTIONS:
            out.append(tok)
            continue
        v = tok
        for rx, rep in GB_RULES:
            v = rx.sub(rep, v)
        out.append(v)
        changed |= v != tok
    return " ".join(out) if changed else None


MIN_DEVERBAL_LEN = 4  # a shorter '-ed' token has too little stem left


def morph_variants(norm: str) -> list[str]:
    """Deverbal rephrasings of a normalised atom, most-conservative first"""
    toks = norm.split(" ")
    variants = []
    for i, tok in enumerate(toks):
        if len(tok) > MIN_DEVERBAL_LEN and tok.endswith("ed") and not tok.endswith("eed"):
            stem = tok[:-2]
            for form in (stem + "eing", stem + "ing"):
                out = toks.copy()
                out[i] = form
                variants.append(" ".join(out))
    return variants


# 'engraving on paper': fields whose tail is out of field
COMPOUND_FIELDS = {"spectrum/technique", "spectrum/inscription_method"}
COMPOUND_PREPS = ("on", "onto", "in", "into", "over", "with")
COMPOUND_MAX_TOKENS = 4  # either side longer than this is prose, not a compound
COMPOUND_RX = re.compile(rf"^(?P<head>.+?)\s+(?P<prep>{'|'.join(COMPOUND_PREPS)})\s+(?P<tail>.+)$")


def compound_head(norm: str) -> tuple[str, str, str] | None:
    """`'engraving on paper'` -> `('engraving', 'paper', 'on paper')`, else None"""
    m = COMPOUND_RX.match(norm)
    if m is None:
        return None
    head, tail = m["head"].strip(), m["tail"].strip()
    if not head or not tail:
        return None
    if len(head.split()) > COMPOUND_MAX_TOKENS or len(tail.split()) > COMPOUND_MAX_TOKENS:
        return None
    if any(c.isdigit() for c in norm):
        return None
    return head, tail, f"{m['prep']} {tail}"


LLM_COVERAGE = 0.95
LLM_MAX_NEW_TOKENS = 128
LLM_MAX_ATOM_TOKENS = 6
DATE_LIKE = r"\d{3,4}|\d{1,2}(st|nd|rd|th)\b|centur|circa|\bc\.|\bbc\b|\bad\b"

# One per group in vocab_indexes.CONCEPT_GROUPS, rendered into the prompt
GROUP_DESC = {
    "aat": "object type / technique / subject classification",
    "aat+fish_building_materials": "material or medium",
    "periodo": "named historical period",
    "fish_event_types": "fieldwork or collection method",
    "fish_event_types+local_field_collection_method": "fieldwork or collection method",
    "local_persons_association+loc_relators": "person's or organisation's association with the object "
    "(role, occupation or relationship)",
    "local_association_event": "event in the object's history that a date or place attaches to",
    "local_acquisition_method": "means by which the museum acquired the object",
    "local_condition": "physical condition of the object",
    "local_location_type": "kind of location the object is stored or displayed in",
    "local_object_status+gbif_type_status": "status of the object in the collection, "
    "including nomenclatural type status",
    "local_place_status+fish_monument_types": "status or type of a place associated with the object",
    "local_title_type": "kind of title the object is known by",
    "gbif_rank+local_object_name_level": "level of the object name (taxonomic rank or naming level)",
    "dcmi_type": "type of a reproduction of the object",
    "iana_media_types": "file format of a reproduction of the object",
}
LLM_PROMPT = (
    'The following is one value from the "{desc}" field of a UK museum catalogue record.\n'
    "\n"
    'Value: "{value}"\n'
    "\n"
    "Extract the standalone controlled-vocabulary terms it contains ({desc} terms). Rules:\n"
    "- A string is an atom only if you would expect it, as written, as a label in some vocabulary — the "
    "field's target scheme or an institution's own classification system (`Tea, Coffee & Chocolate wares`).\n"
    "- Media coordinations split (`pen and ink` is two atoms), and support constructions contribute the noun, "
    "not the preposition (`etching on paper` → `etching`, `paper`; `pencil and ink on paper` → `pencil`, "
    "`ink`, `paper`).\n"
    "- For prose samples, only extract terms that are genuinely assertable (`double glass with engraved gold "
    "and silver foil…` → `glass`, `engraved`, `gold`, `silver foil`, …). Never return the whole sentence as a "
    "single atom.\n"
    "- Strip terminal punctuation from atoms (`Bivalvia.` → `Bivalvia`) unless it is part of the atom itself "
    "(e.g. `C.B.M.`, the abbreviation `man.`)\n"
    "- Null markers in lists (`wood; unknown`) should be dropped but when already atomic "
    "(`Unknown`/`Anonymous`) they carry real semantics and should persist.\n"
    "- Identifiers are never atoms. Date-shaped tokens are (`1880s`, `19th Century`, `c. 202 BC – AD 24`) are "
    "vocabulary-shaped period labels and stay as atoms even outside period fields.\n"
    "- Copy each term verbatim as a contiguous substring of the value; never rephrase, translate, or reorder.\n"
    "- If the value contains no vocabulary terms, return [].\n"
    "\n"
    "Answer with only a JSON array of strings."
)


def parse_llm_atoms(value: str, completion: str) -> list[dict] | None:
    m = re.search(r"\[.*?\]", completion, re.DOTALL)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list):
        return None
    low, out, seen, cursor = value.lower(), [], set(), 0
    for item in arr:
        if not isinstance(item, str):
            continue
        a = item.strip()
        if not a or a.lower() in seen or a.lower() == low or len(a.split()) > LLM_MAX_ATOM_TOKENS:
            continue
        pos = low.find(a.lower(), cursor)
        if pos < 0:
            pos = low.find(a.lower())
        if pos < 0:
            continue  # not a verbatim substring -> discard, never trust a rewrite
        seen.add(a.lower())
        out.append({"sub_atom": value[pos : pos + len(a)], "sub_start": pos, "sub_end": pos + len(a)})
        cursor = pos + len(a)
    return out or None
