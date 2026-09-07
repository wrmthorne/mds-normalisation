from __future__ import annotations

import re

import probablepeople

NAME_FIELDS = ["prefix", "given", "middle", "nickname", "surname", "suffix"]
PARTICLES = {
    "van",
    "von",
    "der",
    "den",
    "de",
    "du",
    "da",
    "di",
    "del",
    "della",
    "des",
    "dos",
    "das",
    "la",
    "le",
    "los",
    "ten",
    "ter",
    "te",
    "zu",
    "af",
    "av",
    "el",
    "al",
}
PREFIXES = {
    "mr",
    "mrs",
    "ms",
    "miss",
    "mme",
    "mlle",
    "dr",
    "sir",
    "dame",
    "lady",
    "lord",
    "rev",
    "revd",
    "fr",
    "prof",
    "professor",
    "capt",
    "captain",
    "col",
    "colonel",
    "maj",
    "major",
    "sgt",
    "lt",
    "cdr",
    "hon",
    # ranks and civic titles seen in inverted names
    "baron",
    "cpl",
    "pte",
    "mne",
    "cllr",
    "councillor",
    "alderman",
    "canon",
    "bishop",
    "archdeacon",
    "sister",
    "brother",
    "gen",
    "general",
    "adm",
    "admiral",
    "wg cdr",
    "flt lt",
    "lord provost",
    "provost",
    "mayor",
}
# Spelled-out trailing name additions; post-nominals have their own rule
SUFFIX_WORDS = {
    "esq",
    "esquire",
    "jnr",
    "jr",
    "junior",
    "snr",
    "sr",
    "senior",
    "the elder",
    "the younger",
    "the elder ii",
    "ii",
    "iii",
    "iv",
}
# Capitals with optional dots ('OBE', 'F.S.I.A.'); title-case tails excluded
POST_NOMINAL = re.compile(r"^(?:[A-Z]\.?){2,6}$")

# any letter, not just ASCII: 'Dürer', 'Vivarès'
NAME_TOK = re.compile(r"^[^\W\d_](?:[^\W\d_]|['’-])*\.?$")
# a dotted initials run ('J.W.') is one forename token
INITIALS_RUN = re.compile(r"^(?:[A-Za-z]\.){1,4}$")
ALPHA_TOK = re.compile(r"[^\W\d_]+")
CAP_TOK = re.compile(r"\b[A-Z]")
# initials after a comma are a person, never an organisation
INVERTED_INITIALS = re.compile(r",\s*(?:[A-Za-z]\.\s*){1,4}(?:,|$)")
# a lone capitalised word of three or more letters
MONONYM = re.compile(r"^[^\W\d_](?:[^\W\d_]|['’-]){2,}$")

FIELD_FOR = {
    "PrefixMarital": "prefix",
    "PrefixOther": "prefix",
    "GivenName": "given",
    "FirstInitial": "given",
    "MiddleName": "middle",
    "MiddleInitial": "middle",
    "Nickname": "nickname",
    "Surname": "surname",
    "LastInitial": "surname",
    "SuffixGenerational": "suffix",
    "SuffixOther": "suffix",
}


def _finish(name: dict) -> dict:
    name["display"] = " ".join(v for f in NAME_FIELDS if (v := name.get(f)))
    return name


def _name_token(tok: str) -> bool:
    return bool(NAME_TOK.match(tok) or INITIALS_RUN.match(tok))


INVERTED_SURNAME_MAX_TOKENS = 4
INVERTED_GIVEN_MAX_TOKENS = 5


def parse_inverted(value: str) -> dict | None:
    if value.count(",") != 1:
        return None
    left, right = (p.strip() for p in value.split(","))
    lt, rt = left.split(), right.split()
    if not (1 <= len(lt) <= INVERTED_SURNAME_MAX_TOKENS and 1 <= len(rt) <= INVERTED_GIVEN_MAX_TOKENS) or not all(
        _name_token(t) for t in lt + rt
    ):
        return None
    name = dict.fromkeys(NAME_FIELDS)
    prefix = []
    while rt and rt[0].lower().rstrip(".") in PREFIXES:
        prefix.append(rt.pop(0))
    particles = []
    while rt and rt[-1].lower() in PARTICLES:
        particles.insert(0, rt.pop())
    if not rt:
        return None
    name.update(
        prefix=" ".join(prefix) or None, given=rt[0], middle=" ".join(rt[1:]) or None, surname=" ".join(particles + lt)
    )
    return _finish(name)


def read_tail(segment: str) -> tuple[str, str] | None:
    """A trailing comma segment as (slot, value), or None if it is not one"""
    seg = segment.strip()
    if not seg:
        return None
    low = seg.lower().rstrip(".")
    if low in PREFIXES:
        return "prefix", seg
    if low in SUFFIX_WORDS or POST_NOMINAL.match(seg):
        return "suffix", seg
    return None


def parse_inverted_tail(value: str) -> dict | None:
    """'White, Francis Buchanan, Dr' / 'Reedie, Kenneth, Mr, MBE'"""
    if value.count(",") <= 1:  # the inversion takes one comma, a tail needs another
        return None
    head, tails = value, {"prefix": [], "suffix": []}
    while head.count(",") > 1:
        rest, _, last = head.rpartition(",")
        read = read_tail(last)
        if read is None:
            return None
        slot, text = read
        tails[slot].insert(0, text)
        head = rest
    if not any(tails.values()):
        return None
    name = parse_inverted(head)
    if name is None:
        return None
    for slot, parts in tails.items():
        if parts:
            existing = [name[slot]] if name[slot] else []
            name[slot] = " ".join(parts + existing if slot == "prefix" else existing + parts)
    return _finish(name)


# 'Surname/Forenames' with an optional '<Honorific' tail; the slash is tight
SLASH_NAME = re.compile(r"^([^/<]+)/([^/<]+?)(?:\s*<\s*(.+))?$")
ANGLE_TAIL = re.compile(r"^([^<]+?)\s*<\s*(.+)$")
ROMAN_TOK = re.compile(r"^[IVX]+$")  # 'HENRY I/King' — regnal, not a surname
HONORIFIC_MAX_TOKENS = 3
SLASHED_SURNAME_MAX_TOKENS = 3
SLASHED_GIVEN_MAX_TOKENS = 4


def _honorific(text: str) -> str | None:
    """A short qualification is an addition to the name; anything longer is provenance text"""
    h = text.strip()
    if not h or "," in h or len(h.split()) > HONORIFIC_MAX_TOKENS or re.search(r"\d", h):
        return None
    return h


def parse_slashed(value: str) -> dict | None:
    m = SLASH_NAME.match(value)
    if m is None or "," in value:
        return None
    left, right, honorific = m.group(1), m.group(2), m.group(3)
    if left != left.strip() or right != right.strip():
        return None  # spaced slash: an alternatives list
    lt, rt = left.split(), right.split()
    if not (1 <= len(lt) <= SLASHED_SURNAME_MAX_TOKENS and 1 <= len(rt) <= SLASHED_GIVEN_MAX_TOKENS):
        return None
    if not all(NAME_TOK.match(t) for t in lt) or not all(_name_token(t) for t in rt):
        return None
    if any(ROMAN_TOK.match(t) for t in lt) or "and" in (t.lower() for t in rt):
        return None  # a regnal left side, or a couple
    suffix = None
    if honorific is not None:
        suffix = _honorific(honorific)
        if suffix is None:
            return None
    name = dict.fromkeys(NAME_FIELDS)
    prefix = []
    while rt and rt[0].lower().rstrip(".") in PREFIXES:
        prefix.append(rt.pop(0))
    if not rt and not prefix:
        return None
    name.update(
        prefix=" ".join(prefix) or None,
        given=rt[0] if rt else None,
        middle=" ".join(rt[1:]) or None,
        surname=" ".join(lt),
        suffix=suffix,
    )
    return _finish(name)


def parse_angle_tail(value: str) -> dict | None:
    """'Carr, J.W. < M.A.', the same '<honorific' tail on a comma-inverted name"""
    m = ANGLE_TAIL.match(value)
    if m is None or "/" in value:
        return None
    head, tail = m.group(1).strip(), m.group(2)
    suffix = _honorific(tail)
    if suffix is None:
        return None
    name = parse_inverted(head) or parse_inverted_tail(head)
    if name is None:
        return None
    name["suffix"] = " ".join(filter(None, [name["suffix"], suffix]))
    return _finish(name)


def _org_guards(value: str) -> bool:
    """Shared refusals: a form that is person-shaped is never read as an org"""
    tokens = value.split()
    if len(tokens) <= 1 or not CAP_TOK.search(value):
        return False
    if set("/:") & set(value) or INVERTED_INITIALS.search(value):
        return False
    return not {t.lower().strip(".,") for t in tokens} & PREFIXES


def parse_corporation(value: str) -> dict | None:
    """The CRF parsed this as an organisation rather than a person"""
    try:
        tags, entity = probablepeople.tag(value)
    except Exception:  # RepeatedLabelError and friends: defer
        return None
    if entity != "Corporation" or "CorporationName" not in tags:
        return None
    return {"display": value.strip()} if _org_guards(value) else None


COORDINATION = re.compile(r"\s(?:and|&)\s|\s?&\s?")


def parse_and_organisation(value: str) -> dict | None:
    """'Harland and Wolff', 'Davidson & Kay', 'W. & D. Downey'"""
    if "," in value or not _org_guards(value):
        return None
    sides = [s.strip() for s in COORDINATION.split(value) if s.strip()]
    if len(sides) <= 1:
        return None
    if all(_crf_once(side)[0] is not None for side in sides):
        return None  # two parseable people, not a firm
    return {"display": value.strip()}


def parse_mononym_organisation(value: str) -> dict | None:
    """A lone capitalised word in an organisation-only field ('Wolseley')"""
    token = value.strip()
    if not MONONYM.match(token) or not token[0].isupper() or token.isupper():
        return None
    return {"display": token}


def _crf_once(text: str) -> tuple[dict | None, str | None]:
    runs = []
    for raw, label in probablepeople.parse(text):
        field = FIELD_FOR.get(label)
        if field is None:
            return None, f"unmapped_label:{label}"
        tok = raw.strip(",;")
        if runs and runs[-1][0] == field:
            runs[-1][1].append(tok)
        else:
            runs.append((field, [tok]))

    name = dict.fromkeys(NAME_FIELDS)
    for field, toks in runs:
        if name[field] is None:
            name[field] = " ".join(toks)
        elif field == "surname" and all(t.lower() in PARTICLES for t in toks):
            name["surname"] = " ".join(toks) + " " + name["surname"]
        else:
            return None, f"repeated_component:{field}"

    for field in ("middle", "given"):
        toks, moved = (name[field] or "").split(), []
        while toks and toks[-1].lower() in PARTICLES:
            moved.insert(0, toks.pop())
        if moved:
            name[field] = " ".join(toks) or None
            name["surname"] = " ".join(moved + ([name["surname"]] if name["surname"] else []))

    if not name["surname"]:
        return None, "no_surname"
    src = sorted(t.lower() for t in ALPHA_TOK.findall(text))
    out = sorted(t.lower() for t in ALPHA_TOK.findall(" ".join(v for v in name.values() if v)))
    if src != out:
        return None, "copy_check_failed"
    return _finish(name), None


def parse_crf(value: str) -> tuple[dict | None, str | None]:
    name, reason = _crf_once(value)
    if name is not None:
        return name, None
    # retry without leading honorifics; the CRF mislabels around them
    toks, lead = value.split(), []
    while toks and toks[0].lower().rstrip(".") in PREFIXES:
        lead.append(toks.pop(0))
    if lead and toks:
        name, _ = _crf_once(" ".join(toks))
        if name is not None:
            name["prefix"] = " ".join(lead + ([name["prefix"]] if name["prefix"] else []))
            return _finish(name), None
    return None, reason


def parse_person(value: str, *, org_field: bool = False) -> tuple[dict | None, str, str, str | None]:
    """Returns (name, entity_type, sub_component, defer_reason)"""
    if (name := parse_inverted(value)) is not None:
        return name, "person", "inverted", None
    if (name := parse_inverted_tail(value)) is not None:
        return name, "person", "inverted_tail", None
    if "/" in value or "<" in value:
        # convention-encoded: parse it or defer, never reaching the CRF
        if (name := parse_slashed(value)) is not None:
            return name, "person", "slashed", None
        if (name := parse_angle_tail(value)) is not None:
            return name, "person", "angle_tail", None
        return None, "person", "slashed", "slash_convention"
    name, reason = parse_crf(value)
    if name is not None:
        return name, "person", "crf", None
    if reason.startswith("unmapped_label:Corporation") and (org := parse_corporation(value)):
        return org, "organisation", "crf_corporation", None
    if reason == "unmapped_label:And" and (org := parse_and_organisation(value)):
        return org, "organisation", "crf_and_organisation", None
    if reason == "no_surname" and org_field and (org := parse_mononym_organisation(value)):
        return org, "organisation", "mononym_organisation", None
    return None, "person", "crf", reason
