from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from html import unescape

import polars as pl

from mds_norm.paths import VOCABS
from mds_norm.utils.atomise import norm_term

SCHEMA = ["subject", "term", "lang", "kind"]


def _frame(rows: list[tuple[str, str, str | None, str]]) -> pl.DataFrame:
    return (
        pl.DataFrame(rows, schema=SCHEMA, orient="row")
        .with_columns(norm=norm_term(pl.col("term")))
        .filter(pl.col("norm") != "")
        .unique()
    )


def build_gbif(dirname: str) -> pl.DataFrame:
    """GBIF XML thesaurus (rank, type_status) -> term index"""
    ns = "{http://rs.gbif.org/thesaurus/}"
    dc = "{http://purl.org/dc/terms/}"
    lang_attr = "{http://www.w3.org/XML/1998/namespace}lang"
    root = ET.parse(next((VOCABS / dirname).glob("*.xml"))).getroot()
    rows = []
    for concept in root.iter(ns + "concept"):
        subject = concept.get(dc + "identifier")
        for group, kind in ((ns + "preferred", "prefLabel"), (ns + "alternative", "altLabel")):
            for el in concept.iterfind(f"{group}/{ns}term"):
                term = (el.get(dc + "title") or "").strip()
                if term:
                    rows.append((subject, term, el.get(lang_attr), kind))
    return _frame(rows)


def build_dcmi_type() -> pl.DataFrame:
    """DCMI Type Vocabulary Turtle -> term index"""
    text = (VOCABS / "dcmi_type" / "dublin_core_type.ttl").read_text()
    rows = []
    for block in re.finditer(r"<http://purl\.org/dc/dcmitype/(\w+)>(.*?)(?=\n<|\Z)", text, re.DOTALL):
        subject, body = block.group(1), block.group(2)
        labels = re.findall(r'rdfs:label\s+"([^"]+)"@(\w+)', body)
        if not labels:
            continue
        for i, (label, lang) in enumerate(labels):
            rows.append((subject, label, lang, "prefLabel" if i == 0 else "altLabel"))
        spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", subject)
        rows.append((subject, subject, "en", "altLabel"))
        rows.append((subject, spaced, "en", "altLabel"))
    return _frame(rows)


def build_iana_media_types() -> pl.DataFrame:
    """IANA media type registry CSVs -> term index"""
    rows = []
    for path in sorted((VOCABS / "iana_media_types").glob("*.csv")):
        frame = pl.read_csv(path)
        for name, template in zip(frame["Name"], frame["Template"], strict=True):
            if not template or "/" not in template or "(" in name:
                continue
            rows.append((template, template, "en", "prefLabel"))
            rows.append((template, name, "en", "altLabel"))
            rows.append((template, template.split("/", 1)[1], "en", "altLabel"))
    return _frame(rows)


def build_loc_relators() -> pl.DataFrame:
    """MARC Relator terms (maintained HTML edition) -> term index"""
    text = (VOCABS / "loc_relators" / "relaterm.html").read_text(errors="replace")
    rows: list[tuple[str, str, str | None, str]] = []
    code_for: dict[str, str] = {}
    for entry in re.finditer(
        r'<dt[^>]*>\s*<span class="authorized">(?P<term>.*?)</span>'
        r'\s*<span class="relator-code">\s*\[(?P<code>\w+)\]</span>\s*</dt>'
        r"(?P<body>.*?)(?=<dt|</dl>)",
        text,
        re.DOTALL,
    ):
        term = unescape(re.sub(r"<[^>]+>", "", entry["term"])).strip()
        code_for[term.lower()] = entry["code"]
        rows.append((entry["code"], term, "en", "prefLabel"))
        for uf in re.finditer(r'<span class="use-for-ref">(.*?)</span>', entry["body"], re.DOTALL):
            alt = unescape(re.sub(r"<[^>]+>", "", uf.group(1))).strip()
            if alt:
                rows.append((entry["code"], alt, "en", "altLabel"))

    for entry in re.finditer(
        r'<dt class="unauthorized">(?P<term>.*?)</dt>(?P<body>.*?)(?=<dt|</dl>)', text, re.DOTALL
    ):
        term = unescape(re.sub(r"<[^>]+>", "", entry["term"])).strip()
        use = re.search(r'<span class="use-ref">(.*?)</span>', entry["body"], re.DOTALL)
        if not use:
            continue
        target = unescape(re.sub(r"<[^>]+>", "", use.group(1))).strip().lower()
        if target in code_for and term:
            rows.append((code_for[target], term, "en", "altLabel"))
    return _frame(rows)


def build_fish_monument_types() -> pl.DataFrame:
    """FISH Monument Types thesaurus CSV export -> term index"""
    base = VOCABS / "fish_monument_types"
    terms = pl.read_csv(base / "ThesaurusTerms.csv")
    prefs = pl.read_csv(base / "ThesaurusTermPreferences.csv")
    non_pref = dict(zip(prefs["THE_TE_UID_1"], prefs["THE_TE_UID_2"], strict=True))
    indexable = {
        uid
        for uid, index_term in zip(terms["THE_TE_UID"], terms["INDEX_TERM"], strict=True)
        if index_term == "Y" and uid not in non_pref
    }
    rows = []
    for uid, raw_term in zip(terms["THE_TE_UID"], terms["TERM"], strict=True):
        term = (raw_term or "").strip()
        if not term:
            continue
        if uid in non_pref:
            if non_pref[uid] in indexable:
                rows.append((str(non_pref[uid]), term, "en", "altLabel"))
        elif uid in indexable:
            rows.append((str(uid), term, "en", "prefLabel"))
    return _frame(rows)


BUILDERS = {
    "gbif_rank": lambda: build_gbif("gbif_rank"),
    "gbif_type_status": lambda: build_gbif("gbif_type_status"),
    "dcmi_type": build_dcmi_type,
    "iana_media_types": build_iana_media_types,
    "loc_relators": build_loc_relators,
    "fish_monument_types": build_fish_monument_types,
}


def build(vocab: str) -> pl.DataFrame:
    """Index a released vocabulary"""
    return BUILDERS[vocab]()


def main() -> None:
    for name, builder in BUILDERS.items():
        index = builder()
        print(f"{name}: {index['subject'].n_unique():,} concepts, {len(index):,} surface forms")


if __name__ == "__main__":
    main()
