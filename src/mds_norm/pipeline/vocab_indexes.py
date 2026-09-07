from __future__ import annotations

import codecs
import gzip
import json
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Callable
from functools import cache, partial
from pathlib import Path
from typing import IO

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from codecarbon import EmissionsTracker

from mds_norm.paths import EMISSIONS_LOG, VOCAB_INDEXES, VOCABS
from mds_norm.pipeline.build_local_termlists import load_termlist_index
from mds_norm.pipeline.build_parent_chains import broader_preferred, parent_chains
from mds_norm.pipeline.build_parent_chains import periodo_bounds as build_periodo_bounds
from mds_norm.pipeline.build_standard_vocabs import BUILDERS as STANDARD_BUILDERS
from mds_norm.pipeline.institutional_vocab_detect import BUILDERS as HOUSE_BUILDERS
from mds_norm.utils.atomise import norm_term

XL = "http://www.w3.org/2008/05/skos-xl#"
GVP = "http://vocab.getty.edu/ontology#"
SKOS_CORE = "http://www.w3.org/2004/02/skos/core#"
NT_LINE = r"^<([^>]*)> <([^>]*)> (.*) \.$"
KIND_PRIORITY = {"prefLabelGVP": 0, "prefLabel": 1, "altLabel": 2}

FIELD_VOCAB_MAP = {
    "spectrum/" + field: vocabs for field, vocabs in json.loads((VOCABS / "field_vocab_map.json").read_text()).items()
}
# Agent fields are linked by the persons stage, not cascaded
AGENT_FIELDS = [f for f, v in FIELD_VOCAB_MAP.items() if "ulan" in v]
GROUP_FOR = {f: "+".join(v) for f, v in FIELD_VOCAB_MAP.items() if f not in AGENT_FIELDS}
GROUP_VOCABS = {g: g.split("+") for g in set(GROUP_FOR.values())}
CASCADE_FIELDS = list(GROUP_FOR)
ALL_VOCABS = {v for vocabs in FIELD_VOCAB_MAP.values() for v in vocabs}
LOCAL_VOCABS = sorted(v for v in ALL_VOCABS if v.startswith("local_"))
STANDARD_VOCABS = sorted(v for v in ALL_VOCABS if v in STANDARD_BUILDERS)
PLACE_GROUPS = {g for g, v in GROUP_VOCABS.items() if "tgn" in v}
CONCEPT_GROUPS = set(GROUP_VOCABS) - PLACE_GROUPS


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cached(name: str, build: Callable[[], pl.DataFrame | pl.LazyFrame]) -> pl.LazyFrame:
    """Build once, cache as parquet, and only ever hand back a lazy scan"""
    path = VOCAB_INDEXES / f"{name}.parquet"
    if not path.exists():
        VOCAB_INDEXES.mkdir(parents=True, exist_ok=True)
        result = build()
        if isinstance(result, pl.LazyFrame):
            result.sink_parquet(path)
        else:
            result.write_parquet(path)
        log(f"{name}: built → {path}")
    return pl.scan_parquet(path)


def cached_sink(name: str, build: Callable[[Path], None]) -> pl.LazyFrame:
    """`cached` for a source too large to materialise: the builder writes the parquet itself"""
    path = VOCAB_INDEXES / f"{name}.parquet"
    if not path.exists():
        VOCAB_INDEXES.mkdir(parents=True, exist_ok=True)
        build(path)
        log(f"{name}: built → {path}")
    return pl.scan_parquet(path)


def scan_nt(path: Path) -> pl.LazyFrame:
    # One triple per line; \x00 never occurs
    return pl.scan_csv(path, separator="\x00", has_header=False, new_columns=["line"], quote_char=None).select(
        s=pl.col("line").str.extract(NT_LINE, 1),
        p=pl.col("line").str.extract(NT_LINE, 2),
        o=pl.col("line").str.extract(NT_LINE, 3),
    )


def unescape(col: pl.Expr) -> pl.Expr:
    return (
        pl.when(col.str.contains(r"\\"))
        .then(
            col.map_elements(
                lambda s: unicodedata.normalize("NFC", codecs.decode(s, "unicode_escape")), return_dtype=pl.String
            )
        )
        .otherwise(col)
    )


def getty_term_index(nt_path: Path) -> pl.LazyFrame:
    lf = scan_nt(nt_path)
    links = lf.filter(pl.col("p").is_in([XL + "prefLabel", XL + "altLabel", GVP + "prefLabelGVP"])).select(
        subject=pl.col("s").str.extract(r"/(\d+)$"),
        term_id=pl.col("o").str.extract(r"term/(\d+)"),
        kind=pl.col("p").str.extract(r"[#/](\w+)$"),
    )
    terms = lf.filter(pl.col("p") == GVP + "term").select(
        term_id=pl.col("s").str.extract(r"term/(\d+)"),
        term=unescape(pl.col("o").str.extract(r'^"(.*)"(?:@[\w-]+)?$', 1)),
        lang=pl.col("o").str.extract(r"@([\w-]+)$"),
    )
    return (
        links.join(terms, on="term_id")
        .select("subject", "term", "lang", "kind", norm=norm_term(pl.col("term")))
        .filter(pl.col("norm") != "")
        .unique()
    )


# GVP-preferred names US places, so prefer UK unless world-renowned
TGN_UK = "7008591"
TGN_EUROPE = "1000003"
TGN_MAJOR_MIN = 10  # narrower places before a candidate counts as a major place
TGN_STANDING_LANGS = 2  # label languages before a UK place beats a hamlet
TGN_RENOWN_LANGS = 5  # label languages before a place anywhere outranks a UK one
MAX_ANCESTOR_DEPTH = 25  # cycle guard; TGN's broaderPreferred tree is far shallower


def build_tgn_ancestors() -> pl.DataFrame:
    """Transitive broaderPreferred closure: (subject, ancestor) pairs"""
    up = broader_preferred(VOCABS / "tgn/TGNOut_HierarchicalRels.nt", "tgn")
    hop = up.rename({"parent": "ancestor"})
    frames = [hop]
    for _ in range(MAX_ANCESTOR_DEPTH):
        hop = hop.join(up, left_on="ancestor", right_on="subject").select("subject", ancestor="parent").unique()
        if not hop.height:
            break
        frames.append(hop)
    else:
        raise RuntimeError("broaderPreferred closure did not converge — cycle?")
    closure = pl.concat(frames).unique()
    log(f"tgn_ancestors: {closure.height:,} (subject, ancestor) pairs, depth ≤ {len(frames)}")
    return closure


def build_tgn_spatial() -> pl.LazyFrame:
    """Preference per TGN subject: 0 = UK with standing, 1 = renowned, 2 = other UK, 3 = major, 4 = Europe, 5 = rest"""
    reach = (
        tgn_ancestors()
        .group_by("subject")
        .agg(uk=(pl.col("ancestor") == TGN_UK).any(), eu=(pl.col("ancestor") == TGN_EUROPE).any())
    )
    languages = indexes()["tgn"].group_by("subject").agg(n_lang=pl.col("lang").n_unique())
    uk = pl.col("uk").fill_null(False) | (pl.col("subject") == TGN_UK)
    eu = pl.col("eu").fill_null(False) | (pl.col("subject") == TGN_EUROPE)
    major = pl.col("prominence").fill_null(0) >= TGN_MAJOR_MIN
    return (
        languages.join(reach, on="subject", how="left")
        .join(tgn_children(), on="subject", how="left")
        .select(
            "subject",
            preference=pl.when(uk & ((pl.col("n_lang") >= TGN_STANDING_LANGS) | major))
            .then(pl.lit(0, dtype=pl.Int8))
            .when(pl.col("n_lang") >= TGN_RENOWN_LANGS)
            .then(pl.lit(1, dtype=pl.Int8))
            .when(uk)
            .then(pl.lit(2, dtype=pl.Int8))
            .when(major)
            .then(pl.lit(3, dtype=pl.Int8))
            .when(eu)
            .then(pl.lit(4, dtype=pl.Int8))
            .otherwise(pl.lit(5, dtype=pl.Int8)),
            # prominence is the tie-break within a class
            pref_tiebreak=pl.lit(0, dtype=pl.Int32),
        )
    )


def build_tgn_children() -> pl.LazyFrame:
    """Direct narrower-place counts per TGN subject — the prominence prior"""
    # Getty also reifies relations; matching those subjects loses most broaders
    return (
        pl.scan_csv(
            VOCABS / "tgn/TGNOut_HierarchicalRels.nt",
            separator="\x00",
            has_header=False,
            new_columns=["line"],
            quote_char=None,
        )
        .select(subject=pl.col("line").str.extract(r"ontology#broaderPreferred> <http://vocab\.getty\.edu/tgn/(\d+)>"))
        .drop_nulls()
        .group_by("subject")
        .agg(prominence=pl.len().cast(pl.UInt32))
    )


# Families and firms sit under Corporate Bodies; Non-Artists are persons
ULAN_FACETS = {"500000002": "person", "500299802": "person", "500000003": "corporate"}
ULAN_FACET_HOPS = 8


def build_ulan_facets() -> pl.DataFrame:
    """Person or corporate body per ULAN subject, walked up broaderPreferred to a top-level facet"""
    up = broader_preferred(VOCABS / "ulan/ULANOut_HierarchicalRels.nt", "ulan")
    facets = pl.DataFrame({"parent": list(ULAN_FACETS), "facet": list(ULAN_FACETS.values())})
    chain, reached = up, []
    for _ in range(ULAN_FACET_HOPS):
        hit = chain.join(facets, on="parent", how="inner").select("subject", "facet")
        reached.append(hit)
        chain = (
            chain.join(hit.select("subject"), on="subject", how="anti")
            .join(up, left_on="parent", right_on="subject", how="inner")
            .select("subject", parent=pl.col("parent_right"))
        )
        if not chain.height:
            break
    return pl.concat(reached)


def build_ulan_agent_types() -> pl.LazyFrame:
    """The occupation AAT concepts ULAN assigns each agent, encoded in the reified relation URI"""
    return (
        pl.scan_csv(
            VOCABS / "ulan/ULANOut_AgentTypes.nt",
            separator="\x00",
            has_header=False,
            new_columns=["line"],
            quote_char=None,
        )
        .select(rel=pl.col("line").str.extract(r"ulan/rel/(\d+-agentType-\d+)"))
        .drop_nulls()
        .unique()
        .select(subject=pl.col("rel").str.extract(r"^(\d+)"), agent_type_aat=pl.col("rel").str.extract(r"(\d+)$"))
    )


def build_ulan_wikidata() -> pl.LazyFrame:
    """ULAN subject to Wikidata QID, the one identifier route out of ULAN the dump publishes"""
    return (
        scan_nt(VOCABS / "ulan/ULANOut_WikidataAlignment.nt")
        .filter(pl.col("p") == SKOS_CORE + "exactMatch")
        .select(subject=pl.col("s").str.extract(r"ulan/(\d+)$"), qid=pl.col("o").str.extract(r"(Q\d+)>?$"))
        .drop_nulls()
        .unique()
    )


def build_ulan_biographies() -> pl.DataFrame:
    """Birth and death years off the preferred biography, for tier-3 name verification"""
    lf = scan_nt(VOCABS / "ulan/ULANOut_Biographies.nt")
    # the biography hangs off the -agent URI, not the subject
    preferred = lf.filter(pl.col("p") == GVP + "biographyPreferred").select(
        subject=pl.col("s").str.extract(r"/(\d+)-agent$"), bio=pl.col("o").str.extract(r"bio/(\d+)")
    )
    years = (
        lf.filter(pl.col("p").is_in([GVP + "estStart", GVP + "estEnd"]))
        .select(
            bio=pl.col("s").str.extract(r"bio/(\d+)"),
            pred=pl.col("p").str.extract(r"#(\w+)$"),
            year=pl.col("o").str.extract(r'^"(-?\d+)"').cast(pl.Int32),
        )
        .collect(engine="streaming")
        .pivot("pred", index="bio", values="year", aggregate_function="first")
        .lazy()
    )
    return (
        preferred.join(years, on="bio")
        .select("subject", birth_year="estStart", death_year="estEnd")
        .collect(engine="streaming")
    )


SKOS = "{http://www.w3.org/2004/02/skos/core#}"
RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


def fish_concepts(dirname: str) -> list[ET.Element]:
    return list(ET.parse(next((VOCABS / dirname).glob("*.rdf"))).getroot().iter(SKOS + "Concept"))


def fish_subject(concept: ET.Element) -> str:
    return concept.get(RDF + "about").rstrip("/").rsplit("/", 1)[-1]


def build_fish(dirname: str) -> pl.DataFrame:
    rows = []
    for concept in fish_concepts(dirname):
        subject = fish_subject(concept)
        for kind in ("prefLabel", "altLabel"):
            rows.extend((subject, el.text.strip(), el.get(XML_LANG), kind) for el in concept.findall(SKOS + kind))
    return (
        pl.DataFrame(rows, schema=["subject", "term", "lang", "kind"], orient="row")
        .with_columns(norm=norm_term(pl.col("term")))
        .filter(pl.col("norm") != "")
        .unique()
    )


def build_periodo() -> pl.DataFrame:
    data = json.loads((VOCABS / "periodo/periodo-dataset.json").read_text())
    rows = []
    for authority in data["authorities"].values():
        for pid, period in authority.get("periods", {}).items():
            label = period.get("label")
            if label:
                rows.append((pid, label, (period.get("languageTag") or "").split("-")[0] or None, "prefLabel"))
            for tag, alts in (period.get("localizedLabels") or {}).items():
                rows.extend((pid, alt, tag.split("-")[0], "altLabel") for alt in alts if alt != label)
    return (
        pl.DataFrame(rows, schema=["subject", "term", "lang", "kind"], orient="row")
        .with_columns(norm=norm_term(pl.col("term")))
        .filter(pl.col("norm") != "")
        .unique()
    )


def build_periodo_spatial() -> pl.DataFrame:
    """Spatial preference per period: 0 = UK-covering, 1 = Europe-covering, 2 = other"""
    uk_labels = {
        "united kingdom",
        "great britain",
        "britain",
        "england",
        "wales",
        "scotland",
        "northern ireland",
        "ireland",
        "british isles",
        "isle of man",
        "channel islands",
    }
    data = json.loads((VOCABS / "periodo/periodo-dataset.json").read_text())
    rows = []
    for authority in data["authorities"].values():
        periods = authority.get("periods", {})
        for pid, period in periods.items():
            labels = {c.get("label", "").lower() for c in (period.get("spatialCoverage") or []) if isinstance(c, dict)}
            desc = (period.get("spatialCoverageDescription") or "").lower()
            uk = bool(labels & uk_labels) or any(t in desc for t in uk_labels)
            eu = "europe" in " ".join(labels) or "europe" in desc
            rows.append((pid, 0 if uk else (1 if eu else 2), -len(periods)))
    return pl.DataFrame(
        rows, schema={"subject": pl.String, "preference": pl.Int8, "pref_tiebreak": pl.Int32}, orient="row"
    )


# Too large and not valid XML, so filtered and streamed

RDFS = "{http://www.w3.org/2000/01/rdf-schema#}"
SCHEMA_ORG = "{http://schema.org/}"
MADS = "{http://www.loc.gov/mads/rdf/v1#}"
OWL = "{http://www.w3.org/2002/07/owl#}"
ABOUT, RESOURCE = RDF + "about", RDF + "resource"
ISNI_URI = re.compile(r"^https://isni\.org/isni/(\d{15}[\dX])$")
ISNI_EXPORTS = {"person": "ISNI_persons.rdf.gz", "organisation": "ISNI_organizations.rdf.gz"}
ISNI_DATES = {
    SCHEMA_ORG + "birthDate": "begin_date",
    SCHEMA_ORG + "foundingDate": "begin_date",
    SCHEMA_ORG + "deathDate": "end_date",
    SCHEMA_ORG + "dissolutionDate": "end_date",
}
ISNI_BATCH = 250_000
ISNI_SCHEMA = pa.schema(
    [
        ("isni", pa.string()),
        ("names", pa.list_(pa.string())),
        ("begin_date", pa.string()),
        ("end_date", pa.string()),
        ("same_as", pa.list_(pa.string())),
        ("authorities", pa.list_(pa.string())),
        ("replaced_by", pa.string()),
    ]
)


class StripSeparators:
    """Drop the 0x1E record separators the ISNI export puts between elements, which XML 1.0 forbids"""

    def __init__(self, raw: IO[bytes]) -> None:
        self.raw = raw
        self.prev = b""

    def read(self, size: int = -1) -> bytes:
        chunk = self.raw.read(size)
        if b"\x1e" not in chunk:
            self.prev = chunk[-1:] or self.prev
            return chunk
        # A separator between elements always follows a '>'
        for run in re.finditer(b"\x1e+", chunk):
            before = chunk[run.start() - 1 : run.start()] if run.start() else self.prev
            if before != b">":
                raise ValueError("0x1E inside a text node; stripping it would corrupt the value it sits in")
        self.prev = chunk.rstrip(b"\x1e")[-1:] or self.prev
        return chunk.replace(b"\x1e", b"")


def read_isni(desc: ET.Element, isni: str) -> dict:
    row: dict = {
        "isni": isni,
        "names": [],
        "begin_date": None,
        "end_date": None,
        "same_as": [],
        "authorities": [],
        "replaced_by": None,
    }
    for child in desc:
        # rdfs:label is the ISNI itself; names are alternates
        if child.tag == SCHEMA_ORG + "alternateName" and child.text:
            row["names"].append(child.text)
        elif child.tag in ISNI_DATES:
            row[ISNI_DATES[child.tag]] = child.text
        elif child.tag == OWL + "sameAs":
            row["same_as"].append(child.get(RESOURCE, ""))
        elif child.tag == MADS + "isIdentifiedByAuthority":
            row["authorities"].append(child.get(RESOURCE, ""))
        elif child.tag == RDFS + "seeAlso":
            row["replaced_by"] = child.get(RESOURCE, "").rsplit("/", 1)[-1]
    if not row["names"] and row["replaced_by"] is None:
        raise ValueError(f"ISNI {isni} carries neither a name nor a replacement; the export's shape has changed")
    return row


def sink_isni(source_path: Path, out: Path) -> None:
    """Stream one gzipped export to parquet, holding a single record's tree in memory at a time"""
    batch: list[dict] = []
    root = None
    with gzip.open(source_path, "rb") as raw, pq.ParquetWriter(out, ISNI_SCHEMA, compression="zstd") as writer:
        for event, elem in ET.iterparse(StripSeparators(raw), events=("start", "end")):
            if root is None:
                root = elem
                continue
            if event == "start" or elem.tag != RDF + "RDF":
                continue
            for desc in elem:
                match = ISNI_URI.match(desc.get(ABOUT, ""))
                # each block also describes its metadata document, without an ISNI
                if match is not None:
                    batch.append(read_isni(desc, match.group(1)))
            elem.clear()
            root.clear()
            if len(batch) >= ISNI_BATCH:
                writer.write_table(pa.Table.from_pylist(batch, schema=ISNI_SCHEMA))
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=ISNI_SCHEMA))


def build_isni(entity_type: str) -> pl.LazyFrame:
    """Live ISNI name forms in term-index shape; the export publishes no preferred form, so all are alternates"""
    return (
        isni_records(entity_type)
        .filter(pl.col("replaced_by").is_null())
        .select(subject="isni", term=pl.col("names"))
        .explode("term", empty_as_null=False)
        .select(
            "subject", "term", lang=pl.lit(None, pl.String), kind=pl.lit("altLabel"), norm=norm_term(pl.col("term"))
        )
        .filter(pl.col("norm") != "")
        .unique()
    )


def isni_records(entity_type: str) -> pl.LazyFrame:
    """One row per ISNI. Deprecated records carry only `replaced_by`, the ISNI that superseded them"""
    return cached_sink(f"isni_{entity_type}_records", partial(sink_isni, VOCABS / "isni" / ISNI_EXPORTS[entity_type]))


def isni_index(entity_type: str) -> pl.LazyFrame:
    return cached(f"isni_{entity_type}", partial(build_isni, entity_type))


# One line of context per subject for the rerank rung

GLOSS_MAX_CHARS = 220
GLOSS_MAX_ALTERNATES = 4


def build_fish_gloss(dirname: str) -> pl.DataFrame:
    """FISH scope notes, falling back to the broader concept's preferred label"""
    labels, rows = {}, []
    for concept in fish_concepts(dirname):
        subject = fish_subject(concept)
        pref = concept.find(SKOS + "prefLabel")
        if pref is not None and pref.text:
            labels[subject] = pref.text.strip()
        note = concept.find(SKOS + "scopeNote")
        broader = concept.find(SKOS + "broader")
        parent = broader.get(RDF + "resource", "").rstrip("/").rsplit("/", 1)[-1] if broader is not None else None
        rows.append((subject, (note.text or "").strip() or None if note is not None else None, parent))
    return pl.DataFrame(rows, schema=["subject", "note", "parent"], orient="row").select(
        "subject",
        gloss=pl.coalesce(
            pl.col("note"), pl.lit("narrower term of ") + pl.col("parent").replace_strict(labels, default=None)
        ),
    )


def build_monument_types_gloss() -> pl.DataFrame:
    """FISH Monument Types ships its scope notes in the term export itself"""
    return (
        pl.read_csv(VOCABS / "fish_monument_types" / "ThesaurusTerms.csv")
        .select(subject=pl.col("THE_TE_UID").cast(pl.String), gloss=pl.col("SCOPE_NOTE").str.strip_chars())
        .filter(pl.col("gloss").is_not_null() & (pl.col("gloss") != ""))
    )


def era(year: pl.Expr) -> pl.Expr:
    return pl.when(year < 0).then(pl.format("{} BC", year.abs())).otherwise(pl.format("AD {}", year))


def build_periodo_gloss() -> pl.DataFrame:
    """Extent and defining authority — what separates one authority's Iron Age from another's"""
    bounds = periodo_bounds().with_columns(
        span=pl.when(pl.col("start_year").is_not_null() & pl.col("stop_year").is_not_null()).then(
            pl.format("{} to {}", era(pl.col("start_year")), era(pl.col("stop_year")))
        )
    )
    return bounds.select(
        "subject",
        gloss=pl.concat_str(
            [pl.col("span"), pl.lit("defined by ") + pl.col("authority")], separator="; ", ignore_nulls=True
        ),
    ).filter(pl.col("gloss") != "")


def alternate_gloss(vocab: str) -> pl.LazyFrame:
    """Every vocabulary's fallback: the subject's own alternate labels"""
    return (
        english_only(indexes()[vocab])
        .filter(pl.col("kind") == "altLabel")
        .group_by("subject")
        .agg(alts=pl.col("term").unique().sort().head(GLOSS_MAX_ALTERNATES))
        .select("subject", gloss=pl.lit("also called ") + pl.col("alts").list.join(", "))
    )


def getty_parent_gloss(voc: str) -> pl.DataFrame:
    return parent_chains(voc, VOCABS / f"{voc}/{voc.upper()}Out_HierarchicalRels.nt", indexes()[voc])


GLOSS_BUILDERS: dict[str, Callable[[], pl.DataFrame]] = {
    "aat": lambda: getty_parent_gloss("aat").select("subject", gloss=pl.lit("narrower term of ") + pl.col("parents")),
    "periodo": build_periodo_gloss,
    "fish_building_materials": lambda: build_fish_gloss("fish_building_materials"),
    "fish_event_types": lambda: build_fish_gloss("fish_event_types"),
    "fish_monument_types": build_monument_types_gloss,
}


def gloss_index(vocab: str) -> pl.LazyFrame:
    """Subject to one line of context, from the richest source that vocabulary publishes"""
    fallback = alternate_gloss(vocab)
    if vocab not in GLOSS_BUILDERS:
        return fallback
    published = cached(f"{vocab}_gloss", GLOSS_BUILDERS[vocab]).filter(pl.col("gloss").is_not_null())
    return pl.concat([published, fallback.join(published, on="subject", how="anti")]).with_columns(
        pl.col("gloss").str.slice(0, GLOSS_MAX_CHARS)
    )


@cache
def indexes() -> dict[str, pl.LazyFrame]:
    """Every vocabulary named by `field_vocab_map.json`, keyed by vocabulary name"""
    built = {
        "aat": cached("aat", lambda: getty_term_index(VOCABS / "aat/AATOut_2Terms.nt")),
        "tgn": cached("tgn", lambda: getty_term_index(VOCABS / "tgn/TGNOut_2Terms.nt")),
        "ulan": cached("ulan", lambda: getty_term_index(VOCABS / "ulan/ULANOut_2Terms.nt")),
        "fish_building_materials": cached("fish_building_materials", lambda: build_fish("fish_building_materials")),
        "fish_event_types": cached("fish_event_types", lambda: build_fish("fish_event_types")),
        "periodo": cached("periodo", build_periodo),
    }
    # Registering a field in field_vocab_map is the only step needed
    built |= {v: cached(v, partial(load_termlist_index, v.removeprefix("local_"))) for v in LOCAL_VOCABS}
    return built | {v: cached(v, STANDARD_BUILDERS[v]) for v in STANDARD_VOCABS}


def house_index(vocab: str) -> pl.LazyFrame:
    """A published list an institution catalogues from, shaped like a group index"""
    return cached(vocab, HOUSE_BUILDERS[vocab]).with_columns(
        # these lists carry code/codeLabel surface kinds; rank them as alternates
        kind=pl.col("kind").replace({"code": "altLabel", "codeLabel": "altLabel"}),
        vocab=pl.lit(vocab),
        vocab_priority=pl.lit(0),
    )


def group_index(group: str) -> pl.LazyFrame:
    """The group's vocabularies stacked, tagged with their `field_vocab_map` priority"""
    return pl.concat(
        [
            indexes()[vocab].with_columns(vocab=pl.lit(vocab), vocab_priority=pl.lit(priority))
            for priority, vocab in enumerate(GROUP_VOCABS[group])
        ]
    )


def english_only(index: pl.LazyFrame) -> pl.LazyFrame:
    return index.filter(pl.col("lang").str.starts_with("en").fill_null(True))


def tgn_children() -> pl.LazyFrame:
    return cached("tgn_children", build_tgn_children)


def tgn_ancestors() -> pl.LazyFrame:
    return cached("tgn_ancestors", build_tgn_ancestors)


def tgn_spatial() -> pl.LazyFrame:
    return cached("tgn_spatial", build_tgn_spatial)


def periodo_spatial() -> pl.LazyFrame:
    return cached("periodo_spatial", build_periodo_spatial)


def periodo_bounds() -> pl.LazyFrame:
    return cached("periodo_bounds", build_periodo_bounds)


def ulan_facets() -> pl.LazyFrame:
    return cached("ulan_facets", build_ulan_facets)


def ulan_wikidata() -> pl.LazyFrame:
    return cached("ulan_wikidata", build_ulan_wikidata)


def isni_wikidata(entity_type: str) -> pl.LazyFrame:
    """Live ISNIs that name a Wikidata entity, the other half of the ULAN bridge"""
    return (
        isni_records(entity_type)
        .filter(pl.col("replaced_by").is_null())
        .select("isni", qid=pl.col("same_as"))
        .explode("qid", empty_as_null=False)
        .select("isni", qid=pl.col("qid").str.extract(r"(Q\d+)$"))
        .drop_nulls()
        .unique()
    )


def main() -> None:
    EMISSIONS_LOG.mkdir(parents=True, exist_ok=True)
    with EmissionsTracker(project_name="vocab_index_build", output_dir=str(EMISSIONS_LOG), log_level="error"):
        for name, lf in indexes().items():
            log(f"{name}: {lf.select(pl.len()).collect(engine='streaming').item():,} surface forms")
        log(f"tgn_children: {tgn_children().select(pl.len()).collect(engine='streaming').item():,} subjects")
        log(f"tgn_spatial: {tgn_spatial().select(pl.len()).collect(engine='streaming').item():,} subjects")
        log(f"periodo_spatial: {periodo_spatial().select(pl.len()).collect(engine='streaming').item():,} periods")
        agent_types = cached("ulan_agent_types", build_ulan_agent_types)
        biographies = cached("ulan_biographies", build_ulan_biographies)
        facets = ulan_facets().group_by("facet").agg(pl.len()).collect(engine="streaming")
        log(
            f"ulan: {agent_types.select(pl.len()).collect(engine='streaming').item():,} agent types / "
            f"{biographies.select(pl.len()).collect(engine='streaming').item():,} preferred biographies / "
            f"{dict(facets.iter_rows())} facets"
        )
        for entity_type in ISNI_EXPORTS:
            records = isni_records(entity_type).select(pl.len(), deprecated=pl.col("replaced_by").is_not_null().sum())
            records = records.collect(engine="streaming").row(0)
            forms = isni_index(entity_type).select(pl.len()).collect(engine="streaming").item()
            log(f"isni_{entity_type}: {records[0]:,} records ({records[1]:,} superseded) / {forms:,} surface forms")


if __name__ == "__main__":
    main()
