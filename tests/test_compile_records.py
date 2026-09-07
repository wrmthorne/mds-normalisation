import json

import polars as pl
import pytest

import mds_norm.pipeline.compile_records as c
from mds_norm.parsers.parse_dimensions import parse_dimensions


def dehtml(value):
    return pl.select(c._dehtml(pl.lit(value))).item() if value is not None else None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # tags stripped, whitespace collapsed
        ("<p>Oak chest</p>", "Oak chest"),
        ("A carved<br/>oak table", "A carved oak table"),
        ('<span style="x:1">gilt</span> frame', "gilt frame"),
        # entities unescaped
        ("Smith &amp; Sons", "Smith & Sons"),
        ("brass&nbsp;plate", "brass plate"),
        # escaped markup must decode before the strip runs
        ("a ballet &lt;i&gt;The Dryad&lt;/i&gt; postcard", "a ballet The Dryad postcard"),
        ("Smith &amp;lt;b&amp;gt;bold&amp;lt;/b&amp;gt;", "Smith bold"),
        # an escaped bracketed convention decodes to content
        ("&lt;fire-making and controlling equipment&gt;", "fire-making and controlling equipment"),
        ("temperature &lt; 5 degrees", "temperature < 5 degrees"),
        # values merely starting with '<' or '>' are not markup
        ("<1g", "<1g"),
        ("<1760", "<1760"),
        (">2cm", ">2cm"),
        # bracketed phrase text is content: unwrap, never delete
        ("<lutelike chordophones with long neck: plucked>", "lutelike chordophones with long neck: plucked"),
        ("<train protection equipment>", "train protection equipment"),
        ("<medical equipment>", "medical equipment"),
        # a value that is nothing but markup nulls
        ("<br/>", None),
        ("<br/><br/><br/>", None),
        ("<p></p>", None),
        ("<u></u>", None),
        ("&nbsp;", None),
        # tag-shaped whole values (attribute shape) are not phrase content
        ('<span style="x:1"></span>', None),
        # bracketed photographic qualifiers are protected recording practice
        ("print <b/w>", "print <b/w>"),
        ("print <sepia>", "print <sepia>"),
        ("oak <mahogany>", "oak <mahogany>"),
        # markup-free values pass through untouched (no space-collapse)
        ("plain  value", "plain  value"),
        (None, None),
    ],
)
def test_dehtml(raw, expected):
    assert dehtml(raw) == expected


@pytest.mark.parametrize(
    ("raw", "is_ph"),
    [
        # whole-value recording-practice placeholders carry no content
        ("-", True),
        ("?", True),
        ("x", True),
        ("X", True),
        ("n/a", True),
        (".", True),
        # empty dimension form-templates: labels and colons, no digits
        ("length: width: height: depth: diameter:", True),
        ("Height: Width: Length: Depth:", True),
        ("length / mm: width / mm: thickness / mm:", True),
        ("weight / g:", True),
        ("measures needed:", True),
        # header leak: 'Period' in a date slot
        ("Period", True),
        # knowledge-state markers are NOT placeholders — they persist (routed)
        ("unknown", False),
        ("Not recorded", False),
        # genuine content, including a filled template
        ("oak", False),
        ("Height: 5 cm", False),
        ("1885", False),
        (None, False),
    ],
)
def test_placeholder(raw, is_ph):
    assert pl.select(c._placeholder(pl.lit(raw, dtype=pl.String))).item() is is_ph


@pytest.mark.parametrize(
    ("field", "raw", "leaks"),
    [
        # a bare unit token is the unit column leaked
        ("spectrum/dimension_value", "mm", True),
        ("spectrum/dimension_value", "= mm", True),
        ("spectrum/dimension_value", "g", True),
        ("spectrum/dimension_value", "cms", True),
        ("spectrum/dimension_value", "23", False),
        ("spectrum/dimension_value", "23 mm", False),
        # a template or note in the magnitude slot
        ("spectrum/dimension_value", "ins x ins", True),
        ("spectrum/dimension_value", '" x "', True),
        ("spectrum/dimension_value", "*numeric value*", True),
        ("spectrum/dimension_value", "various heights", True),
        # a magnitude the parser cannot read is coverage, not absence
        ("spectrum/dimension_value", "25 cm (diameter)", False),
        # unit tokens belong in the unit slot
        ("spectrum/dimension_measurement_unit", "mm", False),
    ],
)
def test_unit_leak(field, raw, leaks):
    df = pl.DataFrame({"field_type": [field], "v": [raw]})
    assert df.select(c._unit_leak(pl.col("v"))).item() is leaks


@pytest.mark.parametrize(
    ("field", "raw", "expected"),
    [
        # an export joined an empty column, orphaning the separator
        ("spectrum/normal_location", ", cabinet 2 drawer 18", "cabinet 2 drawer 18"),
        ("spectrum/condition", "good :", "good"),
        ("spectrum/associated_place", "Perthshire :", "Perthshire"),
        ("spectrum/material", "wood; ", "wood"),
        ("spectrum/associated_concept", "| Fine Art", "Fine Art"),
        # BCE years and identifier structure must not be trimmed
        ("spectrum/date_earliest_single", "-0118", "-0118"),
        ("spectrum/other_number", "66/1017", "66/1017"),
        ("spectrum/acquisition_reference_number", "1893.01? 1934.02?", "1893.01? 1934.02?"),
        # a bilingual pair keeps its internal separator
        ("spectrum/associated_concept", "Celf Gain | Fine Art", "Celf Gain | Fine Art"),
        ("spectrum/material", "oak", "oak"),
        # manual-only and free-text fields are never edited
        ("spectrum/object_name_type", "·4>Numismatic standard reference:", "·4>Numismatic standard reference:"),
        ("spectrum/other_number_type", "Small finds number:", "Small finds number:"),
        ("spectrum/physical_description", "Obverse design:", "Obverse design:"),
    ],
)
def test_strip_dangling(field, raw, expected):
    df = pl.DataFrame({"field_type": [field], "v": [raw]})
    assert df.select(c._strip_dangling(pl.col("v"))).item() == expected


@pytest.mark.parametrize(
    ("field", "raw", "is_label"),
    [
        # a label with no number, in a number field
        ("spectrum/other_number", "Small finds number:", True),
        ("spectrum/other_number", ":", True),
        ("spectrum/other_number", "Publication number::", True),
        ("spectrum/dimension", "w(cm) secondary support:", True),
        # a number was recorded — not a bare label
        ("spectrum/other_number", "Small finds number: 42", False),
        ("spectrum/other_number", "TRURI 1830", False),
        # the same shape elsewhere is trimmed, not nulled
        ("spectrum/condition", "good:", False),
        ("spectrum/associated_place", "Perthshire :", False),
    ],
)
def test_label_only(field, raw, is_label):
    df = pl.DataFrame({"field_type": [field], "v": [raw]})
    assert df.select(c._label_only(pl.col("v"))).item() is is_label


@pytest.mark.parametrize(
    ("field", "label", "raw", "is_echo"),
    [
        # the export wrote the column header into the column
        ("spectrum/material", "Material", "material", True),
        ("spectrum/material", "Material", "Material", True),
        ("spectrum/dimension", "Dimension", "dimension", True),
        ("spectrum/brief_description", "Brief Description", "Brief Description", True),
        ("spectrum/persons_association", "Person's Association", "persons association", True),
        # a value containing the field name is content
        ("spectrum/material", "Material", "material culture fragment", False),
        ("spectrum/technique", "Technique", "engraving", False),
    ],
)
def test_field_name_echo(field, label, raw, is_echo):
    df = pl.DataFrame({"field_type": [field], "label": [label], "v": [raw]})
    assert df.select(c._field_name_echo(pl.col("v"))).item() is is_echo


@pytest.mark.parametrize(
    ("field", "label", "raw", "kept"),
    [
        # generated person parts skip build_base, so screen them here
        ("spectrum/persons_forenames", "Given", "No", None),
        ("spectrum/persons_forenames", "Given", "X", None),
        ("spectrum/persons_forenames", "Given", "Given", None),
        ("spectrum/persons_additions_to_name", "Additions", "see notes", None),
        ("spectrum/dimension_measured_part", "Measured Part", "various", None),
        # a label leak's dangling colon exposes the placeholder
        ("spectrum/persons_forenames", "Given", "artist:", "artist"),
        ("spectrum/persons_surname", "Surname", "Mousley,", "Mousley"),
        # real decomposition survives untouched
        ("spectrum/persons_surname", "Surname", "Noble", "Noble"),
        ("spectrum/persons_forenames", "Given", "Xavier", "Xavier"),
        ("spectrum/dimension_measurement_unit", "Unit", "cm", "cm"),
        # the release channel records the cataloguer's mark verbatim
        ("wrmthorne/notation", "Notation", "?", "?"),
    ],
)
def test_screen_generated(field, label, raw, kept):
    part = pl.LazyFrame({"field_type": [field], "label": [label], "value": [raw]})
    out = c.screen_generated(part).collect()
    assert (out["value"].to_list() or [None])[0] == kept


def test_filtered_annotations_refuses_placeholder_terms(tmp_path):
    # placeholder terms drop the informative half of a value
    ann = pl.DataFrame(
        {
            "field_type": [
                "spectrum/associated_concept",
                "spectrum/material",
                "spectrum/material",
                "spectrum/technique",
            ],
            "value": ["miscellaneous (other industries)", "brass other metal", "oak", "engraving"],
            "matched_term": ["miscellaneous", "other", "oak", "engraving"],
            "data_source": ["Amgueddfa", "V&A", "V&A", "V&A"],
            "span_start": [0, 6, 0, 0],
            "span_end": [31, 11, 3, 9],
        }
    )
    path = tmp_path / "ann.parquet"
    ann.write_parquet(path)
    kept = c._filtered_annotations(path).collect()
    assert kept["matched_term"].to_list() == ["oak", "engraving"]


def ms(raw):
    parsed = parse_dimensions(raw)
    assert parsed is not None, raw
    return parsed["measurements"]


@pytest.mark.parametrize(
    ("expr", "raw", "expected"),
    [
        # one stated type over a range pair types both bounds
        ("Diameter", "8.7 - 8.75", ["diameter", "diameter"]),
        ("diameter", "23-23.5", ["diameter", "diameter"]),
        # genuine multi-axis correspondence is unchanged
        ("height x width", "68 x 51 cm", ["height", "width"]),
        ("height x width x depth", "12.9 x 26 x 0.2cm", ["height", "width", "depth"]),
        # under-specified expression never types more measurements than it names
        ("height x width", "12.9 x 26 x 0.2cm", None),
        # one type over two distinct axes is not a range
        ("Diameter", "23 x 24", None),
        # an unreadable type expression defers
        ("mystery", "68 x 51 cm", None),
        # CRLF-aligned parallel lists zip line i with line i
        ("Length\r\nBreadth\r\nWeight", "21.5\r\n18\r\n1.92", ["length", "breadth", "weight"]),
        ("Height\nWidth", "10\n20", ["height", "width"]),
        # unequal line counts still defer
        ("Length\r\nBreadth", "21.5\r\n18\r\n1.92", None),
    ],
)
def test_stated_types(expr, raw, expected):
    stated = c._stated_types(expr, ms(raw))
    assert (None if stated is None else [t for t, _, _ in stated]) == expected


@pytest.mark.parametrize(
    ("expr", "raw", "expected"),
    [
        # a parenthetical qualifies the type it follows
        ("diameter (max)", "23 cm", [("diameter", "maximum", None)]),
        ("height (min)", "7 cm", [("height", "minimum", None)]),
        ("width (reel)", "25mm", [("width", None, "reel")]),
        ("height (overall)", "12 cm", [("height", None, "overall")]),
        # held out of the split, so nothing cuts the segment
        ("length (extended)", "45cm", [("length", None, "extended")]),
        ("height (max) x width (max)", "10 x 20 cm", [("height", "maximum", None), ("width", "maximum", None)]),
        # the parenthetical never rescues an expression naming no type
        ("size (max)", "185mm", None),
    ],
)
def test_stated_types_parenthetical(expr, raw, expected):
    assert c._stated_types(expr, ms(raw)) == expected


@pytest.mark.parametrize(
    ("raw", "types", "expected"),
    [
        # a restated reading inherits both type and part
        (
            "Dimensions: Barrel length: 39 in (991 mm)",
            ["length", None],
            [("length", "dimensions barrel"), ("length", "dimensions barrel")],
        ),
        # equal sides are ambiguous: left untyped, group defers
        (
            "height 100 mm x width 100 mm x 3.94 in",
            ["height", "width", None],
            [("height", None), ("width", None), (None, None)],
        ),
        # the same unit twice is a second measurement
        ("length 40 mm x 40 mm", ["length", None], [("length", None), (None, None)]),
    ],
)
def test_inherit_restated(raw, types, expected):
    measurements = ms(raw)
    parts = [m["dimension_measured_part"] for m in measurements]
    c._inherit_restated(measurements, types, parts)
    assert list(zip(types, parts, strict=True)) == expected


def test_demote_uncertain_destinations():
    # uncertain destinations compile to flagged, never applied
    rp = pl.LazyFrame(
        {
            "op": ["add", "add", "move", "move", "add"],
            "source_field": [
                "spectrum/description",
                "spectrum/description",
                "spectrum/technical_attribute",
                "spectrum/acquisition_note",
                "spectrum/description",
            ],
            "field": [
                "spectrum/object_production_date",
                "spectrum/material",
                "spectrum/dimension",
                "spectrum/acquisition_date",
                "spectrum/object_purchase_price",
            ],
            "status": ["resolved", "resolved", "resolved", "resolved", "flagged"],
        }
    )
    out = c.demote_uncertain_destinations(rp).collect()
    assert out["status"].to_list() == ["flagged", "resolved", "flagged", "resolved", "flagged"]
    # only the demotions carry the marker
    assert out["_uncertain_dest"].to_list() == [True, False, True, False, False]


def _base(rows):
    """Minimal tier-0 base table for compile-stage tests"""
    schema = {
        "record_id": pl.String,
        "data_source": pl.String,
        "node_id": pl.Binary,
        "parent_id": pl.Binary,
        "depth": pl.UInt8,
        "label": pl.String,
        "field_type": pl.String,
        "base_value": pl.String,
    }
    return pl.LazyFrame(rows, schema=schema, orient="row")


def test_compile_associated_dates():
    # retype only on an unambiguous production association
    base = _base(
        [
            ("r1", "M", b"a1", None, 0, "Associated Date", "spectrum/associated_date", "1889-1939"),
            ("r1", "M", b"c1", b"a1", 1, "Date Association", "spectrum/date_association", "Creation"),
            ("r2", "M", b"a2", None, 0, "Associated Date", "spectrum/associated_date", "1978"),
            ("r2", "M", b"c2", b"a2", 1, "Date Association", "spectrum/date_association", "determination date"),
            ("r3", "M", b"a3", None, 0, "Associated Date", "spectrum/associated_date", "1900"),
            ("r3", "M", b"c3", b"a3", 1, "Date Association", "spectrum/date_association", "creation"),
            ("r3", "M", b"c4", b"a3", 1, "Date Association", "spectrum/date_association", "date collected"),
            # a production association under a non-associated_date parent is not touched
            ("r4", "M", b"p1", None, 0, "Object Production Date", "spectrum/object_production_date", "1900"),
            ("r4", "M", b"c5", b"p1", 1, "Date Association", "spectrum/date_association", "made"),
        ]
    )
    out = c.compile_associated_dates(base, {})
    assert out["node_id"].to_list() == [b"a1"]
    assert out["retype_field"].to_list() == ["spectrum/object_production_date"]


def test_compile_measured_part_qualifiers():
    # a qualifier in the measured-part slot retypes and canonicalises
    base = _base(
        [
            ("r1", "M", b"m1", b"g1", 1, "Dimension Measured Part", "spectrum/dimension_measured_part", "approx."),
            ("r2", "M", b"m2", b"g2", 1, "Dimension Measured Part", "spectrum/dimension_measured_part", "approx"),
            ("r2", "M", b"q1", b"g2", 1, "Dimension Value Qualifier", "spectrum/dimension_value_qualifier", "approx"),
            ("r3", "M", b"m3", b"g3", 1, "Dimension Measured Part", "spectrum/dimension_measured_part", "mount"),
        ]
    )
    retypes, props = c.compile_measured_part_qualifiers(base, {})
    assert retypes["node_id"].to_list() == [b"m1"]
    assert retypes["retype_field"].to_list() == ["spectrum/dimension_value_qualifier"]
    assert props["node_id"].to_list() == [b"m1"]
    assert props["new_value"].to_list() == ["approx"]


def test_compile_admin_concepts():
    base = _base(
        [
            (
                "r1",
                "M",
                b"c1",
                None,
                0,
                "Associated Concept",
                "spectrum/associated_concept",
                "record verified by E.A. Walker",
            ),
            ("r2", "M", b"c2", None, 0, "Associated Concept", "spectrum/associated_concept", "not verified"),
            ("r3", "M", b"c3", None, 0, "Associated Concept", "spectrum/associated_concept", "Verified"),
            # a genuine concept mentioning the word is not admin text
            ("r4", "M", b"c4", None, 0, "Associated Concept", "spectrum/associated_concept", "verified gold assay"),
            # the shape in another field stays put
            ("r5", "M", b"c5", None, 0, "Comments", "spectrum/comments", "not verified"),
        ]
    )
    out = c.compile_admin_concepts(base, {})
    assert sorted(out["node_id"].to_list()) == [b"c1", b"c2", b"c3"]
    assert set(out["retype_field"].to_list()) == {"spectrum/comments"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1934.02?", ("1934.02", 1)),
        ("1893.01? 1934.02?", (None, 2)),
        ("1890.70?, 1891.33?", (None, 2)),
        ("1875.01? 1887.16? 1914.10?", (None, 3)),
        ("1934.02", None),  # unmarked
        ("?", None),  # a disguised null, nulled at tier 0
        ("shic:?", None),  # a label, not an identifier
        ("? in accession register", None),  # a note
        ("1927.12?@", None),  # trailing junk — not read as marked
        ("1893.01? 1934.02? cf. undated register", None),  # mixed with prose
    ],
)
def test_read_marked_identifier(raw, expected):
    assert c.read_marked_identifier(raw) == expected


def test_compile_identifier_certainty():
    # a marked reference number is cleaned and qualified
    base = _base(
        [
            (
                "r1",
                "M",
                b"i1",
                None,
                0,
                "Acquisition Reference Number",
                "spectrum/acquisition_reference_number",
                "1934.02?",
            ),
            (
                "r2",
                "M",
                b"i2",
                None,
                0,
                "Acquisition Reference Number",
                "spectrum/acquisition_reference_number",
                "1893.01? 1934.02?",
            ),
            (
                "r3",
                "M",
                b"i3",
                None,
                0,
                "Acquisition Reference Number",
                "spectrum/acquisition_reference_number",
                "1934.02",
            ),
            # the same notation in a non-identifier field belongs to compile_notation
            ("r4", "M", b"i4", None, 0, "Material", "spectrum/material", "oak?"),
        ]
    )
    props, cert = c.compile_identifier_certainty(base, {})
    assert props["node_id"].to_list() == [b"i1"]
    assert props["new_value"].to_list() == ["1934.02"]
    assert dict(zip(cert["target_node_id"], cert["kind"], strict=True)) == {
        b"i1": "cataloguer_marked",
        b"i2": "ambiguous_identifier",
    }
    # the alternatives carry the count, and no rewrite
    assert "2 alternative identifiers" in cert.filter(pl.col("target_node_id") == b"i2")["detail"].item()


def test_edtf_patch_dates():
    # an add into a date field re-enters the parser
    add_schema = {
        "record_id": pl.String,
        "data_source": pl.String,
        "node_id": pl.Binary,
        "parent_id": pl.Binary,
        "depth": pl.UInt8,
        "label": pl.String,
        "path": pl.String,
        "field_type": pl.String,
        "value": pl.String,
        "as_recorded": pl.String,
        "component": pl.String,
    }
    adds = pl.DataFrame(
        [
            (
                "r1",
                "M",
                b"a1",
                None,
                0,
                "Field Collection Date",
                None,
                "spectrum/field_collection_date",
                "April 1993",
                None,
                "record_fixes",
            ),
            (
                "r2",
                "M",
                b"a2",
                None,
                0,
                "Acquisition Date",
                None,
                "spectrum/acquisition_date",
                "unknown date",
                None,
                "record_fixes",
            ),
            (
                "r3",
                "M",
                b"a3",
                None,
                0,
                "Acquisition Date",
                None,
                "spectrum/acquisition_date",
                "gifted by the artist",
                None,
                "record_fixes",
            ),
            ("r4", "M", b"a4", None, 0, "Title", None, "spectrum/title", "April 1993", None, "record_fixes"),
        ],
        schema=add_schema,
        orient="row",
    )
    out = c.edtf_patch_dates(adds, {}).sort("node_id")
    assert out["value"].to_list() == ["1993-04", "unknown date", "gifted by the artist", "April 1993"]
    assert out["as_recorded"].to_list() == ["April 1993", None, None, None]


def test_date_period_nodes():
    parsed = pl.LazyFrame(
        {
            "record_id": ["r1", "r2", "r3", "r4"],
            "data_source": ["M"] * 4,
            "node_id": [b"n1", b"n2", b"n3", b"n4"],
            "depth": pl.Series([0, 0, 0, 1], dtype=pl.UInt8),
            "field_type": ["spectrum/object_production_date"] * 3 + ["spectrum/date_earliest_single"],
            "period": ["Victorian", "Victorian", None, "Roman"],
        }
    )
    base = _base(
        [
            # n2 already has a date_period child
            ("r2", "M", b"k1", b"n2", 1, "Date Period", "spectrum/date_period", "Victorian")
        ]
    )
    out = c.date_period_nodes(parsed, base)
    # n1 gains a child; n2 deduped; n3, n4 skipped
    assert out["parent_id"].to_list() == [b"n1"]
    assert out["field_type"].to_list() == ["spectrum/date_period"]
    assert out["value"].to_list() == ["Victorian"]
    assert out["depth"].to_list() == [1]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # the motivating Ashmolean shapes
        ("c. 1760 - after 1820", ("1760~", "1820/..")),
        ("1823 - 1891", ("1823", "1891")),
        ("c. 1537-1546", None),  # 9-year gap: reign/activity, not a lifespan
        ("active 1793 - 1798", None),  # activity range
        ("1960 - 1967", None),  # short gap
        ("-336 - -323", None),  # unsigned-BCE sides do not parse
        ("286 (or 287?) - 293", None),  # side does not parse cleanly
        ("1760", None),  # no dash
        ("1650 - 1900", None),  # 250-year gap is not one life
    ],
)
def test_parse_lifespan(raw, expected):
    assert c.parse_lifespan(raw) == expected


def test_compile_person_dates():
    base = _base(
        [
            ("r1", "M", b"b1", b"p1", 1, "Birth Date", "spectrum/persons_birth_date", "c. 1760 - after 1820"),
            # parent already records a death date: never split
            ("r2", "M", b"b2", b"p2", 1, "Birth Date", "spectrum/persons_birth_date", "1823 - 1891"),
            ("r2", "M", b"d2", b"p2", 1, "Death Date", "spectrum/persons_death_date", "1891"),
            ("r3", "M", b"b3", b"p3", 1, "Birth Date", "spectrum/persons_birth_date", "1745"),
            # a death date in the birth slot is retyped
            ("r4", "M", b"b4", b"p4", 1, "Birth Date", "spectrum/persons_birth_date", "died 1831"),
            # parent already records a death date: never retyped
            ("r5", "M", b"b5", b"p5", 1, "Birth Date", "spectrum/persons_birth_date", "died 1900"),
            ("r5", "M", b"d5", b"p5", 1, "Death Date", "spectrum/persons_death_date", "1900"),
        ]
    )
    props, deaths, retypes = c.compile_person_dates(base, {})
    assert props["node_id"].to_list() == [b"b1", b"b4"]
    assert props["new_value"].to_list() == ["1760~", "1831"]
    assert deaths["parent_id"].to_list() == [b"p1"]
    assert deaths["field_type"].to_list() == ["spectrum/persons_death_date"]
    assert deaths["value"].to_list() == ["1820/.."]
    assert deaths["as_recorded"].to_list() == ["c. 1760 - after 1820"]
    assert retypes["node_id"].to_list() == [b"b4"]
    assert retypes["retype_field"].to_list() == ["spectrum/persons_death_date"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("died 1593", "1593"),
        ("died c. 1632", "1632~"),
        ("died AD 958", "0958"),
        ("died after 1736", "1736/.."),  # an open bound is one life event
        ("d. 1782", "1782"),
        ("died 1.8.1920", "1920-08-01"),
        ("1831", None),  # no death verb
        ("born 1831", None),  # a start verb the parser already strips
        ("died", None),  # no date behind the verb
    ],
)
def test_parse_death_only(raw, expected):
    assert c.parse_death_only(raw) == expected


def test_compile_dates_semantic_markers_and_verified(tmp_path):
    # knowledge-state markers defer; canonical values are verified
    base = _base(
        [
            ("r1", "M", b"n1", None, 0, "Production Date", "spectrum/object_production_date", "n/k"),
            ("r2", "M", b"n2", None, 0, "Production Date", "spectrum/object_production_date", "Unknown date"),
            ("r3", "M", b"n3", None, 0, "Production Date", "spectrum/object_production_date", "n.d."),
            ("r4", "M", b"n4", None, 0, "Production Date", "spectrum/object_production_date", "1885"),
            ("r5", "M", b"n5", None, 0, "Production Date", "spectrum/object_production_date", "c. 1850"),
        ]
    )
    report = {}
    props, _cert, _periods, semantic, verified, _bounds, edtf = c.compile_dates(
        base, tmp_path / "cache.parquet", report
    )
    assert set(semantic["node_id"]) == {b"n1", b"n2", b"n3"}
    assert set(verified["node_id"]) == {b"n4"}
    # no Spectrum slot holds 1850~; the EDTF child carries it
    assert props.is_empty()
    assert dict(zip(edtf.collect()["parent_id"], edtf.collect()["value"], strict=True)) == {
        b"n4": "1885",
        b"n5": "1850~",
    }
    assert report["dates"]["semantic_markers"] == 3
    assert report["dates"]["verified"] == 1
    assert report["dates"]["edtf_nodes"] == 2


def test_date_substructure(tmp_path):
    # a date group decomposes into bound children and native slots
    base = _base(
        [
            # a range → earliest + latest children
            ("r1", "M", b"g1", None, 0, "Associated Date", "spectrum/associated_date", "1850-1900"),
            # a circa point never populates date_latest
            ("r2", "M", b"g2", None, 0, "Production Date", "spectrum/object_production_date", "c. 1885"),
            # an existing bound child blocks decomposition entirely
            ("r3", "M", b"g3", None, 0, "Production Date", "spectrum/object_production_date", "1850-1900"),
            ("r3", "M", b"e3", b"g3", 1, "Date Earliest Single", "spectrum/date_earliest_single", "1850"),
            # a marked bound slot gains a native certainty child
            ("r4", "M", b"e4", b"g4", 1, "Date Earliest Single", "spectrum/date_earliest_single", "c. 1760"),
            # a short BC year pads to the ISO form
            ("r5", "M", b"e5", b"g5", 1, "Date Latest", "spectrum/date_latest", "-118"),
            # an open bound → qualifier lands in the native slot
            ("r6", "M", b"g6", None, 0, "Production Date", "spectrum/object_production_date", "after 1820"),
        ]
    )
    props, _cert, _periods, _semantic, _verified, bounds, edtf = c.compile_dates(base, tmp_path / "cache.parquet", {})
    b = bounds.collect()
    by_parent = {p: sub for (p,), sub in b.partition_by("parent_id", as_dict=True).items()}

    g1 = by_parent[b"g1"]
    assert dict(zip(g1["field_type"], g1["value"], strict=True)) == {
        "spectrum/date_earliest_single": "1850",
        "spectrum/date_latest": "1900",
    }
    assert g1["depth"].to_list() == [1, 1]

    g2 = by_parent[b"g2"]
    assert dict(zip(g2["field_type"], g2["value"], strict=True)) == {"spectrum/date_earliest_single": "1885"}
    grand = by_parent[g2["node_id"][0]]
    assert dict(zip(grand["field_type"], grand["value"], strict=True)) == {
        "spectrum/date_earliest_single_certainty": "circa"
    }
    assert grand["depth"].to_list() == [2]

    assert b"g3" not in by_parent
    assert b"e3" not in by_parent

    e4 = by_parent[b"e4"]
    assert dict(zip(e4["field_type"], e4["value"], strict=True)) == {
        "spectrum/date_earliest_single_certainty": "circa"
    }
    prop = dict(zip(props["node_id"], props["new_value"], strict=True))
    assert prop[b"e4"] == "1760"
    assert prop[b"e5"] == "-0118"

    g6 = by_parent[b"g6"]
    assert dict(zip(g6["field_type"], g6["value"], strict=True)) == {"spectrum/date_earliest_single": "1820"}
    grand6 = by_parent[g6["node_id"][0]]
    assert dict(zip(grand6["field_type"], grand6["value"], strict=True)) == {
        "spectrum/date_earliest_single_qualifier": "after"
    }

    # group nodes carry the expression; bound slots get none
    e = edtf.collect()
    assert set(e["field_type"]) == {"wrmthorne/date_edtf"}
    assert dict(zip(e["parent_id"], e["value"], strict=True)) == {
        b"g1": "1850/1900",
        b"g2": "1885~",
        b"g3": "1850/1900",
        b"g6": "1820/..",
    }
    assert b"g1" not in set(props["node_id"])


def test_compile_persons_native_fields_only(tmp_path, monkeypatch):
    # person decomposition uses Spectrum's own sub-fields only
    parts = ("prefix", "given", "middle", "nickname", "surname", "suffix")
    ann = pl.DataFrame(
        {
            "record_id": ["r1", "r2"],
            "data_source": ["M", "M"],
            "node_id": [b"p1", b"o1"],
            "status": ["resolved", "resolved"],
            "entity_type": ["person", "organisation"],
            "value": ["Smith, John", "Acme Ltd"],
            "display": ["John Smith", "Acme Ltd"],
            **{p: [None, None] for p in parts},
        },
        schema_overrides=dict.fromkeys(parts, pl.String),
    ).with_columns(given=pl.Series(["John", None]), surname=pl.Series(["Smith", None]))
    path = tmp_path / "person_annotations.parquet"
    ann.write_parquet(path)
    monkeypatch.setattr(c, "PERSON_ANN", path)
    base = _base(
        [
            ("r1", "M", b"p1", None, 0, "Maker", "spectrum/maker", "Smith, John"),
            ("r2", "M", b"o1", None, 0, "Maker", "spectrum/maker", "Acme Ltd"),
        ]
    )
    children, _stale, _touched = c.compile_persons(base, {})
    ch = children.collect()
    assert "spectrum/value" not in ch["field_type"].to_list()
    person = ch.filter(pl.col("parent_id") == b"p1")
    assert dict(zip(person["field_type"], person["value"], strict=True)) == {
        "spectrum/persons_forenames": "John",
        "spectrum/persons_surname": "Smith",
    }
    org = ch.filter(pl.col("parent_id") == b"o1")
    assert dict(zip(org["field_type"], org["value"], strict=True)) == {"spectrum/organisations_main_body": "Acme Ltd"}


def test_authority_nodes(tmp_path):
    # a resolved alignment publishes one authority reference group
    ann = pl.DataFrame(
        {
            "record_id": ["r1", "r1", "r2", "r3", "r4"],
            "data_source": ["M"] * 5,
            "node_id": [b"n1", b"n1", b"n2", b"n3", b"n4"],
            "field_type": [
                "spectrum/material",
                "spectrum/material",
                "spectrum/persons_association",
                "spectrum/dimension_measured_part",
                "spectrum/material",
            ],
            "value": ["oak", "oak", "determiner", "rim", "elm"],
            "matched_term": ["oak", "oak", "determiner", "rim", "elm"],
            "span_start": [0, 0, 0, 0, 0],
            "span_end": [3, 3, 10, 3, 3],
            "status": ["resolved", "resolved", "resolved", "resolved", "flagged"],
            "vocab": ["aat", "aat", "local_persons_association", None, "aat"],
            "subject": ["300012264", "300012264", "persons_association/determiner", None, "300012263"],
        }
    )
    path = tmp_path / "vocab_annotations.parquet"
    ann.write_parquet(path)
    base = _base(
        [
            ("r1", "M", b"n1", None, 0, "Material", "spectrum/material", "oak"),
            ("r2", "M", b"n2", None, 0, "Association", "spectrum/persons_association", "determiner"),
            ("r3", "M", b"n3", None, 0, "Part", "spectrum/dimension_measured_part", "rim"),
            ("r4", "M", b"n4", None, 0, "Material", "spectrum/material", "elm"),
        ]
    )
    report = {}
    out = c.authority_nodes(base, [path], report).collect()
    groups = out.filter(pl.col("field_type") == c.AUTH_FIELD)
    # duplicate atoms dedupe; flagged atoms publish nothing
    assert set(groups["parent_id"]) == {b"n1", b"n2"}
    assert report["authorities"]["references"] == 2

    gid = dict(zip(groups["parent_id"], groups["node_id"], strict=True))
    n1 = out.filter(pl.col("parent_id") == gid[b"n1"])
    assert dict(zip(n1["field_type"], n1["value"], strict=True)) == {
        "wrmthorne/authority_source": "aat",
        "wrmthorne/authority_id": "300012264",
        "wrmthorne/authority_uri": "http://vocab.getty.edu/aat/300012264",
    }
    n2 = out.filter(pl.col("parent_id") == gid[b"n2"])
    assert dict(zip(n2["field_type"], n2["value"], strict=True)) == {
        "wrmthorne/authority_source": "local_persons_association",
        "wrmthorne/authority_id": "persons_association/determiner",
    }


def test_compile_encoding_damage():
    # U+FFFD-damaged values are annotated, never rewritten
    base = _base(
        [
            ("r1", "M", b"n1", None, 0, "Maker", "spectrum/persons_surname", "Hamonic, No�l"),
            ("r2", "M", b"n2", None, 0, "Place", "spectrum/field_collection_place", "N�remberg"),
            ("r3", "M", b"n3", None, 0, "Maker", "spectrum/persons_surname", "Smith"),
        ]
    ).with_columns(value=pl.col("base_value"))
    rows = c.compile_encoding_damage(base, {})
    assert set(rows["target_node_id"]) == {b"n1", b"n2"}
    assert rows["kind"].unique().to_list() == ["encoding_damage"]


def test_compile_encoding_damage_repaired():
    # a tier-0-repaired value is annotated encoding_repair
    base = _base([("r1", "M", b"n1", None, 0, "Place", "spectrum/field_collection_place", "Nüremberg")]).with_columns(
        value=pl.lit("N�remberg")
    )
    rows = c.compile_encoding_damage(base, {})
    assert rows["target_node_id"].to_list() == [b"n1"]
    assert rows["kind"].to_list() == ["encoding_repair"]


def test_free_text_class_is_flat_strings():
    # all_free_text_fields() is keyed by model, so flatten to names
    assert all(isinstance(f, str) for f in c.FREE_TEXT), (
        "FREE_TEXT leaked a model class — flatten all_free_text_fields().values()"
    )
    assert any("inscription" in f for f in c.FREE_TEXT)
    excluded = c.DATE_FIELDS + c.FREE_TEXT + c.PROTECTED + c.IDENTIFIERS + c.MEASUREMENT + c.MONETARY
    assert all(isinstance(f, str) for f in excluded)


def _cached(values: list[str]) -> pl.DataFrame:
    """The dimension parse cache for a handful of values, as compile_dimensions builds it"""
    rows = []
    for v in values:
        r = parse_dimensions(v)
        rows.append((v, "ft", r["status"], json.dumps(r["measurements"]), json.dumps(r["residue"])))
    return pl.DataFrame(
        rows,
        orient="row",
        schema={
            "value": pl.String,
            "prime": pl.String,
            "status": pl.String,
            "payload": pl.String,
            "residue": pl.String,
        },
    )


def test_measurement_rows_publishes_an_untyped_single_measurement():
    # an untyped whole measurement publishes rather than defers
    values = ["18 mm", "25 cm (diameter)", "18"]
    pairs = pl.DataFrame({"value": values, "ptype": ["", "", ""], "prime": ["ft"] * 3})
    rows, deferred = c._measurement_rows(_cached(values), pairs)

    published = dict(
        zip(rows["value"], zip(rows["dimension_value"], rows["dimension_measurement_unit"], strict=True), strict=True)
    )
    assert published["18 mm"] == ("18", "mm")
    assert published["25 cm (diameter)"] == ("25", "cm")
    # an unknown type is left unset, never guessed
    assert rows.filter(pl.col("value") == "18 mm")["dimension_type"].to_list() == [None]

    # a bare magnitude settles neither type nor unit: defer
    assert set(deferred["value"]) == {"18"}
    assert set(deferred["reason"]) == {"untyped"}


def test_compile_dates_completes_two_digit_years(tmp_path, monkeypatch):
    # a two-digit year takes its century from the accession year
    years = tmp_path / "accession_years.parquet"
    pl.DataFrame(
        {
            "record_id": ["r1", "r2", "r2"],
            "data_source": ["M", "M", "M"],
            "accession_year": [1995, 2005, 2011],
            "year_source": ["object_number"] * 3,
        }
    ).write_parquet(years)
    monkeypatch.setattr(c, "ACCESSION_YEARS", years)

    base = _base(
        [
            ("r1", "M", b"n1", None, 0, "Production Date", "spectrum/object_production_date", "31/12/79"),
            ("r2", "M", b"n2", None, 0, "Production Date", "spectrum/object_production_date", "1/1/98"),
            # no accession year on record: the century stays unsettled
            ("r3", "M", b"n3", None, 0, "Production Date", "spectrum/object_production_date", "28.9.83"),
        ]
    )
    _, _, _, _, _, _, edtf = c.compile_dates(base, tmp_path / "cache.parquet", {})
    written = dict(zip(edtf.collect()["parent_id"], edtf.collect()["value"], strict=True))
    assert written[b"n1"] == "1979-12-31"
    # accession 2005 makes 1998 the latest century fitting
    assert written[b"n2"] == "1998-01-01"
    assert b"n3" not in written


def test_accession_ceilings_takes_the_earliest(tmp_path, monkeypatch):
    years = tmp_path / "accession_years.parquet"
    pl.DataFrame(
        {
            "record_id": ["r1", "r1", "r2"],
            "data_source": ["M", "M", "M"],
            "accession_year": [1990, 1975, 2001],
            "year_source": ["object_number", "acquisition_date", "object_number"],
        }
    ).write_parquet(years)
    monkeypatch.setattr(c, "ACCESSION_YEARS", years)
    out = c.accession_ceilings().collect().sort("record_id")
    assert out["ceiling"].to_list() == [1975, 2001]
