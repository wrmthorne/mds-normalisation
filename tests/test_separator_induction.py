import polars as pl

from mds_norm.pipeline import vocab_alignment as va

MATERIAL = "aat+fish_building_materials"
INST = "Test Museum"


def values(rows: list[tuple[str, int]]) -> pl.DataFrame:
    """Routed (group, institution, value) rows shaped like the cascade's whole-value frame"""
    return pl.DataFrame(
        [{"group": MATERIAL, "data_source": INST, "value": v, "count": c, "norm": v.lower()} for v, c in rows]
    ).with_columns(route=pl.lit("cascade"))


def test_discovery_and_second_round(tmp_path, monkeypatch):
    monkeypatch.setattr(va, "SEPARATORS", tmp_path / "sep.parquet")
    monkeypatch.setattr(va, "SEPARATOR_CANDIDATES", tmp_path / "cand.parquet")
    monkeypatch.setattr(va, "group_index", lambda g: pl.LazyFrame({"norm": ["wood", "glass", "paper", "iron"]}))
    monkeypatch.setattr(va, "MIN_LITERAL_SUPPORT", 20)
    monkeypatch.setattr(va, "MIN_SUPPORT_RECORDS", 20)
    monkeypatch.setattr(va, "MIN_DISTINCT_FRAGMENTS", 2)
    monkeypatch.setattr(va, "CONNECTIVE_MIN_RECORDS", 20)
    terms = ["wood", "glass", "paper", "iron", "steel", "tin", "silk", "lead"]
    rows = [(t, 10) for t in terms]
    # colon lists, a convention no closed set learned
    rows += [(f"{a}:{b}", 5) for a in terms for b in terms if a != b]
    # "and" attests only once commas are split away
    rows += [(f"{a}, {b} and {c}, {d}", 5) for a, b, c, d in zip(terms, terms[1:], terms[2:], terms[3:], strict=False)]
    # only the spaced hyphen is a list convention
    rows += [(f"{a}-{b}", 5) for a in terms for b in terms if a != b]
    rows += [(f"{a} - {b}", 5) for a in terms for b in terms if a != b]
    # an uncertainty mark is content, never a join
    rows += [(f"{a}, ? {b}", 5) for a in terms for b in terms if a != b]
    # a qualifier bracket never becomes a separator
    rows += [(f"{a} (probably {b})", 5) for a in terms for b in terms if a != b]
    whole = values(rows)

    accepted = va.induce_separators(whole)

    assert set(accepted["separator"]) == {":", ",", "and", " - "}
    scored = pl.read_parquet(tmp_path / "sep.parquet")
    by_sep = {r["separator"]: r for r in scored.iter_rows(named=True)}
    assert by_sep[":"]["round"] == 1
    assert by_sep[","]["round"] == 1
    assert by_sep["and"]["round"] == 2, "and passes only after the comma lists are split"
    candidates = pl.read_parquet(tmp_path / "cand.parquet")
    assert not candidates.filter(pl.col("separator").str.contains(r"[()]") & pl.col("admitted")).height
    reasons = dict(zip(candidates["separator"], candidates["reason"], strict=True))
    assert reasons["-"] == "word orthography"
    assert reasons[" - "] is None
    assert reasons[", ?"] == "uncertainty mark"
    assert "probably" in set(candidates.filter(pl.col("kind") == "word")["separator"])
    assert not candidates.filter((pl.col("separator") == "probably") & pl.col("admitted")).height
