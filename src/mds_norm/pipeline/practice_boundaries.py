"""Tested boundaries in each field's use over acquisition years, for the units the analysis paper draws"""

import itertools
import json

import numpy as np
import polars as pl
from scipy.sparse.csgraph import connected_components

from mds_norm.paths import INSTITUTION_PSEUDONYMS, INSTITUTIONAL, RAW_RECORDS
from mds_norm.pipeline.accession_schemes import field_years

# the units drawn, with the stem their figures are saved under: an institution, or one department of an institution
# whose records name one, chosen because most of their records carry a recorded date
UNITS = {"Norfolk Museums Service | Museum of Norwich": "department", "Trowbridge Museum": "museum"}
# a record's year is read from its accession date, else its acquisition date; never from its object number
DATE_FIELDS = ("spectrum/accession_date", "spectrum/acquisition_date")
YEAR_LABEL = {"accession_date": "accession year", "acquisition_date": "acquisition year"}
DEPARTMENT_FIELD = "spectrum/responsible_department_section"
# a year is read only where it holds this many dated records
MIN_YEAR_RECORDS = 30
# a boundary needs this many readable years on each side and a permutation p-value below this, where years are
# shuffled and the best split taken each time so the search is part of the null; survivors must then pass
# Benjamini-Hochberg across all of a unit's tests
BOUNDARY_SIDE_YEARS, BOUNDARY_ALPHA, BOUNDARY_FDR = 5, 0.01, 0.05
BOUNDARY_PERMUTATIONS = 999
# the mean fill rate must also change by this share of the field's own peak, its best run of side-length years, and by
# at least this many points, below which the change cannot be seen on the figure
BOUNDARY_EFFECT, BOUNDARY_EFFECT_FLOOR = 0.3, 0.10
# each search also tries every shorter window whose ends lie on a grid this many years apart, so a plateau that later
# falls back is split at its own edges rather than lost in the mean of the whole series
BOUNDARY_WINDOW_GRID = 5
# boundaries in different fields that move the same way this close together form one run
BOUNDARY_GROUP_YEARS = 2
# a boundary is kept only where this share of its change in fill rate survives holding the object name fixed, at the
# full name and at its head noun, so a front that is really a change in what was collected is dropped
COMPOSITION_SHARE = 0.5
# fields filled on the same records are one decision, as a dimension's value and its unit are: two fields join one
# group where their presence on the unit's dated records correlates by at least this phi coefficient
FAMILY_PHI = 0.8
# a boundary is shared where a run holds more field groups stepping the same way than chance puts together: the
# smallest run size that shuffling every step's year produces anywhere in fewer than this share of shuffles
SHARED_ALPHA, SHARED_PERMUTATIONS = 0.05, 999
# a field is marked where its commonest value is at least this share of its entries, since a field that mostly repeats
# one value says little about cataloguing effort
DOMINANT_SHARE = 0.6
# a field that fills fewer records than this in every span is left off the figure
MIN_FILL = 0.05
# fields left off the figures, beside the aggregator's own: the dates the year axis is read from, which are filled on
# every dated record by construction, and the department that names a unit
HIDDEN_FIELDS = (*DATE_FIELDS, DEPARTMENT_FIELD)
# a field's boundaries give it one of five signatures, in the order the figures draw them: fronts only is a practice
# adopted and kept, a front then a fall a practice bounded to one stratum, falls only a practice retired, boundaries
# that all vanish once the object name is held fixed a change in what was collected, and no boundary a steady fill
SIGNATURES = ("adopted", "bounded", "retired", "collected", "steady")

raw = pl.scan_parquet(RAW_RECORDS)


def describe(unit: str) -> str:
    """The unit as its institution's pseudonym describes it, without the pseudonym's serial number"""
    pseudonyms = json.loads(INSTITUTION_PSEUDONYMS.read_text())
    institution, *department = unit.split(" | ")
    name = pseudonyms[institution].rstrip(" 0123456789").lower()
    return f"one department of a {name}" if department else f"a {name}"


def unit_members() -> pl.DataFrame:
    """Every record of each unit, dated or not, with the unit it belongs to"""
    institutions = sorted({u.split(" | ")[0] for u in UNITS})
    records = raw.filter(pl.col("data_source").cast(pl.String).is_in(institutions))
    whole = records.select("record_id", data_source=pl.col("data_source").cast(pl.String)).unique()
    departments = (
        records.filter(pl.col("field_type") == DEPARTMENT_FIELD)
        .group_by("record_id")
        .agg(data_source=pl.format("{} | {}", pl.col("data_source").cast(pl.String).first(), pl.col("value").first()))
    )
    return pl.concat([whole, departments]).filter(pl.col("data_source").is_in(list(UNITS))).collect(engine="streaming")


def unit_values(members: pl.DataFrame) -> pl.DataFrame:
    """Every non-empty value on the units' records, with the unit each record belongs to"""
    institutions = sorted({u.split(" | ")[0] for u in UNITS})
    return (
        raw.filter(
            pl.col("data_source").cast(pl.String).is_in(institutions)
            & pl.col("value").is_not_null()
            & (pl.col("value").str.strip_chars().str.len_chars() > 0)
        )
        .select("record_id", "field_type", "value")
        .join(members.lazy(), on="record_id")
        .collect(engine="streaming")
    )


def recorded_years() -> pl.DataFrame:
    """One year per record that records one, from its accession date, else its acquisition date"""
    return (
        pl.concat(
            [
                field_years(f).select("record_id", "data_source", "year", year_source=pl.lit(f.split("/")[1]))
                for f in DATE_FIELDS
            ]
        )
        .sort("year_source")
        .unique("record_id", keep="first")
    )


def unit_years(members: pl.DataFrame, recorded: pl.DataFrame) -> pl.DataFrame:
    """One year per dated record of each unit"""
    return members.join(recorded.drop("data_source"), on="record_id")


def coverage(years: pl.DataFrame, present: pl.DataFrame) -> pl.DataFrame:
    """Records filling each field per unit and year, with the dated records the year holds"""
    return (
        present.join(years.select("record_id", "data_source", "year"), on="record_id")
        .group_by("data_source", "year", "field_type")
        .agg(n_with_field=pl.col("record_id").n_unique())
        .join(years.group_by("data_source", "year").len(name="records"), on=["data_source", "year"])
    )


def split_changes(rates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The change in mean fill rate at every admissible split of each row of `rates`, raw and scaled by precision"""
    m = rates.shape[-1]
    c = np.concatenate([np.zeros((*rates.shape[:-1], 1)), np.cumsum(rates, axis=-1)], axis=-1)
    s = np.arange(BOUNDARY_SIDE_YEARS, m - BOUNDARY_SIDE_YEARS + 1)
    change = (c[..., [m]] - c[..., s]) / (m - s) - c[..., s] / s
    # a lopsided split's mean difference is noisier than an even one's, so splits compete on the scaled change
    return change, change / np.sqrt(1 / s + 1 / (m - s))


def abrupt(series: np.ndarray, split: int) -> bool:
    """Whether one step at `split` explains a run of yearly fill rates better than a straight line, by BIC"""
    m = len(series)
    step = np.concatenate([np.full(split, series[:split].mean()), np.full(m - split, series[split:].mean())])
    x = np.arange(m)
    line = np.polyval(np.polyfit(x, series, 1), x)
    rss_step, rss_line = ((series - step) ** 2).sum(), ((series - line) ** 2).sum()
    # the step spends one parameter more than the line, on where it sits
    return bool(m * np.log(rss_step / m + 1e-12) + 3 * np.log(m) < m * np.log(rss_line / m + 1e-12) + 2 * np.log(m))


def field_tests(rates: np.ndarray) -> list[dict]:
    """Every split tested while segmenting one field's yearly fill rates, recursing only into the halves of a pass"""
    # the same permutations for every field, so a verdict depends on the field's own series and nothing else
    rng = np.random.default_rng(0)
    peak = float(np.convolve(rates, np.ones(BOUNDARY_SIDE_YEARS) / BOUNDARY_SIDE_YEARS, mode="valid").max())
    effect = max(BOUNDARY_EFFECT * peak, BOUNDARY_EFFECT_FLOOR)
    tests = []

    def test(lo: int, hi: int) -> None:
        m = hi - lo
        if m < 2 * BOUNDARY_SIDE_YEARS:
            return
        window = rates[lo:hi]
        grid = sorted({*range(0, m, BOUNDARY_WINDOW_GRID), m})
        intervals = [(a, b) for a in grid for b in grid if b - a >= 2 * BOUNDARY_SIDE_YEARS]
        shuffled = rng.permuted(np.tile(window, (BOUNDARY_PERMUTATIONS, 1)), axis=1)
        best, null = (-1.0, 0, m, 0), np.zeros(BOUNDARY_PERMUTATIONS)
        for a, b in intervals:
            _, scaled = split_changes(window[a:b])
            i = int(np.argmax(np.abs(scaled)))
            best = max(best, (float(abs(scaled[i])), a, b, i))
            null = np.maximum(null, np.abs(split_changes(shuffled[:, a:b])[1]).max(axis=1))
        stat, a, b, i = best
        p = (1 + int((null >= stat).sum())) / (BOUNDARY_PERMUTATIONS + 1)
        split = lo + a + BOUNDARY_SIDE_YEARS + i
        before, after = float(window[a : split - lo].mean()), float(window[split - lo : b].mean())
        change = after - before
        passed = p < BOUNDARY_ALPHA and abs(change) >= effect
        tests.append(
            {
                "split": split,
                "lo": lo + a,
                "hi": lo + b,
                "change": change,
                "p": p,
                "passed": passed,
                "abrupt": abrupt(window[a:b], split - lo - a),
                "before": before,
                "after": after,
            }
        )
        if passed:
            test(lo, split)
            test(split, hi)

    test(0, len(rates))
    return tests


def unit_counts(cov: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """One unit's years, the dated records in each, its testable fields, and the records filling each field by year"""
    totals = cov.select("year", "records").unique().sort("year")
    years, n = totals["year"].to_numpy(), totals["records"].to_numpy().astype(float)
    fields = sorted(f for f in cov["field_type"].unique() if not f.startswith("ciim/") and f not in HIDDEN_FIELDS)
    k = np.zeros((len(fields), len(years)))
    fidx, yidx = {f: i for i, f in enumerate(fields)}, {int(y): i for i, y in enumerate(years)}
    for f, y, v in cov.filter(pl.col("field_type").is_in(fields)).select("field_type", "year", "n_with_field").rows():
        k[fidx[f], yidx[int(y)]] = v
    return years, n, fields, k


def detect_boundaries(cov: pl.DataFrame) -> pl.DataFrame:
    """Every passing split in each field's yearly fill rate over a unit's readable years"""
    rows = []
    for unit in cov["data_source"].unique().sort():
        years, n, fields, k = unit_counts(cov.filter(pl.col("data_source") == unit))
        readable = n >= MIN_YEAR_RECORDS
        years, n, k = years[readable], n[readable], k[:, readable]
        tests = [
            {"data_source": unit, "field_type": field, **t}
            for field, counts in zip(fields, k, strict=True)
            for t in field_tests(counts / n)
        ]
        if not tests:
            continue
        p = np.array([t["p"] for t in tests])
        order = np.argsort(p)
        bh = np.empty_like(p)
        bh[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
        # the tested window is kept as its first and last readable years, so the boundary can be re-examined on records
        rows += [
            {**t, "year": int(years[t["split"]]), "from_year": int(years[t["lo"]]), "to_year": int(years[t["hi"] - 1])}
            for t, q in zip(tests, bh, strict=True)
            if t["passed"] and q < BOUNDARY_FDR
        ]
    return (
        pl.DataFrame(rows)
        .drop("split", "lo", "hi", "passed")
        .with_columns(direction=pl.when(pl.col("change") > 0).then(pl.lit("up")).otherwise(pl.lit("down")))
    )


def field_families(present: pl.DataFrame, years: pl.DataFrame) -> pl.DataFrame:
    """Each unit's fields with the group they belong to, joining fields whose presence correlates at FAMILY_PHI"""
    rows = []
    for unit in years["data_source"].unique().sort():
        dated = years.filter(pl.col("data_source") == unit).select("record_id")
        p = present.join(dated, on="record_id").filter(~pl.col("field_type").str.starts_with("ciim/"))
        fields = sorted(p["field_type"].unique())
        fidx, ridx = {f: i for i, f in enumerate(fields)}, {r: i for i, r in enumerate(dated["record_id"])}
        m = np.zeros((len(ridx), len(fields)))
        for r, f in p.select("record_id", "field_type").iter_rows():
            m[ridx[r], fidx[f]] = 1.0
        share = m.mean(axis=0)
        spread = np.sqrt(share * (1 - share))
        with np.errstate(invalid="ignore", divide="ignore"):
            phi = (m.T @ m / len(ridx) - np.outer(share, share)) / np.outer(spread, spread)
        _, family = connected_components(np.nan_to_num(phi) >= FAMILY_PHI, directed=False)
        rows += [{"data_source": unit, "field_type": f, "family": int(g)} for f, g in zip(fields, family, strict=True)]
    return pl.DataFrame(rows)


def composition_test(marks: pl.DataFrame, records: pl.DataFrame, present: pl.DataFrame) -> pl.DataFrame:
    """Each boundary's change recomputed within `name` over its tested window, and whether enough of it survives"""
    rows = []
    for m in marks.iter_rows(named=True):
        window = records.filter(
            (pl.col("data_source") == m["data_source"]) & pl.col("year").is_between(m["from_year"], m["to_year"])
        ).with_columns(after=pl.col("year") >= m["year"])
        filled = window.join(
            present.filter(pl.col("field_type") == m["field_type"]), on="record_id", how="left"
        ).with_columns(filled=pl.col("field_type").is_not_null())
        strata = (
            filled.group_by("name")
            .agg(
                n_before=(~pl.col("after")).sum(),
                n_after=pl.col("after").sum(),
                f_before=(pl.col("filled") & ~pl.col("after")).sum(),
                f_after=(pl.col("filled") & pl.col("after")).sum(),
            )
            .with_columns(weight=pl.col("n_before") * pl.col("n_after") / (pl.col("n_before") + pl.col("n_after")))
        )
        both = strata.filter(pl.col("weight") > 0)
        raw_change = float(
            strata["f_after"].sum() / strata["n_after"].sum() - strata["f_before"].sum() / strata["n_before"].sum()
        )
        within = both["weight"] * (both["f_after"] / both["n_after"] - both["f_before"] / both["n_before"])
        adjusted = float(within.sum() / both["weight"].sum()) if both.height else 0.0
        rows.append(
            {
                **m,
                "raw_change": raw_change,
                "within_change": adjusted,
                "stratified_share": float((both["n_before"].sum() + both["n_after"].sum()) / window.height),
                "survives": adjusted / raw_change >= COMPOSITION_SHARE if raw_change else False,
            }
        )
    return pl.DataFrame(rows)


def screened_boundaries(cov: pl.DataFrame, years: pl.DataFrame, values: pl.DataFrame) -> pl.DataFrame:
    """Every passing boundary with its field group and whether it survives the object name held fixed at both grains"""
    present = values.select("record_id", "field_type").unique()
    marks = detect_boundaries(cov).join(field_families(present, years), on=["data_source", "field_type"])
    names = (
        values.filter(pl.col("field_type") == "spectrum/object_name")
        .group_by("record_id")
        .agg(name=pl.col("value").str.to_lowercase().first())
        .with_columns(head=pl.col("name").str.extract(r"([a-z]+)$"))
    )
    records = years.join(names, on="record_id", how="left")
    by_name = composition_test(marks, records.with_columns(name=pl.col("name").fill_null("(none)")), present)
    by_head = composition_test(marks, records.with_columns(name=pl.col("head").fill_null("(none)")), present)
    return by_name.with_columns(
        within_change_head=by_head["within_change"],
        stratified_share_head=by_head["stratified_share"],
        kept=pl.col("survives") & by_head["survives"],
    ).drop("survives")


def field_profile(members: pl.DataFrame, values: pl.DataFrame, years: pl.DataFrame) -> pl.DataFrame:
    """Each unit's fields with their fill among its undated records and whether one value dominates their entries"""
    undated = members.join(years.select("record_id"), on="record_id", how="anti")
    fill = (
        values.select("record_id", "data_source", "field_type")
        .unique()
        .join(undated.select("record_id"), on="record_id")
        .group_by("data_source", "field_type")
        .len()
        .join(undated.group_by("data_source").len(name="records"), on="data_source")
        .select("data_source", "field_type", undated_fill=pl.col("len") / pl.col("records"))
    )
    dominant = (
        values.group_by("data_source", "field_type", "value")
        .len()
        .group_by("data_source", "field_type")
        .agg(dominant=pl.col("len").max() / pl.col("len").sum() >= DOMINANT_SHARE)
    )
    return dominant.join(fill, on=["data_source", "field_type"], how="left").with_columns(
        pl.col("undated_fill").fill_null(0.0)
    )


def runs(years: np.ndarray, families: np.ndarray) -> list[tuple[float, int]]:
    """Each run of steps no further apart than BOUNDARY_GROUP_YEARS, as its median year and distinct field groups"""
    order = np.argsort(years, kind="stable")
    y, f = years[order], families[order]
    breaks = np.flatnonzero(np.diff(y) > BOUNDARY_GROUP_YEARS) + 1
    return [
        (float(np.median(ys)), len(set(fs)))
        for ys, fs in zip(np.split(y, breaks), np.split(f, breaks), strict=True)
        if len(ys)
    ]


def shared_boundaries(steps: pl.DataFrame, readable: np.ndarray) -> tuple[list[int], int, float]:
    """Years where more field groups step the same way than chance puts together, with the run size that takes"""
    rng = np.random.default_rng(0)
    # grouped in a fixed order, so the seeded null does not depend on which direction polars happens to yield first
    by_direction = {
        d: (g["year"].to_numpy(), g["family"].to_numpy())
        for (d,), g in steps.sort("direction", "year", "family").group_by("direction", maintain_order=True)
    }
    largest = np.zeros(SHARED_PERMUTATIONS, dtype=int)
    for i in range(SHARED_PERMUTATIONS):
        for _, families in by_direction.values():
            # each group keeps its number of steps but lands anywhere in the unit's readable years
            shuffled = np.concatenate(
                [rng.choice(readable, size=int(c), replace=False) for c in np.unique(families, return_counts=True)[1]]
            )
            largest[i] = max(largest[i], max((n for _, n in runs(shuffled, np.sort(families))), default=0))
    observed = [r for years, families in by_direction.values() for r in runs(years, families)]
    min_groups = next((k for k in range(2, max(largest.max(), 1) + 2) if (largest >= k).mean() < SHARED_ALPHA), 2)
    chance = float((largest >= min_groups).mean())
    edges: list[list[float]] = []
    for year in sorted(y for y, n in observed if n >= min_groups):
        if edges and year - edges[-1][-1] <= BOUNDARY_GROUP_YEARS:
            edges[-1].append(year)
        else:
            edges.append([year])
    return [round(float(np.mean(e))) for e in edges], min_groups, chance


def unit_summary(
    unit: str, cov: pl.DataFrame, marks: pl.DataFrame, profile: pl.DataFrame
) -> tuple[dict, pl.DataFrame]:
    """One unit's shared boundaries and counts, and its drawn fields with their signature, in figure order"""
    years, n, fields, k = unit_counts(cov)
    readable = n >= MIN_YEAR_RECORDS
    start = int(years[readable][0])
    kept = marks.filter("kept")
    edges, min_groups, chance = shared_boundaries(kept.filter("abrupt"), years[readable])
    bounds = [start, *edges, int(years[-1]) + 1]
    in_span = [(years >= a) & (years < b) & readable for a, b in itertools.pairwise(bounds)]
    span_fill = np.array([[row[m].sum() / n[m].sum() if m.any() else np.nan for m in in_span] for row in k])
    collected = set(marks["field_type"]) - set(kept["field_type"])
    by_field = {f: g.sort("year") for (f,), g in kept.group_by("field_type")}
    rows = []
    for i, f in enumerate(fields):
        if np.nanmax(span_fill[i]) < MIN_FILL:
            continue
        directions = set(by_field[f]["direction"].to_list()) if f in by_field else set()
        signature = {
            frozenset({"up"}): "adopted",
            frozenset({"up", "down"}): "bounded",
            frozenset({"down"}): "retired",
        }.get(frozenset(directions), "collected" if f in collected else "steady")
        first_change = int(by_field[f]["year"][0]) if f in by_field else 0
        rows.append((SIGNATURES.index(signature), first_change, -np.nanmean(span_fill[i]), f, signature))
    rows.sort()
    drawn = pl.DataFrame(
        {"data_source": unit, "field_type": [r[3] for r in rows], "signature": [r[4] for r in rows]}
    ).join(profile, on=["data_source", "field_type"], how="left")
    summary = {
        "data_source": unit,
        "start": start,
        "shared": edges,
        "min_groups": min_groups,
        "chance": chance,
        "records_before_start": int(n[years < start].sum()),
        "hidden_fields": len(fields) - len(rows),
    }
    return summary, drawn


def main() -> None:
    recorded = recorded_years()
    per_institution = (
        raw.group_by(pl.col("data_source").cast(pl.String))
        .agg(records=pl.col("record_id").n_unique())
        .collect(engine="streaming")
        .join(recorded.group_by("data_source").len(name="dated"), on="data_source", how="left")
        .with_columns(pl.col("dated").fill_null(0))
    )
    any_dated = (per_institution["dated"] > 0).sum()
    half_dated = (per_institution["dated"] >= per_institution["records"] / 2).sum()
    print(
        f"{recorded.height:,} records at {any_dated} of {per_institution.height} institutions carry a recorded"
        f" accession or acquisition date; {half_dated} institutions date half or more of their records"
    )
    members = unit_members()
    values = unit_values(members)
    years = unit_years(members, recorded)
    present = values.select("record_id", "field_type").unique()
    cov = coverage(years, present)
    marks = screened_boundaries(cov, years, values)
    profile = field_profile(members, values, years)
    summaries, fields = [], []
    for unit, stem in UNITS.items():
        used = sorted(years.filter(pl.col("data_source") == unit)["year_source"].unique())
        summary, drawn = unit_summary(
            unit,
            cov.filter(pl.col("data_source") == unit),
            marks.filter(pl.col("data_source") == unit),
            profile.filter(pl.col("data_source") == unit),
        )
        summary |= {
            "stem": stem,
            "label": describe(unit),
            "year_label": YEAR_LABEL[used[0]] if len(used) == 1 else "accession or acquisition year",
            "records": members.filter(pl.col("data_source") == unit).height,
            "dated": years.filter(pl.col("data_source") == unit).height,
        }
        summaries.append(summary)
        fields.append(drawn)
        print(
            f"{unit}: {summary['dated']:,} of {summary['records']:,} records dated;"
            f" shared boundaries {summary['shared']} at {summary['min_groups']} or more groups"
            f" (chance {summary['chance']:.3f}); {drawn['signature'].value_counts().sort('signature').rows()}"
        )
    unit_marks = marks.filter(pl.col("data_source").is_in(list(UNITS)))
    print(
        f"{unit_marks.height} splits pass, {unit_marks['kept'].sum()} survive the object name,"
        f" {unit_marks.filter(pl.col('kept') & pl.col('abrupt')).height} of those abrupt"
    )
    cov.write_parquet(INSTITUTIONAL / "practice_coverage.parquet")
    unit_marks.write_parquet(INSTITUTIONAL / "practice_boundaries.parquet")
    pl.concat(fields).write_parquet(INSTITUTIONAL / "practice_fields.parquet")
    pl.DataFrame(summaries).write_parquet(INSTITUTIONAL / "practice_units.parquet")


if __name__ == "__main__":
    main()
