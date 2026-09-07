from __future__ import annotations

from collections.abc import Sequence

import plotly.express as px
import plotly.graph_objects as go
import polars as pl


def histbar(
    s: pl.Series | Sequence[float],
    nbins: int = 50,
    lo: float | None = None,
    hi: float | None = None,
    title: str | None = None,
    xlabel: str | None = None,
) -> go.Figure:
    """Pre-binned histogram; the full value array runs to millions of points"""
    s = pl.Series(s).drop_nulls()
    lo = float(s.min()) if lo is None else lo
    hi = float(s.max()) if hi is None else hi
    width = (hi - lo) / nbins or 1.0
    binned = (
        s.to_frame("v")
        .select((((pl.col("v") - lo) / width).floor().clip(0, nbins - 1) * width + lo).alias("bin"))
        .group_by("bin")
        .len()
        .sort("bin")
    )
    fig = px.bar(
        binned,
        x="bin",
        y="len",
        title=title,
        labels={"bin": xlabel or "value", "len": "count"},
        color_discrete_sequence=["#4c78a8"],
    )
    fig.update_traces(marker_line_width=0, hovertemplate="%{x}<br>%{y:,} records<extra></extra>")
    fig.update_layout(
        template="plotly_white",
        height=520,
        bargap=0,
        title={"x": 0.5, "xanchor": "center", "font": {"size": 18}},
        font={"family": "Inter, Helvetica, Arial, sans-serif", "size": 13, "color": "#333"},
        margin={"l": 60, "r": 30, "t": 70, "b": 55},
        plot_bgcolor="white",
        bargroupgap=0,
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor="#ccc", ticks="outside")
    fig.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False, separatethousands=True)
    return fig
