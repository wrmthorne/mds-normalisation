from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2a78d6"  # after / kept / pipeline
ORANGE = "#eb6834"  # before / demoted / attention
PALE = "#7fb0e6"  # a second series that belongs with BLUE
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"
LIGHT = "#b9b7b0"

RC = {
    "font.size": 8.5,
    "font.family": "sans-serif",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.7,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
}


def use_theme() -> None:
    plt.rcParams.update(RC)


def saver(out: Path) -> Callable[[plt.Figure, str], None]:
    """A save(fig, name) bound to one figure directory, writing both PDF and PNG"""
    out.mkdir(parents=True, exist_ok=True)

    def save(fig: plt.Figure, name: str) -> None:
        fig.savefig(out / f"{name}.pdf")
        fig.savefig(out / f"{name}.png")
        plt.close(fig)

    return save


def grid(ax: plt.Axes, axis: str = "x") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
