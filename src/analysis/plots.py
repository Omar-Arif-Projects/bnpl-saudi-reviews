"""Shared matplotlib style and chart helpers for static report figures.

Design rules used in every chart:
- Each app keeps one color everywhere (``viz.colors`` in config, a pair checked
  for colorblind separation). Color follows the app, never the rank.
- One y-axis per panel; different scales go in separate panels.
- Light gridlines, no top/right spines, text in neutral ink (never series color).
- Two series always get a legend; line charts also get end-of-line labels.

Arabic text: matplotlib >= 3.11 joins Arabic letters and lays them out right to
left by itself (checked on 3.11.2), so labels are passed in as plain text. Do
NOT pre-process them with arabic-reshaper / python-bidi: that reverses them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.config import get_path, load_config

VIZ = load_config()["viz"]
APPS = load_config()["apps"]


def app_color(app: str) -> str:
    """Return the fixed color for an app key."""
    return VIZ["colors"][app]


def app_name(app: str) -> str:
    """Return the display name for an app key."""
    return APPS[app]["display_name"]


def setup_style() -> None:
    """Apply the shared rcParams (call once per notebook or script)."""
    plt.rcParams.update(
        {
            "figure.facecolor": VIZ["surface"],
            "axes.facecolor": VIZ["surface"],
            "savefig.facecolor": VIZ["surface"],
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "font.family": "DejaVu Sans",  # ships with matplotlib, has Arabic glyphs
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titlelocation": "left",
            "axes.titlecolor": VIZ["ink"],
            "axes.labelcolor": VIZ["muted"],
            "axes.edgecolor": VIZ["baseline"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": VIZ["grid"],
            "grid.linewidth": 0.8,
            "xtick.color": VIZ["muted"],
            "ytick.color": VIZ["muted"],
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "text.color": VIZ["ink"],
            "lines.linewidth": 2,
        }
    )


def caption(fig: Figure, text: str) -> None:
    """Add a small source / note line at the bottom-left of a figure."""
    fig.text(0.01, 0.005, text, fontsize=7.5, color=VIZ["muted"], va="bottom")


def lines_by_app(ax: Axes, wide: pd.DataFrame, label_ends: bool = True, fmt: str = "{:,.0f}") -> None:
    """Draw one line per app column with a legend and end-of-line value labels.

    Args:
        ax: Target axes.
        wide: Table indexed by time, one column per app key.
        label_ends: Write "<App> <value>" at each line's last point.
        fmt: Format for the end label value.
    """
    for app in wide.columns:
        series = wide[app].dropna()
        ax.plot(series.index, series.values, color=app_color(app), label=app_name(app))
        if label_ends and len(series):
            ax.annotate(
                f"{app_name(app)} {fmt.format(series.iloc[-1])}",
                (series.index[-1], series.iloc[-1]),
                xytext=(5, 0), textcoords="offset points", va="center", fontsize=8,
            )
    ax.legend(loc="upper left")
    ax.margins(x=0.01)


def grouped_bars(
    ax: Axes,
    wide: pd.DataFrame,
    fmt: str | None = None,
    xlabels: list[str] | None = None,
    legend_loc: str = "upper right",
) -> None:
    """Draw grouped vertical bars, one bar per app within each category.

    Args:
        ax: Target axes.
        wide: Table indexed by category, one column per app key.
        fmt: If given, a format string for value labels above the bars.
        xlabels: Category labels (defaults to the index as strings).
        legend_loc: Legend position; pick a corner without tall bars.
    """
    n = len(wide.columns)
    width = 0.8 / n
    x = range(len(wide.index))
    for i, app in enumerate(wide.columns):
        offsets = [xi - 0.4 + width * (i + 0.5) for xi in x]
        bars = ax.bar(
            offsets, wide[app].values, width=width * 0.92,  # small gap between bars
            color=app_color(app), label=app_name(app),
        )
        if fmt:
            ax.bar_label(bars, labels=[fmt.format(v) for v in wide[app].values], padding=2, fontsize=7.5)
    ax.set_xticks(list(x), xlabels or [str(i) for i in wide.index])
    ax.legend(loc=legend_loc, ncols=n)


def terms_panel(ax: Axes, terms: pd.DataFrame, color: str, title: str) -> None:
    """Horizontal bars of distinctive terms (strongest at the top).

    Args:
        ax: Target axes.
        terms: Rows for one side from :func:`src.analysis.eda.distinctive_terms`.
        color: Bar color (the app's color).
        title: Panel title.
    """
    terms = terms.sort_values("z")
    ax.barh(terms["term"], terms["z"], color=color, height=0.7)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlabel("distinctiveness (z-score)")


def save(fig: Figure, name: str) -> Path:
    """Save a figure to ``paths.figures`` as PNG (the figure stays open so a
    notebook still displays it).

    Args:
        fig: Figure to save.
        name: File name without extension.

    Returns:
        Path of the written file.
    """
    out = get_path("figures") / f"{name}.png"
    fig.savefig(out, bbox_inches="tight")
    return out
