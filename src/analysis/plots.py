"""Shared matplotlib style and chart helpers for report figures.

Colors and neutral tones are read from the ``viz`` section of ``config.yaml``.
Arabic labels are passed as plain text; matplotlib >= 3.11 handles shaping and
right-to-left layout.
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
    """Return the configured color for an app key."""
    return VIZ["colors"][app]


def app_name(app: str) -> str:
    """Return the display name for an app key."""
    return APPS[app]["display_name"]


def setup_style() -> None:
    """Apply the project's matplotlib rcParams."""
    plt.rcParams.update(
        {
            "figure.facecolor": VIZ["surface"],
            "axes.facecolor": VIZ["surface"],
            "savefig.facecolor": VIZ["surface"],
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "font.family": "DejaVu Sans",
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
    """Add a source note at the bottom left of a figure.

    Args:
        fig: Target figure.
        text: Note text.
    """
    fig.text(0.01, 0.005, text, fontsize=7.5, color=VIZ["muted"], va="bottom")


def lines_by_app(ax: Axes, wide: pd.DataFrame, label_ends: bool = True, fmt: str = "{:,.0f}") -> None:
    """Plot one line per app with a legend and optional end labels.

    Args:
        ax: Target axes.
        wide: Table indexed by time with one column per app key.
        label_ends: Annotate the last point of each line with its value.
        fmt: Format string for end labels.
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
    """Plot grouped vertical bars with one bar per app in each category.

    Args:
        ax: Target axes.
        wide: Table indexed by category with one column per app key.
        fmt: Format string for value labels; no labels if ``None``.
        xlabels: Category labels. Defaults to the index values.
        legend_loc: Legend location.
    """
    n = len(wide.columns)
    width = 0.8 / n
    x = range(len(wide.index))
    for i, app in enumerate(wide.columns):
        offsets = [xi - 0.4 + width * (i + 0.5) for xi in x]
        bars = ax.bar(
            offsets, wide[app].values, width=width * 0.92,
            color=app_color(app), label=app_name(app),
        )
        if fmt:
            ax.bar_label(bars, labels=[fmt.format(v) for v in wide[app].values], padding=2, fontsize=7.5)
    ax.set_xticks(list(x), xlabels or [str(i) for i in wide.index])
    ax.legend(loc=legend_loc, ncols=n)


def terms_panel(ax: Axes, terms: pd.DataFrame, color: str, title: str) -> None:
    """Plot distinctive terms as horizontal bars, strongest at the top.

    Args:
        ax: Target axes.
        terms: Rows for one side from :func:`src.analysis.eda.distinctive_terms`.
        color: Bar color.
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
    """Save a figure as PNG in ``paths.figures``. The figure is not closed.

    Args:
        fig: Figure to save.
        name: File name without extension.

    Returns:
        Path of the written file.
    """
    out = get_path("figures") / f"{name}.png"
    fig.savefig(out, bbox_inches="tight")
    return out
