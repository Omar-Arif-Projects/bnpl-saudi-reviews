"""Clean manual Google Trends CSV exports.

Input files are exported from https://trends.google.com/explore; export
settings are recorded under ``trends`` in ``config.yaml``.

Processing:

- Locate the header row and skip any preamble lines.
- Rename search-term columns to app keys using ``trends.terms``.
- Replace ``"<1"`` with ``0.5``.
- Convert dates to month start, averaging sub-monthly rows, and flag the
  download month as partial.

Values are Google's relative interest index (0-100), not search volumes.

Outputs (paths from ``config.yaml``):

- ``paths.trends_time``: ``month``, one column per app, ``is_partial``.
- ``paths.trends_region``: ``region``, one column per app.
- ``paths.trends_figure``: line chart of monthly interest.

Usage::

    python -m src.collect.google_trends
"""

from __future__ import annotations

import csv
import re
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import get_logger, get_path, load_config

logger = get_logger("collect.google_trends")

BELOW_ONE = "<1"
BELOW_ONE_VALUE = 0.5
HEADER_KEYS = {"time", "month", "week", "day", "region", "city", "metro", "country"}
TERM_SUFFIX = re.compile(r":\s*\(.*\)\s*$")  # e.g. "تابي: (Saudi Arabia)"


def read_trends_csv(path: Path) -> pd.DataFrame:
    """Read a Google Trends export, skipping lines before the header row.

    Args:
        path: Exported CSV file.

    Returns:
        DataFrame with the original header and string values.

    Raises:
        ValueError: If no header row is found.
    """
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    for i, line in enumerate(lines):
        fields = next(csv.reader([line]), [])
        if len(fields) >= 2 and fields[0].strip().lower() in HEADER_KEYS:
            if i:
                logger.info("%s: skipped %d preamble line(s)", path.name, i)
            return pd.read_csv(StringIO("\n".join(lines[i:])), dtype=str)
    raise ValueError(f"No Google Trends header row found in {path}")


def rename_term_columns(df: pd.DataFrame, terms: dict[str, str], index_name: str) -> pd.DataFrame:
    """Rename the first column and the search-term columns.

    Args:
        df: Output of :func:`read_trends_csv`.
        terms: Mapping of app key to search term.
        index_name: New name for the first column.

    Returns:
        DataFrame with columns ``[index_name, *terms]``.

    Raises:
        ValueError: If a configured term is missing from the file.
    """
    term_to_key = {term: key for key, term in terms.items()}
    mapping = {df.columns[0]: index_name}
    for col in df.columns[1:]:
        term = TERM_SUFFIX.sub("", col).strip()
        if term in term_to_key:
            mapping[col] = term_to_key[term]
    missing = sorted(set(terms) - set(mapping.values()))
    if missing:
        raise ValueError(f"Terms for {missing} not found in columns {list(df.columns)}")
    return df.rename(columns=mapping)[[index_name, *terms]]


def to_numeric_interest(series: pd.Series) -> pd.Series:
    """Convert interest values to float, mapping ``"<1"`` to 0.5.

    Args:
        series: Raw string values.

    Returns:
        Float series.

    Raises:
        ValueError: If a value cannot be parsed.
    """
    cleaned = series.str.strip().replace(BELOW_ONE, str(BELOW_ONE_VALUE))
    return pd.to_numeric(cleaned, errors="raise").astype(float)


def clean_time(path: Path, terms: dict[str, str], downloaded_on: str) -> pd.DataFrame:
    """Clean an interest-over-time export to one row per month.

    Args:
        path: Raw time-series CSV.
        terms: Mapping of app key to search term.
        downloaded_on: ISO date of the export, used to flag the partial month.

    Returns:
        DataFrame with ``month``, one float column per app, and ``is_partial``.
    """
    df = rename_term_columns(read_trends_csv(path), terms, "month")
    n_below_one = int((df[list(terms)] == BELOW_ONE).sum().sum())
    for key in terms:
        df[key] = to_numeric_interest(df[key])

    dates = pd.to_datetime(df["month"], format="mixed")
    df["month"] = dates.dt.to_period("M").dt.to_timestamp()
    rows_in = len(df)
    df = df.groupby("month", as_index=False)[list(terms)].mean()
    if len(df) < rows_in:
        logger.info("Averaged %d sub-monthly rows into %d months", rows_in, len(df))

    download_month = pd.Timestamp(downloaded_on).to_period("M").to_timestamp()
    df["is_partial"] = df["month"] == download_month
    logger.info(
        "Time series: %d months (%s to %s), %d '<1' value(s) set to %.1f",
        len(df), df["month"].min().date(), df["month"].max().date(), n_below_one, BELOW_ONE_VALUE,
    )
    return df.sort_values("month").reset_index(drop=True)


def clean_region(path: Path, terms: dict[str, str]) -> pd.DataFrame:
    """Clean an interest-by-region export.

    Args:
        path: Raw region CSV.
        terms: Mapping of app key to search term.

    Returns:
        DataFrame with ``region`` and one float column per app.
    """
    df = rename_term_columns(read_trends_csv(path), terms, "region")
    for key in terms:
        df[key] = to_numeric_interest(df[key])
    df["region"] = df["region"].str.strip()
    logger.info("Regions: %d rows", len(df))
    return df.sort_values(list(terms)[0], ascending=False).reset_index(drop=True)


def plot_time(df: pd.DataFrame, cfg: dict[str, Any], out_path: Path) -> None:
    """Save a line chart of monthly interest per app, excluding the partial month.

    Args:
        df: Output of :func:`clean_time`.
        cfg: Full project configuration.
        out_path: Output PNG path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz = cfg["viz"]
    full = df[~df["is_partial"]]

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    fig.patch.set_facecolor(viz["surface"])
    ax.set_facecolor(viz["surface"])
    for key, color in viz["colors"].items():
        name = cfg["apps"][key]["display_name"]
        term = cfg["trends"]["terms"][key]
        ax.plot(full["month"], full[key], color=color, linewidth=2, label=f"{name}: {term}")
        last = full.iloc[-1]
        ax.annotate(
            f"{name} {last[key]:.0f}", (last["month"], last[key]),
            xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=viz["ink"],
        )

    t = cfg["trends"]
    ax.set_title(
        f"Google search interest in Saudi Arabia, {full['month'].min():%b %Y} to {full['month'].max():%b %Y}",
        loc="left", fontsize=12, color=viz["ink"],
    )
    ax.set_ylabel("Relative interest (0–100)", color=viz["muted"], fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color=viz["grid"], linewidth=0.8)
    ax.tick_params(colors=viz["muted"], labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(viz["baseline"])
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=viz["ink"])
    ax.margins(x=0.01)
    fig.text(
        0.01, 0.01,
        f"Source: Google Trends, manual export {t['downloaded_on']}. 100 = peak across both terms. "
        "Arabic search terms only. Partial current month excluded.",
        fontsize=7.5, color=viz["muted"],
    )
    fig.tight_layout(rect=(0, 0.03, 0.97, 1))
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved chart -> %s", out_path)


def run() -> None:
    """Clean both exports, write Parquet outputs and save the chart."""
    cfg = load_config()
    t = cfg["trends"]

    time_df = clean_time(get_path("trends_time_raw"), t["terms"], t["downloaded_on"])
    region_df = clean_region(get_path("trends_region_raw"), t["terms"])

    get_path("processed")
    for df, key in ((time_df, "trends_time"), (region_df, "trends_region")):
        out = get_path(key)
        df.to_parquet(out, index=False)
        logger.info("Saved %d rows -> %s", len(df), out)

    get_path("figures")
    plot_time(time_df, cfg, get_path("trends_figure"))


if __name__ == "__main__":
    run()
