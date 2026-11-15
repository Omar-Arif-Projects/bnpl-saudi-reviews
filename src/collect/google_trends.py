"""Clean manually downloaded Google Trends CSV exports into tidy files.

Google Trends has no official API, so the exports are downloaded by hand from
https://trends.google.com/explore (settings recorded under ``trends`` in
``config.yaml``). This script only reads those files; it never contacts Google.

What it does:
- Skips Google's preamble lines (e.g. ``Category: All categories`` and blank lines)
  by locating the real header row. Newer exports have no preamble; both work.
- Renames the search-term columns (e.g. ``تابي: (Saudi Arabia)``) to app keys
  (``tabby``, ``tamara``) using ``trends.terms`` in config.
- Replaces Google's ``"<1"`` (interest above zero but below 1) with ``0.5``.
- Time series: parses dates to month start; daily or weekly exports are averaged
  per month. Adds ``is_partial`` for the month the file was downloaded in, since
  Google reports that month on incomplete data.

Values are Google's relative 0-100 index (100 = peak popularity in the selected
range and region, across all compared terms), not search counts.

Outputs (paths from ``config.yaml``):
- ``paths.trends_time``: columns ``month, tabby, tamara, is_partial``.
- ``paths.trends_region``: columns ``region, tabby, tamara``.
- ``paths.trends_figure``: a line chart of monthly interest.

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
# First-column names Google uses for the header row of each export type.
HEADER_KEYS = {"time", "month", "week", "day", "region", "city", "metro", "country"}
# Google appends the geo to term columns in some exports: "تابي: (Saudi Arabia)".
TERM_SUFFIX = re.compile(r":\s*\(.*\)\s*$")


def read_trends_csv(path: Path) -> pd.DataFrame:
    """Read a Google Trends export, skipping any preamble before the header row.

    Args:
        path: CSV file as downloaded from trends.google.com.

    Returns:
        DataFrame with the original header and all values as strings.

    Raises:
        ValueError: If no recognizable header row is found.
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
    """Rename the first column to ``index_name`` and term columns to app keys.

    Args:
        df: Output of :func:`read_trends_csv`.
        terms: Mapping of app key -> search term as typed in Google Trends.
        index_name: New name for the first column (``month`` or ``region``).

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
    """Convert interest values to floats, mapping ``"<1"`` to 0.5.

    Args:
        series: Raw string values from the export.

    Returns:
        Float series. Unparseable values raise rather than silently becoming NaN.
    """
    cleaned = series.str.strip().replace(BELOW_ONE, str(BELOW_ONE_VALUE))
    return pd.to_numeric(cleaned, errors="raise").astype(float)


def clean_time(path: Path, terms: dict[str, str], downloaded_on: str) -> pd.DataFrame:
    """Clean the interest-over-time export into one row per month.

    Args:
        path: Raw ``trends_time.csv``.
        terms: Mapping of app key -> search term.
        downloaded_on: ISO date of the manual download (marks the partial month).

    Returns:
        DataFrame with ``month`` (month start), one float column per app, and
        ``is_partial``.
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
    """Clean the interest-by-region export.

    In a multi-term comparison, Google splits each region's interest between the
    terms so a row sums to about 100.

    Args:
        path: Raw ``trends_region.csv``.
        terms: Mapping of app key -> search term.

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
    """Save a line chart of monthly interest, one line per app.

    Colors are a palette checked for colorblind separation. Each line is also
    labeled at its end, so color is not the only cue. The partial month is
    dropped from the lines and mentioned in the caption.

    Args:
        df: Output of :func:`clean_time`.
        cfg: Full project config (app display names, trends settings).
        out_path: PNG file to write.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"tabby": "#2a78d6", "tamara": "#eb6834"}
    ink, muted, grid = "#0b0b0b", "#52514e", "#e6e5e0"
    full = df[~df["is_partial"]]

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for key, color in colors.items():
        # matplotlib >= 3.11 lays out Arabic correctly from plain text.
        name = cfg["apps"][key]["display_name"]
        term = cfg["trends"]["terms"][key]
        ax.plot(full["month"], full[key], color=color, linewidth=2, label=f"{name}: {term}")
        last = full.iloc[-1]
        ax.annotate(
            f"{name} {last[key]:.0f}", (last["month"], last[key]),
            xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=ink,
        )

    t = cfg["trends"]
    ax.set_title(
        f"Google search interest in Saudi Arabia, {full['month'].min():%b %Y} to {full['month'].max():%b %Y}",
        loc="left", fontsize=12, color=ink,
    )
    ax.set_ylabel("Relative interest (0–100)", color=muted, fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.tick_params(colors=muted, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=ink)
    ax.margins(x=0.01)
    fig.text(
        0.01, 0.01,
        f"Source: Google Trends, manual export {t['downloaded_on']}. 100 = peak across both terms. "
        "Arabic search terms only. Partial current month excluded.",
        fontsize=7.5, color=muted,
    )
    fig.tight_layout(rect=(0, 0.03, 0.97, 1))
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved chart -> %s", out_path)


def run() -> None:
    """Clean both exports, write tidy Parquet files, and save the chart."""
    cfg = load_config()
    t = cfg["trends"]

    time_df = clean_time(get_path("trends_time_raw"), t["terms"], t["downloaded_on"])
    region_df = clean_region(get_path("trends_region_raw"), t["terms"])

    get_path("processed")  # make sure the folder exists
    for df, key in ((time_df, "trends_time"), (region_df, "trends_region")):
        out = get_path(key)
        df.to_parquet(out, index=False)
        logger.info("Saved %d rows -> %s", len(df), out)

    get_path("figures")
    plot_time(time_df, cfg, get_path("trends_figure"))


if __name__ == "__main__":
    run()
