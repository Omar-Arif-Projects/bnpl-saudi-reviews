"""Download SAMA POS and mada e-commerce statistics from the KAPSARC open data portal.

Source: https://data.kapsarc.org (dataset ``pos-transactions``), which republishes
Saudi Central Bank (SAMA) payment statistics. Public data; no API key is needed,
but ``KAPSARC_API_KEY`` is sent if set in the environment.

Outputs (paths from ``config.yaml``):
- ``paths.sama_long``: the full dataset as published (long format, all periodicities).
- ``paths.sama_wide_monthly``: monthly data from ``kapsarc.wide_start`` onward, one
  column per indicator with short snake_case names, sales in millions of SAR, plus
  derived ``ecom_avg_ticket_sar`` and ``ecom_share_of_card_spending``.

Note: the e-commerce series covers **mada debit cards only**; it excludes Visa,
Mastercard and credit-card online spending.

Usage::

    python -m src.collect.sama_kapsarc
"""

from __future__ import annotations

import os
from io import StringIO
from typing import Any

import pandas as pd
import requests

from src.config import get_logger, get_path, load_config

logger = get_logger("collect.sama_kapsarc")

THOUSAND_TO_MILLION = 1_000
SALES_SUFFIX = "_m_sar"


def build_headers() -> dict[str, str]:
    """Return request headers, adding the API key only if ``KAPSARC_API_KEY`` is set.

    Returns:
        Header dict (empty for anonymous access).
    """
    api_key = os.environ.get("KAPSARC_API_KEY")
    return {"Authorization": f"Apikey {api_key}"} if api_key else {}


def download_long(cfg: dict[str, Any]) -> pd.DataFrame:
    """Download the whole dataset in one request via the CSV export endpoint.

    Args:
        cfg: The ``kapsarc`` section of config.yaml.

    Returns:
        Long-format DataFrame exactly as published.
    """
    url = f"{cfg['base_url']}/{cfg['dataset_id']}/exports/csv"
    response = requests.get(
        url,
        params={"lang": "en", "delimiter": ","},
        headers=build_headers(),
        timeout=cfg["timeout_seconds"],
    )
    response.raise_for_status()
    response.encoding = "utf-8"
    return pd.read_csv(StringIO(response.text))


def normalize_indicator(name: str) -> str:
    """Collapse repeated whitespace so indicator names match config keys reliably.

    Args:
        name: Raw indicator name (SAMA uses inconsistent double spaces).

    Returns:
        Whitespace-normalized name.
    """
    return " ".join(str(name).split())


def to_wide_monthly(long_df: pd.DataFrame, indicators: dict[str, str], start: str) -> pd.DataFrame:
    """Reshape monthly rows to one column per indicator and add derived metrics.

    Args:
        long_df: Long-format data from :func:`download_long`.
        indicators: Mapping of normalized indicator name -> short column name.
        start: First month to keep (ISO date, inclusive).

    Returns:
        Wide DataFrame indexed by month start (column ``month``), sorted ascending.

    Raises:
        ValueError: If a configured indicator is missing from the monthly data.
    """
    monthly = long_df[long_df["periodicity"] == "Monthly"].copy()
    monthly["indicator"] = monthly["indicator"].map(normalize_indicator)

    unknown = sorted(set(monthly["indicator"]) - set(indicators))
    if unknown:
        logger.warning("Ignoring %d indicator(s) not in config: %s", len(unknown), unknown)
    missing = sorted(set(indicators) - set(monthly["indicator"]))
    if missing:
        raise ValueError(f"Configured indicators not found in monthly data: {missing}")

    monthly["month"] = pd.to_datetime(monthly["date_object"])
    monthly = monthly[(monthly["month"] >= pd.Timestamp(start)) & monthly["indicator"].isin(indicators)]

    wide = (
        monthly.pivot(index="month", columns="indicator", values="value_in_different_units")
        .rename(columns=indicators)
        .sort_index()
    )
    wide = wide[[c for c in indicators.values()]]  # stable column order from config
    wide.columns.name = None

    sales_cols = [c for c in wide.columns if c.endswith(SALES_SUFFIX)]
    wide[sales_cols] = wide[sales_cols] / THOUSAND_TO_MILLION

    # Average e-commerce ticket in SAR: (million SAR * 1e6) / transactions.
    wide["ecom_avg_ticket_sar"] = wide["ecom_mada_sales_m_sar"] * 1_000_000 / wide["ecom_mada_txn_count"]
    # Online share of mada card spending (online vs. in-store POS).
    wide["ecom_share_of_card_spending"] = wide["ecom_mada_sales_m_sar"] / (
        wide["ecom_mada_sales_m_sar"] + wide["pos_sales_m_sar"]
    )

    count_cols = [c for c in wide.columns if c.endswith("_count")]
    wide[count_cols] = wide[count_cols].round().astype("Int64")
    return wide.reset_index()


def run() -> None:
    """Download the dataset and write the long and wide monthly CSV files."""
    cfg = load_config()["kapsarc"]
    indicators = {normalize_indicator(k): v for k, v in cfg["indicators"].items()}

    long_df = download_long(cfg)
    long_path = get_path("sama_long")
    long_df.to_csv(long_path, index=False, encoding="utf-8")
    logger.info("Saved %d rows x %d columns -> %s", *long_df.shape, long_path.name)

    wide = to_wide_monthly(long_df, indicators, cfg["wide_start"])
    wide_path = get_path("sama_wide_monthly")
    wide.to_csv(wide_path, index=False, encoding="utf-8", date_format="%Y-%m-%d")
    logger.info(
        "Saved %d months (%s to %s) x %d columns -> %s",
        len(wide),
        wide["month"].min().date(),
        wide["month"].max().date(),
        wide.shape[1],
        wide_path.name,
    )


if __name__ == "__main__":
    run()
