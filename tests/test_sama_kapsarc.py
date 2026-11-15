"""Offline tests for the SAMA/KAPSARC wide-monthly transform."""

from __future__ import annotations

import pandas as pd
import pytest

from src.collect.sama_kapsarc import normalize_indicator, to_wide_monthly

INDICATORS = {
    "Total POS : Sales (In Thousand Riyals)": "pos_sales_m_sar",
    "E-Commerce Transactions Using Mada Cards : Sales (In Thousand Riyals)": "ecom_mada_sales_m_sar",
    "E-Commerce Transactions Using Mada Cards : Number of Transactions": "ecom_mada_txn_count",
}


def _long() -> pd.DataFrame:
    """Small long-format sample with messy spacing, an old month and an annual row."""
    rows = []
    for month, pos, ecom, txn in [("2018-12-01", 1, 1, 1), ("2019-01-01", 30_000, 10_000, 50_000)]:
        rows += [
            ("Monthly", "Total POS :  Sales (In Thousand Riyals)", pos, month),
            ("Monthly", "E-Commerce Transactions Using Mada Cards : Sales (In Thousand Riyals)", ecom, month),
            ("Monthly", "E-Commerce Transactions Using Mada Cards : Number of Transactions", txn, month),
        ]
    rows.append(("Annually", "Total POS :  Sales (In Thousand Riyals)", 999, "2019-01-01"))
    return pd.DataFrame(rows, columns=["periodicity", "indicator", "value_in_different_units", "date_object"])


def test_normalize_indicator() -> None:
    """Double spaces in SAMA names collapse to single spaces."""
    assert normalize_indicator("Total POS :  Sales") == "Total POS : Sales"


def test_wide_monthly_units_filter_and_derived() -> None:
    """Start filter, thousand->million conversion and derived columns are correct."""
    wide = to_wide_monthly(_long(), INDICATORS, "2019-01-01")
    assert len(wide) == 1 and wide["month"].iloc[0] == pd.Timestamp("2019-01-01")
    row = wide.iloc[0]
    assert row["pos_sales_m_sar"] == pytest.approx(30.0)          # 30,000 thousand = 30 million
    assert row["ecom_mada_sales_m_sar"] == pytest.approx(10.0)
    assert row["ecom_mada_txn_count"] == 50_000
    assert row["ecom_avg_ticket_sar"] == pytest.approx(200.0)     # 10m SAR / 50k txns
    assert row["ecom_share_of_card_spending"] == pytest.approx(0.25)


def test_missing_indicator_raises() -> None:
    """A configured indicator absent from the data is an error, not silent NaNs."""
    with pytest.raises(ValueError):
        to_wide_monthly(_long(), {**INDICATORS, "Not A Real Indicator": "x_count"}, "2019-01-01")
