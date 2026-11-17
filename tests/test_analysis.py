"""Tests for EDA summary tables and the Google Trends export cleaner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.eda import country_mentions, distinctive_terms, phrase_share, reply_time
from src.collect.google_trends import clean_region, clean_time


def test_distinctive_terms_separates_groups() -> None:
    counts = {"سيئه": (38, 2), "ممتازه": (2, 38), "خدمه": (20, 20)}   # (negative, positive) reviews
    rows = [
        (side, term)
        for term, (n_neg, n_pos) in counts.items()
        for side, n in (("negative", n_neg), ("positive", n_pos))
        for _ in range(n)
    ]
    df = pd.DataFrame(rows, columns=["sentiment", "text_norm"]).assign(app="tabby", lang="arabic")
    table = distinctive_terms(df, "tabby", "arabic", n=1, min_count=1, top_n=1).set_index("side")
    assert table.loc["negative", "term"] == "سيئه"
    assert table.loc["positive", "term"] == "ممتازه"
    assert (table.loc["negative", "count_neg"], table.loc["negative", "count_pos"]) == (38, 2)
    assert table.loc["negative", "z"] == table.loc["positive", "z"] > 3


def test_reply_time_excludes_replies_dated_before_review() -> None:
    df = pd.DataFrame({
        "app": ["a"] * 5,
        "has_reply": [True, True, True, True, False],
        "reply_hours": [1.0, 3.0, 30.0, -2.0, np.nan],
    })
    row = reply_time(df).loc["a"]
    assert row["replied"] == 4 and row["excluded"] == 1
    assert row["median_h"] == 3.0
    assert row["within_24h"] == round(2 / 3, 3)


def test_phrase_share_and_country_mentions() -> None:
    df = pd.DataFrame({
        "review_id": range(4),
        "app": ["tabby", "tabby", "tamara", "tamara"],
        "lang": ["english", "english", "arabic", "arabic"],
        "sentiment": ["negative", "negative", "negative", "positive"],
        "text_norm": ["customer service is slow, paid 300 aed", "white screen", "رسوم التاخير بالريال", "ممتاز"],
    })
    shares = phrase_share(df)
    assert shares.loc["customer_service", "tabby"] == 0.5
    assert shares.loc["late_fees", "tamara"] == 1.0
    places = country_mentions(df)
    assert places.loc["english", "with_marker"] == 1 and places.loc["english", "uae"] == 1.0
    assert places.loc["arabic", "saudi_arabia"] == 1.0


def test_clean_time_handles_preamble_below_one_and_weeks(tmp_path: Path) -> None:
    path = tmp_path / "time.csv"
    path.write_text(
        "Category: All categories\n\n"
        "Week,تابي: (Saudi Arabia),تمارا: (Saudi Arabia)\n"
        "2026-01-04,40,<1\n2026-01-11,60,2\n2026-02-01,50,3\n",
        encoding="utf-8",
    )
    df = clean_time(path, {"tabby": "تابي", "tamara": "تمارا"}, downloaded_on="2026-02-10")
    assert df["month"].dt.strftime("%Y-%m").tolist() == ["2026-01", "2026-02"]
    assert df.loc[0, "tabby"] == 50.0 and df.loc[0, "tamara"] == (0.5 + 2) / 2
    assert df["is_partial"].tolist() == [False, True]


def test_clean_region(tmp_path: Path) -> None:
    path = tmp_path / "region.csv"
    path.write_text("Region,تابي: (1/1/26 - 2/1/26),تمارا: (1/1/26 - 2/1/26)\nRiyadh,60%,40%\n".replace("%", ""), encoding="utf-8")
    df = clean_region(path, {"tabby": "تابي", "tamara": "تمارا"})
    assert df.loc[0, "region"] == "Riyadh" and df.loc[0, "tabby"] == 60.0
