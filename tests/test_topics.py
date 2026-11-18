"""Tests for topic naming, deduplication and topic summaries (no model downloads)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clean.normalize import normalize_text
from src.config import load_config
from src.models.topics import (
    GENERAL, OTHER, UNASSIGNED, KeywordMatcher, choose_topic, default_matcher, keyword_topic, method_shares,
    name_clusters, quarterly, share_table, token_variants, topic_agreement, topic_scores, unique_texts,
)

KEYWORDS = {
    "fees_late_payment": ["رسوم", "غرامه", "late", "penalt*"],
    "customer_service": ["العملاء", "customer*", "خدمه العملاء"],
    "ease_of_use": ["سهل*", "easy"],
    GENERAL: ["ممتاز*", "good"],
}
PREFIXES = ["وال", "بال", "ال", "و", "ب"]


@pytest.fixture()
def matcher() -> KeywordMatcher:
    return KeywordMatcher(KEYWORDS, PREFIXES)


def test_token_variants_strip_one_prefix() -> None:
    assert token_variants("والرسوم", PREFIXES) == {"والرسوم", "الرسوم", "رسوم"}
    assert "رسوم" in token_variants("بالرسوم", PREFIXES)
    assert token_variants("بس", PREFIXES) == {"بس"}           # a prefix never leaves a single letter


@pytest.mark.parametrize(
    ("term", "topics"),
    [
        ("الرسوم", {"fees_late_payment"}),
        ("penalties", {"fees_late_payment"}),
        ("customers", {"customer_service"}),
        ("سهله", {"ease_of_use"}),
        ("latest", set()),                                  # exact keyword, no wildcard
        ("خدمه العملاء", {"customer_service"}),
        ("ممتاز سهل", {GENERAL, "ease_of_use"}),
    ],
)
def test_matcher_term_topics(matcher: KeywordMatcher, term: str, topics: set[str]) -> None:
    assert matcher.term_topics(term) == topics


def test_topic_scores_are_weight_shares(matcher: KeywordMatcher) -> None:
    scores = topic_scores([("رسوم", 0.6), ("good", 0.2), ("app", 0.2)], matcher)
    assert scores["fees_late_payment"] == pytest.approx(0.6)
    assert scores[GENERAL] == pytest.approx(0.2)
    assert scores["customer_service"] == 0
    assert topic_scores([], matcher)[GENERAL] == 0


def test_choose_topic_prefers_specific_topics() -> None:
    assert choose_topic({"fees_late_payment": 0.2, GENERAL: 0.7}, 0.15) == "fees_late_payment"
    assert choose_topic({"fees_late_payment": 0.1, GENERAL: 0.7}, 0.15) == GENERAL
    assert choose_topic({"fees_late_payment": 0.1, GENERAL: 0.1}, 0.15) == OTHER
    # Ties go to the topic listed first.
    assert choose_topic({"fees_late_payment": 0.3, "customer_service": 0.3}, 0.15) == "fees_late_payment"


def test_keyword_topic_for_single_reviews(matcher: KeywordMatcher) -> None:
    assert keyword_topic("ممتاز وسهل", matcher) == "ease_of_use"
    assert keyword_topic("good", matcher) == GENERAL
    assert keyword_topic("خدمه العملاء ما ترد", matcher) == "customer_service"
    assert keyword_topic("", matcher) == OTHER


def test_name_clusters_marks_unclustered_texts() -> None:
    terms = {-1: [], 0: [("الرسوم", 0.5), ("غرامه", 0.3), ("ممتاز", 0.2)], 1: [("ممتاز", 0.9), ("good", 0.1)]}
    table = name_clusters(terms).set_index("cluster")
    assert table.loc[-1, "topic"] == UNASSIGNED
    assert table.loc[0, "topic"] == "fees_late_payment"
    assert table.loc[1, "topic"] == GENERAL


def test_config_keywords_are_normalized() -> None:
    t = load_config()["topics"]
    for topic, words in t["keywords"].items():
        assert topic in load_config()["labels"]["topics"]
        for word in words:
            assert normalize_text(word.rstrip("*")) == word.rstrip("*"), word
    assert default_matcher().topics == list(t["keywords"])


def test_unique_texts_maps_reviews_back() -> None:
    df = pd.DataFrame({"text_norm": ["ممتاز", "سيء", "ممتاز", "good"], "text": ["مُمتاز", "سيء", "ممتاز", "Good"]})
    unique, codes = unique_texts(df)
    assert unique["text_norm"].tolist() == ["ممتاز", "سيء", "good"]
    assert unique["text"].tolist() == ["مُمتاز", "سيء", "Good"]           # first original text
    assert unique["n_reviews"].tolist() == [2, 1, 1]
    assert (unique["text_norm"].to_numpy()[codes] == df["text_norm"].to_numpy()).all()


def test_share_table_counts_reviews() -> None:
    df = pd.DataFrame({"app": ["a", "a", "a", "b"], "topic": ["x", "x", "y", "y"]})
    table = share_table(df, ["app"]).set_index(["app", "topic"])
    assert table.loc[("a", "x"), ["n", "total"]].tolist() == [2, 3]
    assert table.loc[("a", "x"), "share"] == pytest.approx(2 / 3)
    assert table.loc[("b", "y"), "share"] == 1


def test_quarterly_drops_incomplete_quarters() -> None:
    months = pd.to_datetime(["2026-04-01", "2026-06-01", "2026-07-01", "2026-10-01"])
    df = pd.DataFrame({"group": "negative", "app": "a", "month": months, "topic": ["x", "x", "y", "y"]})
    table = quarterly(df, pd.Timestamp("2026-11-15"))
    assert sorted(table["quarter"].unique()) == ["2026Q2", "2026Q3"]
    assert table.loc[table["quarter"] == "2026Q2", "n"].tolist() == [2]


def test_topic_agreement_metrics() -> None:
    hand = pd.Series(["general"] * 10 + ["fees"] * 10 + ["other"])
    pred = pd.Series(["general"] * 9 + [UNASSIGNED] + ["fees"] * 8 + ["general"] * 2 + ["other"])
    table = topic_agreement(hand, pred, min_support=10)
    overall = table[table["topic"] == "all"].set_index("metric")["value"]
    assert overall["accuracy"] == pytest.approx(18 / 21)
    # Unassigned and other count as "no specific topic", like general.
    assert overall["specific_vs_general"] == pytest.approx(19 / 21)
    fees = table[table["topic"] == "fees"].set_index("metric")["value"]
    assert fees["precision"] == 1 and fees["recall"] == pytest.approx(0.8)
    assert "other" not in set(table["topic"])                       # below min_support
    assert np.isfinite(overall["kappa"])


def test_method_shares_weights_hand_labels() -> None:
    reviews = pd.DataFrame({
        "group": "negative", "topic": ["fees", "fees", GENERAL, GENERAL], "keyword_topic": ["fees", OTHER, OTHER, OTHER],
    })
    gold = pd.DataFrame({"group": "negative", "hand": ["fees", GENERAL, GENERAL], "weight": [3.0, 1.0, 1.0]})
    table = method_shares(reviews, gold).set_index(["method", "topic"])
    assert table.loc[("clusters", "fees"), "share"] == pytest.approx(0.5)
    assert table.loc[("keywords", OTHER), "n"] == 3
    # Weighted share 3 / 5, but n and total stay unweighted counts.
    assert table.loc[("hand_labels", "fees"), "share"] == pytest.approx(0.6)
    assert table.loc[("hand_labels", GENERAL), ["n", "total"]].tolist() == [2, 3]


def test_specific_vs_general_counts_other_as_no_topic() -> None:
    hand = pd.Series([GENERAL, GENERAL, "fees", "fees"])
    pred = pd.Series([OTHER, UNASSIGNED, "fees", OTHER])
    table = topic_agreement(hand, pred, min_support=1)
    overall = table[table["topic"] == "all"].set_index("metric")["value"]
    assert overall["specific_vs_general"] == pytest.approx(3 / 4)
