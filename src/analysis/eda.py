"""Summary tables for exploratory data analysis (Phase 2).

Every function takes the clean review DataFrame (from :func:`load_clean`) and
returns a tidy table; plotting lives in :mod:`src.analysis.plots` and the
narrative in ``notebooks/02_eda.ipynb``.

Notes on the data that affect interpretation:
- ``review_date`` on Google Play is the date of the latest version of a review;
  users can edit reviews, so some dates are later than the original posting.
- ``reply_date`` is the date of the latest developer reply, so response times
  are upper bounds when a reply was edited.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.clean.normalize import extract_emojis, strip_emojis
from src.config import get_path, load_config

ARABIC_TOKEN = re.compile(r"[ء-ي]+")
LATIN_TOKEN = re.compile(r"[a-z][a-z']+")
STAR_ORDER = [1, 2, 3, 4, 5]
SENTIMENT_ORDER = ["negative", "neutral", "positive"]


def load_clean(source: str | None = "google_play") -> pd.DataFrame:
    """Load the clean dataset and add analysis columns.

    Args:
        source: Keep only this source (``"google_play"`` or ``"app_store_rss"``);
            ``None`` keeps both. Default is Google Play, the main dataset.

    Returns:
        DataFrame with extra columns ``month`` (month start, naive),
        ``sentiment`` (from stars) and ``reply_hours``.
    """
    df = pd.read_parquet(get_path("reviews_clean"))
    if source is not None:
        df = df[df["source"] == source].copy()
    star_map = {int(k): v for k, v in load_config()["labels"]["star_to_sentiment"].items()}
    df["month"] = df["review_date"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    df["sentiment"] = pd.Categorical(df["rating"].astype(int).map(star_map), SENTIMENT_ORDER)
    df["reply_hours"] = (df["reply_date"] - df["review_date"]).dt.total_seconds() / 3600
    return df.reset_index(drop=True)


def overview(df: pd.DataFrame) -> pd.DataFrame:
    """Headline numbers per app.

    Args:
        df: Output of :func:`load_clean`.

    Returns:
        One row per app: reviews, first/last month, mean rating, share of
        1-2 star reviews, share Arabic, developer reply rate.
    """
    g = df.groupby("app")
    return pd.DataFrame(
        {
            "reviews": g.size(),
            "first_month": g["month"].min().dt.strftime("%Y-%m"),
            "last_month": g["month"].max().dt.strftime("%Y-%m"),
            "mean_rating": g["rating"].mean().round(2),
            "share_negative": g["sentiment"].apply(lambda s: (s == "negative").mean()).round(3),
            "share_arabic": g["lang"].apply(lambda s: (s == "arabic").mean()).round(3),
            "reply_rate": g["has_reply"].mean().round(3),
        }
    )


def monthly_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Reviews per month per app.

    Args:
        df: Output of :func:`load_clean`.

    Returns:
        Wide table indexed by month, one column per app (0 where none).
    """
    return df.pivot_table(index="month", columns="app", values="review_id", aggfunc="count", fill_value=0)


def monthly_by_sentiment(df: pd.DataFrame, sentiment: str) -> pd.DataFrame:
    """Reviews per month per app for one star-based sentiment group.

    Counts (not shares) are used so that a change in how many happy users
    review cannot hide or fake a change in complaints.

    Args:
        df: Output of :func:`load_clean`.
        sentiment: ``"negative"``, ``"neutral"`` or ``"positive"``.

    Returns:
        Wide table indexed by month, one column per app.
    """
    return monthly_volume(df[df["sentiment"] == sentiment]).reindex(
        monthly_volume(df).index, fill_value=0
    )


def top_emojis(df: pd.DataFrame, sentiment: str, n: int = 10) -> pd.DataFrame:
    """Most common emoji in reviews of one sentiment group, per app.

    Args:
        df: Output of :func:`load_clean`.
        sentiment: Sentiment group to count.
        n: Emoji to return per app.

    Returns:
        Columns ``app, emoji, reviews`` (reviews containing it at least once).
    """
    rows = []
    for app, group in df[df["sentiment"] == sentiment].groupby("app"):
        counts: Counter[str] = Counter()
        for text in group["text"]:
            counts.update(set(extract_emojis(text)))
        rows += [{"app": app, "emoji": e, "reviews": c} for e, c in counts.most_common(n)]
    return pd.DataFrame(rows)


def monthly_mean_rating(df: pd.DataFrame, min_reviews: int = 30) -> pd.DataFrame:
    """Mean star rating per month per app, hiding thin months.

    Args:
        df: Output of :func:`load_clean`.
        min_reviews: Months with fewer reviews are set to NaN.

    Returns:
        Wide table indexed by month, one column per app.
    """
    g = df.groupby(["month", "app"])["rating"]
    mean = g.mean().where(g.size() >= min_reviews)
    return mean.unstack("app")


def rating_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Share of each star rating per app.

    Args:
        df: Output of :func:`load_clean`.

    Returns:
        Table indexed by rating (1-5), one column per app, values sum to 1.
    """
    counts = pd.crosstab(df["rating"].astype(int), df["app"])
    return (counts / counts.sum()).reindex(STAR_ORDER, fill_value=0)


def language_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Share of each detected language per app.

    Args:
        df: Output of :func:`load_clean`.

    Returns:
        Table indexed by language label, one column per app, sorted by total.
    """
    shares = pd.crosstab(df["lang"], df["app"], normalize="columns")
    return shares.loc[shares.sum(axis=1).sort_values(ascending=False).index]


def language_vs_query(df: pd.DataFrame) -> pd.DataFrame:
    """Compare the store's language bucket with the detected language.

    Args:
        df: Output of :func:`load_clean`.

    Returns:
        Counts with ``lang_query`` rows and detected ``lang`` columns.
    """
    return pd.crosstab(df["lang_query"], df["lang"], margins=True)


def length_by_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Median review length in words per star rating and app.

    Emoji-only reviews are excluded (they have zero words by definition).

    Args:
        df: Output of :func:`load_clean`.

    Returns:
        Table indexed by rating, one column per app.
    """
    words = df[~df["is_emoji_only"]]
    return words.pivot_table(index="rating", columns="app", values="n_words", aggfunc="median")


def reply_by_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Developer reply rate per star rating and app.

    Args:
        df: Output of :func:`load_clean`.

    Returns:
        Table indexed by rating, one column per app (share replied).
    """
    return df.pivot_table(index="rating", columns="app", values="has_reply", aggfunc="mean")


def reply_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Developer reply rate per year and app.

    Args:
        df: Output of :func:`load_clean`.

    Returns:
        Table indexed by year, one column per app.
    """
    year = df["month"].dt.year.rename("year")
    return df.groupby([year, "app"])["has_reply"].mean().unstack("app")


def reply_time(df: pd.DataFrame) -> pd.DataFrame:
    """Developer response time (hours) for replied reviews, per app.

    Negative gaps (reply dated before an edited review) are excluded.

    Args:
        df: Output of :func:`load_clean`.

    Returns:
        Table indexed by app with replied count, median and 75th / 90th
        percentile hours, and share answered within 24 hours.
    """
    replied = df[df["has_reply"] & (df["reply_hours"] >= 0)]
    g = replied.groupby("app")["reply_hours"]
    return pd.DataFrame(
        {
            "replied": g.size(),
            "median_h": g.median().round(1),
            "p75_h": g.quantile(0.75).round(1),
            "p90_h": g.quantile(0.90).round(1),
            "within_24h": g.apply(lambda s: (s <= 24).mean()).round(3),
        }
    )


def tokenize(text: str, script: str, stopwords: frozenset[str]) -> list[str]:
    """Split normalized text into words of one script, minus stop words.

    Args:
        text: Normalized review text (``text_norm``).
        script: ``"arabic"`` or ``"english"``.
        stopwords: Words to drop.

    Returns:
        List of tokens (length >= 2).
    """
    pattern = ARABIC_TOKEN if script == "arabic" else LATIN_TOKEN
    return [t for t in pattern.findall(strip_emojis(text)) if len(t) >= 2 and t not in stopwords]


def stopwords_for(script: str) -> frozenset[str]:
    """Return the stop-word set for a script from config.

    Args:
        script: ``"arabic"`` or ``"english"``.

    Returns:
        Frozen set of stop words.
    """
    cfg = load_config()["eda"]
    if script == "arabic":
        return frozenset(cfg["stopwords_ar"])
    return frozenset(ENGLISH_STOP_WORDS) | frozenset(cfg["stopwords_en_extra"])


def ngram_counts(texts: pd.Series, script: str, n: int) -> Counter[str]:
    """Count n-grams (counted once per review) across texts.

    Args:
        texts: Normalized review texts.
        script: ``"arabic"`` or ``"english"``.
        n: 1 for words, 2 for bigrams.

    Returns:
        Counter of n-gram -> number of reviews containing it.
    """
    stop = stopwords_for(script)
    counts: Counter[str] = Counter()
    for text in texts:
        tokens = tokenize(text, script, stop)
        grams = {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}
        counts.update(grams)
    return counts


def distinctive_terms(
    df: pd.DataFrame,
    app: str,
    script: str,
    n: int = 1,
    min_count: int | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Words or bigrams most typical of negative vs. positive reviews.

    Uses the weighted log-odds ratio with an informative Dirichlet prior
    (Monroe, Colaresi & Quinn, 2008). Unlike raw frequency, it down-weights
    words common in both groups ("تطبيق", "good") and gives z-scores, so rare
    words with noisy ratios do not dominate.

    Args:
        df: Output of :func:`load_clean`.
        app: App key (``"tabby"`` / ``"tamara"``).
        script: ``"arabic"`` (rows tagged Arabic) or ``"english"``.
        n: 1 for words, 2 for bigrams.
        min_count: Minimum reviews containing the term (default from config).
        top_n: Terms to return per side (default from config).

    Returns:
        Rows with ``term, side ('negative'/'positive'), z, count_neg, count_pos``,
        the top ``top_n`` per side, strongest first.
    """
    cfg = load_config()["eda"]
    min_count = cfg["min_term_count"] if min_count is None else min_count
    top_n = cfg["top_n_terms"] if top_n is None else top_n

    rows = df[(df["app"] == app) & (df["lang"] == script)]
    neg = ngram_counts(rows.loc[rows["sentiment"] == "negative", "text_norm"], script, n)
    pos = ngram_counts(rows.loc[rows["sentiment"] == "positive", "text_norm"], script, n)

    vocab = [t for t in set(neg) | set(pos) if neg[t] + pos[t] >= min_count]
    if not vocab:
        return pd.DataFrame(columns=["term", "side", "z", "count_neg", "count_pos"])
    y_neg = np.array([neg[t] for t in vocab], dtype=float)
    y_pos = np.array([pos[t] for t in vocab], dtype=float)

    background = y_neg + y_pos
    alpha = background * cfg["log_odds_prior"] / background.sum() * len(vocab)
    alpha = np.maximum(alpha, 0.01)
    a0, n_neg, n_pos = alpha.sum(), y_neg.sum(), y_pos.sum()

    log_odds_neg = np.log((y_neg + alpha) / (n_neg + a0 - y_neg - alpha))
    log_odds_pos = np.log((y_pos + alpha) / (n_pos + a0 - y_pos - alpha))
    variance = 1 / (y_neg + alpha) + 1 / (y_pos + alpha)
    z = (log_odds_neg - log_odds_pos) / np.sqrt(variance)

    table = pd.DataFrame({"term": vocab, "z": z, "count_neg": y_neg.astype(int), "count_pos": y_pos.astype(int)})
    top_neg = table.nlargest(top_n, "z").assign(side="negative")
    top_pos = table.nsmallest(top_n, "z").assign(side="positive")
    out = pd.concat([top_neg, top_pos], ignore_index=True)
    out["z"] = out["z"].abs().round(1)
    return out[["term", "side", "z", "count_neg", "count_pos"]]
