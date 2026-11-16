"""Summary tables for exploratory data analysis.

All functions take the DataFrame returned by :func:`load_clean` and return a
table. Plotting is in :mod:`src.analysis.plots`.

Data caveats:

- ``review_date`` is the date of the latest version of a review; edited
  reviews carry the edit date.
- ``reply_date`` is the date of the latest version of the developer reply, so
  response times are upper bounds.
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
        source: Source to keep (``"google_play"`` or ``"app_store_rss"``), or
            ``None`` for both.

    Returns:
        DataFrame with added ``month`` (timezone-naive month start),
        ``sentiment`` (from star rating) and ``reply_hours``.
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
    """Summarize each app.

    Args:
        df: Output of :func:`load_clean`.

    Returns:
        One row per app with review count, first and last month, mean rating,
        negative share, Arabic share and developer reply rate.
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
    """Count reviews per month and app.

    Args:
        df: Output of :func:`load_clean`.

    Returns:
        Table indexed by month with one column per app.
    """
    return df.pivot_table(index="month", columns="app", values="review_id", aggfunc="count", fill_value=0)


def monthly_by_sentiment(df: pd.DataFrame, sentiment: str) -> pd.DataFrame:
    """Count reviews per month and app for one sentiment group.

    Args:
        df: Output of :func:`load_clean`.
        sentiment: ``"negative"``, ``"neutral"`` or ``"positive"``.

    Returns:
        Table indexed by month with one column per app.
    """
    return monthly_volume(df[df["sentiment"] == sentiment]).reindex(
        monthly_volume(df).index, fill_value=0
    )


def top_emojis(df: pd.DataFrame, sentiment: str, n: int = 10) -> pd.DataFrame:
    """Find the most frequent emoji per app within one sentiment group.

    Args:
        df: Output of :func:`load_clean`.
        sentiment: Sentiment group.
        n: Number of emoji per app.

    Returns:
        Columns ``app``, ``emoji`` and ``reviews`` (number of reviews
        containing the emoji).
    """
    rows = []
    for app, group in df[df["sentiment"] == sentiment].groupby("app"):
        counts: Counter[str] = Counter()
        for text in group["text"]:
            counts.update(set(extract_emojis(text)))
        rows += [{"app": app, "emoji": e, "reviews": c} for e, c in counts.most_common(n)]
    return pd.DataFrame(rows)


def monthly_mean_rating(df: pd.DataFrame, min_reviews: int = 30) -> pd.DataFrame:
    """Compute the mean star rating per month and app.

    Args:
        df: Output of :func:`load_clean`.
        min_reviews: Months with fewer reviews are set to NaN.

    Returns:
        Table indexed by month with one column per app.
    """
    g = df.groupby(["month", "app"])["rating"]
    mean = g.mean().where(g.size() >= min_reviews)
    return mean.unstack("app")


def rating_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the share of each star rating per app.

    Args:
        df: Output of :func:`load_clean`.

    Returns:
        Table indexed by rating (1-5) with one column per app.
    """
    counts = pd.crosstab(df["rating"].astype(int), df["app"])
    return (counts / counts.sum()).reindex(STAR_ORDER, fill_value=0)


def language_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the share of each detected language per app.

    Args:
        df: Output of :func:`load_clean`.

    Returns:
        Table indexed by language label with one column per app, sorted by
        overall share.
    """
    shares = pd.crosstab(df["lang"], df["app"], normalize="columns")
    return shares.loc[shares.sum(axis=1).sort_values(ascending=False).index]


def language_vs_query(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate the store query language against the detected language.

    Args:
        df: Output of :func:`load_clean`.

    Returns:
        Counts with ``lang_query`` as rows and ``lang`` as columns, with totals.
    """
    return pd.crosstab(df["lang_query"], df["lang"], margins=True)


def length_by_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the median review length in words per rating and app.

    Emoji-only reviews are excluded.

    Args:
        df: Output of :func:`load_clean`.

    Returns:
        Table indexed by rating with one column per app.
    """
    words = df[~df["is_emoji_only"]]
    return words.pivot_table(index="rating", columns="app", values="n_words", aggfunc="median")


def reply_by_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the developer reply rate per rating and app.

    Args:
        df: Output of :func:`load_clean`.

    Returns:
        Table indexed by rating with one column per app.
    """
    return df.pivot_table(index="rating", columns="app", values="has_reply", aggfunc="mean")


def reply_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the developer reply rate per year and app.

    Args:
        df: Output of :func:`load_clean`.

    Returns:
        Table indexed by year with one column per app.
    """
    year = df["month"].dt.year.rename("year")
    return df.groupby([year, "app"])["has_reply"].mean().unstack("app")


def reply_time(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize developer response time in hours per app.

    Replies dated before the review (possible after a review edit) are excluded.

    Args:
        df: Output of :func:`load_clean`.

    Returns:
        Table indexed by app with reply count, median, 75th and 90th
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


def stopwords_for(script: str) -> frozenset[str]:
    """Return the stop-word set for a script.

    Args:
        script: ``"arabic"`` or ``"english"``.

    Returns:
        Stop words from config; English also includes scikit-learn's list.
    """
    cfg = load_config()["eda"]
    if script == "arabic":
        return frozenset(cfg["stopwords_ar"])
    return frozenset(ENGLISH_STOP_WORDS) | frozenset(cfg["stopwords_en_extra"])


def tokenize(text: str, script: str, stopwords: frozenset[str]) -> list[str]:
    """Extract words of one script from normalized text.

    Args:
        text: Normalized review text.
        script: ``"arabic"`` or ``"english"``.
        stopwords: Words to exclude.

    Returns:
        Tokens of at least two characters.
    """
    pattern = ARABIC_TOKEN if script == "arabic" else LATIN_TOKEN
    return [t for t in pattern.findall(strip_emojis(text)) if len(t) >= 2 and t not in stopwords]


def ngram_counts(texts: pd.Series, script: str, n: int) -> Counter[str]:
    """Count the number of reviews containing each n-gram.

    Args:
        texts: Normalized review texts.
        script: ``"arabic"`` or ``"english"``.
        n: N-gram length.

    Returns:
        Document frequency per n-gram.
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
    """Rank n-grams by how strongly they distinguish negative from positive reviews.

    Uses the weighted log-odds ratio with an informative Dirichlet prior
    (Monroe, Colaresi & Quinn, 2008), with document frequencies as counts.

    Args:
        df: Output of :func:`load_clean`.
        app: App key.
        script: ``"arabic"`` or ``"english"``; selects reviews with that ``lang``.
        n: N-gram length.
        min_count: Minimum document frequency across both groups. Defaults to
            ``eda.min_term_count``.
        top_n: Terms per side. Defaults to ``eda.top_n_terms``.

    Returns:
        Columns ``term``, ``side``, ``z`` (absolute z-score), ``count_neg`` and
        ``count_pos``, with the top ``top_n`` terms per side.
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
