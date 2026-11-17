"""Summary tables for exploratory data analysis.

All functions take the DataFrame returned by :func:`load_clean` and return a
table. Plotting is in :mod:`src.analysis.plots`.

Data caveats:

- ``review_date`` is the date of the latest version of a review; edited
  reviews carry the edit date.
- ``reply_date`` is the date of the latest version of the developer reply.
  A response time can therefore be longer than the time to the first reply
  (the reply was edited later) or shorter (the review was edited before the
  reply). Replies dated before the review are excluded and counted.
- Times are in UTC.
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
        DataFrame with added ``month`` (timezone-naive month start, UTC),
        ``year``, ``sentiment`` (from star rating) and ``reply_hours``.
    """
    df = pd.read_parquet(get_path("reviews_clean"))
    if source is not None:
        df = df[df["source"] == source].copy()
    star_map = {int(k): v for k, v in load_config()["labels"]["star_to_sentiment"].items()}
    df["month"] = df["review_date"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    df["year"] = df["month"].dt.year
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


def reply_time(df: pd.DataFrame, by: str | list[str] = "app") -> pd.DataFrame:
    """Summarize developer response time in hours.

    Replies dated before the review (possible when the review was edited after
    the reply) are excluded from the time statistics and counted separately.

    Args:
        df: Output of :func:`load_clean`, possibly filtered (e.g. to 1-2 star
            reviews).
        by: Grouping column(s), e.g. ``"app"`` or ``["year", "app"]``.

    Returns:
        Table with the number of replies, the number and share excluded,
        median, 75th and 90th percentile hours, and the share answered within
        24 hours.
    """
    replied = df[df["has_reply"]].assign(before_review=lambda d: d["reply_hours"] < 0)
    valid = replied[~replied["before_review"]]
    g = valid.groupby(by)["reply_hours"]
    out = pd.DataFrame(
        {
            "replied": replied.groupby(by).size(),
            "excluded": replied.groupby(by)["before_review"].sum(),
            "median_h": g.median().round(1),
            "p75_h": g.quantile(0.75).round(1),
            "p90_h": g.quantile(0.90).round(1),
            "within_24h": g.apply(lambda s: (s <= 24).mean()).round(3),
        }
    )
    out["excluded_share"] = (out["excluded"] / out["replied"]).round(3)
    return out


def mentions(df: pd.DataFrame, phrase: str) -> pd.Series:
    """Flag reviews that mention a configured complaint phrase.

    Args:
        df: Output of :func:`load_clean`.
        phrase: Key under ``eda.complaint_phrases``.

    Returns:
        Boolean Series aligned to ``df``.
    """
    pattern = load_config()["eda"]["complaint_phrases"][phrase]
    return df["text_norm"].str.contains(pattern, regex=True)


def phrase_share(df: pd.DataFrame, sentiment: str = "negative") -> pd.DataFrame:
    """Compute the share of reviews that mention each complaint phrase.

    Args:
        df: Output of :func:`load_clean`.
        sentiment: Sentiment group to count within.

    Returns:
        Table indexed by phrase key with one column per app (share of that
        app's reviews in the group).
    """
    group = df[df["sentiment"] == sentiment]
    phrases = load_config()["eda"]["complaint_phrases"]
    return pd.DataFrame(
        {key: mentions(group, key).groupby(group["app"]).mean() for key in phrases}
    ).T


def country_mentions(df: pd.DataFrame, by: str | list[str] = "lang") -> pd.DataFrame:
    """Count reviews that name a country, city or currency, and which country.

    Markers (place names and currencies) are listed under
    ``eda.country_markers``. Only a small share of reviews contain one, so the
    result indicates where reviewers are rather than measuring it.

    Args:
        df: Output of :func:`load_clean`.
        by: Grouping column(s).

    Returns:
        Table with the number of reviews, the number and share with any
        marker, and, among reviews with a marker, the share mentioning each
        country (a review can mention several).
    """
    markers = load_config()["eda"]["country_markers"]
    countries = list(markers)
    flagged = df.assign(
        **{country: df["text_norm"].str.contains(pattern, regex=True) for country, pattern in markers.items()}
    )
    flagged["any_marker"] = flagged[countries].any(axis=1)
    out = flagged.groupby(by).agg(reviews=("review_id", "size"), with_marker=("any_marker", "sum"))
    out["marker_share"] = (out["with_marker"] / out["reviews"]).round(4)
    shares = flagged[flagged["any_marker"]].groupby(by)[countries].mean().round(3)
    return out.join(shares)


def source_language_share(df: pd.DataFrame) -> pd.DataFrame:
    """Compare the Arabic share of Google Play and App Store reviews over the same dates.

    For each app, the window starts at the earliest App Store review in the
    data. Only reviews with words (Arabic, English or mixed) are counted.

    Args:
        df: Output of :func:`load_clean` with ``source=None``.

    Returns:
        One row per app and source with the window start, review count and
        Arabic share.
    """
    words = df[df["lang"].isin(["arabic", "english", "mixed"])]
    rows = []
    for app, group in words.groupby("app"):
        start = group.loc[group["source"] == "app_store_rss", "review_date"].min()
        window = group[group["review_date"] >= start]
        for source, part in window.groupby("source"):
            rows.append({
                "app": app, "source": source, "since": start.strftime("%Y-%m-%d"),
                "reviews": len(part), "arabic_share": round((part["lang"] == "arabic").mean(), 3),
            })
    return pd.DataFrame(rows).set_index(["app", "source"])


def version_summary(
    df: pd.DataFrame, app: str, start: str, end: str, min_reviews: int = 15,
) -> pd.DataFrame:
    """Summarize reviews per app version within a date range.

    Args:
        df: Output of :func:`load_clean`.
        app: App key.
        start: First review date to include (inclusive).
        end: Last review date (exclusive).
        min_reviews: Versions with fewer reviews in the range are left out.

    Returns:
        One row per app version, ordered by first review, with review count,
        positive share, number of negative reviews and first and last review dates.
    """
    rows = df[(df["app"] == app) & (df["review_date"] >= start) & (df["review_date"] < end)]
    g = rows.groupby("app_version")
    out = pd.DataFrame(
        {
            "reviews": g.size(),
            "positive_share": g["sentiment"].apply(lambda s: (s == "positive").mean()).round(2),
            "negative": g["sentiment"].apply(lambda s: int((s == "negative").sum())),
            "first": g["review_date"].min().dt.strftime("%Y-%m-%d"),
            "last": g["review_date"].max().dt.strftime("%Y-%m-%d"),
        }
    )
    return out[out["reviews"] >= min_reviews].sort_values("first")


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
    (Monroe, Colaresi & Quinn, 2008), with three adaptations: counts are
    document frequencies (reviews containing the n-gram); group totals are
    sums over the n-grams kept by ``min_count``; and the prior is weak
    (``eda.log_odds_prior`` of the background counts), so the ranking is close
    to a smoothed log-odds ratio. The ranking measures how distinctive a
    phrase is, not how often it occurs (see :func:`phrase_share`).

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
