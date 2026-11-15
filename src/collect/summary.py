"""Load all raw review files and summarize what has been collected."""

from __future__ import annotations

import pandas as pd

from src.config import get_path


def load_raw(pattern: str = "*_reviews_*.parquet") -> pd.DataFrame:
    """Load and concatenate all raw review parquet files, de-duplicated per source.

    Args:
        pattern: Glob pattern inside ``paths.raw``.

    Returns:
        Combined DataFrame (Play Store and App Store rows share the core columns).
    """
    files = sorted(get_path("raw").glob(pattern))
    if not files:
        raise FileNotFoundError(f"No raw files matching {pattern!r}; run src.collect.play_store first.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.drop_duplicates(["source", "review_id"]).reset_index(drop=True)


def collection_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per source / app / query-language counts, date range, mean rating and reply rate.

    Args:
        df: Output of :func:`load_raw`.

    Returns:
        Summary table.
    """
    return (
        df.assign(lang_query=df["lang_query"].fillna("n/a"))
        .groupby(["source", "app", "lang_query"])
        .agg(
            reviews=("review_id", "size"),
            first_review=("review_date", "min"),
            last_review=("review_date", "max"),
            mean_rating=("rating", "mean"),
            reply_rate=("has_reply", "mean"),
            empty_text=("text", lambda s: (s.str.strip() == "").sum()),
        )
        .round(3)
    )
