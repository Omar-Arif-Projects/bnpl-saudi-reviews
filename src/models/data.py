"""Model datasets: star-labeled training data, dev split and comparison sample.

Labels come from star ratings (``labels.star_to_sentiment``). Reviews in the
gold set are always removed, so the gold set stays an unseen test set.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import get_path, load_config
from src.label.gold import exclude_gold, load_gold

KEEP = ["review_id", "app", "lang", "rating", "review_date", "text", "text_norm"]


def load_star_labeled() -> pd.DataFrame:
    """Load trainable reviews with star-based sentiment labels, gold set excluded.

    Returns:
        DataFrame with :data:`KEEP` columns plus ``label``.
    """
    cfg = load_config()
    m = cfg["models"]
    star_map = {int(k): v for k, v in cfg["labels"]["star_to_sentiment"].items()}
    df = pd.read_parquet(get_path("reviews_clean"), columns=[*KEEP, "source", "trainable"])
    df = df[(df["source"] == m["source"]) & df["trainable"] & df["lang"].isin(m["languages"])]
    gold_path = get_path("gold")
    if gold_path.exists():
        df = exclude_gold(df, load_gold(gold_path))
    df = df[KEEP].copy()
    df["label"] = df["rating"].astype(int).map(star_map)
    return df.reset_index(drop=True)


def split_dev(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train and dev sets, stratified by label and language.

    Args:
        df: Output of :func:`load_star_labeled`.
        seed: Random seed.

    Returns:
        ``(train, dev)``.
    """
    strata = df["label"] + "|" + df["lang"]
    train, dev = train_test_split(
        df, test_size=load_config()["models"]["dev_share"], stratify=strata, random_state=seed,
    )
    return train.reset_index(drop=True), dev.reset_index(drop=True)


def comparison_sample(dev: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Draw a balanced sample from the dev set for comparing all models.

    Args:
        dev: Dev split.
        seed: Random seed.

    Returns:
        Up to ``compare_per_cell`` reviews per language x label cell.
    """
    m = load_config()["models"]
    pool = dev[dev["lang"].isin(m["compare_languages"])]
    parts = [
        rows.sample(n=min(len(rows), m["compare_per_cell"]), random_state=seed)
        for _, rows in pool.groupby(["lang", "label"], sort=True)
    ]
    return pd.concat(parts).reset_index(drop=True)
