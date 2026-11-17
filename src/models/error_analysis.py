"""Error analysis of sentiment models on the gold set.

All functions take the output of :func:`src.models.evaluate.gold_frame`:
usable gold rows with hand labels (``sentiment``), the star-based label
(``star_sentiment``) and ``<model>_pred`` columns. A model is only judged on
rows it scores (CAMeLBERT scores Arabic reviews only).

The gold set has a few hundred rows and each model makes a few dozen errors,
so the tables report counts next to rates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.models.evaluate import model_keys


def length_group(texts: pd.Series, bins: list[int] | None = None) -> pd.Series:
    """Group reviews by length in words.

    Args:
        texts: Review texts.
        bins: Upper bounds of the groups except the last, e.g. ``[3, 10, 30]``
            gives ``1-3``, ``4-10``, ``11-30`` and ``31+``. Defaults to
            ``models.error_analysis.length_bins``.

    Returns:
        Ordered categorical Series of group labels.
    """
    bins = bins or load_config()["models"]["error_analysis"]["length_bins"]
    edges = [0, *bins, np.inf]
    labels = [f"{lo + 1}-{hi}" for lo, hi in zip(edges[:-2], edges[1:-1])] + [f"{bins[-1] + 1}+"]
    n_words = texts.fillna("").str.split().str.len()
    return pd.cut(n_words, edges, labels=labels)


def error_table(df: pd.DataFrame, by: str, label_col: str = "sentiment") -> pd.DataFrame:
    """Count errors per model within groups of reviews.

    Args:
        df: Gold rows with predictions.
        by: Column to group by (e.g. ``sentiment``, ``lang``, ``length``).
        label_col: Column with true labels.

    Returns:
        One row per model and group with ``n`` (rows the model scores),
        ``errors`` and ``error_rate``. A categorical ``by`` column keeps its
        category order.
    """
    rows = []
    for key in model_keys(df):
        scored = df[df[f"{key}_pred"].notna()]
        wrong = scored[f"{key}_pred"] != scored[label_col]
        for group, mask in wrong.groupby(scored[by], observed=True):
            rows.append({
                "model": key, by: group, "n": len(mask), "errors": int(mask.sum()),
                "error_rate": float(mask.mean()),
            })
    table = pd.DataFrame(rows)
    if isinstance(df[by].dtype, pd.CategoricalDtype) and not table.empty:
        table[by] = pd.Categorical(table[by], categories=df[by].cat.categories, ordered=True)
    return table


def star_vs_hand(df: pd.DataFrame, label_col: str = "sentiment") -> pd.DataFrame:
    """Compare predictions with hand and star labels where the two disagree.

    A model trained on star labels could learn to repeat the star rating
    instead of reading the text. On rows where the hand label differs from
    the star-based label, this counts how often each model agrees with
    either one.

    Args:
        df: Gold rows with predictions and ``star_sentiment``.
        label_col: Column with hand labels.

    Returns:
        One row per model with ``n`` and the counts ``matches_hand``,
        ``matches_star`` and ``matches_neither``.
    """
    differ = df[df["star_sentiment"] != df[label_col]]
    rows = []
    for key in model_keys(df):
        part = differ[differ[f"{key}_pred"].notna()]
        pred = part[f"{key}_pred"]
        hand, star = pred == part[label_col], pred == part["star_sentiment"]
        rows.append({
            "model": key, "n": len(part), "matches_hand": int(hand.sum()),
            "matches_star": int(star.sum()), "matches_neither": int((~hand & ~star).sum()),
        })
    return pd.DataFrame(rows)


def wrong_for_all(df: pd.DataFrame, label_col: str = "sentiment") -> pd.DataFrame:
    """Return rows that every model scoring them gets wrong.

    Args:
        df: Gold rows with predictions.
        label_col: Column with true labels.

    Returns:
        Rows scored by at least one model and misclassified by all models
        that score them.
    """
    keys = model_keys(df)
    scored = np.zeros(len(df), dtype=bool)
    all_wrong = np.ones(len(df), dtype=bool)
    for key in keys:
        pred = df[f"{key}_pred"]
        scored |= pred.notna().to_numpy()
        all_wrong &= (pred.isna() | (pred != df[label_col])).to_numpy()
    return df[scored & all_wrong]


def predictions_for(df: pd.DataFrame, label: str, label_col: str = "sentiment") -> pd.DataFrame:
    """Count each model's predictions for reviews with one true label.

    Args:
        df: Gold rows with predictions.
        label: True label to select (e.g. ``"neutral"``).
        label_col: Column with true labels.

    Returns:
        Table with one row per model and one column per predicted class.
    """
    part = df[df[label_col] == label]
    classes = load_config()["labels"]["sentiment_classes"]
    counts = {
        key: part[f"{key}_pred"].dropna().value_counts().reindex(classes, fill_value=0)
        for key in model_keys(df)
    }
    return pd.DataFrame(counts).T.rename_axis("model")
