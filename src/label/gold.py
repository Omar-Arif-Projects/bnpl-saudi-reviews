"""Build and manage the hand-labeled gold set.

The gold set is a stratified random sample of clean reviews (app x language x
star group) that is labeled by hand for sentiment and main topic. All final
model metrics are reported on it.

Privacy:

- Review IDs are replaced by a truncated SHA-256 hash (``gold_id``), which is
  enough to exclude gold reviews from training data.
- E-mail addresses, URLs and long digit sequences (phone, order or card
  numbers) are masked in the stored text.
- Rows flagged ``personal_info`` have their text removed when saved.

File columns (``paths.gold``):

``gold_id, app, lang, month, rating, star_sentiment, stratum, weight, text,
sentiment, topic, flag``

``sentiment``, ``topic`` and ``flag`` are empty until labeled. ``rating`` and
``star_sentiment`` are kept for analysis but hidden in the labeling tool.

Consistency check: once every row is labeled, ``--relabel`` copies a random
set of usable rows, with empty labels, to ``paths.gold_relabel``. Labeling
them again without seeing the first labels gives the agreement reported by
:mod:`src.models.evaluate`.

Usage::

    python -m src.label.gold --sample          # create the unlabeled sample
    python -m src.label.gold --sample --force  # replace it, discarding labels
    python -m src.label.gold --relabel         # create the relabel sample
    python -m src.label.gold                   # print labeling progress
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_logger, get_path, load_config

logger = get_logger("label.gold")

COLUMNS = [
    "gold_id", "app", "lang", "month", "rating", "star_sentiment", "stratum", "weight",
    "text", "sentiment", "topic", "flag",
]
LABEL_COLUMNS = ["sentiment", "topic", "flag"]
PII_FLAG = "personal_info"

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
# Seven or more digits, optionally separated by spaces or dashes. ``\d``
# also matches Arabic-Indic digits.
LONG_NUMBER = re.compile(r"\+?\d(?:[\s-]?\d){6,}")


def gold_id(review_id: str, length: int | None = None) -> str:
    """Return the hashed identifier of a review.

    Args:
        review_id: Original review ID.
        length: Number of hex characters to keep; defaults to
            ``gold.id_hash_length``.

    Returns:
        Truncated SHA-256 hex digest.
    """
    length = length or load_config()["gold"]["id_hash_length"]
    return hashlib.sha256(str(review_id).encode("utf-8")).hexdigest()[:length]


def scrub(text: str | None) -> str:
    """Mask e-mail addresses, URLs and long digit sequences.

    Args:
        text: Review text.

    Returns:
        Text with ``[EMAIL]``, ``[URL]`` and ``[NUMBER]`` placeholders.
    """
    if not text:
        return ""
    text = EMAIL.sub("[EMAIL]", text)
    text = URL.sub("[URL]", text)
    return LONG_NUMBER.sub("[NUMBER]", text)


def allocate(n_total: int, shares: dict[str, float], n_cells: int) -> dict[str, int]:
    """Split the sample size across star groups, per app x language cell.

    Args:
        n_total: Total sample size.
        shares: Share of the sample per star group; must sum to 1.
        n_cells: Number of app x language cells.

    Returns:
        Sample size per star group within one cell. Rounding remainders go to
        the groups with the largest fractional parts.

    Raises:
        ValueError: If the shares do not sum to 1.
    """
    if not np.isclose(sum(shares.values()), 1.0):
        raise ValueError(f"star_group_shares must sum to 1, got {sum(shares.values())}")
    per_cell = n_total / n_cells
    exact = {group: per_cell * share for group, share in shares.items()}
    counts = {group: int(np.floor(value)) for group, value in exact.items()}
    remainder = round(per_cell) - sum(counts.values())
    for group in sorted(exact, key=lambda g: exact[g] - counts[g], reverse=True)[:remainder]:
        counts[group] += 1
    return counts


def candidate_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Select reviews eligible for the gold set.

    Args:
        df: Clean dataset (``paths.reviews_clean``).

    Returns:
        Trainable reviews from the configured source and languages, with
        ``star_sentiment`` and ``stratum`` columns.
    """
    cfg = load_config()
    g = cfg["gold"]
    star_map = {int(k): v for k, v in cfg["labels"]["star_to_sentiment"].items()}
    pool = df[(df["source"] == g["source"]) & df["trainable"] & df["lang"].isin(g["languages"])].copy()
    pool["star_sentiment"] = pool["rating"].astype(int).map(star_map)
    pool["stratum"] = pool["app"] + "|" + pool["lang"] + "|" + pool["star_sentiment"]
    return pool


def draw_sample(pool: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Draw the stratified sample and compute population weights.

    Args:
        pool: Output of :func:`candidate_pool`.
        seed: Random seed.

    Returns:
        Sample in :data:`COLUMNS` order, shuffled, with empty label columns.
        ``weight`` is the stratum population divided by the stratum sample size.

    Raises:
        ValueError: If a stratum has fewer reviews than requested.
    """
    g = load_config()["gold"]
    cells = pool[["app", "lang"]].drop_duplicates()
    per_group = allocate(g["n_total"], g["star_group_shares"], len(cells))

    parts = []
    for stratum, rows in pool.groupby("stratum", sort=True):
        n = per_group[stratum.split("|")[-1]]
        if len(rows) < n:
            raise ValueError(f"Stratum {stratum} has {len(rows)} reviews, {n} requested")
        part = rows.sample(n=n, random_state=seed)
        part["weight"] = len(rows) / n
        parts.append(part)

    sample = pd.concat(parts).sample(frac=1, random_state=seed)
    sample["gold_id"] = sample["review_id"].map(gold_id)
    sample["month"] = sample["review_date"].dt.strftime("%Y-%m")
    sample["text"] = sample["text"].map(scrub)
    for col in LABEL_COLUMNS:
        sample[col] = ""
    return sample[COLUMNS].reset_index(drop=True)


def load_gold(path: Path | None = None) -> pd.DataFrame:
    """Read the gold file with label columns as strings.

    Args:
        path: Gold CSV; defaults to ``paths.gold``.

    Returns:
        Gold DataFrame. Empty label cells are ``""``.
    """
    path = path or get_path("gold")
    df = pd.read_csv(path, dtype={"gold_id": str, "month": str}, keep_default_na=False)
    for col in ["text", *LABEL_COLUMNS]:
        df[col] = df[col].astype(str)
    return df


def save_gold(df: pd.DataFrame, path: Path | None = None) -> None:
    """Write the gold file atomically, removing text of personal-info rows.

    Args:
        df: Gold DataFrame.
        path: Output CSV; defaults to ``paths.gold``.
    """
    path = path or get_path("gold")
    out = df[COLUMNS].copy()
    out.loc[out["flag"] == PII_FLAG, "text"] = ""
    tmp = path.with_suffix(".tmp")
    out.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)


def is_done(df: pd.DataFrame) -> pd.Series:
    """Return a mask of rows that are labeled or flagged.

    Args:
        df: Gold DataFrame.

    Returns:
        Boolean Series; a row is done when it has sentiment and topic, or a flag.
    """
    return ((df["sentiment"] != "") & (df["topic"] != "")) | (df["flag"] != "")


def usable(df: pd.DataFrame) -> pd.DataFrame:
    """Return labeled, unflagged rows for evaluation.

    Args:
        df: Gold DataFrame.

    Returns:
        Rows with sentiment and topic and no flag.
    """
    return df[(df["sentiment"] != "") & (df["topic"] != "") & (df["flag"] == "")].copy()


def exclude_gold(df: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Remove gold reviews from a dataset, e.g. before training.

    Args:
        df: DataFrame with a ``review_id`` column.
        gold: Gold DataFrame.

    Returns:
        ``df`` without rows whose hashed review ID is in the gold set.
    """
    ids = set(gold["gold_id"])
    return df[~df["review_id"].map(gold_id).isin(ids)].copy()


def progress(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize labeling progress per stratum.

    Args:
        df: Gold DataFrame.

    Returns:
        Table with ``total`` and ``done`` per stratum, plus a total row.
    """
    table = (
        df.assign(done=is_done(df))
        .groupby("stratum")
        .agg(total=("gold_id", "size"), done=("done", "sum"))
    )
    table.loc["all"] = table.sum()
    return table


def create_sample(force: bool = False) -> pd.DataFrame:
    """Create the unlabeled gold sample file.

    Args:
        force: Replace an existing file even if it contains labels.

    Returns:
        The saved sample.

    Raises:
        FileExistsError: If the file exists and ``force`` is False.
    """
    path = get_path("gold")
    if path.exists() and not force:
        existing = load_gold(path)
        raise FileExistsError(
            f"{path} exists with {int(is_done(existing).sum())} labeled rows; use --force to replace it"
        )
    seed = load_config()["project"]["random_seed"]
    pool = candidate_pool(pd.read_parquet(get_path("reviews_clean")))
    sample = draw_sample(pool, seed)
    get_path("labeled")
    save_gold(sample, path)
    logger.info("Saved %d unlabeled reviews from a pool of %d -> %s", len(sample), len(pool), path)
    logger.info("Per stratum:\n%s", sample["stratum"].value_counts().sort_index().to_string())
    return sample


def create_relabel_sample(gold_path: Path | None = None, out_path: Path | None = None) -> pd.DataFrame:
    """Copy a random set of usable gold rows, with empty labels, for a second labeling.

    Args:
        gold_path: Gold CSV; defaults to ``paths.gold``.
        out_path: Output CSV; defaults to ``paths.gold_relabel``.

    Returns:
        The saved relabel sample (``gold.relabel_n`` rows, or all usable rows
        if there are fewer), in random order.

    Raises:
        RuntimeError: If some gold rows are not labeled yet.
        FileExistsError: If the relabel file already exists.
    """
    out_path = out_path or get_path("gold_relabel")
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; delete it to draw a new relabel sample")
    gold = load_gold(gold_path)
    if not is_done(gold).all():
        raise RuntimeError(f"{int((~is_done(gold)).sum())} gold rows are not labeled yet")
    cfg = load_config()
    pool = usable(gold)
    sample = pool.sample(n=min(cfg["gold"]["relabel_n"], len(pool)), random_state=cfg["project"]["random_seed"])
    for col in LABEL_COLUMNS:
        sample[col] = ""
    sample = sample.reset_index(drop=True)
    save_gold(sample, out_path)
    logger.info("Saved %d rows for relabeling -> %s", len(sample), out_path)
    return sample


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", action="store_true", help="create the unlabeled sample")
    parser.add_argument("--force", action="store_true", help="replace an existing gold file")
    parser.add_argument("--relabel", action="store_true", help="create the relabel sample")
    args = parser.parse_args()
    if args.sample:
        create_sample(force=args.force)
    elif args.relabel:
        create_relabel_sample()
    else:
        print(progress(load_gold()).to_string())


if __name__ == "__main__":
    main()
