"""Build the clean review dataset from the raw collection files.

Steps (each removal is counted and written to the cleaning log):
1. Load every raw Parquet file (Google Play and App Store).
2. Drop repeated review IDs (the same review collected twice).
3. Drop double submissions: same source, app, text, rating and timestamp.
   Repeated *text* alone is NOT a duplicate: thousands of different people
   write "جيد" or "👍", and each is a real opinion.
4. Drop empty reviews.
5. Normalize text, extract emoji, tag language, count characters and words.
6. Drop reviews with no usable content: fewer than ``min_letters`` letters and
   no emoji (e.g. ".", "1", a single letter).
7. Flag rows for model training: languages in ``exclude_from_training`` (emoji-only
   reviews) stay in the dataset for EDA but get ``trainable = False``.

Outputs (paths from ``config.yaml``):
- ``paths.reviews_clean``: the clean dataset (gitignored).
- ``paths.cleaning_log``: one row per step with counts (committed).

Usage::

    python -m src.clean.pipeline
"""

from __future__ import annotations

import re

import pandas as pd

from src.clean.language import LanguageRules, detect_language
from src.clean.normalize import extract_emojis, normalize_text, strip_emojis
from src.config import get_logger, get_path, load_config

logger = get_logger("clean.pipeline")

LETTER = re.compile(r"[a-zء-يٱ-ۓ]")
WORD = re.compile(r"[a-z0-9'ء-يٱ-ۓ]+")


class RemovalLog:
    """Collect row counts before and after each cleaning step."""

    def __init__(self, start_rows: int) -> None:
        """Start the log.

        Args:
            start_rows: Number of rows loaded.
        """
        self.rows: list[dict[str, object]] = [
            {"step": "loaded", "reason": "all raw files", "removed": 0, "rows_after": start_rows}
        ]

    def record(self, step: str, reason: str, before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
        """Record one step and return the filtered frame.

        Args:
            step: Short step name.
            reason: Plain-language reason for the removal.
            before: Frame before the step.
            after: Frame after the step.

        Returns:
            ``after``, so calls can be chained.
        """
        removed = len(before) - len(after)
        self.rows.append({"step": step, "reason": reason, "removed": removed, "rows_after": len(after)})
        logger.info("%-22s removed %7d  -> %d rows", step, removed, len(after))
        return after

    def to_frame(self) -> pd.DataFrame:
        """Return the log as a DataFrame.

        Returns:
            One row per step.
        """
        return pd.DataFrame(self.rows)


def load_raw_files() -> pd.DataFrame:
    """Concatenate all raw review files without dropping anything.

    Returns:
        Combined raw DataFrame.

    Raises:
        FileNotFoundError: If ``data/raw`` has no review files.
    """
    files = sorted(get_path("raw").glob("*_reviews_*.parquet"))
    if not files:
        raise FileNotFoundError("No raw review files; run src.collect.play_store first.")
    logger.info("Loading %d raw files", len(files))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def add_text_features(df: pd.DataFrame, rules: LanguageRules, repeat_max: int) -> pd.DataFrame:
    """Add normalized text, emoji, language and length columns.

    Args:
        df: Reviews with a ``text`` column.
        rules: Language-tagging rules.
        repeat_max: Longest run of a repeated character to keep.

    Returns:
        Copy of ``df`` with ``text_norm, emojis, n_emoji, lang, n_letters,
        n_chars, n_words``.
    """
    out = df.copy()
    out["text_norm"] = out["text"].map(lambda t: normalize_text(t, repeat_max))
    emojis = out["text"].map(extract_emojis)
    out["emojis"] = emojis.map("".join)
    out["n_emoji"] = emojis.map(len).astype("int32")
    out["lang"] = out["text"].map(lambda t: detect_language(t, rules))
    words_only = out["text_norm"].map(strip_emojis)
    out["n_letters"] = words_only.map(lambda t: len(LETTER.findall(t))).astype("int32")
    out["n_chars"] = out["text"].str.strip().str.len().astype("int32")
    out["n_words"] = words_only.map(lambda t: len(WORD.findall(t))).astype("int32")
    return out


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all cleaning steps on the raw reviews.

    Args:
        raw: Output of :func:`load_raw_files`.

    Returns:
        ``(clean_df, log_df)``.
    """
    cfg = load_config()["cleaning"]
    rules = LanguageRules.from_config(cfg["language"])
    log = RemovalLog(len(raw))

    df = log.record(
        "duplicate_id", "same review collected twice (source + review_id)",
        raw, raw.drop_duplicates(["source", "review_id"]),
    )
    df = log.record(
        "double_submission", "same app, text, rating and timestamp",
        df, df.drop_duplicates(["source", "app", "text", "rating", "review_date"]),
    )
    df = log.record("empty", "no text at all", df, df[df["text"].fillna("").str.strip() != ""])

    df = add_text_features(df, rules, int(cfg["repeat_max"]))
    has_content = ((df["n_letters"] >= int(cfg["min_letters"])) | (df["n_emoji"] > 0)) & (
        df["lang"] != "other"
    )
    df = log.record(
        "no_content",
        f"fewer than {cfg['min_letters']} Arabic/Latin letters and no emoji "
        "(punctuation, numbers, single letters, other scripts)",
        df, df[has_content],
    )

    df["is_emoji_only"] = df["lang"] == "emoji_only"
    df["trainable"] = ~df["lang"].isin(cfg["exclude_from_training"])
    df = df.sort_values(["app", "source", "review_date"]).reset_index(drop=True)

    logger.info(
        "Kept %d rows; %d flagged not trainable (%s)",
        len(df), int((~df["trainable"]).sum()), ", ".join(cfg["exclude_from_training"]),
    )
    return df, log.to_frame()


def run() -> None:
    """Clean the raw reviews and write the dataset and the removal log."""
    df, log_df = clean(load_raw_files())

    get_path("processed")
    out = get_path("reviews_clean")
    df.to_parquet(out, index=False)
    logger.info("Saved %d rows x %d columns -> %s", *df.shape, out)

    log_path = get_path("cleaning_log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_path, index=False, encoding="utf-8")
    logger.info("Saved cleaning log -> %s", log_path)
    logger.info("Language mix:\n%s", df["lang"].value_counts().to_string())


if __name__ == "__main__":
    run()
