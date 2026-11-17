"""Build the clean review dataset from the raw collection files.

Steps:

1. Load all raw Parquet files (Google Play and App Store) and convert Google
   Play times stored as local time to UTC (see
   :func:`src.collect.play_store.fix_legacy_times`).
2. Keep the latest copy of each review (reviews read again by a later
   collection run are stored again).
3. Drop double submissions (same source, app, text, rating and timestamp).
   Identical text from different reviews is retained.
4. Drop empty reviews.
5. Add normalized text, emoji, language and length features.
6. Drop reviews written only in scripts other than Arabic and Latin.
7. Drop reviews with fewer than ``min_letters`` Arabic/Latin letters and no emoji.
8. Set ``trainable = False`` for languages listed in ``exclude_from_training``.

Outputs (paths from ``config.yaml``):

- ``paths.reviews_clean``: cleaned dataset.
- ``paths.cleaning_log``: rows removed per step.

Usage::

    python -m src.clean.pipeline
"""

from __future__ import annotations

import re

import pandas as pd

from src.clean.language import LanguageRules, detect_language, has_other_script
from src.clean.normalize import extract_emojis, normalize_text, strip_emojis
from src.collect.play_store import fix_legacy_times, latest_copies
from src.config import get_logger, get_path, load_config

logger = get_logger("clean.pipeline")

LETTER = re.compile(r"[a-zء-يٱ-ۓ]")
WORD = re.compile(r"[a-z0-9'ء-يٱ-ۓ]+")


class RemovalLog:
    """Record row counts before and after each cleaning step."""

    def __init__(self, start_rows: int) -> None:
        """Initialize the log.

        Args:
            start_rows: Number of rows loaded.
        """
        self.rows: list[dict[str, object]] = [
            {"step": "loaded", "reason": "all raw files", "removed": 0, "rows_after": start_rows}
        ]

    def record(self, step: str, reason: str, before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
        """Record one step.

        Args:
            step: Step name.
            reason: Description of the removal criterion.
            before: DataFrame before the step.
            after: DataFrame after the step.

        Returns:
            ``after``, unchanged.
        """
        removed = len(before) - len(after)
        self.rows.append({"step": step, "reason": reason, "removed": removed, "rows_after": len(after)})
        logger.info("%-22s removed %7d  -> %d rows", step, removed, len(after))
        return after

    def to_frame(self) -> pd.DataFrame:
        """Return the log as a DataFrame with one row per step."""
        return pd.DataFrame(self.rows)


def load_raw_files() -> pd.DataFrame:
    """Concatenate all raw review files, with all times in UTC.

    Returns:
        Combined raw DataFrame.

    Raises:
        FileNotFoundError: If no raw review files exist.
    """
    files = sorted(get_path("raw").glob("*_reviews_*.parquet"))
    if not files:
        raise FileNotFoundError("No raw review files found; run src.collect.play_store first.")
    logger.info("Loading %d raw files", len(files))
    return fix_legacy_times(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True))


def add_text_features(df: pd.DataFrame, rules: LanguageRules, repeat_max: int) -> pd.DataFrame:
    """Add normalized text, emoji, language and length columns.

    Args:
        df: Reviews with a ``text`` column.
        rules: Language tagging rules.
        repeat_max: Maximum run length of a repeated character.

    Returns:
        Copy of ``df`` with ``text_norm``, ``emojis``, ``n_emoji``, ``lang``,
        ``n_letters``, ``n_chars`` and ``n_words``.
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
    """Apply all cleaning steps.

    Args:
        raw: Output of :func:`load_raw_files`.

    Returns:
        Tuple of the cleaned DataFrame and the removal log.
    """
    cfg = load_config()["cleaning"]
    rules = LanguageRules.from_config(cfg["language"])
    log = RemovalLog(len(raw))

    df = log.record(
        "older_copy", "earlier copy of a review collected again (latest kept)",
        raw, latest_copies(raw, ["source", "review_id"]),
    )
    df = log.record(
        "double_submission", "same source, app, text, rating and timestamp",
        df, df.drop_duplicates(["source", "app", "text", "rating", "review_date"]),
    )
    df = log.record("empty", "empty text", df, df[df["text"].fillna("").str.strip() != ""])

    df = add_text_features(df, rules, int(cfg["repeat_max"]))
    other_script = (df["lang"] == "other") & df["text"].map(has_other_script)
    df = log.record(
        "other_script", "only letters of scripts other than Arabic and Latin (e.g. Bengali)",
        df, df[~other_script],
    )
    has_content = ((df["n_letters"] >= int(cfg["min_letters"])) | (df["n_emoji"] > 0)) & (
        df["lang"] != "other"
    )
    df = log.record(
        "no_content",
        f"fewer than {cfg['min_letters']} Arabic/Latin letters and no emoji",
        df, df[has_content],
    )

    df["is_emoji_only"] = df["lang"] == "emoji_only"
    df["trainable"] = ~df["lang"].isin(cfg["exclude_from_training"])
    df = df.sort_values(["app", "source", "review_date"]).reset_index(drop=True)

    logger.info("Kept %d rows; %d not trainable", len(df), int((~df["trainable"]).sum()))
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
