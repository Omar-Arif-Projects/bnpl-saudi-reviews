"""Collect Google Play reviews for Tabby and Tamara (Saudi Arabia, Arabic + English).

Privacy: only whitelisted, non-personal fields are kept. Reviewer names, profile
images and any user identifiers are dropped the moment a batch is received and
are never written to disk.

Incremental: each run loads the reviews already saved in ``data/raw`` and pages
through the newest reviews only until it reaches dates it has already covered.
New reviews are written to ``data/raw/play_reviews_<app>_<YYYYMMDD>.parquet``.

Usage::

    python -m src.collect.play_store                      # both apps, all languages
    python -m src.collect.play_store --apps tabby --max-reviews 500   # quick test
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from google_play_scraper import Sort, reviews

from src.config import get_logger, get_path, load_config

logger = get_logger("collect.play_store")

SOURCE = "google_play"

# Final column order of every raw Play Store parquet file.
COLUMNS: list[str] = [
    "review_id",
    "app",
    "source",
    "country",
    "lang_query",
    "text",
    "rating",
    "review_date",
    "thumbs_up",
    "app_version",
    "has_reply",
    "reply_date",
    "collected_at",
]

# Overlap window when resuming, to catch reviews posted around the last run.
RESUME_OVERLAP = timedelta(days=1)


def to_record(raw: dict[str, Any], app: str, lang: str, country: str, collected_at: datetime) -> dict[str, Any]:
    """Convert one google-play-scraper review into a privacy-safe record.

    Only whitelisted fields are copied, so ``userName`` and ``userImage`` are never kept.
    The developer reply text is not stored either: only whether a reply exists and when.

    Args:
        raw: Review dict as returned by ``google_play_scraper.reviews``.
        app: Project app key (e.g. ``"tabby"``).
        lang: Language code used in the query.
        country: Country code used in the query.
        collected_at: Timestamp of this collection run (UTC).

    Returns:
        A dict with exactly the keys in ``COLUMNS``.
    """
    replied_at = raw.get("repliedAt")
    return {
        "review_id": raw["reviewId"],
        "app": app,
        "source": SOURCE,
        "country": country,
        "lang_query": lang,
        "text": raw.get("content") or "",
        "rating": int(raw["score"]) if raw.get("score") is not None else None,
        "review_date": raw.get("at"),
        "thumbs_up": int(raw.get("thumbsUpCount") or 0),
        "app_version": raw.get("reviewCreatedVersion") or raw.get("appVersion"),
        "has_reply": bool(raw.get("replyContent")),
        "reply_date": replied_at,
        "collected_at": collected_at,
    }


def _as_utc(value: Any) -> pd.Timestamp | None:
    """Convert a datetime-like value to a UTC Timestamp (naive values are assumed UTC).

    Args:
        value: datetime, Timestamp, string or None.

    Returns:
        UTC Timestamp, or None if ``value`` is empty.
    """
    if value is None or pd.isna(value):
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a DataFrame with a stable schema and UTC timestamps.

    Args:
        records: Records produced by :func:`to_record`.

    Returns:
        DataFrame with columns ``COLUMNS``.
    """
    df = pd.DataFrame.from_records(records, columns=COLUMNS)
    for col in ("review_date", "reply_date", "collected_at"):
        # google-play-scraper returns naive datetimes in UTC.
        df[col] = pd.to_datetime(df[col], utc=True)
    df["rating"] = df["rating"].astype("Int8")
    df["thumbs_up"] = df["thumbs_up"].astype("int64")
    df["has_reply"] = df["has_reply"].astype(bool)
    return df


def load_existing(app: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """Load every raw Play Store file already collected for an app.

    Args:
        app: Project app key.
        raw_dir: Directory holding raw parquet files (defaults to config ``paths.raw``).

    Returns:
        Concatenated, de-duplicated DataFrame (empty with ``COLUMNS`` if nothing exists).
    """
    raw_dir = raw_dir or get_path("raw")
    files = sorted(raw_dir.glob(f"play_reviews_{app}_*.parquet"))
    if not files:
        return to_frame([])
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.drop_duplicates("review_id", keep="first").reset_index(drop=True)


def fetch_batches(
    app_id: str,
    lang: str,
    country: str,
    batch_size: int,
    sleep_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
) -> Iterator[list[dict[str, Any]]]:
    """Yield batches of reviews, newest first, until Google Play has no more pages.

    Args:
        app_id: Google Play package ID.
        lang: Review language code.
        country: Store country code.
        batch_size: Reviews per request.
        sleep_seconds: Delay between requests (rate limiting).
        max_retries: Retries per request before giving up.
        retry_backoff_seconds: Base wait between retries (multiplied by attempt number).

    Yields:
        Lists of raw review dicts.
    """
    token = None
    while True:
        for attempt in range(1, max_retries + 1):
            try:
                batch, token = reviews(
                    app_id,
                    lang=lang,
                    country=country,
                    sort=Sort.NEWEST,
                    count=batch_size,
                    continuation_token=token,
                )
                break
            except Exception as exc:  # network errors, throttling, parse errors
                if attempt == max_retries:
                    logger.error("Giving up on %s/%s after %d attempts: %s", app_id, lang, attempt, exc)
                    return
                wait = retry_backoff_seconds * attempt
                logger.warning("Request failed (%s); retry %d/%d in %.0fs", exc, attempt, max_retries, wait)
                time.sleep(wait)
        if not batch:
            return
        yield batch
        if token is None or getattr(token, "token", None) is None:
            return
        time.sleep(sleep_seconds)


def collect_app_lang(
    app: str,
    app_id: str,
    lang: str,
    known_ids: set[str],
    resume_from: pd.Timestamp | None,
    max_reviews: int | None,
    collected_at: datetime,
) -> list[dict[str, Any]]:
    """Collect new reviews for one app and one language.

    Stops when reviews become older than ``resume_from`` (incremental mode),
    when ``max_reviews`` new reviews are collected, or when pages run out.

    Args:
        app: Project app key.
        app_id: Google Play package ID.
        lang: Review language code.
        known_ids: Review IDs already stored (updated in place with new IDs).
        resume_from: Only reviews on or after this time are needed; None = full history.
        max_reviews: Optional cap on new reviews (for testing).
        collected_at: Timestamp of this collection run.

    Returns:
        List of new privacy-safe records.
    """
    cfg = load_config()["collection"]
    ps = cfg["play_store"]
    country = cfg["country"]
    new: list[dict[str, Any]] = []
    seen_total = 0

    for batch in fetch_batches(
        app_id, lang, country, ps["batch_size"], ps["sleep_seconds"], ps["max_retries"], ps["retry_backoff_seconds"]
    ):
        reached_old = False
        for raw in batch:
            record = to_record(raw, app, lang, country, collected_at)  # names/images dropped here
            seen_total += 1
            review_date = _as_utc(record["review_date"])
            if resume_from is not None and review_date is not None and review_date < resume_from:
                reached_old = True
                break
            if record["review_id"] in known_ids:
                continue
            known_ids.add(record["review_id"])
            new.append(record)
            if max_reviews is not None and len(new) >= max_reviews:
                reached_old = True
                break
        logger.info("%s/%s: scanned %d, new %d", app, lang, seen_total, len(new))
        if reached_old:
            break
    return new


def save_new(app: str, new_df: pd.DataFrame, run_date: str, raw_dir: Path | None = None) -> Path | None:
    """Write new reviews to ``play_reviews_<app>_<run_date>.parquet``, merging a same-day file.

    Args:
        app: Project app key.
        new_df: New reviews to save.
        run_date: Date stamp ``YYYYMMDD`` for the file name.
        raw_dir: Output directory (defaults to config ``paths.raw``).

    Returns:
        Path written, or None if there was nothing new.
    """
    if new_df.empty:
        return None
    raw_dir = raw_dir or get_path("raw")
    out = raw_dir / f"play_reviews_{app}_{run_date}.parquet"
    if out.exists():
        new_df = pd.concat([pd.read_parquet(out), new_df], ignore_index=True)
        new_df = new_df.drop_duplicates("review_id", keep="first")
    new_df.to_parquet(out, index=False)
    return out


def run(
    apps: list[str] | None = None,
    langs: list[str] | None = None,
    max_reviews: int | None = None,
    full: bool = False,
) -> None:
    """Run incremental collection for the given apps and languages.

    Args:
        apps: App keys from config (default: all).
        langs: Language codes (default: ``collection.languages``).
        max_reviews: Cap on new reviews per app/language (default: config value).
        full: Ignore the resume date and page through the full history
            (already stored IDs are still skipped). Use after a capped test run.
    """
    cfg = load_config()
    apps = apps or list(cfg["apps"])
    langs = langs or cfg["collection"]["languages"]
    if max_reviews is None:
        max_reviews = cfg["collection"]["play_store"]["max_reviews_per_lang"]
    collected_at = datetime.now(timezone.utc)
    run_date = collected_at.strftime("%Y%m%d")

    for app in apps:
        app_id = cfg["apps"][app]["play_store_id"]
        existing = load_existing(app)
        known_ids: set[str] = set(existing["review_id"])
        logger.info("%s (%s): %d reviews already stored", app, app_id, len(existing))

        all_new: list[dict[str, Any]] = []
        for lang in langs:
            prev = existing.loc[existing["lang_query"] == lang, "review_date"]
            resume_from = None if (full or prev.empty) else prev.max() - RESUME_OVERLAP
            logger.info("%s/%s: %s", app, lang, f"resuming from {resume_from}" if resume_from is not None else "full history")
            all_new += collect_app_lang(app, app_id, lang, known_ids, resume_from, max_reviews, collected_at)

        out = save_new(app, to_frame(all_new), run_date)
        logger.info("%s: %d new reviews%s", app, len(all_new), f" -> {out.name}" if out else "")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    cfg = load_config()
    p = argparse.ArgumentParser(description="Incrementally collect Google Play reviews.")
    p.add_argument("--apps", nargs="+", choices=list(cfg["apps"]), help="Apps to collect (default: all)")
    p.add_argument("--langs", nargs="+", help="Language codes (default: from config)")
    p.add_argument("--max-reviews", type=int, default=None, help="Cap new reviews per app/language (testing)")
    p.add_argument("--full", action="store_true", help="Scan full history instead of resuming (skips known IDs)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.apps, args.langs, args.max_reviews, args.full)
