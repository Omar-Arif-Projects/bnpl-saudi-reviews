"""Collect Google Play reviews for Tabby and Tamara (Arabic and English queries).

Google Play returns the same reviews whatever store country is requested, so
the data covers users in every country where the apps are used. The
``country`` column records the store country passed to the scraper.

Privacy: only whitelisted, non-personal fields are kept. Reviewer names, profile
images and any user identifiers are dropped the moment a batch is received and
are never written to disk.

Incremental collection:

- Reviews are read newest first. A run stops at the first review older than
  the resume point minus ``resume_overlap_days``.
- Reviews read again inside that window are stored again with the current
  ``collected_at``. Readers keep the latest copy of each review, so edits and
  new developer replies are picked up.
- The resume point (``paths.play_state``) moves forward only after a scan ends
  normally: no failed request and no ``--max-reviews`` cap. A failed or capped
  run is therefore repeated from the old resume point by the next run.
- Without a resume point, the full history is read.

Times are stored in UTC. Each run writes
``data/raw/play_reviews_<app>_<YYYYMMDD>.parquet``.

Usage::

    python -m src.collect.play_store                      # both apps, all languages
    python -m src.collect.play_store --apps tabby --max-reviews 500   # quick test
    python -m src.collect.play_store --full               # read the full history again
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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


class FetchError(RuntimeError):
    """Raised when a request to Google Play still fails after all retries."""


@dataclass
class ScanResult:
    """Outcome of reading the reviews of one app in one query language.

    Attributes:
        records: Privacy-safe records read in this scan, new and already stored.
        n_new: Number of records whose review ID was not stored before.
        complete: True if the scan reached the stop time (or the end of the
            history when there is no stop time) without a failed request or a cap.
        newest: Latest review time read, or None if no review was read.
    """

    records: list[dict[str, Any]]
    n_new: int
    complete: bool
    newest: pd.Timestamp | None


def to_utc(value: datetime | None) -> datetime | None:
    """Convert a time returned by google-play-scraper to timezone-aware UTC.

    The scraper builds review and reply times with ``datetime.fromtimestamp``,
    which gives naive local time of the machine running it. ``astimezone``
    reads naive values as local time in the same way, so the conversion is exact.

    Args:
        value: Naive local or timezone-aware datetime, or None.

    Returns:
        UTC datetime, or None.
    """
    if value is None:
        return None
    return value.astimezone(timezone.utc)


def to_record(raw: dict[str, Any], app: str, lang: str, country: str, collected_at: datetime) -> dict[str, Any]:
    """Convert one google-play-scraper review into a privacy-safe record.

    Only whitelisted fields are copied, so ``userName`` and ``userImage`` are never kept.
    The developer reply text is not stored either: only whether a reply exists and when.

    Args:
        raw: Review dict as returned by ``google_play_scraper.reviews``.
        app: Project app key (e.g. ``"tabby"``).
        lang: Language code used in the query.
        country: Store country code used in the query.
        collected_at: Timestamp of this collection run (UTC).

    Returns:
        A dict with exactly the keys in ``COLUMNS``; times are in UTC.
    """
    return {
        "review_id": raw["reviewId"],
        "app": app,
        "source": SOURCE,
        "country": country,
        "lang_query": lang,
        "text": raw.get("content") or "",
        "rating": int(raw["score"]) if raw.get("score") is not None else None,
        "review_date": to_utc(raw.get("at")),
        "thumbs_up": int(raw.get("thumbsUpCount") or 0),
        "app_version": raw.get("reviewCreatedVersion") or raw.get("appVersion"),
        "has_reply": bool(raw.get("replyContent")),
        "reply_date": to_utc(raw.get("repliedAt")),
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
        df[col] = pd.to_datetime(df[col], utc=True)
    df["rating"] = df["rating"].astype("Int8")
    df["thumbs_up"] = df["thumbs_up"].astype("int64")
    df["has_reply"] = df["has_reply"].astype(bool)
    return df


def latest_copies(df: pd.DataFrame, keys: list[str] | None = None) -> pd.DataFrame:
    """Keep the most recently collected copy of each review.

    Args:
        df: Reviews with a ``collected_at`` column.
        keys: Columns that identify a review; defaults to ``["review_id"]``.

    Returns:
        One row per review, in the original row order.
    """
    keys = keys or ["review_id"]
    latest = df.sort_values("collected_at", kind="stable").drop_duplicates(keys, keep="last")
    return latest.sort_index().reset_index(drop=True)


def fix_legacy_times(df: pd.DataFrame) -> pd.DataFrame:
    """Convert review and reply times of rows that hold local time labeled as UTC.

    Google Play rows collected before
    ``collection.play_store.legacy_local_time.collected_before`` store the
    collecting machine's local time (``legacy_local_time.timezone``) as if it
    were UTC. Their ``review_date`` and ``reply_date`` are converted to UTC.
    Other rows are unchanged.

    Args:
        df: Raw reviews with ``source``, ``collected_at``, ``review_date`` and
            ``reply_date``.

    Returns:
        Copy of ``df`` with corrected times.
    """
    legacy = load_config()["collection"]["play_store"]["legacy_local_time"]
    out = df.copy()
    mask = (out["source"] == SOURCE) & (out["collected_at"] < pd.Timestamp(legacy["collected_before"]))
    if not mask.any():
        return out
    for col in ("review_date", "reply_date"):
        local = out.loc[mask, col].dt.tz_localize(None).dt.tz_localize(legacy["timezone"])
        out[col] = out[col].where(~mask, local.dt.tz_convert("UTC").reindex(out.index))
    return out


def load_existing(app: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """Load the latest stored copy of every review of an app.

    Args:
        app: Project app key.
        raw_dir: Directory holding raw parquet files (defaults to config ``paths.raw``).

    Returns:
        One row per review (empty with ``COLUMNS`` if nothing exists).
    """
    raw_dir = raw_dir or get_path("raw")
    files = sorted(raw_dir.glob(f"play_reviews_{app}_*.parquet"))
    if not files:
        return to_frame([])
    return latest_copies(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True))


def load_state(path: Path | None = None) -> dict[str, dict[str, str]]:
    """Read the resume points.

    Args:
        path: State file; defaults to ``paths.play_state``.

    Returns:
        ``{app: {lang: ISO timestamp}}``; empty if the file does not exist.
    """
    path = path or get_path("play_state")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(state: dict[str, dict[str, str]], path: Path | None = None) -> None:
    """Write the resume points atomically.

    Args:
        state: ``{app: {lang: ISO timestamp}}``.
        path: State file; defaults to ``paths.play_state``.
    """
    path = path or get_path("play_state")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


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
        max_retries: Attempts per request before giving up.
        retry_backoff_seconds: Base wait between retries (multiplied by attempt number).

    Yields:
        Lists of raw review dicts.

    Raises:
        FetchError: If a request fails ``max_retries`` times.
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
                    raise FetchError(f"{app_id}/{lang}: gave up after {attempt} attempts: {exc}") from exc
                wait = retry_backoff_seconds * attempt
                logger.warning("Request failed (%s); retry %d/%d in %.0fs", exc, attempt, max_retries, wait)
                time.sleep(wait)
        if not batch:
            return
        yield batch
        if token is None or getattr(token, "token", None) is None:
            return
        time.sleep(sleep_seconds)


def scan(
    batches: Iterable[list[dict[str, Any]]],
    app: str,
    lang: str,
    country: str,
    known_ids: set[str],
    seen_ids: set[str],
    stop_at: pd.Timestamp | None,
    max_new: int | None,
    collected_at: datetime,
) -> ScanResult:
    """Read review batches until the stop time, the cap or the end of the history.

    Args:
        batches: Raw review batches, newest first (see :func:`fetch_batches`).
        app: Project app key.
        lang: Query language code.
        country: Store country code passed to the scraper.
        known_ids: Review IDs stored before this run.
        seen_ids: Review IDs already read in this run; updated in place, so a
            review returned by two query languages is stored once.
        stop_at: Stop at the first review older than this; None reads the full history.
        max_new: Stop after this many new reviews; None for no cap.
        collected_at: Timestamp of this collection run (UTC).

    Returns:
        Scan result. ``complete`` is False after a failed request, after the
        cap is hit, or if the history ends before ``stop_at`` is reached.
    """
    records: list[dict[str, Any]] = []
    n_new = 0
    newest: pd.Timestamp | None = None
    reached_stop = capped = False
    try:
        for batch in batches:
            for raw in batch:
                record = to_record(raw, app, lang, country, collected_at)  # names/images dropped here
                review_date = _as_utc(record["review_date"])
                if review_date is not None:
                    newest = review_date if newest is None else max(newest, review_date)
                    if stop_at is not None and review_date < stop_at:
                        reached_stop = True
                        break
                if record["review_id"] in seen_ids:
                    continue
                seen_ids.add(record["review_id"])
                records.append(record)
                if record["review_id"] not in known_ids:
                    n_new += 1
                    if max_new is not None and n_new >= max_new:
                        capped = True
                        break
            logger.info("%s/%s: %d reviews read, %d new", app, lang, len(records), n_new)
            if reached_stop or capped:
                break
    except FetchError as exc:
        logger.error("%s; keeping %d reviews read so far, resume point unchanged", exc, len(records))
        return ScanResult(records, n_new, False, newest)
    complete = not capped and (reached_stop or stop_at is None)
    return ScanResult(records, n_new, complete, newest)


def save_records(app: str, df: pd.DataFrame, run_date: str, raw_dir: Path | None = None) -> Path | None:
    """Write reviews to ``play_reviews_<app>_<run_date>.parquet``, merging a same-day file.

    Args:
        app: Project app key.
        df: Reviews to save (new and refreshed).
        run_date: Date stamp ``YYYYMMDD`` for the file name.
        raw_dir: Output directory (defaults to config ``paths.raw``).

    Returns:
        Path written, or None if there was nothing to save.
    """
    if df.empty:
        return None
    raw_dir = raw_dir or get_path("raw")
    out = raw_dir / f"play_reviews_{app}_{run_date}.parquet"
    if out.exists():
        df = latest_copies(pd.concat([pd.read_parquet(out), df], ignore_index=True))
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    return out


def run(
    apps: list[str] | None = None,
    langs: list[str] | None = None,
    max_reviews: int | None = None,
    full: bool = False,
    raw_dir: Path | None = None,
    state_path: Path | None = None,
) -> None:
    """Run incremental collection for the given apps and languages.

    Args:
        apps: App keys from config (default: all).
        langs: Language codes (default: ``collection.languages``).
        max_reviews: Cap on new reviews per app/language (default: config value).
        full: Ignore the resume points and read the full history.
        raw_dir: Raw data directory (default: ``paths.raw``).
        state_path: Resume-point file (default: ``paths.play_state``).
    """
    cfg = load_config()
    coll = cfg["collection"]
    ps = coll["play_store"]
    apps = apps or list(cfg["apps"])
    langs = langs or coll["languages"]
    if max_reviews is None:
        max_reviews = ps["max_reviews_per_lang"]
    overlap = timedelta(days=ps["resume_overlap_days"])
    collected_at = datetime.now(timezone.utc)
    run_date = collected_at.strftime("%Y%m%d")
    state = load_state(state_path)

    for app in apps:
        app_id = cfg["apps"][app]["play_store_id"]
        known_ids: set[str] = set(load_existing(app, raw_dir)["review_id"])
        logger.info("%s (%s): %d reviews already stored", app, app_id, len(known_ids))

        seen_ids: set[str] = set()
        records: list[dict[str, Any]] = []
        new_resume: dict[str, str] = {}
        for lang in langs:
            resume = state.get(app, {}).get(lang)
            stop_at = None if (full or resume is None) else pd.Timestamp(resume) - overlap
            logger.info("%s/%s: %s", app, lang, f"reading back to {stop_at}" if stop_at is not None else "full history")
            batches = fetch_batches(
                app_id, lang, coll["country"], ps["batch_size"], ps["sleep_seconds"],
                ps["max_retries"], ps["retry_backoff_seconds"],
            )
            result = scan(batches, app, lang, coll["country"], known_ids, seen_ids, stop_at, max_reviews, collected_at)
            records += result.records
            if result.complete and result.newest is not None:
                previous = pd.Timestamp(resume) if resume is not None else result.newest
                new_resume[lang] = max(previous, result.newest).isoformat()
            else:
                logger.warning("%s/%s: scan incomplete; resume point unchanged", app, lang)

        out = save_records(app, to_frame(records), run_date, raw_dir)
        n_new = sum(r["review_id"] not in known_ids for r in records)
        logger.info("%s: %d reviews saved (%d new)%s", app, len(records), n_new, f" -> {out.name}" if out else "")
        # Resume points move only after the reviews they cover are on disk.
        if new_resume:
            state.setdefault(app, {}).update(new_resume)
            save_state(state, state_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    cfg = load_config()
    p = argparse.ArgumentParser(description="Incrementally collect Google Play reviews.")
    p.add_argument("--apps", nargs="+", choices=list(cfg["apps"]), help="Apps to collect (default: all)")
    p.add_argument("--langs", nargs="+", help="Language codes (default: from config)")
    p.add_argument("--max-reviews", type=int, default=None, help="Cap new reviews per app/language (testing)")
    p.add_argument("--full", action="store_true", help="Read the full history instead of resuming")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.apps, args.langs, args.max_reviews, args.full)
