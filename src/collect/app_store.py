"""Optional: collect recent Apple App Store reviews from Apple's public RSS feed.

Limitations (state these wherever the data is used):
- The RSS feed only exposes the most recent reviews (at most 10 pages x 50 reviews per
  app and country), so this is NOT a full history and is not comparable in volume to
  the Google Play data.
- It has no developer-reply information, so ``has_reply`` and ``reply_date`` are null.
- Reviews are not filtered by language; ``lang_query`` is null.

Every row is labelled ``source = "app_store_rss"``. Reviewer names and profile URLs are
dropped at parse time and never written to disk.

Output: ``data/raw/appstore_reviews_<app>_<YYYYMMDD>.parquet`` (only reviews not stored before).

Usage::

    python -m src.collect.app_store
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from src.config import get_logger, get_path, load_config

logger = get_logger("collect.app_store")

SOURCE = "app_store_rss"
RSS_URL = "https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"
USER_AGENT = "bnpl-reviews-research/0.1 (academic project)"

COLUMNS: list[str] = [
    "review_id",
    "app",
    "source",
    "country",
    "lang_query",
    "title",
    "text",
    "rating",
    "review_date",
    "thumbs_up",
    "app_version",
    "has_reply",
    "reply_date",
    "collected_at",
]


def _label(entry: dict[str, Any], key: str) -> str | None:
    """Read the ``label`` field of an RSS JSON node.

    Args:
        entry: One RSS ``entry`` dict.
        key: Field name, e.g. ``"im:rating"``.

    Returns:
        The label string, or None if missing.
    """
    node = entry.get(key)
    return node.get("label") if isinstance(node, dict) else None


def parse_entry(entry: dict[str, Any], app: str, country: str, collected_at: datetime) -> dict[str, Any] | None:
    """Convert one RSS entry into a privacy-safe record (author fields are ignored).

    Args:
        entry: One element of ``feed.entry``.
        app: Project app key.
        country: Store country code.
        collected_at: Timestamp of this collection run (UTC).

    Returns:
        A record with keys ``COLUMNS``, or None if the entry is not a review.
    """
    rating = _label(entry, "im:rating")
    if rating is None:  # the feed sometimes includes an app-metadata entry
        return None
    return {
        "review_id": _label(entry, "id"),
        "app": app,
        "source": SOURCE,
        "country": country,
        "lang_query": None,
        "title": _label(entry, "title") or "",
        "text": _label(entry, "content") or "",
        "rating": int(rating),
        "review_date": _label(entry, "updated"),
        "thumbs_up": int(_label(entry, "im:voteSum") or 0),
        "app_version": _label(entry, "im:version"),
        "has_reply": None,
        "reply_date": None,
        "collected_at": collected_at,
    }


def fetch_app(app: str, app_id: str, country: str, max_pages: int, sleep_seconds: float) -> list[dict[str, Any]]:
    """Fetch all available RSS review pages for one app.

    Args:
        app: Project app key.
        app_id: Numeric App Store ID.
        country: Store country code.
        max_pages: Maximum number of pages (Apple serves at most 10).
        sleep_seconds: Delay between requests.

    Returns:
        List of records.
    """
    collected_at = datetime.now(timezone.utc)
    records: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        url = RSS_URL.format(country=country, page=page, app_id=app_id)
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            resp.raise_for_status()
            entries = resp.json().get("feed", {}).get("entry", [])
        except (requests.RequestException, ValueError) as exc:
            logger.warning("%s page %d failed: %s", app, page, exc)
            break
        if isinstance(entries, dict):  # a single entry is returned as an object
            entries = [entries]
        page_records = [r for e in entries if (r := parse_entry(e, app, country, collected_at))]
        if not page_records:
            break
        records += page_records
        logger.info("%s: page %d -> %d reviews", app, page, len(page_records))
        time.sleep(sleep_seconds)
    return records


def to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a DataFrame with a stable schema and UTC timestamps.

    Args:
        records: Records from :func:`parse_entry`.

    Returns:
        DataFrame with columns ``COLUMNS``.
    """
    df = pd.DataFrame.from_records(records, columns=COLUMNS)
    for col in ("review_date", "reply_date", "collected_at"):
        df[col] = pd.to_datetime(df[col], utc=True)
    df["rating"] = df["rating"].astype("Int8")
    df["has_reply"] = df["has_reply"].astype("boolean")
    df["lang_query"] = df["lang_query"].astype("string")
    return df


def run(apps: list[str] | None = None) -> None:
    """Collect App Store RSS reviews and save only reviews not stored before.

    Args:
        apps: App keys from config (default: all).
    """
    cfg = load_config()
    store_cfg = cfg["collection"]["app_store"]
    if not store_cfg["enabled"]:
        logger.info("App Store collection disabled in config.yaml")
        return
    country = cfg["collection"]["country"]
    raw_dir = get_path("raw")
    run_date = datetime.now(timezone.utc).strftime("%Y%m%d")

    for app in apps or list(cfg["apps"]):
        existing_files = sorted(raw_dir.glob(f"appstore_reviews_{app}_*.parquet"))
        known = set(pd.concat([pd.read_parquet(f, columns=["review_id"]) for f in existing_files])["review_id"]) if existing_files else set()

        df = to_frame(fetch_app(app, cfg["apps"][app]["app_store_id"], country, store_cfg["max_pages"], store_cfg["sleep_seconds"]))
        df = df[~df["review_id"].isin(known)].drop_duplicates("review_id")
        if df.empty:
            logger.info("%s: no new App Store reviews", app)
            continue
        out = raw_dir / f"appstore_reviews_{app}_{run_date}.parquet"
        if out.exists():
            df = pd.concat([pd.read_parquet(out), df], ignore_index=True).drop_duplicates("review_id")
        df.to_parquet(out, index=False)
        logger.info("%s: %d new App Store reviews -> %s", app, len(df), out.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect recent App Store reviews from Apple's public RSS feed.")
    parser.add_argument("--apps", nargs="+", choices=list(load_config()["apps"]), help="Apps to collect (default: all)")
    run(parser.parse_args().apps)
