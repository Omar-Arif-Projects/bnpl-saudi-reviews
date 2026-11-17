"""Offline tests for the collection modules (no network calls)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.collect import app_store, play_store
from src.config import load_config

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
UTC = timezone.utc


def raw_review(i: int, day: int, reply: bool = False) -> dict[str, Any]:
    """A google-play-scraper review dict posted on 2026-01-<day> at 12:00 UTC."""
    at = datetime(2026, 1, day, 12, tzinfo=UTC)
    return {
        "reviewId": f"r{i}", "content": f"text {i}", "score": 5, "at": at,
        "replyContent": "thanks" if reply else None, "repliedAt": at if reply else None,
    }


def test_play_record_drops_personal_fields() -> None:
    """Reviewer name and image must never reach the stored record."""
    raw = {
        "reviewId": "abc",
        "userName": "Some Person",
        "userImage": "https://example.com/img.png",
        "content": "تطبيق ممتاز",
        "score": 5,
        "thumbsUpCount": 3,
        "reviewCreatedVersion": "1.2.3",
        "at": datetime(2025, 5, 1, 12, 0),
        "replyContent": "شكرا",
        "repliedAt": datetime(2025, 5, 2, 9, 0),
    }
    rec = play_store.to_record(raw, "tabby", "ar", "sa", NOW)
    assert set(rec) == set(play_store.COLUMNS)
    assert "Some Person" not in map(str, rec.values())
    assert rec["has_reply"] is True and rec["rating"] == 5

    df = play_store.to_frame([rec])
    assert list(df.columns) == play_store.COLUMNS
    assert str(df["review_date"].dt.tz) == "UTC"


def test_play_record_without_reply() -> None:
    """Missing reply fields become has_reply=False and a null date."""
    raw = {"reviewId": "x", "content": None, "score": 1, "at": datetime(2025, 1, 1)}
    rec = play_store.to_record(raw, "tamara", "en", "sa", NOW)
    assert rec["has_reply"] is False and rec["reply_date"] is None and rec["text"] == ""


@pytest.mark.parametrize("epoch", [1_700_000_000, 1_750_000_000])
def test_to_utc_undoes_local_fromtimestamp(epoch: int) -> None:
    """The scraper's naive local times convert to the true UTC instant."""
    assert play_store.to_utc(datetime.fromtimestamp(epoch)) == datetime.fromtimestamp(epoch, UTC)


def test_fix_legacy_times_shifts_only_old_play_rows() -> None:
    legacy = load_config()["collection"]["play_store"]["legacy_local_time"]
    cutoff = pd.Timestamp(legacy["collected_before"])
    day = pd.Timedelta(days=1)
    df = pd.DataFrame({
        "source": ["google_play", "google_play", "app_store_rss"],
        "collected_at": [cutoff - day, cutoff + day, cutoff - day],
        "review_date": pd.to_datetime(["2025-03-01 02:00"] * 3, utc=True),
        "reply_date": pd.to_datetime(["2025-03-01 05:00", None, None], utc=True),
    })
    out = play_store.fix_legacy_times(df)
    offset = pd.Timestamp("2025-03-01 02:00").tz_localize(legacy["timezone"]).utcoffset()
    assert out.loc[0, "review_date"] == df.loc[0, "review_date"] - offset
    assert out.loc[0, "reply_date"] == df.loc[0, "reply_date"] - offset
    assert out.loc[1, "review_date"] == df.loc[1, "review_date"]
    assert out.loc[2, "review_date"] == df.loc[2, "review_date"]
    assert pd.isna(out.loc[1, "reply_date"])


def test_latest_copies_keeps_most_recent_collection() -> None:
    df = pd.DataFrame({
        "review_id": ["a", "b", "a"],
        "collected_at": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-02-01"], utc=True),
        "has_reply": [False, False, True],
    })
    out = play_store.latest_copies(df)
    assert sorted(out["review_id"]) == ["a", "b"]
    assert bool(out.set_index("review_id").loc["a", "has_reply"])


def test_scan_refreshes_known_reviews_inside_window() -> None:
    batches = [[raw_review(3, 10), raw_review(2, 9, reply=True)], [raw_review(1, 5)]]
    result = play_store.scan(
        batches, "tabby", "ar", "sa", known_ids={"r1", "r2"}, seen_ids=set(),
        stop_at=pd.Timestamp("2026-01-08", tz="UTC"), max_new=None, collected_at=NOW,
    )
    assert [r["review_id"] for r in result.records] == ["r3", "r2"]
    assert result.records[1]["has_reply"] is True
    assert result.n_new == 1
    assert result.complete
    assert result.newest == pd.Timestamp("2026-01-10 12:00", tz="UTC")


def test_scan_skips_reviews_seen_in_another_language() -> None:
    result = play_store.scan(
        [[raw_review(1, 10), raw_review(2, 9)]], "tabby", "en", "sa", known_ids=set(), seen_ids={"r1"},
        stop_at=None, max_new=None, collected_at=NOW,
    )
    assert [r["review_id"] for r in result.records] == ["r2"]


def test_scan_after_failed_request_is_incomplete() -> None:
    def batches() -> Iterator[list[dict[str, Any]]]:
        yield [raw_review(3, 10)]
        raise play_store.FetchError("request failed")

    result = play_store.scan(
        batches(), "tabby", "ar", "sa", known_ids=set(), seen_ids=set(),
        stop_at=pd.Timestamp("2026-01-01", tz="UTC"), max_new=None, collected_at=NOW,
    )
    assert not result.complete
    assert [r["review_id"] for r in result.records] == ["r3"]


def test_scan_with_cap_is_incomplete() -> None:
    result = play_store.scan(
        [[raw_review(3, 10), raw_review(2, 9)]], "tabby", "ar", "sa", known_ids=set(), seen_ids=set(),
        stop_at=None, max_new=1, collected_at=NOW,
    )
    assert not result.complete and len(result.records) == 1


def test_scan_completeness_at_end_of_history() -> None:
    def scan(stop_at: pd.Timestamp | None) -> play_store.ScanResult:
        return play_store.scan(
            [[raw_review(3, 10)]], "tabby", "ar", "sa", known_ids=set(), seen_ids=set(),
            stop_at=stop_at, max_new=None, collected_at=NOW,
        )

    assert scan(None).complete
    # The history ended before reaching the stop time: something is missing.
    assert not scan(pd.Timestamp("2026-01-05", tz="UTC")).complete


def test_run_moves_resume_point_only_after_complete_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pages: list[list[dict[str, Any]]] = [[raw_review(2, 10), raw_review(1, 5)]]
    fail = {"on": False}

    def fake_fetch(*args: Any, **kwargs: Any) -> Iterator[list[dict[str, Any]]]:
        yield from pages
        if fail["on"]:
            raise play_store.FetchError("request failed")

    monkeypatch.setattr(play_store, "fetch_batches", fake_fetch)
    state_path = tmp_path / "state.json"

    play_store.run(apps=["tabby"], langs=["ar"], raw_dir=tmp_path, state_path=state_path)
    first = pd.Timestamp("2026-01-10 12:00", tz="UTC").isoformat()
    assert play_store.load_state(state_path) == {"tabby": {"ar": first}}
    assert len(play_store.load_existing("tabby", tmp_path)) == 2

    # A newer review arrives and a reply is added, but the run then fails:
    # both are saved, and the resume point stays where it was.
    pages[:] = [[raw_review(3, 12), raw_review(2, 10, reply=True)]]
    fail["on"] = True
    play_store.run(apps=["tabby"], langs=["ar"], raw_dir=tmp_path, state_path=state_path)
    assert play_store.load_state(state_path) == {"tabby": {"ar": first}}
    stored = play_store.load_existing("tabby", tmp_path).set_index("review_id")
    assert sorted(stored.index) == ["r1", "r2", "r3"]
    assert bool(stored.loc["r2", "has_reply"])


def test_app_store_entry_drops_author_and_skips_metadata() -> None:
    """RSS author fields are ignored, and non-review entries are skipped."""
    entry = {
        "author": {"name": {"label": "Some Person"}, "uri": {"label": "https://itunes.apple.com/sa/reviews/id1"}},
        "id": {"label": "123"},
        "title": {"label": "رائع"},
        "content": {"label": "خدمة ممتازة"},
        "im:rating": {"label": "4"},
        "im:version": {"label": "5.0"},
        "im:voteSum": {"label": "0"},
        "updated": {"label": "2026-11-13T01:15:15-07:00"},
    }
    rec = app_store.parse_entry(entry, "tamara", "sa", NOW)
    assert rec is not None and set(rec) == set(app_store.COLUMNS)
    assert "Some Person" not in map(str, rec.values())
    assert app_store.parse_entry({"id": {"label": "meta"}}, "tamara", "sa", NOW) is None
