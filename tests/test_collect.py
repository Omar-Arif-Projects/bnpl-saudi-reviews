"""Offline tests for the collection modules (no network calls)."""

from __future__ import annotations

from datetime import datetime, timezone

from src.collect import app_store, play_store

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


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
