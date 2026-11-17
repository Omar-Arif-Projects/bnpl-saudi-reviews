"""Tests for gold-set sampling, anonymization and the labeling widget."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.label.gold import (
    COLUMNS, allocate, candidate_pool, draw_sample, exclude_gold, gold_id, is_done, load_gold,
    save_gold, scrub, usable,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("call me 0551234567 please", "call me [NUMBER] please"),
        ("رقمي ٠٥٥١٢٣٤٥٦٧", "رقمي [NUMBER]"),
        ("+966 55 123 4567", "[NUMBER]"),
        ("mail a.b@x.com now", "mail [EMAIL] now"),
        ("see https://x.co/a?b=1", "see [URL]"),
        ("paid 500 SAR in 4 payments", "paid 500 SAR in 4 payments"),
        (None, ""),
    ],
)
def test_scrub(raw: str | None, expected: str) -> None:
    assert scrub(raw) == expected


def test_gold_id_is_stable() -> None:
    assert gold_id("abc", 12) == gold_id("abc", 12)
    assert len(gold_id("abc", 12)) == 12
    assert gold_id("abc", 12) != gold_id("abd", 12)


def test_allocate() -> None:
    assert allocate(400, {"negative": 0.4, "neutral": 0.25, "positive": 0.35}, 4) == {
        "negative": 40, "neutral": 25, "positive": 35,
    }
    counts = allocate(100, {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, 1)
    assert sum(counts.values()) == 100
    with pytest.raises(ValueError):
        allocate(10, {"a": 0.5, "b": 0.6}, 1)


def make_clean(n_per_cell: int = 60) -> pd.DataFrame:
    """Synthetic clean data with every app x language x rating combination."""
    rows = []
    for app in ("tabby", "tamara"):
        for lang in ("arabic", "english"):
            for rating in (1, 2, 3, 4, 5):
                for i in range(n_per_cell):
                    rows.append({
                        "review_id": f"{app}-{lang}-{rating}-{i}", "app": app, "lang": lang,
                        "rating": rating, "source": "google_play", "trainable": True,
                        "review_date": pd.Timestamp("2025-01-15", tz="UTC"), "text": f"text {i}",
                    })
    return pd.DataFrame(rows)


def test_draw_sample_strata_and_weights() -> None:
    clean = make_clean()
    sample = draw_sample(candidate_pool(clean), seed=1)
    assert list(sample.columns) == COLUMNS
    assert len(sample) == 400
    counts = sample.groupby("stratum").size()
    assert counts["tabby|arabic|negative"] == 40
    assert counts["tamara|english|neutral"] == 25
    # 120 negative reviews per cell, 40 sampled.
    assert sample.loc[sample["stratum"] == "tabby|arabic|negative", "weight"].iloc[0] == 3.0
    assert sample["gold_id"].is_unique
    assert (sample[["sentiment", "topic", "flag"]] == "").all().all()
    assert len(exclude_gold(clean, sample)) == len(clean) - 400


def test_save_load_roundtrip_and_pii(tmp_path: Path) -> None:
    sample = draw_sample(candidate_pool(make_clean()), seed=1)
    sample.loc[0, ["sentiment", "topic"]] = ["negative", "customer_service"]
    sample.loc[1, "flag"] = "personal_info"
    path = tmp_path / "gold.csv"
    save_gold(sample, path)
    loaded = load_gold(path)
    assert loaded.loc[1, "text"] == ""
    assert int(is_done(loaded).sum()) == 2
    assert len(usable(loaded)) == 1


def test_labeler_saves_and_advances(tmp_path: Path) -> None:
    pytest.importorskip("ipywidgets")
    from src.label.widget import GoldLabeler

    path = tmp_path / "gold.csv"
    save_gold(draw_sample(candidate_pool(make_clean()), seed=1), path)
    labeler = GoldLabeler(path)
    labeler.label(sentiment="positive")              # no topic chosen: not saved
    assert not is_done(load_gold(path)).any()
    labeler.topic.value = "ease_of_use"
    labeler.label(sentiment="positive")
    labeler.label(flag="unclear")
    saved = load_gold(path)
    assert saved.loc[0, ["sentiment", "topic"]].tolist() == ["positive", "ease_of_use"]
    assert saved.loc[1, "flag"] == "unclear"
    assert labeler.pos == 2


def test_labelers_keep_each_others_rows(tmp_path: Path) -> None:
    pytest.importorskip("ipywidgets")
    from src.label.widget import GoldLabeler

    path = tmp_path / "gold.csv"
    save_gold(draw_sample(candidate_pool(make_clean()), seed=1), path)
    first, second = GoldLabeler(path), GoldLabeler(path)
    first.topic.value = "general"
    first.label(sentiment="positive")              # row 0
    second.move(5)
    second.topic.value = "customer_service"
    second.label(sentiment="negative")             # row 5, from a table loaded before row 0 was saved
    saved = load_gold(path)
    assert saved.loc[0, "sentiment"] == "positive"
    assert saved.loc[5, "sentiment"] == "negative"


def test_labeler_shows_failed_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("ipywidgets")
    from src.label import widget

    path = tmp_path / "gold.csv"
    save_gold(draw_sample(candidate_pool(make_clean()), seed=1), path)
    labeler = widget.GoldLabeler(path)

    def locked(*args: object, **kwargs: object) -> None:
        raise PermissionError("file is locked")

    monkeypatch.setattr(widget, "save_gold", locked)
    labeler.topic.value = "general"
    labeler.label(sentiment="positive")
    assert "Not saved" in labeler.message.value
    assert labeler.pos == 0
    assert not is_done(load_gold(path)).any()


def test_relabel_sample_needs_complete_labels(tmp_path: Path) -> None:
    from src.label.gold import create_relabel_sample

    gold_path, out_path = tmp_path / "gold.csv", tmp_path / "relabel.csv"
    sample = draw_sample(candidate_pool(make_clean()), seed=1)
    save_gold(sample, gold_path)
    with pytest.raises(RuntimeError):
        create_relabel_sample(gold_path, out_path)

    sample[["sentiment", "topic"]] = ["positive", "general"]
    sample.loc[0, "flag"] = "unclear"
    save_gold(sample, gold_path)
    relabel = create_relabel_sample(gold_path, out_path)
    assert len(relabel) == 50
    assert (load_gold(out_path)[["sentiment", "topic", "flag"]] == "").all().all()
    assert sample.loc[0, "gold_id"] not in set(relabel["gold_id"])
    with pytest.raises(FileExistsError):
        create_relabel_sample(gold_path, out_path)
