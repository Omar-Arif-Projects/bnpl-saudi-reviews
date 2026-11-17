"""Tests for model evaluation helpers and the baseline pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.baseline import build_pipeline, predict_frame
from src.models.evaluate import bootstrap_ci, metrics, model_keys, results_table

CLASSES = ["negative", "neutral", "positive"]


def test_metrics_perfect_and_partial() -> None:
    y = pd.Series(["negative", "neutral", "positive", "positive"])
    assert metrics(y, y, CLASSES)["macro_f1"] == 1.0
    wrong = pd.Series(["negative", "positive", "positive", "positive"])
    m = metrics(y, wrong, CLASSES)
    assert m["recall_neutral"] == 0.0
    assert m["accuracy"] == 0.75


def test_bootstrap_ci_brackets_point_estimate() -> None:
    y = pd.Series(["negative", "neutral", "positive"] * 30)
    pred = y.copy()
    pred.iloc[::7] = "positive"
    low, high = bootstrap_ci(y, pred, CLASSES, n_boot=200, seed=0)
    point = metrics(y, pred, CLASSES)["macro_f1"]
    assert low <= point <= high


def test_results_table_skips_partial_coverage() -> None:
    df = pd.DataFrame({
        "lang": ["arabic", "arabic", "english", "english"],
        "label": ["negative", "positive", "negative", "positive"],
        "baseline_pred": ["negative", "positive", "positive", "positive"],
        "camelbert_pred": ["negative", "positive", None, None],
    })
    assert model_keys(df) == ["baseline", "camelbert"]
    table = results_table(df, "label", "test", ci=False)
    assert set(table.loc[table["model"] == "camelbert", "subset"]) == {"arabic"}
    assert set(table.loc[table["model"] == "baseline", "subset"]) == {"all", "arabic", "english"}


def test_baseline_pipeline_learns_toy_data() -> None:
    texts = ["ممتاز جدا", "سيء جدا", "great app", "terrible app", "عادي", "okay i guess"] * 10
    labels = ["positive", "negative", "positive", "negative", "neutral", "neutral"] * 10
    model = build_pipeline(seed=0)
    model.set_params(features__char__min_df=1, features__word__min_df=1)
    model.fit(texts, labels)
    out = predict_frame(model, pd.Series(["ممتاز", "terrible"]))
    assert out["pred"].tolist() == ["positive", "negative"]
    assert out.filter(like="p_").sum(axis=1).round(6).eq(1).all()


@pytest.mark.parametrize("text", ["حلووووو", None])
def test_baseline_accepts_raw_text(text: str | None) -> None:
    from src.clean.normalize import normalize_batch

    assert isinstance(normalize_batch([text])[0], str)
