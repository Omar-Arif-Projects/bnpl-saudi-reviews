"""Tests for model evaluation helpers, the baseline pipeline and label mapping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.models.baseline import build_pipeline, predict_frame, resolve_class_weight
from src.models.evaluate import (
    agreement, binary_view, bootstrap_scores, encode, macro_f1_codes, metrics, model_keys, results_table,
    stratified_resamples,
)

CLASSES = ["negative", "neutral", "positive"]


def test_metrics_perfect_and_partial() -> None:
    y = pd.Series(["negative", "neutral", "positive", "positive"])
    assert metrics(y, y, CLASSES)["macro_f1"] == 1.0
    wrong = pd.Series(["negative", "positive", "positive", "positive"])
    m = metrics(y, wrong, CLASSES)
    assert m["recall_neutral"] == 0.0
    assert m["accuracy"] == 0.75


def test_weighted_metrics_follow_weights() -> None:
    y = pd.Series(["negative", "positive"])
    pred = pd.Series(["negative", "negative"])
    assert metrics(y, pred, CLASSES)["accuracy"] == 0.5
    assert metrics(y, pred, CLASSES, np.array([3.0, 1.0]))["accuracy"] == 0.75


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_macro_f1_codes_matches_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y, pred = rng.integers(0, 3, 60), rng.integers(0, 3, 60)
    pred[pred == 1] = 2                          # one class never predicted
    w = rng.uniform(0.5, 3, 60)
    for weights in (None, w):
        expected = f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0, sample_weight=weights)
        assert macro_f1_codes(y, pred, 3, weights) == pytest.approx(expected)


def test_stratified_resamples_keep_stratum_sizes() -> None:
    strata = np.array(["a"] * 5 + ["b"] * 3)
    for idx in stratified_resamples(strata, n_boot=20, seed=0):
        assert (strata[idx] == "a").sum() == 5 and (strata[idx] == "b").sum() == 3


def test_bootstrap_scores_bracket_point_estimate() -> None:
    y = pd.Series(["negative", "neutral", "positive"] * 30)
    pred = y.copy()
    pred.iloc[::7] = "positive"
    scores = bootstrap_scores(
        encode(y, CLASSES), {"m": encode(pred, CLASSES)}, 3, np.asarray(y), n_boot=200, seed=0,
    )["m"]
    point = metrics(y, pred, CLASSES)["macro_f1"]
    assert np.percentile(scores, 2.5) <= point <= np.percentile(scores, 97.5)


def frame_with_two_models() -> pd.DataFrame:
    """Labels plus a perfect model ``good`` and a weaker model ``weak``."""
    y = ["negative", "neutral", "positive", "positive"] * 10
    weak = ["negative", "positive", "positive", "negative"] * 10
    df = pd.DataFrame({"lang": ["arabic", "english"] * 20, "label": y, "good_pred": y, "weak_pred": weak})
    for key, preds in (("good", y), ("weak", weak)):
        df[f"{key}_p_negative"] = [0.8 if p == "negative" else 0.1 for p in preds]
        df[f"{key}_p_positive"] = [0.8 if p == "positive" else 0.1 for p in preds]
    df["stratum"] = df["label"] + "|" + df["lang"]
    df["weight"] = 1.0
    return df


def test_results_table_skips_partial_coverage() -> None:
    df = pd.DataFrame({
        "lang": ["arabic", "arabic", "english", "english"],
        "label": ["negative", "positive", "negative", "positive"],
        "baseline_pred": ["negative", "positive", "positive", "positive"],
        "camelbert_pred": ["negative", "positive", None, None],
    })
    assert model_keys(df) == ["baseline", "camelbert"]
    table, paired = results_table(df, "label", "test", ci=False)
    assert set(table.loc[table["model"] == "camelbert", "subset"]) == {"arabic"}
    assert set(table.loc[table["model"] == "baseline", "subset"]) == {"all", "arabic", "english"}
    assert paired.empty


def test_results_table_paired_differences_and_weights() -> None:
    table, paired = results_table(frame_with_two_models(), "label", "test", "stratum", "weight")
    good_all = table[(table["model"] == "good") & (table["subset"] == "all")].iloc[0]
    assert good_all["macro_f1"] == 1.0 and good_all["macro_f1_weighted"] == 1.0
    assert good_all["binary_macro_f1"] == 1.0
    row = paired[(paired["subset"] == "all") & (paired["metric"] == "macro_f1")].iloc[0]
    assert (row["model_a"], row["model_b"]) == ("good", "weak")
    assert row["difference"] > 0 and row["ci_low"] > 0


def test_binary_view_uses_polar_probabilities() -> None:
    df = pd.DataFrame({
        "sentiment": ["negative", "neutral", "positive"],
        "m_p_negative": [0.4, 0.5, 0.2], "m_p_neutral": [0.5, 0.4, 0.5], "m_p_positive": [0.1, 0.1, 0.3],
    })
    y, pred = binary_view(df, "m", "sentiment")
    assert y.tolist() == ["negative", "positive"]
    assert pred.tolist() == ["negative", "positive"]
    assert binary_view(df, "missing", "sentiment") is None


def test_agreement_kappa() -> None:
    first = pd.DataFrame({
        "gold_id": list("abcd"), "sentiment": ["negative", "positive", "neutral", "positive"],
        "topic": ["general", "fees_late_payment", "general", "customer_service"], "flag": [""] * 4,
    })
    second = first.copy()
    second.loc[3, "flag"] = "unclear"            # flagged rows are left out
    table = agreement(first, second).set_index("label")
    assert table.loc["sentiment", "n"] == 3
    assert table.loc["sentiment", "kappa"] == pytest.approx(1.0)
    assert agreement(first, second.iloc[0:0]).empty


def test_label_order_matches_model_labels() -> None:
    from src.models.transformer import label_order

    camelbert = {0: "positive", 1: "negative", 2: "neutral"}
    xlmr = {0: "Negative", 1: "Neutral", 2: "Positive"}
    assert label_order(camelbert, CLASSES) == [1, 2, 0]
    assert label_order(xlmr, CLASSES) == [0, 1, 2]
    with pytest.raises(ValueError):
        label_order({0: "pos", 1: "neg"}, CLASSES)


def test_resolve_class_weight() -> None:
    labels = pd.Series(["positive"] * 8 + ["negative"] * 2)
    assert resolve_class_weight(None, labels) is None
    assert resolve_class_weight("balanced", labels) == "balanced"
    weights = resolve_class_weight("sqrt_balanced", labels)
    assert weights["negative"] == pytest.approx(np.sqrt(10 / (2 * 2)))
    assert weights["positive"] == pytest.approx(np.sqrt(10 / (2 * 8)))
    with pytest.raises(ValueError):
        resolve_class_weight("unknown", labels)


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


def error_frame() -> pd.DataFrame:
    """Gold-like rows: ``a`` scores every row, ``b`` only Arabic rows."""
    return pd.DataFrame({
        "lang": ["arabic", "arabic", "english", "english"],
        "text": ["ممتاز", "لا يوجد تذكير بالدفع", "please add stc pay", "good"],
        "sentiment": ["positive", "negative", "neutral", "positive"],
        "star_sentiment": ["positive", "neutral", "neutral", "negative"],
        "a_pred": ["positive", "neutral", "positive", "positive"],
        "b_pred": ["positive", "positive", None, None],
    })


def test_length_group_labels() -> None:
    from src.models.error_analysis import length_group

    groups = length_group(pd.Series(["a", "a b c d", "a " * 12, None]), bins=[3, 10])
    assert groups.iloc[:3].tolist() == ["1-3", "4-10", "11+"]
    assert pd.isna(groups.iloc[3])                 # empty text has no group


def test_error_table_counts_scored_rows_only() -> None:
    from src.models.error_analysis import error_table

    table = error_table(error_frame(), "lang").set_index(["model", "lang"])
    assert table.loc[("a", "arabic"), ["n", "errors"]].tolist() == [2, 1]
    assert table.loc[("a", "english"), ["n", "errors"]].tolist() == [2, 1]
    assert ("b", "english") not in table.index

    df = error_frame()
    df["length"] = pd.Categorical(["4-10", "1-3", "11+", "1-3"], categories=["1-3", "4-10", "11+"], ordered=True)
    pivot = error_table(df, "length").pivot(index="length", columns="model", values="n")
    assert pivot.index.tolist() == ["1-3", "4-10", "11+"]          # category order, not alphabetical


def test_star_vs_hand_and_wrong_for_all() -> None:
    from src.models.error_analysis import predictions_for, star_vs_hand, wrong_for_all

    df = error_frame()
    table = star_vs_hand(df).set_index("model")
    # Rows 1 and 3 have hand labels that differ from the star label.
    assert table.loc["a", ["n", "matches_hand", "matches_star", "matches_neither"]].tolist() == [2, 1, 1, 0]
    assert table.loc["b", ["n", "matches_neither"]].tolist() == [1, 1]
    assert wrong_for_all(df)["text"].tolist() == ["لا يوجد تذكير بالدفع", "please add stc pay"]
    neutral = predictions_for(df, "neutral")
    assert neutral.loc["a", "positive"] == 1 and neutral.loc["b"].sum() == 0
