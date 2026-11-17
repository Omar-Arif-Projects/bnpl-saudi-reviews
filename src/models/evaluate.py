"""Evaluate sentiment models on star labels (dev) and on the gold set.

Metrics, per model and language subset (``models.eval_subsets``):

- macro-F1 and accuracy, with per-class precision, recall and F1;
- on the gold set, population-weighted macro-F1 and accuracy, using each
  row's sampling ``weight``, because the sample over-represents negative and
  neutral reviews;
- a two-class view on reviews whose true label is negative or positive: the
  prediction is whichever of the model's negative and positive probabilities
  is higher.

Uncertainty: 95% percentile intervals from a bootstrap that resamples rows
within strata (gold: sampling stratum; dev: label x language). All models are
scored on the same resamples, which gives intervals for the difference in
macro-F1 between each pair of models.

A model is evaluated on a language subset only if it scores every review in
that subset (CAMeLBERT has no ``all`` or ``english`` rows).

Dev results use star labels and are a sanity check. Gold results use hand
labels and are the reported results.

Outputs:

- ``paths.results_dev`` and ``paths.results_dev_paired``: baseline on the full
  dev split and all models on the balanced comparison sample.
- ``paths.results_gold`` and ``paths.results_gold_paired``: all models on
  usable gold rows (once labeled).
- ``paths.gold_agreement``: agreement between the two labelings of the
  relabel sample (once it exists).
- ``paths.figures``/``sentiment_confusion_gold.png``: confusion matrices.

Usage::

    python -m src.models.evaluate
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src.config import get_logger, get_path, load_config
from src.label.gold import load_gold, usable

logger = get_logger("models.evaluate")

# The two polar classes used by the two-class view.
POLAR = ["negative", "positive"]


def metrics(
    y_true: pd.Series, y_pred: pd.Series, classes: list[str], weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute macro-F1, accuracy and per-class precision, recall and F1.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        classes: Class order.
        weights: Optional row weights.

    Returns:
        Flat dictionary of metrics.
    """
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0, sample_weight=weights,
    )
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    out = {
        "n": len(y_true),
        "macro_f1": f1_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0, sample_weight=weights,
        ),
        "accuracy": float(np.average(correct, weights=weights)) if len(correct) else float("nan"),
    }
    for i, c in enumerate(classes):
        out[f"precision_{c}"] = p[i]
        out[f"recall_{c}"] = r[i]
        out[f"f1_{c}"] = f[i]
    return out


def encode(labels: pd.Series, classes: list[str]) -> np.ndarray:
    """Return the position of each label in ``classes``.

    Args:
        labels: Class labels.
        classes: Class order.

    Returns:
        Integer array.

    Raises:
        KeyError: If a label is not in ``classes``.
    """
    index = {c: i for i, c in enumerate(classes)}
    return np.array([index[v] for v in labels], dtype=np.int64)


def macro_f1_codes(y_true: np.ndarray, y_pred: np.ndarray, k: int, weights: np.ndarray | None = None) -> float:
    """Compute macro-F1 from integer-coded labels.

    Gives the same result as scikit-learn's ``f1_score(average="macro",
    zero_division=0)`` with all ``k`` classes listed, but is much faster,
    which matters inside the bootstrap.

    Args:
        y_true: True class codes in ``range(k)``.
        y_pred: Predicted class codes in ``range(k)``.
        k: Number of classes.
        weights: Optional row weights.

    Returns:
        Macro-F1.
    """
    cm = np.bincount(y_true * k + y_pred, weights=weights, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    denom = 2 * tp + (cm.sum(axis=0) - tp) + (cm.sum(axis=1) - tp)
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return float(f1.mean())


def stratified_resamples(strata: np.ndarray, n_boot: int, seed: int) -> Iterator[np.ndarray]:
    """Yield bootstrap row indices that resample with replacement within each stratum.

    Args:
        strata: Stratum of each row.
        n_boot: Number of resamples.
        seed: Random seed.

    Yields:
        Index arrays of the same length as ``strata``.
    """
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    for _ in range(n_boot):
        yield np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])


def bootstrap_scores(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    k: int,
    strata: np.ndarray,
    n_boot: int,
    seed: int,
    weights: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Score every model on the same stratified bootstrap resamples.

    Args:
        y_true: True class codes.
        preds: Model key -> predicted class codes.
        k: Number of classes.
        strata: Stratum of each row.
        n_boot: Number of resamples.
        seed: Random seed.
        weights: Optional row weights.

    Returns:
        Model key -> array of ``n_boot`` macro-F1 values.
    """
    scores = {key: np.empty(n_boot) for key in preds}
    for b, idx in enumerate(stratified_resamples(strata, n_boot, seed)):
        w = None if weights is None else weights[idx]
        for key, pred in preds.items():
            scores[key][b] = macro_f1_codes(y_true[idx], pred[idx], k, w)
    return scores


def interval(values: np.ndarray) -> tuple[float, float]:
    """Return the 2.5th and 97.5th percentiles."""
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def model_keys(df: pd.DataFrame) -> list[str]:
    """Return model prefixes that have a ``<key>_pred`` column."""
    return [c[: -len("_pred")] for c in df.columns if c.endswith("_pred")]


def binary_view(df: pd.DataFrame, key: str, label_col: str) -> tuple[pd.Series, pd.Series] | None:
    """Restrict to reviews labeled negative or positive and take the model's polarity.

    Args:
        df: Rows with labels and ``<key>_p_negative`` / ``<key>_p_positive``.
        key: Model prefix.
        label_col: Column with true labels.

    Returns:
        ``(y_true, y_pred)``, or None if the probability columns are missing.
    """
    p_neg, p_pos = f"{key}_p_{POLAR[0]}", f"{key}_p_{POLAR[1]}"
    if p_neg not in df or p_pos not in df:
        return None
    part = df[df[label_col].isin(POLAR)]
    pred = np.where(part[p_neg] > part[p_pos], POLAR[0], POLAR[1])
    return part[label_col], pd.Series(pred, index=part.index)


def results_table(
    df: pd.DataFrame,
    label_col: str,
    dataset: str,
    strata_col: str | None = None,
    weight_col: str | None = None,
    ci: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every model on every language subset it fully covers.

    Args:
        df: Rows with ``lang``, the label column and ``<model>_pred`` columns.
        label_col: Column with true labels.
        dataset: Name stored in the ``dataset`` column.
        strata_col: Column with bootstrap strata; None treats all rows as one stratum.
        weight_col: Column with population weights; adds weighted metrics.
        ci: Compute bootstrap intervals and paired differences.

    Returns:
        ``(results, paired)``. ``results`` has one row per model and subset;
        ``paired`` has one row per subset, metric and pair of models with the
        difference in macro-F1 (``model_a`` minus ``model_b``) and its interval.
    """
    cfg = load_config()
    classes = cfg["labels"]["sentiment_classes"]
    n_boot, seed = cfg["models"]["bootstrap_samples"], cfg["project"]["random_seed"]
    rows: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    for subset, langs in cfg["models"]["eval_subsets"].items():
        part = df if langs is None else df[df["lang"].isin(langs)]
        keys = [k for k in model_keys(df) if len(part) and part[f"{k}_pred"].notna().all()]
        if not keys:
            continue
        weights = part[weight_col].to_numpy(float) if weight_col else None
        boot: dict[str, dict[str, np.ndarray]] = {}
        if ci:
            y = encode(part[label_col], classes)
            preds = {k: encode(part[f"{k}_pred"], classes) for k in keys}
            strata = part[strata_col].to_numpy() if strata_col else np.zeros(len(part))
            boot["macro_f1"] = bootstrap_scores(y, preds, len(classes), strata, n_boot, seed)
            if weights is not None:
                boot["macro_f1_weighted"] = bootstrap_scores(y, preds, len(classes), strata, n_boot, seed, weights)

        points: dict[str, dict[str, float]] = {}
        for k in keys:
            m = metrics(part[label_col], part[f"{k}_pred"], classes)
            row: dict[str, object] = {
                "dataset": dataset, "model": k, "subset": subset, "n": m.pop("n"), "macro_f1": m.pop("macro_f1"),
            }
            if ci:
                row["ci_low"], row["ci_high"] = interval(boot["macro_f1"][k])
            row["accuracy"] = m.pop("accuracy")
            points[k] = {"macro_f1": float(row["macro_f1"])}
            if weights is not None:
                mw = metrics(part[label_col], part[f"{k}_pred"], classes, weights)
                row["macro_f1_weighted"] = mw["macro_f1"]
                if ci:
                    row["ci_low_weighted"], row["ci_high_weighted"] = interval(boot["macro_f1_weighted"][k])
                row["accuracy_weighted"] = mw["accuracy"]
                points[k]["macro_f1_weighted"] = float(mw["macro_f1"])
            binary = binary_view(part, k, label_col)
            if binary is not None and len(binary[0]):
                mb = metrics(*binary, POLAR)
                row.update(n_binary=mb["n"], binary_macro_f1=mb["macro_f1"], binary_accuracy=mb["accuracy"])
            row.update(m)
            rows.append(row)

        for metric, scores in boot.items():
            for a, b in combinations(keys, 2):
                low, high = interval(scores[a] - scores[b])
                pairs.append({
                    "dataset": dataset, "subset": subset, "metric": metric, "model_a": a, "model_b": b,
                    "difference": points[a][metric] - points[b][metric], "ci_low": low, "ci_high": high,
                })
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def gold_frame() -> pd.DataFrame:
    """Join usable gold labels with model predictions.

    Returns:
        Usable gold rows (with text) and all prediction columns.
    """
    preds = pd.read_csv(get_path("gold_predictions"), dtype={"gold_id": str})
    gold = usable(load_gold())
    return gold.merge(preds.drop(columns="lang"), on="gold_id", how="left")


def agreement(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    """Measure agreement between two labelings of the same gold rows.

    Rows flagged or unlabeled in either labeling are left out.

    Args:
        first: Gold file with the first labels.
        second: Relabel file with the second labels.

    Returns:
        One row per label column (``sentiment``, ``topic``) with the number of
        rows, raw agreement and Cohen's kappa; empty if no row is in both.
    """
    both = usable(first).merge(usable(second), on="gold_id", suffixes=("_1", "_2"))
    if both.empty:
        return pd.DataFrame()
    rows = []
    for col in ("sentiment", "topic"):
        a, b = both[f"{col}_1"], both[f"{col}_2"]
        rows.append({
            "label": col, "n": len(both), "agreement": float((a == b).mean()),
            "kappa": float(cohen_kappa_score(a, b)),
        })
    return pd.DataFrame(rows)


def errors(df: pd.DataFrame, key: str, label_col: str = "sentiment") -> pd.DataFrame:
    """Return misclassified rows for error analysis, most confident first.

    Args:
        df: Output of :func:`gold_frame`.
        key: Model prefix.
        label_col: Column with true labels.

    Returns:
        Rows where the prediction differs from the label.
    """
    wrong = df[df[f"{key}_pred"].notna() & (df[f"{key}_pred"] != df[label_col])].copy()
    prob_cols = [c for c in df.columns if c.startswith(f"{key}_p_")]
    wrong["confidence"] = wrong[prob_cols].max(axis=1)
    cols = ["gold_id", "app", "lang", "text", label_col, f"{key}_pred", "confidence", "topic"]
    return wrong.sort_values("confidence", ascending=False)[cols]


def plot_confusion(df: pd.DataFrame, label_col: str, out_name: str) -> None:
    """Save row-normalized confusion matrices, one panel per model.

    Each model is shown on the largest subset it fully covers.

    Args:
        df: Rows with labels and predictions.
        label_col: Column with true labels.
        out_name: Figure file name without extension.
    """
    import matplotlib.pyplot as plt

    from src.analysis.plots import caption, save, setup_style

    setup_style()
    cfg = load_config()
    classes = cfg["labels"]["sentiment_classes"]
    panels = []
    for key in model_keys(df):
        for subset, langs in cfg["models"]["eval_subsets"].items():
            part = df if langs is None else df[df["lang"].isin(langs)]
            if len(part) and part[f"{key}_pred"].notna().all():
                panels.append((key, subset, part))
                break

    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.6), squeeze=False)
    for ax, (key, subset, part) in zip(axes[0], panels):
        cm = confusion_matrix(part[label_col], part[f"{key}_pred"], labels=classes)
        share = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9,
                        color="white" if share[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(classes)), classes)
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.grid(False)
        ax.set_title(f"{key} ({subset}, n={len(part)})", fontsize=10)
    caption(fig, "Counts; shading is the share of each true class. Gold set, hand labels.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out_name)
    plt.close(fig)


def run() -> None:
    """Write dev results and, if gold labels exist, gold results, agreement and figure."""
    dev = pd.read_parquet(get_path("dev_predictions"))
    dev["stratum"] = dev["label"] + "|" + dev["lang"]
    baseline_cols = ["lang", "label", "stratum", *[c for c in dev.columns if c.startswith("baseline_")]]
    full, full_pairs = results_table(dev[baseline_cols], "label", "dev_full_star_labels", "stratum")
    comp, comp_pairs = results_table(dev[dev["in_comparison"]], "label", "dev_balanced_star_labels", "stratum")
    dev_results = pd.concat([full, comp], ignore_index=True)
    dev_results.to_csv(get_path("results_dev"), index=False, float_format="%.4f")
    pd.concat([full_pairs, comp_pairs], ignore_index=True).to_csv(
        get_path("results_dev_paired"), index=False, float_format="%.4f",
    )
    logger.info("Dev results:\n%s", dev_results[["dataset", "model", "subset", "n", "macro_f1"]].to_string())

    gold = gold_frame()
    if gold.empty:
        logger.info("No usable gold labels yet; gold evaluation skipped")
        return
    gold_results, gold_pairs = results_table(gold, "sentiment", "gold", "stratum", "weight")
    gold_results.to_csv(get_path("results_gold"), index=False, float_format="%.4f")
    gold_pairs.to_csv(get_path("results_gold_paired"), index=False, float_format="%.4f")
    logger.info("Gold results (%d rows):\n%s", len(gold),
                gold_results[["model", "subset", "n", "macro_f1", "ci_low", "ci_high"]].to_string())
    plot_confusion(gold, "sentiment", "sentiment_confusion_gold")

    relabel_path = get_path("gold_relabel")
    if relabel_path.exists():
        table = agreement(load_gold(), load_gold(relabel_path))
        if not table.empty:
            table.to_csv(get_path("gold_agreement"), index=False, float_format="%.4f")
            logger.info("Label agreement:\n%s", table.to_string())


if __name__ == "__main__":
    run()
