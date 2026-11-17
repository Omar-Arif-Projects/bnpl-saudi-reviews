"""Evaluate sentiment models on star labels (dev) and on the gold set.

Metrics: macro-F1 with a bootstrap 95% confidence interval, accuracy, and
per-class precision, recall and F1. A model is evaluated on a language subset
only if it scores every review in that subset (CAMeLBERT has no ``all`` or
``english`` rows).

Dev results use star labels and are a sanity check. Gold results use hand
labels and are the reported results; the gold sample is stratified with
negative and neutral reviews over-sampled, and metrics are unweighted.

Outputs:

- ``paths.results_dev``: baseline on the full dev split and all models on
  the balanced comparison sample.
- ``paths.results_gold``: all models on usable gold rows (once labeled).
- ``paths.figures``/``sentiment_confusion_gold.png``: confusion matrices.

Usage::

    python -m src.models.evaluate
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from src.config import get_logger, get_path, load_config
from src.label.gold import load_gold, usable

logger = get_logger("models.evaluate")

SUBSETS = {"all": None, "arabic": ["arabic"], "english": ["english"]}


def metrics(y_true: pd.Series, y_pred: pd.Series, classes: list[str]) -> dict[str, float]:
    """Compute macro-F1, accuracy and per-class precision, recall and F1.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        classes: Class order.

    Returns:
        Flat dictionary of metrics.
    """
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    out = {
        "n": len(y_true),
        "macro_f1": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
    }
    for i, c in enumerate(classes):
        out[f"precision_{c}"] = p[i]
        out[f"recall_{c}"] = r[i]
        out[f"f1_{c}"] = f[i]
    return out


def bootstrap_ci(
    y_true: pd.Series, y_pred: pd.Series, classes: list[str], n_boot: int, seed: int,
) -> tuple[float, float]:
    """Return a percentile bootstrap 95% interval for macro-F1.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        classes: Class order.
        n_boot: Number of resamples.
        seed: Random seed.

    Returns:
        ``(low, high)``.
    """
    rng = np.random.default_rng(seed)
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    scores = [
        f1_score(yt[idx], yp[idx], labels=classes, average="macro", zero_division=0)
        for idx in (rng.integers(0, len(yt), len(yt)) for _ in range(n_boot))
    ]
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def model_keys(df: pd.DataFrame) -> list[str]:
    """Return model prefixes that have a ``<key>_pred`` column."""
    return [c[: -len("_pred")] for c in df.columns if c.endswith("_pred")]


def results_table(df: pd.DataFrame, label_col: str, dataset: str, ci: bool = True) -> pd.DataFrame:
    """Evaluate every model on every language subset it fully covers.

    Args:
        df: Rows with ``lang``, the label column and ``<model>_pred`` columns.
        label_col: Column with true labels.
        dataset: Name stored in the ``dataset`` column.
        ci: Compute bootstrap intervals.

    Returns:
        One row per model and subset.
    """
    cfg = load_config()
    classes = cfg["labels"]["sentiment_classes"]
    rows = []
    for key in model_keys(df):
        for subset, langs in SUBSETS.items():
            part = df if langs is None else df[df["lang"].isin(langs)]
            if part.empty or part[f"{key}_pred"].isna().any():
                continue
            row = {"dataset": dataset, "model": key, "subset": subset}
            row.update(metrics(part[label_col], part[f"{key}_pred"], classes))
            if ci:
                row["ci_low"], row["ci_high"] = bootstrap_ci(
                    part[label_col], part[f"{key}_pred"], classes,
                    cfg["models"]["bootstrap_samples"], cfg["project"]["random_seed"],
                )
            rows.append(row)
    return pd.DataFrame(rows)


def gold_frame() -> pd.DataFrame:
    """Join usable gold labels with model predictions.

    Returns:
        Usable gold rows (with text) and all prediction columns.
    """
    preds = pd.read_csv(get_path("gold_predictions"), dtype={"gold_id": str})
    gold = usable(load_gold())
    return gold.merge(preds.drop(columns="lang"), on="gold_id", how="left")


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
    classes = load_config()["labels"]["sentiment_classes"]
    panels = []
    for key in model_keys(df):
        for subset, langs in SUBSETS.items():
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
    """Write dev results and, if gold labels exist, gold results and figure."""
    dev = pd.read_parquet(get_path("dev_predictions"))
    full = results_table(dev[["lang", "label", "baseline_pred"]], "label", "dev_full_star_labels")
    comparison = results_table(dev[dev["in_comparison"]], "label", "dev_balanced_star_labels")
    dev_results = pd.concat([full, comparison], ignore_index=True)
    dev_results.to_csv(get_path("results_dev"), index=False, float_format="%.4f")
    logger.info("Dev results:\n%s", dev_results[["dataset", "model", "subset", "n", "macro_f1"]].to_string())

    gold = gold_frame()
    if gold.empty:
        logger.info("No usable gold labels yet; gold evaluation skipped")
        return
    gold_results = results_table(gold, "sentiment", "gold")
    gold_results.to_csv(get_path("results_gold"), index=False, float_format="%.4f")
    logger.info("Gold results (%d rows):\n%s", len(gold),
                gold_results[["model", "subset", "n", "macro_f1", "ci_low", "ci_high"]].to_string())
    plot_confusion(gold, "sentiment", "sentiment_confusion_gold")


if __name__ == "__main__":
    run()
