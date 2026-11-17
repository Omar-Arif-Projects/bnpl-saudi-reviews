"""TF-IDF + Logistic Regression sentiment baseline trained on star labels.

Features are character n-grams (within word boundaries) and word n-grams,
both computed on normalized text. Character n-grams handle Arabic prefixes,
suffixes and spelling variation without a stemmer. The pipeline normalizes
its input, so it accepts raw review text.

Class weighting is set by ``models.baseline.class_weight``. The options in
``class_weight_options`` are compared on the dev split with
``--compare-class-weights``; the selection rule is stated in ``config.yaml``.

Outputs:

- ``paths.models``/``baseline.joblib``: fitted pipeline.
- ``paths.dev_predictions``: predictions on the dev split (column ``baseline``).
- ``paths.baseline_class_weights``: dev comparison of class-weight options
  (``--compare-class-weights`` only).

Usage::

    python -m src.models.baseline
    python -m src.models.baseline --compare-class-weights
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.clean.normalize import normalize_batch
from src.config import get_logger, get_path, load_config, set_seed
from src.models.data import load_star_labeled, split_dev

logger = get_logger("models.baseline")

MODEL_FILE = "baseline.joblib"


def resolve_class_weight(option: str | None, labels: pd.Series) -> str | dict[str, float] | None:
    """Translate a class-weight option into the form scikit-learn expects.

    Args:
        option: ``None`` (no weighting), ``"balanced"`` (weights inversely
            proportional to class frequency) or ``"sqrt_balanced"`` (square
            root of the balanced weights, a milder correction).
        labels: Training labels.

    Returns:
        Value for ``LogisticRegression(class_weight=...)``.

    Raises:
        ValueError: If the option is unknown.
    """
    if option is None:
        return None
    if option == "balanced":
        return "balanced"
    if option == "sqrt_balanced":
        counts = labels.value_counts()
        balanced = len(labels) / (len(counts) * counts)
        return {str(c): float(np.sqrt(w)) for c, w in balanced.items()}
    raise ValueError(f"Unknown class_weight option {option!r}")


def build_pipeline(seed: int, class_weight: str | dict[str, float] | None = "balanced") -> Pipeline:
    """Create the unfitted baseline pipeline from ``models.baseline`` settings.

    Args:
        seed: Random seed for the classifier.
        class_weight: Resolved class weighting (see :func:`resolve_class_weight`).

    Returns:
        Pipeline of normalization, TF-IDF features and logistic regression.
    """
    b = load_config()["models"]["baseline"]
    features = FeatureUnion([
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=tuple(b["char_ngram"]), min_df=b["min_df"],
            max_features=b["max_features_char"], sublinear_tf=b["sublinear_tf"],
        )),
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=tuple(b["word_ngram"]), min_df=b["min_df"],
            max_features=b["max_features_word"], sublinear_tf=b["sublinear_tf"],
            token_pattern=r"(?u)\b\w+\b",
        )),
    ])
    classifier = LogisticRegression(
        C=b["C"], class_weight=class_weight, max_iter=b["max_iter"], random_state=seed,
    )
    return Pipeline([
        ("normalize", FunctionTransformer(
            normalize_batch, kw_args={"repeat_max": load_config()["cleaning"]["repeat_max"]},
        )),
        ("features", features),
        ("classifier", classifier),
    ])


def predict_frame(model: Pipeline, texts: pd.Series) -> pd.DataFrame:
    """Predict labels and class probabilities.

    Args:
        model: Fitted pipeline.
        texts: Raw review texts.

    Returns:
        DataFrame with ``pred`` and ``p_<class>`` columns, aligned to ``texts``.
    """
    proba = model.predict_proba(texts.tolist())
    classes = model.classes_
    out = pd.DataFrame(proba, columns=[f"p_{c}" for c in classes], index=texts.index)
    out["pred"] = classes[np.argmax(proba, axis=1)]
    return out


def load_model() -> Pipeline:
    """Load the fitted baseline from ``paths.models``."""
    return joblib.load(get_path("models") / MODEL_FILE)


def compare_class_weights() -> pd.DataFrame:
    """Train one baseline per class-weight option and score each on the dev split.

    Only star-labeled train and dev data are used; the gold set is not read.

    Returns:
        One row per option with dev macro-F1, accuracy, and the predicted and
        true share of each class. Also written to ``paths.baseline_class_weights``.
    """
    from src.models.evaluate import metrics

    seed = set_seed()
    cfg = load_config()
    classes = cfg["labels"]["sentiment_classes"]
    train, dev = split_dev(load_star_labeled(), seed)
    rows = []
    for option in cfg["models"]["baseline"]["class_weight_options"]:
        model = build_pipeline(seed, resolve_class_weight(option, train["label"]))
        model.fit(train["text"].tolist(), train["label"])
        pred = predict_frame(model, dev["text"])["pred"]
        m = metrics(dev["label"], pred, classes)
        row = {"class_weight": str(option), "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}
        for c in classes:
            row[f"pred_share_{c}"] = float((pred == c).mean())
            row[f"true_share_{c}"] = float((dev["label"] == c).mean())
        rows.append(row)
        logger.info("class_weight=%s: dev macro-F1 %.4f", option, m["macro_f1"])
    table = pd.DataFrame(rows)
    table.to_csv(get_path("baseline_class_weights"), index=False, float_format="%.4f")
    logger.info("Saved class-weight comparison -> %s", get_path("baseline_class_weights"))
    return table


def run() -> None:
    """Train on the train split, predict the dev split and save both."""
    seed = set_seed()
    train, dev = split_dev(load_star_labeled(), seed)
    logger.info("Train %d, dev %d reviews", len(train), len(dev))
    logger.info("Train labels:\n%s", train["label"].value_counts().to_string())

    option = load_config()["models"]["baseline"]["class_weight"]
    logger.info("class_weight: %s", option)
    model = build_pipeline(seed, resolve_class_weight(option, train["label"]))
    model.fit(train["text"].tolist(), train["label"])
    out = get_path("models") / MODEL_FILE
    joblib.dump(model, out)
    logger.info("Saved model -> %s", out)

    preds = predict_frame(model, dev["text"])
    dev_out = dev[["review_id", "lang", "label"]].join(preds.add_prefix("baseline_"))
    dev_out.to_parquet(get_path("dev_predictions"), index=False)
    logger.info("Saved dev predictions -> %s", get_path("dev_predictions"))


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Train the TF-IDF baseline.")
    parser.add_argument(
        "--compare-class-weights", action="store_true",
        help="compare class-weight options on the dev split instead of training the final model",
    )
    args = parser.parse_args()
    if args.compare_class_weights:
        compare_class_weights()
    else:
        run()


if __name__ == "__main__":
    main()
