"""TF-IDF + Logistic Regression sentiment baseline trained on star labels.

Features are character n-grams (within word boundaries) and word n-grams,
both computed on normalized text. Character n-grams handle Arabic prefixes,
suffixes and spelling variation without a stemmer. The pipeline normalizes
its input, so it accepts raw review text.

Outputs:

- ``paths.models``/``baseline.joblib``: fitted pipeline.
- ``paths.dev_predictions``: predictions on the dev split (column ``baseline``).

Usage::

    python -m src.models.baseline
"""

from __future__ import annotations

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


def build_pipeline(seed: int) -> Pipeline:
    """Create the unfitted baseline pipeline from ``models.baseline`` settings.

    Args:
        seed: Random seed for the classifier.

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
        C=b["C"], class_weight=b["class_weight"], max_iter=b["max_iter"], random_state=seed,
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


def run() -> None:
    """Train on the train split, predict the dev split and save both."""
    seed = set_seed()
    train, dev = split_dev(load_star_labeled(), seed)
    logger.info("Train %d, dev %d reviews", len(train), len(dev))
    logger.info("Train labels:\n%s", train["label"].value_counts().to_string())

    model = build_pipeline(seed)
    model.fit(train["text"].tolist(), train["label"])
    out = get_path("models") / MODEL_FILE
    joblib.dump(model, out)
    logger.info("Saved model -> %s", out)

    preds = predict_frame(model, dev["text"])
    dev_out = dev[["review_id", "lang", "label"]].join(preds.add_prefix("baseline_"))
    dev_out.to_parquet(get_path("dev_predictions"), index=False)
    logger.info("Saved dev predictions -> %s", get_path("dev_predictions"))


if __name__ == "__main__":
    run()
