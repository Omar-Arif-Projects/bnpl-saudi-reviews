"""Zero-shot sentiment from pretrained transformer models.

Models are listed under ``models.transformers.models`` in ``config.yaml``:

- ``camelbert``: CAMeLBERT-DA sentiment, Arabic only.
- ``xlmr``: multilingual XLM-RoBERTa sentiment (Arabic and English).

Neither model is trained on this data. Each model only scores reviews in its
configured languages; other rows get no prediction.

Outputs:

- ``paths.dev_predictions``: adds ``<model>_pred`` and ``<model>_p_<class>``
  columns for the comparison sample.
- ``paths.gold_predictions``: predictions of every model (including the
  baseline) on the gold set, keyed by ``gold_id``. Contains no review text.

Usage::

    python -m src.models.transformer
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import get_logger, get_path, load_config, set_seed
from src.label.gold import load_gold
from src.models.baseline import load_model, predict_frame
from src.models.data import comparison_sample, load_star_labeled, split_dev

logger = get_logger("models.transformer")


class ZeroShotSentiment:
    """Wrapper around a Hugging Face sequence-classification sentiment model.

    Args:
        name: Model name on the Hugging Face Hub.
        classes: Project sentiment classes; model labels are matched to them
            case-insensitively.
        batch_size: Texts per forward pass.
        max_length: Maximum tokens per text.

    Raises:
        ValueError: If the model's labels do not cover ``classes``.
    """

    def __init__(self, name: str, classes: list[str], batch_size: int, max_length: int) -> None:
        self.name = name
        self.classes = classes
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        labels = {i: str(lab).lower() for i, lab in self.model.config.id2label.items()}
        missing = set(classes) - set(labels.values())
        if missing:
            raise ValueError(f"{name} has labels {labels}, missing {missing}")
        # Column order of the model's logits, rearranged to ``classes`` order.
        self.order = [next(i for i, lab in labels.items() if lab == c) for c in classes]

    @torch.inference_mode()
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return class probabilities in ``classes`` order.

        Args:
            texts: Raw review texts.

        Returns:
            Array of shape ``(len(texts), len(classes))``.
        """
        out = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                texts[start:start + self.batch_size], padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            logits = self.model(**batch).logits
            out.append(torch.softmax(logits, dim=-1)[:, self.order].numpy())
        return np.vstack(out) if out else np.empty((0, len(self.classes)))

    def predict_frame(self, texts: pd.Series) -> pd.DataFrame:
        """Predict labels and probabilities.

        Args:
            texts: Raw review texts.

        Returns:
            DataFrame with ``pred`` and ``p_<class>`` columns, aligned to ``texts``.
        """
        proba = self.predict_proba(texts.tolist())
        out = pd.DataFrame(proba, columns=[f"p_{c}" for c in self.classes], index=texts.index)
        out["pred"] = np.array(self.classes)[np.argmax(proba, axis=1)] if len(out) else []
        return out


def score(df: pd.DataFrame, key: str, model: ZeroShotSentiment, languages: list[str]) -> pd.DataFrame:
    """Add one model's predictions for rows in its languages.

    Args:
        df: DataFrame with ``text`` and ``lang``.
        key: Column prefix for the model.
        model: Loaded model.
        languages: Languages the model scores.

    Returns:
        ``df`` with ``<key>_pred`` and ``<key>_p_<class>`` columns; rows in
        other languages are missing.
    """
    mask = df["lang"].isin(languages) & (df["text"] != "")
    preds = model.predict_frame(df.loc[mask, "text"]).add_prefix(f"{key}_")
    return df.join(preds)


def run() -> None:
    """Score the dev comparison sample and the gold set with all models."""
    seed = set_seed()
    cfg = load_config()
    t = cfg["models"]["transformers"]
    classes = cfg["labels"]["sentiment_classes"]

    _, dev = split_dev(load_star_labeled(), seed)
    sample = comparison_sample(dev, seed)
    gold = load_gold()[["gold_id", "lang", "text"]]

    baseline = load_model()
    gold = gold.join(predict_frame(baseline, gold["text"]).add_prefix("baseline_"))

    for key, spec in t["models"].items():
        logger.info("Loading %s (%s)", key, spec["name"])
        model = ZeroShotSentiment(spec["name"], classes, t["batch_size"], t["max_length"])
        sample = score(sample, key, model, spec["languages"])
        gold = score(gold, key, model, spec["languages"])
        logger.info("%s: scored %d dev and %d gold reviews", key,
                    sample[f"{key}_pred"].notna().sum(), gold[f"{key}_pred"].notna().sum())

    dev_path = get_path("dev_predictions")
    dev_preds = pd.read_parquet(dev_path)
    model_cols = [c for c in sample.columns if c.split("_")[0] in t["models"]]
    dev_preds = dev_preds.drop(columns=[c for c in model_cols if c in dev_preds], errors="ignore")
    dev_preds = dev_preds.merge(sample[["review_id", *model_cols]], on="review_id", how="left")
    dev_preds["in_comparison"] = dev_preds["review_id"].isin(sample["review_id"])
    dev_preds.to_parquet(dev_path, index=False)
    logger.info("Updated dev predictions -> %s", dev_path)

    gold.drop(columns="text").to_csv(get_path("gold_predictions"), index=False, float_format="%.4f")
    logger.info("Saved gold predictions -> %s", get_path("gold_predictions"))


if __name__ == "__main__":
    run()
