"""Topic modeling of negative and positive reviews with BERTopic.

Pipeline, run separately for each star group in ``topics.groups``
(1-2 stars and 4-5 stars):

1. Corpus: trainable Google Play reviews in ``topics.languages``. Reviews
   with the same normalized text (thousands of "ممتاز" or "good") are
   modeled once and mapped back, so every share is a share of reviews.
2. Embeddings: a multilingual sentence-transformer on the original text,
   loaded at a pinned revision and cached in ``paths.topic_embeddings``.
3. Clusters: UMAP and HDBSCAN (BERTopic). Keywords per cluster come from
   c-TF-IDF on the normalized text, letters only, without stop words.
4. Outliers: texts that HDBSCAN leaves unclustered join the most similar
   cluster if the cosine similarity reaches ``topics.outlier_similarity``.
   The rest stay ``unassigned``.
5. Naming: each cluster's top keywords are matched against the keyword
   lists in ``topics.keywords`` and the cluster gets the readable topic
   with the largest share of its keyword weight (see :func:`choose_topic`).
   The readable topics are the ones used for the gold set
   (``labels.topics``).

For comparison, every review also gets a topic from the keyword lists alone
(:func:`keyword_topic`), without clustering. Both are checked against the
hand-labeled topics of gold reviews.

Outputs:

- ``paths.review_topics``: cluster and topic per review (not committed).
- ``paths.topic_terms``: top keywords and weights per cluster.
- ``paths.models``/``topics_<group>``: fitted BERTopic model (safetensors)
  and the cluster-to-topic map.
- ``paths.topic_clusters``: one row per cluster with size, keywords, topic.
- ``paths.topic_shares``: topic shares per star group, app and language.
- ``paths.topic_quarterly``: topic counts per star group, app and quarter.
- ``paths.topic_eval_gold``: agreement with gold topic labels.
- ``paths.topic_methods``: topic shares from clusters, keyword rules and
  gold hand labels.
- ``paths.figures``/``topics_*.png``: charts.

Usage::

    python -m src.models.topics            # fit both groups, then summarize
    python -m src.models.topics --rename   # re-apply keyword naming to saved clusters
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

from src.analysis.eda import load_clean, stopwords_for
from src.config import get_logger, get_path, load_config, set_seed
from src.label.gold import gold_id, load_gold, usable

logger = get_logger("models.topics")

GENERAL, OTHER, UNASSIGNED = "general", "other", "unassigned"
# Words of at least two letters (no digits or underscores), any script.
TOKEN_PATTERN = r"(?u)\b[^\W\d_]{2,}\b"
TOKEN = re.compile(TOKEN_PATTERN)
CORPUS_COLUMNS = ["review_id", "app", "lang", "rating", "month", "text", "text_norm"]


# Naming --------------------------------------------------------------------

def token_variants(token: str, prefixes: list[str]) -> set[str]:
    """Return a token and its forms with one leading Arabic prefix removed.

    Args:
        token: Normalized word.
        prefixes: Prefixes such as ``ال`` or ``وال``.

    Returns:
        The token, plus each form without a matching prefix that keeps at
        least two letters.
    """
    out = {token}
    for prefix in prefixes:
        if token.startswith(prefix) and len(token) - len(prefix) >= 2:
            out.add(token[len(prefix):])
    return out


class KeywordMatcher:
    """Find the topics whose keywords match a word or phrase.

    Keywords are normalized words. A trailing ``*`` matches any ending, an
    entry with a space must appear as consecutive words, and Arabic words
    also match after one of the configured prefixes (``الرسوم`` matches
    ``رسوم``). Results per word are cached.
    """

    def __init__(self, keywords: dict[str, list[str]], prefixes: list[str]) -> None:
        """Compile the keyword lists.

        Args:
            keywords: Topic -> keyword list, in priority order for ties.
            prefixes: Arabic prefixes that may precede a keyword.
        """
        self.topics = list(keywords)
        self.prefixes = prefixes
        self.exact: dict[str, set[str]] = {}
        self.stems: dict[str, set[str]] = {}
        self.phrases: list[tuple[str, str]] = []
        for topic, words in keywords.items():
            for word in words:
                if " " in word:
                    self.phrases.append((f" {word} ", topic))
                elif word.endswith("*"):
                    self.stems.setdefault(word[:-1], set()).add(topic)
                else:
                    self.exact.setdefault(word, set()).add(topic)
        self._cache: dict[str, frozenset[str]] = {}

    def token_topics(self, token: str) -> frozenset[str]:
        """Return the topics matching one word."""
        if token not in self._cache:
            found: set[str] = set()
            for form in token_variants(token, self.prefixes):
                found |= self.exact.get(form, set())
                for k in range(1, len(form) + 1):
                    found |= self.stems.get(form[:k], set())
            self._cache[token] = frozenset(found)
        return self._cache[token]

    def term_topics(self, term: str) -> frozenset[str]:
        """Return the topics matching a word or phrase."""
        found: set[str] = set()
        for token in term.split():
            found |= self.token_topics(token)
        padded = f" {term} "
        found |= {topic for phrase, topic in self.phrases if phrase in padded}
        return frozenset(found)


def default_matcher() -> KeywordMatcher:
    """Return a matcher built from ``topics.keywords``."""
    t = load_config()["topics"]
    return KeywordMatcher(t["keywords"], t["arabic_prefixes"])


def topic_scores(terms: list[tuple[str, float]], matcher: KeywordMatcher) -> dict[str, float]:
    """Compute the share of a cluster's keyword weight that matches each topic.

    Args:
        terms: ``(term, weight)`` pairs, e.g. a cluster's top c-TF-IDF terms.
        matcher: Keyword matcher.

    Returns:
        Topic -> share of the total positive weight, between 0 and 1. A term
        that matches several topics counts for each of them.
    """
    scores = dict.fromkeys(matcher.topics, 0.0)
    total = sum(w for _, w in terms if w > 0)
    if total <= 0:
        return scores
    for term, weight in terms:
        if weight > 0:
            for topic in matcher.term_topics(term):
                scores[topic] += weight / total
    return scores


def choose_topic(scores: dict[str, float], min_share: float) -> str:
    """Pick a readable topic from keyword scores.

    A specific topic is preferred over ``general``: the best-scoring
    specific topic is chosen if it reaches ``min_share``. Otherwise the
    result is ``general`` if general opinion words reach ``min_share``, and
    ``other`` if nothing does. Ties go to the topic listed first.

    Args:
        scores: Output of :func:`topic_scores`.
        min_share: Minimum share of keyword weight.

    Returns:
        Topic key.
    """
    specific = {t: s for t, s in scores.items() if t != GENERAL}
    if specific:
        best = max(specific, key=lambda t: specific[t])
        if specific[best] >= min_share:
            return best
    if scores.get(GENERAL, 0.0) >= min_share:
        return GENERAL
    return OTHER


def review_terms(text_norm: str) -> list[tuple[str, float]]:
    """Return the words and two-word phrases of a review, each with weight 1.

    Args:
        text_norm: Normalized review text.

    Returns:
        ``(term, 1.0)`` pairs.
    """
    tokens = TOKEN.findall(text_norm or "")
    bigrams = [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    return [(t, 1.0) for t in tokens] + [(b, 1.0) for b in bigrams]


def keyword_topic(text_norm: str, matcher: KeywordMatcher) -> str:
    """Assign a topic to a single review from keyword matches alone.

    Any specific keyword in the review is enough; the topic with the most
    matching words wins (ties go to the topic listed first).

    Args:
        text_norm: Normalized review text.
        matcher: Keyword matcher.

    Returns:
        Topic key; ``other`` if no keyword matches.
    """
    return choose_topic(topic_scores(review_terms(text_norm), matcher), min_share=1e-9)


def name_clusters(terms: dict[int, list[tuple[str, float]]], matcher: KeywordMatcher | None = None) -> pd.DataFrame:
    """Name each cluster from its top keywords.

    Args:
        terms: Cluster id -> ``(term, weight)`` pairs. Cluster ``-1`` (texts
            left unclustered) is named ``unassigned``.
        matcher: Keyword matcher; defaults to :func:`default_matcher`.

    Returns:
        One row per cluster with ``topic``, its keyword ``share``, and the
        runner-up topic and share.
    """
    matcher = matcher or default_matcher()
    min_share = load_config()["topics"]["min_share"]
    rows = []
    for cluster, pairs in sorted(terms.items()):
        if cluster == -1:
            rows.append({"cluster": -1, "topic": UNASSIGNED})
            continue
        scores = topic_scores(pairs, matcher)
        topic = choose_topic(scores, min_share)
        ranked = sorted(((s, k) for k, s in scores.items() if k != topic), reverse=True)
        rows.append({
            "cluster": cluster, "topic": topic, "share": scores.get(topic, np.nan),
            "runner_up": ranked[0][1], "runner_up_share": ranked[0][0],
        })
    return pd.DataFrame(rows)


# Corpus and model -----------------------------------------------------------

def load_corpus(group: str) -> pd.DataFrame:
    """Load the reviews of one star group.

    Args:
        group: Key of ``topics.groups`` (``negative`` or ``positive``).

    Returns:
        Trainable reviews from ``topics.source`` in ``topics.languages`` with
        a star rating in the group, :data:`CORPUS_COLUMNS` only.
    """
    t = load_config()["topics"]
    df = load_clean(source=t["source"])
    keep = df["trainable"] & df["lang"].isin(t["languages"]) & df["rating"].isin(t["groups"][group])
    return df.loc[keep, CORPUS_COLUMNS].reset_index(drop=True)


def unique_texts(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Collapse reviews with the same normalized text.

    Args:
        df: Reviews with ``text`` and ``text_norm``.

    Returns:
        ``(unique, codes)``: one row per distinct ``text_norm`` (in order of
        first appearance) with the first original ``text`` and ``n_reviews``;
        ``codes[i]`` is the row of ``unique`` for review ``i``.
    """
    codes, uniques = pd.factorize(df["text_norm"])
    _, first = np.unique(codes, return_index=True)
    unique = pd.DataFrame({
        "text_norm": np.asarray(uniques, dtype=object),
        "text": df["text"].to_numpy()[first],
        "n_reviews": np.bincount(codes),
    })
    return unique, codes


def load_embedding_model():  # noqa: ANN201 - optional heavy dependency
    """Load the sentence-transformer at the configured revision."""
    from sentence_transformers import SentenceTransformer

    e = load_config()["topics"]["embedding"]
    return SentenceTransformer(e["name"], revision=e["revision"], device="cpu")


def embed(texts: list[str], group: str, model) -> np.ndarray:  # noqa: ANN001
    """Embed texts, reusing the cached embeddings if the inputs are unchanged.

    Args:
        texts: Original review texts.
        group: Star group, used in the cache file name.
        model: Loaded sentence-transformer.

    Returns:
        L2-normalized embeddings, one row per text.
    """
    e = load_config()["topics"]["embedding"]
    key = hashlib.sha256(json.dumps([e["name"], e["revision"], texts], ensure_ascii=False).encode()).hexdigest()
    folder = get_path("topic_embeddings")
    path, meta = folder / f"{group}.npy", folder / f"{group}.json"
    if path.exists() and meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("key") == key:
        logger.info("%s: using cached embeddings", group)
        return np.load(path)
    logger.info("%s: embedding %d texts", group, len(texts))
    vectors = model.encode(texts, batch_size=e["batch_size"], normalize_embeddings=True, show_progress_bar=False)
    np.save(path, vectors)
    meta.write_text(json.dumps({"key": key, "model": e["name"], "revision": e["revision"]}), encoding="utf-8")
    return vectors


def build_model(group: str, embedding_model):  # noqa: ANN001, ANN201
    """Create an unfitted BERTopic model from the config.

    Args:
        group: Star group (sets HDBSCAN's minimum cluster size).
        embedding_model: Loaded sentence-transformer.

    Returns:
        BERTopic model.
    """
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    cfg = load_config()
    t = cfg["topics"]
    stop = sorted(stopwords_for("arabic") | stopwords_for("english"))
    vectorizer = CountVectorizer(
        ngram_range=tuple(t["vectorizer"]["ngram_range"]), min_df=t["vectorizer"]["min_df"],
        token_pattern=TOKEN_PATTERN, stop_words=stop,
    )
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=UMAP(**t["umap"], random_state=cfg["project"]["random_seed"]),
        hdbscan_model=HDBSCAN(
            min_cluster_size=t["hdbscan"]["min_cluster_size"][group], min_samples=t["hdbscan"]["min_samples"],
            metric="euclidean", cluster_selection_method="eom", prediction_data=True,
        ),
        vectorizer_model=vectorizer,
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        top_n_words=t["top_n_words"],
        calculate_probabilities=False,
    )


def fit_group(group: str, embedding_model) -> tuple[pd.DataFrame, dict[int, list[tuple[str, float]]]]:  # noqa: ANN001
    """Fit BERTopic on one star group and save the model.

    Args:
        group: Star group.
        embedding_model: Loaded sentence-transformer.

    Returns:
        ``(reviews, terms)``: the group's reviews with ``group`` and
        ``cluster`` columns, and cluster id -> top ``(term, weight)`` pairs.
    """
    t = load_config()["topics"]
    corpus = load_corpus(group)
    unique, codes = unique_texts(corpus)
    logger.info("%s: %d reviews, %d distinct texts", group, len(corpus), len(unique))
    vectors = embed(unique["text"].tolist(), group, embedding_model)
    docs = unique["text_norm"].tolist()

    model = build_model(group, embedding_model)
    clusters, _ = model.fit_transform(docs, embeddings=vectors)
    n_outliers = int(np.sum(np.asarray(clusters) == -1))
    clusters = model.reduce_outliers(
        docs, clusters, strategy="embeddings", embeddings=vectors, threshold=t["outlier_similarity"],
    )
    model.update_topics(
        docs, topics=clusters, top_n_words=t["top_n_words"],
        vectorizer_model=model.vectorizer_model, ctfidf_model=model.ctfidf_model,
    )
    clusters = np.asarray(clusters)
    logger.info(
        "%s: %d clusters; %d of %d texts unclustered, %d after reassignment",
        group, len(set(clusters) - {-1}), n_outliers, len(docs), int(np.sum(clusters == -1)),
    )

    terms = {int(c): [(w, float(s)) for w, s in model.get_topic(int(c))] for c in sorted(set(clusters))}
    out = get_path("models") / f"topics_{group}"
    model.save(out, serialization="safetensors", save_ctfidf=True, save_embedding_model=t["embedding"]["name"])

    reviews = corpus.assign(group=group, cluster=clusters[codes])
    return reviews, terms


# Summaries ------------------------------------------------------------------

def assign_topics(reviews: pd.DataFrame, terms: dict[str, dict[int, list[tuple[str, float]]]]) -> pd.DataFrame:
    """Add the cluster topic and the keyword-only topic to each review.

    Args:
        reviews: Reviews with ``group``, ``cluster`` and ``text_norm``.
        terms: Group -> cluster id -> ``(term, weight)`` pairs.

    Returns:
        ``reviews`` with ``topic`` and ``keyword_topic`` columns.
    """
    matcher = default_matcher()
    parts = []
    for group, part in reviews.groupby("group", sort=False):
        names = name_clusters(terms[group], matcher).set_index("cluster")["topic"]
        parts.append(part.assign(topic=part["cluster"].map(names)))
    out = pd.concat(parts).sort_index()
    by_text = {text: keyword_topic(text, matcher) for text in out["text_norm"].unique()}
    out["keyword_topic"] = out["text_norm"].map(by_text)
    return out


def cluster_table(reviews: pd.DataFrame, terms: dict[str, dict[int, list[tuple[str, float]]]]) -> pd.DataFrame:
    """Describe every cluster: size, top keywords and readable topic.

    Args:
        reviews: Output of :func:`assign_topics`.
        terms: Group -> cluster id -> ``(term, weight)`` pairs.

    Returns:
        One row per group and cluster, largest clusters first.
    """
    rows = []
    for group, part in reviews.groupby("group", sort=False):
        sizes = part.groupby("cluster").agg(n_reviews=("review_id", "size"), n_texts=("text_norm", "nunique"))
        named = name_clusters(terms[group]).set_index("cluster")
        table = sizes.join(named, how="left")
        table["top_terms"] = [", ".join(w for w, _ in terms[group].get(c, [])) for c in table.index]
        rows.append(table.reset_index().assign(group=group))
    cols = ["group", "cluster", "n_reviews", "n_texts", "topic", "share", "runner_up", "runner_up_share", "top_terms"]
    out = pd.concat(rows)[cols]
    return out.sort_values(["group", "n_reviews"], ascending=[True, False]).reset_index(drop=True)


def share_table(reviews: pd.DataFrame, by: list[str], topic_col: str = "topic") -> pd.DataFrame:
    """Count reviews per topic within groups and compute each topic's share.

    Args:
        reviews: Reviews with the ``by`` columns and a topic column.
        by: Grouping columns.
        topic_col: Column with the topic.

    Returns:
        ``by`` columns, ``topic``, ``n``, ``total`` and ``share`` (``n`` /
        ``total`` within each group).
    """
    counts = reviews.groupby([*by, topic_col], observed=True).size().rename("n").reset_index()
    counts = counts.rename(columns={topic_col: "topic"})
    counts["total"] = counts.groupby(by)["n"].transform("sum")
    counts["share"] = counts["n"] / counts["total"]
    return counts


def topic_shares(reviews: pd.DataFrame) -> pd.DataFrame:
    """Topic shares per star group: all reviews, per app, and per app and language.

    Args:
        reviews: Output of :func:`assign_topics`.

    Returns:
        Long table from :func:`share_table` with ``app`` and ``lang`` set to
        ``all`` where not split. Languages are those in
        ``topics.share_languages``.
    """
    langs = load_config()["topics"]["share_languages"]
    parts = [
        share_table(reviews.assign(app="all", lang="all"), ["group", "app", "lang"]),
        share_table(reviews.assign(lang="all"), ["group", "app", "lang"]),
        share_table(reviews[reviews["lang"].isin(langs)], ["group", "app", "lang"]),
    ]
    return pd.concat(parts, ignore_index=True)


def quarterly(reviews: pd.DataFrame, last_date: pd.Timestamp) -> pd.DataFrame:
    """Topic counts per star group, app and calendar quarter.

    Args:
        reviews: Output of :func:`assign_topics`.
        last_date: Date of the last review in the data (timezone-naive).
            Quarters that end after it are incomplete and dropped.

    Returns:
        ``group, app, quarter, topic, n, total, share``, with ``quarter`` as
        text such as ``2026Q3``.
    """
    df = reviews.assign(quarter=reviews["month"].dt.to_period("Q"))
    df = df[df["quarter"].dt.end_time <= last_date]
    table = share_table(df, ["group", "app", "quarter"])
    table["quarter"] = table["quarter"].astype(str)
    return table


def topic_agreement(hand: pd.Series, pred: pd.Series, min_support: int) -> pd.DataFrame:
    """Compare predicted topics with hand-labeled topics.

    Args:
        hand: Hand-labeled topic per review.
        pred: Predicted topic per review (may include ``unassigned``).
        min_support: Minimum number of hand-labeled reviews for a topic to
            get its own precision, recall and F1.

    Returns:
        Long table with ``topic`` (``all`` for overall metrics), ``n``,
        ``metric`` and ``value``. Overall metrics: ``accuracy``, Cohen's
        ``kappa`` and ``specific_vs_general`` (agreement on whether a review
        has one of the specific topics; ``general``, ``other`` and
        ``unassigned`` count as no specific topic).
    """
    hand, pred = hand.reset_index(drop=True), pred.reset_index(drop=True)
    no_topic = {GENERAL, OTHER, UNASSIGNED}
    rows = [
        {"topic": "all", "n": len(hand), "metric": "accuracy", "value": float((hand == pred).mean())},
        {"topic": "all", "n": len(hand), "metric": "kappa", "value": float(cohen_kappa_score(hand, pred))},
        {"topic": "all", "n": len(hand), "metric": "specific_vs_general",
         "value": float((hand.isin(no_topic) == pred.isin(no_topic)).mean())},
    ]
    support = hand.value_counts()
    topics = [t for t in support.index if support[t] >= min_support]
    p, r, f, _ = precision_recall_fscore_support(hand, pred, labels=topics, zero_division=0)
    for i, topic in enumerate(topics):
        for metric, values in (("precision", p), ("recall", r), ("f1", f)):
            rows.append({"topic": topic, "n": int(support[topic]), "metric": metric, "value": float(values[i])})
    return pd.DataFrame(rows)


def gold_topics(reviews: pd.DataFrame) -> pd.DataFrame:
    """Join the usable gold reviews with their predicted topics.

    Args:
        reviews: Output of :func:`assign_topics`.

    Returns:
        One row per gold review found in ``reviews``: ``gold_id``, ``app``,
        ``lang``, ``weight``, ``text``, ``hand`` (hand-labeled topic),
        ``group``, ``topic`` and ``keyword_topic``.
    """
    gold = usable(load_gold())[["gold_id", "app", "lang", "weight", "text", "topic"]]
    gold = gold.rename(columns={"topic": "hand"})
    ids = reviews[["review_id", "group", "topic", "keyword_topic"]].assign(
        gold_id=reviews["review_id"].map(gold_id),
    )
    return gold.merge(ids.drop(columns="review_id"), on="gold_id", how="inner")


def gold_evaluation(gold: pd.DataFrame, min_support: int = 10) -> pd.DataFrame:
    """Check cluster topics and keyword topics against gold topic labels.

    Args:
        gold: Output of :func:`gold_topics`.
        min_support: See :func:`topic_agreement`.

    Returns:
        :func:`topic_agreement` rows for each method (``clusters`` and
        ``keywords``) and star group (``all`` for both groups together).
    """
    parts = []
    for method, col in (("clusters", "topic"), ("keywords", "keyword_topic")):
        for group, part in [("all", gold), *gold.groupby("group")]:
            table = topic_agreement(part["hand"], part[col], min_support)
            parts.append(table.assign(method=method, group=group))
    cols = ["method", "group", "topic", "n", "metric", "value"]
    return pd.concat(parts, ignore_index=True)[cols]


def method_shares(reviews: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Topic shares per star group from clusters, keyword rules and hand labels.

    Cluster and keyword shares are over all reviews in a star group.
    Hand-label shares are over the gold reviews in the group, weighted by
    their ``weight`` so that apps and languages count as in the population;
    they are an independent estimate from a small sample.

    Args:
        reviews: Output of :func:`assign_topics`.
        gold: Output of :func:`gold_topics`.

    Returns:
        ``group, method, topic, n, total, share``, where ``n`` and ``total``
        are unweighted review counts.
    """
    hand = gold.groupby(["group", "hand"]).agg(n=("weight", "size"), w=("weight", "sum")).reset_index()
    hand = hand.rename(columns={"hand": "topic"})
    hand["total"] = hand.groupby("group")["n"].transform("sum")
    hand["share"] = hand["w"] / hand.groupby("group")["w"].transform("sum")
    parts = [
        share_table(reviews, ["group"]).assign(method="clusters"),
        share_table(reviews, ["group"], topic_col="keyword_topic").assign(method="keywords"),
        hand.drop(columns="w").assign(method="hand_labels"),
    ]
    return pd.concat(parts, ignore_index=True)[["group", "method", "topic", "n", "total", "share"]]


# Figures --------------------------------------------------------------------

def topic_label(topic: str) -> str:
    """Return the display name of a topic key."""
    names = {**load_config()["labels"]["topics"], UNASSIGNED: "Not assigned"}
    return names.get(topic, topic)


def plot_shares(shares: pd.DataFrame) -> None:
    """Save a chart of specific-topic shares per app for both star groups.

    Args:
        shares: Output of :func:`topic_shares`.
    """
    import matplotlib.pyplot as plt

    from src.analysis.plots import VIZ, app_color, app_name, caption, save, setup_style

    setup_style()
    apps = list(load_config()["apps"])
    groups = list(load_config()["topics"]["groups"])
    fig, axes = plt.subplots(1, len(groups), figsize=(10, 4.2))
    notes = []
    for ax, group in zip(np.atleast_1d(axes), groups):
        part = shares[(shares["group"] == group) & (shares["lang"] == "all") & (shares["app"].isin(apps))]
        wide = part.pivot(index="topic", columns="app", values="share").reindex(columns=apps).fillna(0)
        rest = wide.loc[wide.index.isin([GENERAL, OTHER, UNASSIGNED])]
        wide = wide.drop(index=rest.index).sort_values(apps[0])
        height = 0.38
        y = np.arange(len(wide))
        for i, app in enumerate(apps):
            bars = ax.barh(y + (0.5 - i) * height, wide[app], height=height * 0.92, color=app_color(app),
                           label=app_name(app))
            ax.bar_label(bars, labels=[f"{v:.0%}" for v in wide[app]], padding=2, fontsize=7.5, color=VIZ["muted"])
        ax.set_yticks(y, [topic_label(t) for t in wide.index])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        stars = "1–2" if group == "negative" else "4–5"
        ax.set_title(f"{group.capitalize()} reviews ({stars} stars)", fontsize=10)
        ax.margins(x=0.15)
        other = ", ".join(
            f"{app_name(a)} {rest[a].sum():.0%}" for a in apps
        )
        notes.append(f"{group}: {other}")
    np.atleast_1d(axes)[0].legend(loc="lower right")
    fig.suptitle("Share of reviews about each specific topic", x=0.01, ha="left", fontsize=11)
    caption(fig, "Remaining reviews have no specific topic (general opinion, other or not assigned): "
                 + "; ".join(notes) + ". Google Play reviews, topics from BERTopic clusters.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "topics_share")
    plt.close(fig)


def plot_quarterly(table: pd.DataFrame, group: str = "negative", n_topics: int = 6) -> None:
    """Save small multiples of quarterly review counts for the largest specific topics.

    Args:
        table: Output of :func:`quarterly`.
        group: Star group to plot.
        n_topics: Number of topics (panels).
    """
    import matplotlib.pyplot as plt

    from src.analysis.plots import app_color, app_name, caption, save, setup_style

    setup_style()
    apps = list(load_config()["apps"])
    part = table[(table["group"] == group) & ~table["topic"].isin([GENERAL, OTHER, UNASSIGNED])]
    top = part.groupby("topic")["n"].sum().sort_values(ascending=False).index[:n_topics]
    quarters = pd.PeriodIndex(sorted(table["quarter"].unique()), freq="Q")
    x = quarters.to_timestamp()
    ncols = 3
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.6 * nrows), sharex=True, sharey=True, squeeze=False)
    for ax, topic in zip(axes.flat, top):
        for app in apps:
            series = (part[(part["topic"] == topic) & (part["app"] == app)]
                      .set_index("quarter")["n"].reindex(quarters.astype(str), fill_value=0))
            ax.plot(x, series.to_numpy(), color=app_color(app), label=app_name(app), linewidth=1.8)
        ax.set_title(topic_label(topic), fontsize=9.5)
    for ax in axes.flat[len(top):]:
        ax.set_visible(False)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{group.capitalize()} reviews per quarter, by topic", x=0.01, ha="left", fontsize=11)
    caption(fig, "Number of 1–2 star Google Play reviews per calendar quarter (complete quarters only).")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, f"topics_{group}_by_quarter")
    plt.close(fig)


# Run ------------------------------------------------------------------------

def load_terms() -> dict[str, dict[int, list[tuple[str, float]]]]:
    """Read the saved cluster keywords (JSON keys back to integers)."""
    raw = json.loads(get_path("topic_terms").read_text(encoding="utf-8"))
    return {g: {int(c): [(w, float(s)) for w, s in pairs] for c, pairs in clusters.items()} for g, clusters in raw.items()}


def summarize(reviews: pd.DataFrame, terms: dict[str, dict[int, list[tuple[str, float]]]]) -> pd.DataFrame:
    """Name clusters and write the topic tables and figures.

    Args:
        reviews: Reviews with ``group`` and ``cluster``.
        terms: Group -> cluster id -> ``(term, weight)`` pairs.

    Returns:
        Reviews with ``topic`` and ``keyword_topic``.
    """
    reviews = assign_topics(reviews, terms)
    reviews.to_parquet(get_path("review_topics"), index=False)
    clusters = cluster_table(reviews, terms)
    clusters.to_csv(get_path("topic_clusters"), index=False, float_format="%.3f")
    shares = topic_shares(reviews)
    shares.to_csv(get_path("topic_shares"), index=False, float_format="%.4f")
    dates = pd.read_parquet(get_path("reviews_clean"), columns=["source", "review_date"])
    last = dates.loc[dates["source"] == load_config()["topics"]["source"], "review_date"].max()
    table = quarterly(reviews, last.tz_convert("UTC").tz_localize(None))
    table.to_csv(get_path("topic_quarterly"), index=False, float_format="%.4f")
    gold = gold_topics(reviews)
    evaluation = gold_evaluation(gold)
    evaluation.to_csv(get_path("topic_eval_gold"), index=False, float_format="%.4f")
    method_shares(reviews, gold).to_csv(get_path("topic_methods"), index=False, float_format="%.4f")
    for group in load_config()["topics"]["groups"]:
        topic_map = clusters[clusters["group"] == group].set_index("cluster")["topic"].to_dict()
        folder = get_path("models") / f"topics_{group}"
        if folder.exists():
            (folder / "topic_map.json").write_text(json.dumps({int(k): v for k, v in topic_map.items()}, indent=1))
    plot_shares(shares)
    plot_quarterly(table)
    overall = shares[(shares["app"] == "all")].pivot(index="topic", columns="group", values="share")
    logger.info("Topic shares:\n%s", overall.round(3).to_string())
    logger.info("Gold check:\n%s", evaluation[evaluation["topic"] == "all"].round(3).to_string())
    return reviews


def run(rename: bool = False) -> None:
    """Fit the topic models (unless ``rename``) and write all outputs.

    Args:
        rename: Re-apply keyword naming to the saved clusters without
            fitting again.
    """
    set_seed()
    if rename:
        reviews = pd.read_parquet(get_path("review_topics")).drop(columns=["topic", "keyword_topic"], errors="ignore")
        summarize(reviews, load_terms())
        return
    embedding_model = load_embedding_model()
    parts, terms = [], {}
    for group in load_config()["topics"]["groups"]:
        reviews, terms[group] = fit_group(group, embedding_model)
        parts.append(reviews)
    get_path("topic_terms").write_text(
        json.dumps({g: {str(c): p for c, p in t.items()} for g, t in terms.items()}, ensure_ascii=False),
        encoding="utf-8",
    )
    summarize(pd.concat(parts, ignore_index=True), terms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rename", action="store_true", help="re-apply keyword naming to saved clusters")
    run(rename=parser.parse_args().rename)
