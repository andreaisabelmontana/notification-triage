"""The notification priority model.

Pipeline per message:

    text  --TF-IDF--> sparse term vector  --truncated SVD (LSA)--> dense
          --lexical signals (features.py)--------------------------> dense
    concat[ LSA embedding | scaled lexical signals ]  --LogReg--> P(priority)

"Semantic analysis" here means exactly TF-IDF + Latent Semantic Analysis
(truncated SVD over the term-document matrix) plus a small urgency lexicon.
It is not a large language model and makes no claim to be one — it is a compact,
fully inspectable, CPU-only model that trains in well under a second.

The classifier outputs class probabilities for {low, medium, high}. We collapse
those into a single continuous *priority score* in [0, 1] by taking the
probability-weighted average of tier values, then use that score to rank an
inbox into a stable total order.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, lexical_features

# Ordered low -> high so larger class index == higher priority.
PRIORITY_ORDER = ("low", "medium", "high")
PRIORITY_VALUE = {"low": 0.0, "medium": 0.5, "high": 1.0}

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "notifications.csv"


@dataclass
class Notification:
    text: str
    sender: str = ""


@dataclass
class RankedNotification:
    notification: Notification
    score: float
    predicted_priority: str
    rank: int


def load_dataset(path: Path = DATA_PATH):
    """Read the seed CSV into (texts, senders, labels)."""
    texts, senders, labels = [], [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            texts.append(row["text"])
            senders.append(row["sender"])
            labels.append(row["priority"])
    return texts, senders, labels


class PriorityModel:
    """TF-IDF + LSA semantic embedding + lexical signals -> LogisticRegression."""

    def __init__(self, n_components: int = 40, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
        )
        self.svd: TruncatedSVD | None = None
        self.lex_scaler = StandardScaler()
        self.clf = LogisticRegression(
            max_iter=2000,
            C=4.0,
            class_weight="balanced",
            random_state=random_state,
        )
        self._fitted = False

    # -- feature construction -------------------------------------------------

    def _lexical_matrix(self, texts, senders) -> np.ndarray:
        rows = [
            lexical_features(t, s).as_vector() for t, s in zip(texts, senders)
        ]
        return np.asarray(rows, dtype=float)

    def _embed(self, texts, fit: bool):
        tfidf = (
            self.vectorizer.fit_transform(texts)
            if fit
            else self.vectorizer.transform(texts)
        )
        if fit:
            # Keep SVD components below the rank of the term-document matrix.
            n = min(self.n_components, tfidf.shape[1] - 1, len(texts) - 1)
            n = max(n, 2)
            self.svd = TruncatedSVD(
                n_components=n, random_state=self.random_state
            )
            return self.svd.fit_transform(tfidf)
        return self.svd.transform(tfidf)

    def _design_matrix(self, texts, senders, fit: bool) -> np.ndarray:
        emb = self._embed(texts, fit=fit)
        lex = self._lexical_matrix(texts, senders)
        lex = (
            self.lex_scaler.fit_transform(lex)
            if fit
            else self.lex_scaler.transform(lex)
        )
        return np.hstack([emb, lex])

    # -- public API -----------------------------------------------------------

    def fit(self, texts, senders, labels) -> "PriorityModel":
        X = self._design_matrix(texts, senders, fit=True)
        self.clf.fit(X, labels)
        self._fitted = True
        return self

    def priority_score(self, texts, senders) -> np.ndarray:
        """Continuous priority in [0, 1]: prob-weighted average of tier values."""
        if not self._fitted:
            raise RuntimeError("model is not fitted")
        X = self._design_matrix(texts, senders, fit=False)
        proba = self.clf.predict_proba(X)
        tier_vals = np.array([PRIORITY_VALUE[c] for c in self.clf.classes_])
        return proba @ tier_vals

    def predict_priority(self, texts, senders):
        if not self._fitted:
            raise RuntimeError("model is not fitted")
        X = self._design_matrix(texts, senders, fit=False)
        return list(self.clf.predict(X))

    def rank(self, notifications) -> list:
        """Return notifications as a stable, fully-ordered ranked inbox.

        Ties on the continuous score are broken deterministically by original
        index so the ordering is always a strict total order.
        """
        texts = [n.text for n in notifications]
        senders = [n.sender for n in notifications]
        scores = self.priority_score(texts, senders)
        preds = self.predict_priority(texts, senders)

        order = sorted(
            range(len(notifications)),
            key=lambda i: (-scores[i], i),
        )
        ranked = []
        for rank, i in enumerate(order, start=1):
            ranked.append(
                RankedNotification(
                    notification=notifications[i],
                    score=float(scores[i]),
                    predicted_priority=preds[i],
                    rank=rank,
                )
            )
        return ranked


def topic_boost(model: PriorityModel, texts, senders, topics, weight=0.25):
    """Boost each message's score by its semantic similarity to user topics.

    Similarity is cosine in the LSA embedding space — i.e. genuinely semantic,
    not keyword overlap. Returns boosted scores in roughly [0, 1+weight].
    """
    base = model.priority_score(texts, senders)
    msg_emb = model._embed(texts, fit=False)
    topic_emb = model._embed(list(topics), fit=False)

    def _norm(m):
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)

    sims = _norm(msg_emb) @ _norm(topic_emb).T
    best = sims.max(axis=1) if sims.size else np.zeros(len(texts))
    best = np.clip(best, 0.0, 1.0)
    return base + weight * best


def train_default() -> PriorityModel:
    """Convenience: fit a model on the committed seed dataset."""
    texts, senders, labels = load_dataset()
    return PriorityModel().fit(texts, senders, labels)
