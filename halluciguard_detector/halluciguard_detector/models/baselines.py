"""
models / baselines.py
─────────────────────
Shortcut baselines M0a (token count) and M0b (TF-IDF).
Used to detect style/length artifacts. Never deployed to production.
"""

from __future__ import annotations
from typing import List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


class M0aTokenCountBaseline:
    """M0a: Claim token count -> LogisticRegression."""

    def __init__(self) -> None:
        self.clf = LogisticRegression()
        self.fitted = False

    def fit(self, claims: List[str], labels: List[int]) -> None:
        X = np.array([[len(c.split())] for c in claims])
        self.clf.fit(X, labels)
        self.fitted = True

    def predict_proba(self, claims: List[str]) -> List[float]:
        if not self.fitted:
            return [0.5] * len(claims)
        X = np.array([[len(c.split())] for c in claims])
        probs = self.clf.predict_proba(X)[:, 1]
        return [float(p) for p in probs]


class M0bTfidfBaseline:
    """M0b: TF-IDF on claim text -> LogisticRegression."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.clf = LogisticRegression()
        self.fitted = False

    def fit(self, claims: List[str], labels: List[int]) -> None:
        X = self.vectorizer.fit_transform(claims)
        self.clf.fit(X, labels)
        self.fitted = True

    def predict_proba(self, claims: List[str]) -> List[float]:
        if not self.fitted:
            return [0.5] * len(claims)
        X = self.vectorizer.transform(claims)
        probs = self.clf.predict_proba(X)[:, 1]
        return [float(p) for p in probs]
