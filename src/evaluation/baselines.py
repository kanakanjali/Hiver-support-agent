"""
Baselines — trivial and simple baselines for all three tasks.

Trivial baseline  : random / majority-class / template
Simple baseline   : TF-IDF + LogReg / nearest-neighbor retrieval
"""

import logging
import random
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

import config

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  INTENT CLASSIFICATION BASELINES
# ══════════════════════════════════════════════════════════════════════════════

class RandomIntentBaseline:
    """Trivial baseline: assign a random intent."""

    def __init__(self, seed: int = config.RANDOM_SEED):
        self.rng = random.Random(seed)
        self.labels = config.INTENT_LABELS

    def classify(self, message: str, **kwargs) -> dict:
        return {
            "intent": self.rng.choice(self.labels),
            "confidence": round(1.0 / len(self.labels), 3),
        }

    def classify_batch(self, messages: List[dict]) -> List[dict]:
        return [self.classify(m["customer_message"]) for m in messages]


class TfidfIntentBaseline:
    """
    Simple baseline: TF-IDF features + Logistic Regression.

    Trained on the golden set's labelled examples (leave-one-out or
    on a separate training split).
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000, stop_words="english", ngram_range=(1, 2)
        )
        self.clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.RANDOM_SEED
        )
        self._fitted = False

    def fit(self, messages: List[str], labels: List[str]):
        X = self.vectorizer.fit_transform(messages)
        self.clf.fit(X, labels)
        self._fitted = True
        logger.info("TF-IDF+LogReg fitted on %d examples", len(messages))

    def classify(self, message: str, **kwargs) -> dict:
        if not self._fitted:
            return {"intent": "other", "confidence": 0.0}
        X = self.vectorizer.transform([message])
        pred = self.clf.predict(X)[0]
        proba = self.clf.predict_proba(X).max()
        return {"intent": pred, "confidence": round(float(proba), 3)}

    def classify_batch(self, messages: List[dict]) -> List[dict]:
        return [self.classify(m["customer_message"]) for m in messages]


# ══════════════════════════════════════════════════════════════════════════════
#  REPLY DRAFTING BASELINES
# ══════════════════════════════════════════════════════════════════════════════

class TemplateReplyBaseline:
    """
    Trivial baseline: return a fixed template reply.

    Uses the most common brand reply pattern from historical data.
    """

    DEFAULT_TEMPLATE = (
        "Hi there! We'd like to help you out. "
        "Could you send us a DM with more details about your issue? "
        "We'll take a look and get back to you. ^KA"
    )

    def __init__(self, historical_pairs: Optional[List[dict]] = None):
        if historical_pairs:
            replies = [p["brand_reply"] for p in historical_pairs if p.get("brand_reply")]
            # Find most common reply (or representative one)
            counter = Counter(replies)
            self.template = counter.most_common(1)[0][0] if counter else self.DEFAULT_TEMPLATE
        else:
            self.template = self.DEFAULT_TEMPLATE

    def draft_reply(self, message: str, **kwargs) -> dict:
        return {"reply": self.template, "retrieved_examples": [], "intent": "other"}

    def draft_batch(self, messages: List[dict]) -> List[dict]:
        return [self.draft_reply(m["customer_message"]) for m in messages]


class NearestNeighborReplyBaseline:
    """
    Simple baseline: return the brand reply from the most similar historical message.

    Uses TF-IDF cosine similarity — no LLM generation.
    """

    def __init__(self, historical_pairs: List[dict]):
        self.pairs = [
            p for p in historical_pairs
            if p.get("customer_message", "").strip() and p.get("brand_reply", "").strip()
        ]
        corpus = [p["customer_message"] for p in self.pairs]
        self.vectorizer = TfidfVectorizer(
            max_features=10_000, stop_words="english", ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def draft_reply(self, message: str, **kwargs) -> dict:
        query_vec = self.vectorizer.transform([message])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        best_idx = sims.argmax()
        return {
            "reply": self.pairs[best_idx]["brand_reply"],
            "retrieved_examples": [self.pairs[best_idx]],
            "intent": kwargs.get("intent", "other"),
            "similarity": float(sims[best_idx]),
        }

    def draft_batch(self, messages: List[dict]) -> List[dict]:
        return [
            self.draft_reply(m["customer_message"], intent=m.get("intent", "other"))
            for m in messages
        ]


# ══════════════════════════════════════════════════════════════════════════════
#  ESCALATION DECISION BASELINES
# ══════════════════════════════════════════════════════════════════════════════

class RandomEscalationBaseline:
    """Trivial baseline: random escalation decision (50/50)."""

    def __init__(self, seed: int = config.RANDOM_SEED):
        self.rng = random.Random(seed)

    def decide(self, message: str, **kwargs) -> dict:
        decision = self.rng.choice(["auto_handle", "escalate"])
        return {
            "decision": decision,
            "reason": "Random baseline",
            "method": "random",
            "confidence": 0.5,
        }

    def decide_batch(self, messages: List[dict]) -> List[dict]:
        return [self.decide(m["customer_message"]) for m in messages]


class KeywordEscalationBaseline:
    """
    Simple baseline: escalation based only on keyword rules (no LLM).

    If any escalation keyword is found → escalate.
    If any auto-handle keyword is found → auto_handle.
    Otherwise → escalate (safe default).
    """

    def decide(self, message: str, **kwargs) -> dict:
        msg_lower = message.lower()

        for kw in config.AUTO_ESCALATE_KEYWORDS:
            if kw in msg_lower:
                return {
                    "decision": "escalate",
                    "reason": f"Keyword match: '{kw}'",
                    "method": "keyword",
                    "confidence": 0.8,
                }

        for pattern in config.AUTO_HANDLE_PATTERNS:
            if pattern in msg_lower:
                return {
                    "decision": "auto_handle",
                    "reason": f"Keyword match: '{pattern}'",
                    "method": "keyword",
                    "confidence": 0.6,
                }

        # Default to escalate
        return {
            "decision": "escalate",
            "reason": "No keyword match — default to escalate",
            "method": "keyword_default",
            "confidence": 0.3,
        }

    def decide_batch(self, messages: List[dict]) -> List[dict]:
        return [self.decide(m["customer_message"]) for m in messages]
