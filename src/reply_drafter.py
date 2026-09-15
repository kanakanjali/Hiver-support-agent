"""
Reply Drafter — generate brand-voice replies grounded in historical data.

Architecture (RAG-lite):
  1. Build a TF-IDF index over all historical (customer, brand_reply) pairs.
  2. For a new message, retrieve top-k most similar historical pairs.
  3. Prompt the LLM with the retrieved examples + intent + customer message.
  4. LLM drafts a reply that sounds like the brand and addresses the issue.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config

logger = logging.getLogger(__name__)


class ReplyDrafter:
    """RAG-based reply generator for Apple Support."""

    def __init__(self, historical_pairs: List[dict]):
        """
        Parameters
        ----------
        historical_pairs : list of dict
            Each dict must have ``customer_message`` and ``brand_reply`` keys.
        """
        self.pairs = [
            p for p in historical_pairs
            if p.get("customer_message", "").strip()
            and p.get("brand_reply", "").strip()
        ]
        logger.info("Building TF-IDF index over %d pairs …", len(self.pairs))

        corpus = [p["customer_message"] for p in self.pairs]
        self.vectorizer = TfidfVectorizer(
            max_features=10_000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        logger.info("TF-IDF index ready (vocab size=%d)", len(self.vectorizer.vocabulary_))

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve_similar(
        self, message: str, k: int = 3
    ) -> List[dict]:
        """Return the *k* most similar historical pairs."""
        query_vec = self.vectorizer.transform([message])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_k_idx = sims.argsort()[-k:][::-1]
        return [
            {**self.pairs[i], "similarity": float(sims[i])}
            for i in top_k_idx
            if sims[i] > 0
        ]

    # ── Reply Generation ──────────────────────────────────────────────────────

    def draft_reply(
        self,
        message: str,
        intent: str,
        context: Optional[List[str]] = None,
    ) -> dict:
        """
        Draft a reply for *message*.

        Returns ``{"reply": str, "retrieved_examples": list, "intent": str}``.
        """
        retrieved = self.retrieve_similar(message, k=3)
        prompt = self._build_prompt(message, intent, retrieved, context)

        try:
            raw = self._call_llm(prompt)
            reply = self._parse_reply(raw)
        except Exception as e:
            logger.error("Reply generation failed: %s", e)
            # Fall back to best retrieved reply
            reply = retrieved[0]["brand_reply"] if retrieved else (
                "Hi! We'd like to help. Could you DM us with more details "
                "so we can look into this for you?"
            )

        return {
            "reply": reply,
            "retrieved_examples": retrieved,
            "intent": intent,
        }

    # ── Prompt ─────────────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        message: str,
        intent: str,
        retrieved: List[dict],
        context: Optional[List[str]] = None,
    ) -> str:
        intent_desc = config.INTENT_DEFINITIONS.get(intent, "")

        examples_block = ""
        for i, ex in enumerate(retrieved, 1):
            examples_block += (
                f"  Example {i}:\n"
                f"    Customer: \"{ex['customer_message']}\"\n"
                f"    Brand reply: \"{ex['brand_reply']}\"\n\n"
            )

        context_block = ""
        if context:
            context_block = (
                "\n**Earlier messages in this thread**:\n"
                + "\n".join(f"  - {c}" for c in context[-3:])
                + "\n"
            )

        return f"""You are a customer support agent for **Apple** on Twitter.

**Brand voice guidelines** (derived from historical data):
  - Friendly, empathetic, professional.
  - Use the customer's first name if mentioned.
  - Keep replies concise (under 280 characters when possible, max 560).
  - Offer specific next steps, not vague sympathy.
  - For complex issues, suggest a DM or link to support.
  - Never blame the customer.
  - Sign off with a positive note or offer for further help.

**Customer's intent**: {intent} — {intent_desc}
{context_block}
**Similar past interactions your team handled**:
{examples_block}

**New customer message**:
"{message}"

**Draft a reply** that:
  1. Acknowledges the customer's issue.
  2. Provides a concrete next step or resolution, grounded in how similar issues were resolved above.
  3. Matches Apple's support voice.

Reply (plain text, no JSON, no quotes):"""

    # ── LLM call ───────────────────────────────────────────────────────────────

    @staticmethod
    def _call_llm(prompt: str) -> str:
        from src.llm_helper import call_llm
        return call_llm(prompt, temperature=0.5, max_tokens=300)

    @staticmethod
    def _parse_reply(raw: str) -> str:
        """Clean up the LLM output."""
        reply = raw.strip().strip('"').strip("'")
        # Remove any accidental markdown
        if reply.startswith("```"):
            reply = reply.split("\n", 1)[-1]
        if reply.endswith("```"):
            reply = reply.rsplit("```", 1)[0]
        return reply.strip()

    # ── Batch ──────────────────────────────────────────────────────────────────

    def draft_batch(
        self,
        messages: List[dict],
        delay: float = 0.0,
    ) -> List[dict]:
        """
        Draft replies for a batch.

        Each item should have ``customer_message``, ``intent``, and
        optionally ``conversation_context``.

        Pacing is handled centrally by ``llm_helper.call_llm``; *delay* adds
        an extra sleep on top of that and should normally stay 0.
        """
        results = []
        for i, msg in enumerate(messages):
            result = self.draft_reply(
                msg["customer_message"],
                msg.get("intent", "other"),
                msg.get("conversation_context"),
            )
            results.append(result)

            if delay and i < len(messages) - 1:
                time.sleep(delay)
            if (i + 1) % 10 == 0:
                logger.info("Drafted %d / %d replies", i + 1, len(messages))

        return results
