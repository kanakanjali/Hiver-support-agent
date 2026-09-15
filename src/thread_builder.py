"""
Thread Builder — reconstruct multi-turn conversation threads from flat tweets.

Each thread is a chronological list of messages forming one support conversation.
From threads we extract (customer_message, brand_reply) pairs for downstream use.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

import config

logger = logging.getLogger(__name__)


# ── Thread Reconstruction ──────────────────────────────────────────────────────

def build_threads(df: pd.DataFrame) -> List[List[dict]]:
    """
    Reconstruct conversation threads from the flat tweet DataFrame.

    Strategy
    --------
    1. Build a mapping: tweet_id → row dict.
    2. Build parent→children edges from ``in_response_to_tweet_id``.
    3. Find root tweets (those with no parent in the dataset).
    4. Walk each tree depth-first to produce an ordered thread.
    """
    # Index every tweet
    tweet_map: Dict[str, dict] = {}
    children: Dict[str, List[str]] = defaultdict(list)

    for _, row in df.iterrows():
        tid = str(row.get("tweet_id", ""))
        if not tid or tid == "nan":
            continue
        tweet_map[tid] = row.to_dict()

        parent = str(row.get("in_response_to_tweet_id", ""))
        if parent and parent != "nan":
            children[parent].append(tid)

    # Also handle response_tweet_id (forward pointer)
    if "response_tweet_id" in df.columns:
        for _, row in df.iterrows():
            tid = str(row.get("tweet_id", ""))
            resp = row.get("response_tweet_id")
            if pd.isna(resp) or not resp:
                continue
            for child_id in str(resp).split(","):
                child_id = child_id.strip()
                if child_id and child_id != "nan" and child_id in tweet_map:
                    if child_id not in children[tid]:
                        children[tid].append(child_id)

    # Find roots
    all_child_ids = {cid for cids in children.values() for cid in cids}
    roots = [tid for tid in tweet_map if tid not in all_child_ids]

    # Walk trees
    threads: List[List[dict]] = []
    visited = set()

    def _walk(tid: str) -> List[dict]:
        if tid in visited or tid not in tweet_map:
            return []
        visited.add(tid)
        result = [tweet_map[tid]]
        for child in children.get(tid, []):
            result.extend(_walk(child))
        return result

    for root in roots:
        thread = _walk(root)
        if len(thread) >= 2:  # need at least a customer msg + brand reply
            threads.append(thread)

    logger.info(
        "Built %d threads from %d tweets (%d roots found)",
        len(threads), len(tweet_map), len(roots),
    )
    return threads


# ── Pair Extraction ────────────────────────────────────────────────────────────

def extract_pairs(
    threads: List[List[dict]],
    brand: str = config.BRAND,
) -> List[dict]:
    """
    Extract (customer_message, brand_reply) pairs from threads.

    Each pair also carries:
      - ``conversation_context``: preceding messages in the thread
      - ``thread_id``: the root tweet_id for traceability
    """
    pairs: List[dict] = []

    for thread in threads:
        for i, msg in enumerate(thread):
            is_inbound = msg.get("inbound")
            # Handle bool or string representations
            if isinstance(is_inbound, str):
                is_inbound = is_inbound.lower() == "true"

            if not is_inbound:
                continue

            # Look for a brand reply immediately after this message
            brand_reply = _find_brand_reply(thread, i, brand)
            if brand_reply is None:
                continue

            customer_text = msg.get("clean_text") or msg.get("text", "")
            reply_text = brand_reply.get("clean_text") or brand_reply.get("text", "")

            if not customer_text.strip() or not reply_text.strip():
                continue

            context = [
                (m.get("clean_text") or m.get("text", ""))
                for m in thread[:i]
                if (m.get("clean_text") or m.get("text", "")).strip()
            ]

            pairs.append({
                "thread_id": str(thread[0].get("tweet_id", "")),
                "customer_message": customer_text.strip(),
                "brand_reply": reply_text.strip(),
                "conversation_context": context[-3:],  # last 3 for context window
            })

    logger.info("Extracted %d (customer, brand_reply) pairs", len(pairs))
    return pairs


def _find_brand_reply(
    thread: List[dict], customer_idx: int, brand: str
) -> Optional[dict]:
    """Find the next brand reply after position *customer_idx* in the thread."""
    for j in range(customer_idx + 1, min(customer_idx + 4, len(thread))):
        author = str(thread[j].get("author_id", "")).lower()
        is_inbound = thread[j].get("inbound")
        if isinstance(is_inbound, str):
            is_inbound = is_inbound.lower() == "true"
        # Match by author name OR by outbound flag (not inbound)
        if author == brand.lower() or is_inbound == False:
            return thread[j]
    return None


# ── Convenience ────────────────────────────────────────────────────────────────

def build_pairs_from_df(
    df: pd.DataFrame,
    brand: str = config.BRAND,
) -> Tuple[List[List[dict]], List[dict]]:
    """Build threads then extract pairs. Returns (threads, pairs)."""
    threads = build_threads(df)
    pairs = extract_pairs(threads, brand)
    return threads, pairs
