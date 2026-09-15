"""
LLM-as-Judge — evaluate reply quality using a structured rubric.

Scores each reply on four dimensions (1-5 scale):
  1. Relevance  — Does the reply address the customer's actual issue?
  2. Brand Voice — Does it sound like Apple's support team?
  3. Helpfulness — Does it provide a concrete next step or resolution?
  4. Grounding   — Is the reply factually grounded and not hallucinating?

Also provides a judge–human agreement analysis.
"""

import json
import logging
import time
from typing import Dict, List, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)

# ── Rubric ─────────────────────────────────────────────────────────────────────

RUBRIC = """
You are evaluating AI-generated customer support replies for Apple on Twitter.

**Scoring rubric** (score each dimension 1-5):

### Relevance (Does the reply address the customer's actual issue?)
  1 — Completely off-topic or answers a different question
  2 — Tangentially related but misses the core issue
  3 — Addresses the general topic but misses specific details
  4 — Addresses the issue well with minor gaps
  5 — Directly and precisely addresses the customer's exact issue

### Brand Voice (Does it sound like Apple Support on Twitter?)
  1 — Rude, robotic, or completely wrong tone
  2 — Generic support tone, not distinctly Apple
  3 — Acceptable professional tone, somewhat Apple-like
  4 — Warm, professional, clearly sounds like Apple Support
  5 — Perfect Apple voice: empathetic, concise, helpful, with appropriate warmth

### Helpfulness (Does it provide a concrete next step or resolution?)
  1 — No actionable information whatsoever
  2 — Vague suggestion ("try restarting") with no specifics
  3 — One reasonable step but incomplete guidance
  4 — Clear steps that would likely help resolve the issue
  5 — Excellent guidance with specific, actionable steps tailored to the issue

### Grounding (Is the reply factually accurate and not hallucinating?)
  1 — Contains clear factual errors or fabricated information
  2 — Mostly generic filler with questionable claims
  3 — Generally accurate but some unverifiable claims
  4 — Accurate information, well-grounded in Apple's known practices
  5 — Fully grounded, factually correct, no hallucination
"""


def _build_judge_prompt(
    customer_message: str,
    generated_reply: str,
    intent: str,
    context: Optional[List[str]] = None,
) -> str:
    """Build the LLM judge prompt."""
    context_block = ""
    if context:
        context_block = (
            "\n**Conversation context**:\n"
            + "\n".join(f"  - {c}" for c in context[-3:])
            + "\n"
        )

    return f"""{RUBRIC}

{context_block}
**Customer intent**: {intent}

**Customer message**:
"{customer_message}"

**AI-generated reply**:
"{generated_reply}"

**Score the reply** on each dimension. Output JSON only, no markdown:
{{
  "relevance": <1-5>,
  "brand_voice": <1-5>,
  "helpfulness": <1-5>,
  "grounding": <1-5>,
  "overall": <1-5>,
  "reasoning": "<brief explanation of scores>"
}}
"""


# ── LLM Call ───────────────────────────────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    from src.llm_helper import call_llm
    return call_llm(prompt, temperature=0.1, max_tokens=512)


def _parse_scores(raw: str) -> dict:
    """Parse the judge's JSON scores."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    try:
        result = json.loads(raw)
        scores = {}
        for dim in ["relevance", "brand_voice", "helpfulness", "grounding", "overall"]:
            val = result.get(dim, 3)
            scores[dim] = max(1, min(5, int(val)))
        scores["reasoning"] = result.get("reasoning", "")
        return scores
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("Failed to parse judge scores: %s", raw[:200])
        return {
            "relevance": 3, "brand_voice": 3, "helpfulness": 3,
            "grounding": 3, "overall": 3,
            "reasoning": "Parse error — defaulted to 3",
        }


# ── Public API ─────────────────────────────────────────────────────────────────

def score_reply(
    customer_message: str,
    generated_reply: str,
    intent: str,
    context: Optional[List[str]] = None,
) -> dict:
    """Score a single reply on the 4-dimension rubric."""
    prompt = _build_judge_prompt(customer_message, generated_reply, intent, context)
    raw = _call_gemini(prompt)
    return _parse_scores(raw)


def score_batch(
    items: List[dict],
    delay: float = 0.0,
) -> List[dict]:
    """
    Score a batch of (customer_message, generated_reply, intent) items.

    Each item should have keys: ``customer_message``, ``generated_reply``,
    ``intent``, and optionally ``conversation_context``.

    Pacing is handled centrally by ``llm_helper.call_llm``; *delay* adds an
    extra sleep on top of that and should normally stay 0.
    """
    scores = []
    for i, item in enumerate(items):
        try:
            s = score_reply(
                item["customer_message"],
                item["generated_reply"],
                item.get("intent", "other"),
                item.get("conversation_context"),
            )
        except Exception as e:
            logger.error("Judge failed for item %d: %s", i, e)
            s = {
                "relevance": 3, "brand_voice": 3, "helpfulness": 3,
                "grounding": 3, "overall": 3,
                "reasoning": f"Error: {e}",
            }
        scores.append(s)

        if delay and i < len(items) - 1:
            time.sleep(delay)
        if (i + 1) % 10 == 0:
            logger.info("Judged %d / %d replies", i + 1, len(items))

    return scores


# ── Agreement Analysis ─────────────────────────────────────────────────────────

def compute_judge_human_agreement(
    judge_scores: List[int],
    human_scores: List[int],
) -> dict:
    """
    Compute agreement between LLM judge and human ratings.

    Returns Spearman correlation, Cohen's kappa (linearised), MAE, and
    the percentage of exact matches.
    """
    from scipy import stats

    judge_arr = np.array(judge_scores)
    human_arr = np.array(human_scores)

    # Spearman rank correlation
    spearman_r, spearman_p = stats.spearmanr(judge_arr, human_arr)

    # Mean absolute error
    mae = np.mean(np.abs(judge_arr - human_arr))

    # Exact match %
    exact_match = np.mean(judge_arr == human_arr)

    # Within-1 match %
    within_one = np.mean(np.abs(judge_arr - human_arr) <= 1)

    # Weighted Cohen's kappa
    try:
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(human_arr, judge_arr, weights="linear")
    except Exception:
        kappa = float("nan")

    return {
        "spearman_r": round(float(spearman_r), 4),
        "spearman_p": round(float(spearman_p), 4),
        "cohens_kappa": round(float(kappa), 4),
        "mae": round(float(mae), 4),
        "exact_match_pct": round(float(exact_match) * 100, 1),
        "within_one_pct": round(float(within_one) * 100, 1),
        "n": len(judge_scores),
    }
