"""
Intent Classifier — assign one of ten intents to each customer message.

Uses a Gemini LLM with few-shot prompting.  The prompt contains:
  • The full intent taxonomy with descriptions
  • 2-3 representative examples per intent
  • Instructions to output structured JSON

Also provides a ``classify_batch`` helper that handles rate-limiting.
"""

import json
import logging
import time
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)

# ── Few-shot examples (curated from dataset exploration) ──────────────────────
FEW_SHOT_EXAMPLES: Dict[str, List[str]] = {
    "device_issue": [
        "My iPhone screen is cracked and touch isn't responding at all.",
        "My MacBook charger stopped working after 2 months.",
        "The home button on my iPad is stuck and won't click.",
    ],
    "software_bug": [
        "Ever since the iOS update my phone keeps restarting randomly.",
        "My Mac is stuck on the loading screen after the latest update.",
        "FaceTime keeps crashing every time I try to make a call.",
    ],
    "account_access": [
        "I can't sign into my Apple ID. It says my account is locked.",
        "I forgot my Apple ID password and the reset email never arrives.",
        "Two-factor authentication code isn't being sent to my phone.",
    ],
    "billing_inquiry": [
        "I was charged $9.99 for something I didn't buy on the App Store.",
        "How do I cancel my Apple Music subscription?",
        "I need a refund for an accidental in-app purchase my kid made.",
    ],
    "how_to": [
        "How do I transfer my photos from iPhone to Mac?",
        "Where can I find the serial number of my iPad?",
        "How do I set up Family Sharing?",
    ],
    "app_issue": [
        "I can't download any apps from the App Store. It just spins.",
        "The WhatsApp update won't install on my iPhone 8.",
        "Apps keep closing immediately after I open them.",
    ],
    "connectivity": [
        "My iPhone keeps disconnecting from WiFi every few minutes.",
        "Bluetooth won't pair with my AirPods anymore.",
        "I can't use AirDrop to send files to my Mac.",
    ],
    "performance": [
        "My iPhone 11 battery drains from 100% to 20% in 3 hours.",
        "My Mac is incredibly slow and takes 5 minutes to boot.",
        "I keep getting storage full notifications even after deleting apps.",
    ],
    "feedback_complaint": [
        "Honestly Apple support has been terrible. 3 hours on hold.",
        "Love the new iOS features! Great work team!",
        "This is the worst experience I've ever had with a tech company.",
    ],
    "other": [
        "Hi there!",
        "lol ok whatever",
        "Can you guys help me with something about my Samsung phone?",
    ],
}


# ── Prompt Construction ───────────────────────────────────────────────────────

def _build_classification_prompt(
    message: str,
    context: Optional[List[str]] = None,
) -> str:
    """Build the full prompt for intent classification."""
    # Intent definitions section
    intent_section = "\n".join(
        f"  • **{intent}**: {desc}"
        for intent, desc in config.INTENT_DEFINITIONS.items()
    )

    # Few-shot examples section
    examples_section = ""
    for intent, examples in FEW_SHOT_EXAMPLES.items():
        for ex in examples[:2]:  # Use 2 per intent to keep prompt reasonable
            examples_section += f'  Message: "{ex}"\n  Intent: {intent}\n\n'

    # Context section
    context_section = ""
    if context:
        context_section = (
            "\n**Conversation context** (earlier messages in this thread):\n"
            + "\n".join(f"  - {c}" for c in context[-3:])
            + "\n"
        )

    return f"""You are an intent classifier for Apple customer support on Twitter.

**Task**: Classify the customer message into exactly ONE intent from the list below.

**Intent Taxonomy**:
{intent_section}

**Examples**:
{examples_section}

{context_section}
**Customer message to classify**:
"{message}"

**Output** (JSON only, no markdown):
{{"intent": "<one of the intent labels>", "confidence": <0.0-1.0>}}
"""


# ── LLM Classification ───────────────────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    """Call the Gemini API with retries."""
    from src.llm_helper import call_llm
    return call_llm(prompt, temperature=config.LLM_TEMPERATURE, max_tokens=256)


def _parse_classification(raw: str) -> dict:
    """Parse the LLM JSON output; fall back gracefully."""
    import re

    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    # Try 1: Direct JSON parse
    try:
        result = json.loads(raw)
        intent = result.get("intent", "other").lower().strip()
        confidence = float(result.get("confidence", 0.5))
    except (json.JSONDecodeError, ValueError, TypeError):
        # Try 2: Extract JSON from surrounding text
        json_match = re.search(r'\{[^}]+\}', raw)
        if json_match:
            try:
                result = json.loads(json_match.group())
                intent = result.get("intent", "other").lower().strip()
                confidence = float(result.get("confidence", 0.5))
            except (json.JSONDecodeError, ValueError, TypeError):
                intent = None
                confidence = 0.5
        else:
            intent = None
            confidence = 0.5

        # Try 3: Look for an intent label directly in the text
        if intent is None:
            raw_lower = raw.lower()
            for label in config.INTENT_LABELS:
                if label in raw_lower:
                    intent = label
                    break
            else:
                logger.warning("Failed to parse classification: %s", raw[:200])
                intent = "other"
                confidence = 0.0

    # Validate intent label
    if intent not in config.INTENT_LABELS:
        for label in config.INTENT_LABELS:
            if label in intent or intent in label:
                intent = label
                break
        else:
            intent = "other"

    return {"intent": intent, "confidence": round(confidence, 3)}


# ── Public API ─────────────────────────────────────────────────────────────────

def classify(
    message: str,
    context: Optional[List[str]] = None,
) -> dict:
    """
    Classify a single customer message.

    Returns ``{"intent": str, "confidence": float}``.
    """
    prompt = _build_classification_prompt(message, context)
    raw = _call_gemini(prompt)
    return _parse_classification(raw)


def classify_batch(
    messages: List[dict],
    delay: float = 0.0,
) -> List[dict]:
    """
    Classify a batch of messages.

    Each item in *messages* should have keys ``customer_message`` and
    optionally ``conversation_context``.

    Pacing between API calls is handled centrally by ``llm_helper.call_llm``
    (see MIN_CALL_INTERVAL / set_min_call_interval). *delay* here adds an
    EXTRA sleep on top of that and should normally stay 0 — it exists only
    for callers who want additional spacing beyond the shared throttle.
    """
    results = []
    for i, msg in enumerate(messages):
        try:
            result = classify(
                msg["customer_message"],
                msg.get("conversation_context"),
            )
        except Exception as e:
            logger.error("Classification failed for msg %d: %s", i, e)
            result = {"intent": "other", "confidence": 0.0}
        results.append(result)

        if delay and i < len(messages) - 1:
            time.sleep(delay)

        if (i + 1) % 25 == 0:
            logger.info("Classified %d / %d messages", i + 1, len(messages))

    return results
