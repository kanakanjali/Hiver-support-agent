"""
Escalation Decider — decide whether a customer message should be auto-handled
or escalated to a human agent, with a stated reason.

Architecture (rules + LLM hybrid):
  Layer 1 — Keyword rules for obvious escalations (legal, security, threats)
  Layer 2 — Keyword rules for obvious auto-handles (greetings, acknowledgements)
  Layer 3 — LLM judgment for the grey zone, with structured reasoning
"""

import json
import logging
import time
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)


# ── Rule-based layers ─────────────────────────────────────────────────────────

def _check_auto_escalate(message: str) -> Optional[dict]:
    """Check if the message matches hard escalation triggers."""
    msg_lower = message.lower()
    for keyword in config.AUTO_ESCALATE_KEYWORDS:
        if keyword in msg_lower:
            return {
                "decision": "escalate",
                "reason": f"Matched escalation keyword: '{keyword}'",
                "method": "rule_based",
                "confidence": 0.95,
            }
    return None


def _check_auto_handle(message: str) -> Optional[dict]:
    """Check if the message is clearly auto-handleable."""
    msg_lower = message.lower().strip()

    # Short messages that are just acknowledgements or simple greetings
    if len(msg_lower.split()) <= 6:
        for pattern in config.AUTO_HANDLE_PATTERNS:
            if msg_lower.startswith(pattern) or pattern in msg_lower:
                return {
                    "decision": "auto_handle",
                    "reason": f"Simple acknowledgement/greeting: '{pattern}'",
                    "method": "rule_based",
                    "confidence": 0.90,
                }
    return None


# ── LLM judgment ──────────────────────────────────────────────────────────────

def _build_escalation_prompt(
    message: str,
    intent: str,
    context: Optional[List[str]] = None,
) -> str:
    """Build prompt for the LLM escalation decision."""
    context_block = ""
    if context:
        context_block = (
            "\n**Conversation history**:\n"
            + "\n".join(f"  - {c}" for c in context[-3:])
            + "\n"
        )

    return f"""You are a routing engine for Apple's Twitter support team.

**Task**: Decide if this customer message should be AUTO-HANDLED by the AI agent
or ESCALATED to a human agent.

**Escalate when**:
  - The issue involves account security, potential fraud, or data breach
  - The customer mentions legal action or regulatory complaints
  - The customer is extremely frustrated and has contacted multiple times
  - The issue requires access to internal systems (order lookup, account changes)
  - The message involves a safety concern (overheating, battery swelling)
  - Financial disputes over significant amounts
  - The customer explicitly asks for a manager or human agent

**Auto-handle when**:
  - Simple how-to questions with clear answers
  - Status inquiries the AI can look up
  - Basic troubleshooting the AI can walk through
  - Acknowledgements, thanks, or confirmations
  - Questions about Apple features or services
  - First-contact issues that follow standard resolution scripts

**Classified intent**: {intent}
{context_block}
**Customer message**:
"{message}"

**Output** (JSON only, no markdown):
{{
  "decision": "auto_handle" or "escalate",
  "reason": "one-sentence explanation",
  "confidence": 0.0-1.0
}}
"""


def _call_gemini(prompt: str) -> str:
    """Call Gemini API."""
    from src.llm_helper import call_llm
    return call_llm(prompt, temperature=0.1, max_tokens=256)


def _parse_decision(raw: str) -> dict:
    """Parse LLM JSON output for escalation decision."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    try:
        result = json.loads(raw)
        decision = result.get("decision", "escalate").lower().strip()
        if decision not in ("auto_handle", "escalate"):
            decision = "escalate"  # Safe default
        return {
            "decision": decision,
            "reason": result.get("reason", "LLM did not provide a reason"),
            "method": "llm",
            "confidence": float(result.get("confidence", 0.5)),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("Failed to parse escalation decision: %s", raw[:200])
        return {
            "decision": "escalate",
            "reason": "Failed to parse LLM response — defaulting to escalate for safety",
            "method": "llm_fallback",
            "confidence": 0.0,
        }


# ── Public API ─────────────────────────────────────────────────────────────────

def decide(
    message: str,
    intent: str = "other",
    context: Optional[List[str]] = None,
) -> dict:
    """
    Decide whether to auto-handle or escalate.

    Returns::

        {
            "decision": "auto_handle" | "escalate",
            "reason": str,
            "method": "rule_based" | "llm" | "llm_fallback",
            "confidence": float,
        }
    """
    # Layer 1: hard escalation triggers
    esc = _check_auto_escalate(message)
    if esc:
        return esc

    # Layer 2: obvious auto-handles
    auto = _check_auto_handle(message)
    if auto:
        return auto

    # Layer 3: LLM for the grey zone
    prompt = _build_escalation_prompt(message, intent, context)
    raw = _call_gemini(prompt)
    return _parse_decision(raw)


def decide_batch(
    messages: List[dict],
    delay: float = 0.0,
) -> List[dict]:
    """
    Decide for a batch of messages.

    Each item should have ``customer_message``, ``intent``, and
    optionally ``conversation_context``.

    Pacing is handled centrally by ``llm_helper.call_llm``; *delay* adds an
    extra sleep on top of that and should normally stay 0. Note rule-based
    Layers 1/2 short-circuit before any LLM call, so they incur no delay.
    """
    results = []
    for i, msg in enumerate(messages):
        try:
            result = decide(
                msg["customer_message"],
                msg.get("intent", "other"),
                msg.get("conversation_context"),
            )
        except Exception as e:
            logger.error("Escalation decision failed for msg %d: %s", i, e)
            result = {
                "decision": "escalate",
                "reason": f"Error during decision: {e}",
                "method": "error_fallback",
                "confidence": 0.0,
            }
        results.append(result)

        if delay and i < len(messages) - 1:
            time.sleep(delay)
        if (i + 1) % 25 == 0:
            logger.info("Decided %d / %d messages", i + 1, len(messages))

    return results
