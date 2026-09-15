"""
Shared LLM helper — wraps the google-genai SDK so every module
calls the same thin wrapper instead of duplicating client setup.

Handles rate limiting for free-tier (15 req/min) by tracking
call timestamps and throttling when necessary.
"""

import logging
import time
from typing import Optional

import config

logger = logging.getLogger(__name__)

_client = None
_last_call_time = 0.0  # Track last API call for rate limiting
MIN_CALL_INTERVAL = 4.5  # seconds between calls (safe for 15 req/min free tier)

# NOTE: this is the ONLY throttle applied to LLM calls. The batch helpers in
# intent_classifier.py / reply_drafter.py / escalation_decider.py /
# llm_judge.py used to ALSO sleep(api_delay) between calls on top of this,
# which silently doubled (or worse) the wall-clock time of a run. Those call
# sites now pass delay=0 by default and let this single throttle govern
# pacing — see run_pipeline.py's --api-delay flag, which changes this value
# via set_min_call_interval() instead of adding a second sleep.


def set_min_call_interval(seconds: float) -> None:
    """Override the throttle interval between LLM calls (single source of truth)."""
    global MIN_CALL_INTERVAL
    MIN_CALL_INTERVAL = max(0.0, seconds)


def _get_client():
    """Lazily initialise the google-genai client."""
    global _client
    if _client is not None:
        return _client

    if not config.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not set. Export it:\n"
            "  set GEMINI_API_KEY=your-key-here   (Windows)\n"
            "  export GEMINI_API_KEY=your-key-here (Linux/Mac)"
        )

    from google import genai
    _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def call_llm(
    prompt: str,
    temperature: Optional[float] = None,
    max_tokens: int = 256,
) -> str:
    """
    Call Gemini with retries and rate-limit backoff.
    Returns the raw text response.
    """
    global _last_call_time
    from google.genai import types

    # Rate-limit: wait only if we're calling too fast
    elapsed = time.time() - _last_call_time
    if elapsed < MIN_CALL_INTERVAL:
        time.sleep(MIN_CALL_INTERVAL - elapsed)

    client = _get_client()
    temp = temperature if temperature is not None else config.LLM_TEMPERATURE

    max_retries = 6  # More retries to handle rate limiting
    for attempt in range(max_retries):
        try:
            _last_call_time = time.time()
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temp,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str

            if is_rate_limit:
                # Parse retry delay from error if available
                wait = 15  # Default: wait 15s for rate limit
                if "retryDelay" in err_str:
                    try:
                        import re
                        match = re.search(r'retryDelay.*?(\d+)', err_str)
                        if match:
                            wait = int(match.group(1)) + 1
                    except Exception:
                        pass
                logger.info(
                    "Rate limited (attempt %d/%d). Waiting %ds...",
                    attempt + 1, max_retries, wait
                )
                time.sleep(wait)
            else:
                logger.warning("LLM call attempt %d failed: %s", attempt + 1, e)
                if attempt < max_retries - 1:
                    time.sleep(config.LLM_RETRY_DELAY * (attempt + 1))

    raise RuntimeError("All LLM retries exhausted")

