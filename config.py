"""
Central configuration for the AI Support Agent pipeline.
All tuneable knobs live here — paths, model settings, intent definitions, and escalation rules.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GOLDEN_SET_PATH = DATA_DIR / "golden_set.json"

# Create directories on import
for _d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Brand ──────────────────────────────────────────────────────────────────────
BRAND = os.getenv("BRAND", "AppleSupport")

# ── Subsample ──────────────────────────────────────────────────────────────────
MAX_CONVERSATIONS = int(os.getenv("MAX_CONVERSATIONS", "5000"))
RANDOM_SEED = 42

# ── LLM ────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
LLM_TEMPERATURE = 0.3          # Low for consistency; bump for diversity
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 2            # seconds

# ── Intent Taxonomy ────────────────────────────────────────────────────────────
# Derived from manual analysis of ~500 AppleSupport inbound tweets.
INTENT_DEFINITIONS = {
    "device_issue": (
        "Hardware problems — cracked screen, unresponsive buttons, overheating, "
        "charging failures, speaker/microphone malfunctions, physical damage."
    ),
    "software_bug": (
        "iOS / macOS / watchOS software glitches, app crashes after an OS update, "
        "failed updates, boot loops, frozen screens, unexpected reboots."
    ),
    "account_access": (
        "Apple ID issues — locked account, forgotten password, two-factor "
        "authentication problems, verification codes not arriving, account recovery."
    ),
    "billing_inquiry": (
        "Charges, refunds, subscription management (Apple Music, iCloud+, Apple TV+), "
        "unexpected bills, payment method changes, purchase disputes."
    ),
    "how_to": (
        "General how-to questions about Apple features, settings, tips, or "
        "step-by-step workflows (e.g., 'How do I transfer photos?')."
    ),
    "app_issue": (
        "App Store problems — download failures, stuck installs, app-specific bugs, "
        "compatibility issues, app not updating."
    ),
    "connectivity": (
        "WiFi dropping, Bluetooth pairing failures, cellular/LTE issues, AirDrop "
        "not working, hotspot problems, VPN connectivity."
    ),
    "performance": (
        "Slow device, rapid battery drain, storage full warnings, overheating "
        "during normal use, RAM-related slowdowns."
    ),
    "feedback_complaint": (
        "General sentiment — praise, complaints about service quality, frustration "
        "with wait times, product dissatisfaction, suggestions."
    ),
    "other": (
        "Messages that don't clearly fit any category — greetings, gibberish, "
        "non-Apple queries, or highly ambiguous messages."
    ),
}

INTENT_LABELS = list(INTENT_DEFINITIONS.keys())

# ── Escalation Configuration ──────────────────────────────────────────────────
AUTO_ESCALATE_KEYWORDS = [
    # Legal
    "lawyer", "lawsuit", "legal", "sue", "attorney", "class action",
    "consumer protection", "ftc", "complaint to",
    # Security
    "hack", "hacked", "breach", "stolen", "compromised", "unauthorized access",
    # Safety
    "threat", "threatening",
    # Financial severity
    "unauthorized charge", "fraud", "identity theft",
    # Accessibility
    "disability", "accessibility", "ada",
    # Repeated frustration signals
    "been waiting for weeks", "called multiple times", "no one is helping",
    "worst experience", "going to switch",
]

AUTO_HANDLE_PATTERNS = [
    # Simple acknowledgements
    "thank", "thanks", "thx", "ty",
    "ok", "okay", "got it", "will do",
    # Simple how-to
    "how do i", "how to", "where is", "where can i",
    "what is", "what's the",
    # Greetings
    "hello", "hi ", "hey ",
]
