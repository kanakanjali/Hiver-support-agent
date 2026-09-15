# Golden Evaluation Set — Sampling & Labelling Methodology

## Overview
The golden evaluation set contains **200 hand-labelled examples** used to evaluate the AI support agent across three tasks: intent classification, reply quality assessment, and escalation decision-making.

## Sampling Strategy

### Source — how these 200 examples were actually built
**These examples are author-written, not sampled verbatim from the Kaggle CSV.**
To build them, I first pulled and read a working set of ~500 real inbound
AppleSupport tweets from the Customer Support on Twitter dataset
(`thoughtvector/customer-support-on-twitter`) to learn the real distribution of
issues, phrasing, slang, typos, and tone. I then **wrote 200 new examples by
hand** (`scripts/generate_golden_set.py`) that reproduce the patterns,
vocabulary, and phrasing quirks I observed in that real sample — including
realistic Twitter noise (lowercase, abbreviations, emoji, sarcasm, run-on
sentences) — rather than pasting real tweets in directly.

I made this choice deliberately, for two reasons:
1. **Coverage control.** The real dataset is heavily skewed toward a handful of
   issue types (mostly `device_issue` / `software_bug`). A verbatim sample of
   500 real tweets would have produced only a handful of usable examples for
   intents like `billing_inquiry` or `feedback_complaint`. Writing examples
   by hand let me guarantee ~20 clean examples per intent.
2. **Label confidence.** Real tweets are frequently multi-intent, missing
   context, or truncated mid-thread. Hand-authoring let me control difficulty
   and ambiguity deliberately (see difficulty tiers below) instead of it being
   an accident of what happened to be sampled.

**Trade-off, stated plainly:** this golden set measures how well the agent
generalizes to *AppleSupport-shaped* language, not to the literal noise
distribution of the raw CSV (see the reference replies in
`scripts/add_reference_replies.py`, which are similarly modelled on real
AppleSupport reply patterns rather than copied verbatim). Section 5 of the
report ("What is misleading about my headline number") calls this out
explicitly as a limitation of the headline metrics.

### Stratification
We use **stratified sampling** across two dimensions:

1. **Intent** (10 categories × 20 examples each):
   - `device_issue`, `software_bug`, `account_access`, `billing_inquiry`, `how_to`
   - `app_issue`, `connectivity`, `performance`, `feedback_complaint`, `other`

2. **Difficulty** (3 levels):
   - **Easy** (~40%): Clear, single-intent messages with obvious classification
   - **Medium** (~40%): Messages requiring some interpretation, potential ambiguity
   - **Hard** (~20%): Multi-intent, sarcastic, vague, or edge-case messages

### Construction Process
1. Sampled and read ~500 real inbound AppleSupport tweets to learn the
   taxonomy, vocabulary, and tone (this is also how the 10-intent taxonomy in
   `config.py` was derived).
2. Wrote 20 examples per intent by hand, deliberately varying difficulty and
   phrasing style (typos, sarcasm, ALL CAPS, emoji, run-ons) to mirror what
   was observed in the real sample.
3. Assigned difficulty labels based on ambiguity and complexity.
4. For each example, labelled the escalation decision with a one-line reason.
5. Added reference replies (modelled on real AppleSupport reply conventions —
   concise, signed with an agent tag like `^KA`, DM hand-off for anything
   needing account access) to 47 of the 200 examples for ROUGE-L scoring.

## Labelling Guidelines

### Intent Labelling
- Assign the **primary** intent (the main thing the customer wants resolved)
- If multiple intents are present, label the dominant one
- Use `other` only when no category fits after careful consideration

### Escalation Labelling
- **`escalate`**: The message requires human judgment, system access, or involves sensitive topics (legal, security, financial disputes, extreme frustration, safety concerns)
- **`auto_handle`**: The AI can reasonably address this with standard troubleshooting, how-to guidance, or acknowledgement

### Quality Notes
For each example, we include:
- A brief note on why this intent was chosen (especially for ambiguous cases)
- The escalation reasoning
- Any edge-case observations

## Known Limitations
- **Single annotator**: These labels come from one person, so there's no inter-annotator agreement score. In production, we'd want 2-3 annotators with adjudication.
- **Hand-authored, not verbatim-sampled**: as detailed above, examples are written to mirror real AppleSupport tweet patterns rather than pasted in directly from the dataset. This trades real-world noise for balanced, controllable coverage — see the trade-off note above and Report section 5.
- **Class balance**: The real data is skewed (more device/software issues); our golden set is intentionally balanced for fairer evaluation across intents.
- **Twitter-specific**: Messages are short, informal, and may lack context that a full support ticket would have.
