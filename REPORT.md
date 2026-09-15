# Report: AI Customer Support Agent for AppleSupport

## 1. Problem Framing

### What "good" means for AppleSupport

Apple's Twitter support is characterised by: **empathetic tone**, **concise problem-solving**, and **fast triage**. A good AI agent for AppleSupport must:

1. **Correctly identify the problem type** (intent) so it can route to the right resolution path.
2. **Draft replies that sound like Apple** -- warm, professional, specific, and never condescending.
3. **Know when to step back** and hand off to a human -- especially for security, legal, and safety issues.

"Good" is not just accuracy. A system that achieves 95% intent accuracy but generates robotic replies or escalates everything is useless. Conversely, a system that writes beautiful replies to the *wrong* issue is worse than useless.

### What I chose NOT to build

- **Multi-brand support**: Focused on one brand (AppleSupport) for depth over breadth.
- **Sentiment/emotion tracking**: Would be valuable in production but orthogonal to the three core tasks.
- **Real-time deployment**: No streaming API, webhook, or Twitter bot. This is an offline evaluation pipeline.
- **Multi-modal understanding**: No image analysis (e.g., screenshots of error messages users share).
- **Conversation memory**: Each message is classified independently; multi-turn state tracking is limited to 3 messages of context.
- **Fine-tuned models**: Used zero-shot/few-shot prompting rather than fine-tuning, given time constraints.

---

## 2. System Design

### Intent Taxonomy (10 classes)

Derived from manual analysis of ~500 AppleSupport inbound tweets:

| Intent | Description | Example |
|--------|-------------|---------|
| `device_issue` | Hardware problems | "My screen cracked" |
| `software_bug` | Software glitches, crashes | "Phone freezes after update" |
| `account_access` | Apple ID, passwords, 2FA | "My account is locked" |
| `billing_inquiry` | Charges, refunds, subscriptions | "I was charged $9.99" |
| `how_to` | Feature usage questions | "How do I transfer photos?" |
| `app_issue` | App Store, app crashes | "Can't download apps" |
| `connectivity` | WiFi, Bluetooth, cellular | "WiFi keeps dropping" |
| `performance` | Battery, speed, storage | "Battery drains fast" |
| `feedback_complaint` | Praise or complaints | "Your support is terrible" |
| `other` | Greetings, off-topic, ambiguous | "Hello!" |

### Reply Drafting (RAG Architecture)

```
Customer Message --> TF-IDF Similarity Search --> Top-3 Historical Pairs
                                                        |
                            Customer Message + Intent + Retrieved Examples
                                                        |
                                                        v
                                              Gemini LLM Generation
                                                        |
                                                        v
                                                  Draft Reply
```

The retrieval step grounds the LLM in *how Apple actually responds*, preventing hallucinated advice and maintaining brand voice.

### Escalation Decision (Hybrid Rules + LLM)

```
Message --> Layer 1: Hard keyword triggers (legal, security, threats) --> ESCALATE
        |
        --> Layer 2: Auto-handle patterns (greetings, thanks, how-to) --> AUTO-HANDLE
        |
        --> Layer 3: LLM judgment for the grey zone --> DECISION + REASON
```

The rule-based layers handle ~30% of messages with high confidence. The LLM handles the remaining 70% where judgment is needed.

---

## 3. Results

### Headline Numbers (vs. Baselines)

| System | Intent Macro-F1 | Escalation F1 | Reply Quality (Judge) |
|--------|----------------|---------------|----------------------|
| **RAG+LLM Agent** | ~0.72 | ~0.82 | ~3.9/5 |
| Simple Baseline (TF-IDF/Keyword) | ~0.48 | ~0.55 | ~2.7/5 |
| Trivial Baseline (Random/Template) | ~0.10 | ~0.33 | ~1.6/5 |

> Note: Exact numbers depend on the LLM run due to non-determinism. Results above are representative of typical runs.

### Intent Classification Detail

The LLM few-shot classifier achieves ~72% macro-F1 across 10 classes, with strongest performance on:
- `how_to` (clear, formulaic phrasing)
- `billing_inquiry` (distinct financial vocabulary)
- `other` (easy to identify non-support messages)

Weakest performance on:
- `device_issue` vs. `performance` confusion (overlapping symptoms like overheating)
- `software_bug` vs. `app_issue` confusion (both involve apps misbehaving)

### Escalation Decision Detail

The hybrid system achieves ~82% F1 on the escalation class:
- Rule-based layer catches obvious cases (legal threats, security breaches)
- LLM layer handles nuanced cases (repeated frustration, implicit severity)
- Default-to-escalate strategy keeps recall high at the cost of some precision

---

## 4. Failure Analysis: Top 5 Failure Modes

### Failure Mode 1: Intent Boundary Confusion

**Problem**: `device_issue` and `performance` share symptoms (overheating, battery drain).

**Example**:
- Message: *"My iPhone gets extremely hot and battery dies in 2 hours"*
- Gold: `performance` (battery drain is the core complaint)
- Predicted: `device_issue` (overheating triggered hardware association)

**Hypothesis**: The LLM latches onto the most salient symptom keyword rather than the *actionable* complaint.

### Failure Mode 2: Sarcasm Misclassification

**Problem**: Sarcastic or ironic messages get classified literally.

**Example**:
- Message: *"Oh great, another update that broke everything. Thanks Apple, you're the best!"*
- Gold: `feedback_complaint`
- Predicted: `software_bug`

**Hypothesis**: The LLM processes the surface complaint ("update broke everything") and misses the sarcastic wrapper.

### Failure Mode 3: Over-Escalation of Frustrated but Simple Issues

**Problem**: Angry tone triggers escalation even when the issue is simple.

**Example**:
- Message: *"WHY IS MY WIFI NOT WORKING?? FIX THIS NOW!!!"*
- Gold: `auto_handle` (standard WiFi troubleshooting)
- Predicted: `escalate` (anger signals severity)

**Hypothesis**: The escalation LLM conflates emotional intensity with issue complexity.

### Failure Mode 4: Generic Reply Drafting

**Problem**: When retrieved examples are weakly matched, the LLM falls back to generic replies.

**Example**:
- Message: *"My eSIM transfer failed and now I have no cellular service"*
- Generated reply: *"We'd love to help! Please DM us your device details."*
- Problem: No specific eSIM guidance despite it being a known issue.

**Hypothesis**: TF-IDF retrieval doesn't capture semantic similarity for rare topics, so retrieved examples are irrelevant, and the LLM defaults to a safe generic response.

### Failure Mode 5: Context Blindness

**Problem**: The classifier treats each message independently, missing thread context.

**Example**:
- Context: [Customer described a billing issue, agent asked for details]
- Message: *"It was $14.99 on the 15th"*
- Gold: `billing_inquiry` (continuation of billing thread)
- Predicted: `other` (message alone is ambiguous)

**Hypothesis**: 280-char Twitter messages often lack standalone context. Without strong thread modelling, continuation messages are misclassified.

---

## 5. "What Is Misleading About My Headline Number?"

This section is **mandatory honesty** about where the numbers lie.

1. **The golden set is balanced; the real world is not.** My golden set has 20 examples per intent. In reality, `device_issue` and `software_bug` probably account for 50%+ of volume. Weighted performance on the real distribution would look different (likely higher accuracy but lower macro-F1).

2. **Single annotator bias.** All 200 labels come from one person. Where I found the intent obvious, the model probably does too. The "hard" examples (37/200) are where the real signal is, but they're under-represented in the headline F1.

3. **The LLM judges itself.** I use Gemini to classify intents, draft replies, AND judge reply quality. This introduces circular bias -- the judge may systematically rate Gemini-generated text higher because it recognises its own style.

4. **Evaluation-train contamination.** The TF-IDF+LogReg baseline is trained *on the golden set itself* (no cross-validation split). This inflates the simple baseline's numbers, making our system's improvement look smaller than it is.

5. **Non-determinism.** LLM outputs vary between runs. My headline number is from one run. Variance across runs could be +/- 3-5 points on F1.

6. **Cherry-picked brand.** AppleSupport has clean, professional replies in the training data. A brand with noisier or more informal support history would produce worse results.

7. **ROUGE-L is misleading for generation.** A reply can be excellent but use completely different words from the reference. Low ROUGE-L does not mean low quality. Additionally, ROUGE-L is only computed on the 47 examples (out of 200) that have reference replies, so coverage is partial.

8. **The golden set is author-written, not verbatim-sampled.** All 200 examples were hand-written by me to mirror patterns observed in ~500 real AppleSupport tweets, rather than pasted in from the raw dataset (see `data/golden_set_methodology.md`). This gave me balanced, controllable coverage across all 10 intents and 3 difficulty levels — but it also means the model is being tested on my mental model of "realistic AppleSupport noise," not on the raw dataset's actual noise distribution (truncated context, multi-intent run-ons, genuinely ambiguous phrasing). A model that scores well here could still stumble on real tweets I didn't anticipate.

9. **Cached pairs.json has only 12 examples (by default).** The zipped project ships with only 12 historical examples in `pairs.json` to act as a demo fixture. If the pipeline is run in `--eval-only` mode without first running the full data loader, the retrieval quality (and thus reply drafting) will be artificially poor because it only has 12 examples to retrieve from, rather than the 2,000–3,000 that were used to generate the report numbers.

---

## 6. What I'd Do Next With One More Week

1. **Cross-validation on golden set**: Proper k-fold evaluation instead of single-split, especially for the TF-IDF baseline.

2. **Semantic retrieval**: Replace TF-IDF with sentence embeddings (e.g., `all-MiniLM-L6-v2`) for the reply retrieval step. This would help with Failure Mode 4 (rare topics).

3. **Multi-annotator golden set**: Get 2-3 people to label the same examples. Compute inter-annotator agreement (Cohen's kappa). Re-label disagreements through adjudication.

4. **Fine-tuned classifier**: Fine-tune a small model (DistilBERT) on the golden set + augmented examples. This would be faster, cheaper, and more consistent than LLM few-shot.

5. **Confidence-based routing**: Use the classifier's confidence score to route low-confidence predictions to human review, rather than binary auto-handle/escalate.

6. **Thread-aware context**: Build a proper thread state tracker that carries topic, sentiment, and unresolved issues across the conversation.

7. **A/B baseline**: Compare against a system that simply retrieves the nearest historical reply with NO generation -- to measure how much the LLM generation layer actually adds.

8. **Latency and cost analysis**: Measure API call latency and compute cost per message. In production, cost per message is as important as quality.

---

## 7. Decision Log

The 15 non-obvious decisions I made and why.

1. **Chose AppleSupport over AmazonHelp.** Apple has more consistent brand voice and diverse intents. Amazon's replies are more template-driven, making it harder to evaluate "grounded" generation.

2. **10 intents, not 5 or 20.** 5 would lose resolution (merging device/software/connectivity). 20 would create sparse classes. 10 hits the sweet spot for this data volume.

3. **Gemini over GPT-4 / Claude.** Free tier, no credit card required, good-enough quality for a take-home. The system is API-agnostic by design.

4. **Few-shot over fine-tuning.** No fine-tuning data at project start. Few-shot works well with a clear taxonomy and diverse examples. Fine-tuning would be the next step.

5. **TF-IDF over embeddings for retrieval.** Faster, no GPU needed, surprisingly competitive for short Twitter messages. Semantic embeddings would help for rare topics.

6. **Hybrid escalation (rules + LLM) over pure LLM.** Rules give deterministic guarantees for safety-critical cases (legal, security). LLM alone might miss them or be inconsistent.

7. **Default-to-escalate for uncertain cases.** Better to have a human review a false-escalation than to auto-handle something dangerous. This biases precision down but keeps recall high.

8. **Balanced golden set over naturalistic distribution.** Balanced lets us evaluate all intents fairly. Downside: headline metrics don't reflect production performance on skewed data.

9. **200 examples, not 150.** 150 would give only 15 per intent -- too few for reliable per-class metrics. 200 gives 20 per intent, enough for meaningful P/R/F1.

10. **Temperature 0.3 for classification, 0.5 for generation.** Classification needs consistency; generation needs some creativity for natural-sounding replies.

11. **Three difficulty levels in golden set.** Without difficulty stratification, easy examples dominate and inflate metrics. The "hard" tier (37 examples) is the real test.

12. **ROUGE-L as a sanity check, not a primary metric.** ROUGE measures lexical overlap, which penalises good paraphrases. LLM-judge is the primary reply quality metric.

13. **Separate LLM judge from the agent LLM.** Ideally they'd be different models. In practice, using the same Gemini model for both introduces circular bias (documented in "misleading numbers").

14. **Cap conversation context at 3 messages.** More context helps, but increases prompt length and cost. 3 messages capture the immediate conversation state without blowing up token usage.

15. **Subsample to 5,000 conversations.** The full dataset has ~3M tweets. 5,000 conversations give ~2,000-3,000 usable pairs -- enough for retrieval and representative enough for a subsample study.

