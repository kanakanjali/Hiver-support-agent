# AI Customer Support Agent for AppleSupport

## Quick Start (Reproduce Results in Under 15 Minutes)

### 1. Prerequisites

```bash
# Python 3.10+
python --version

# Clone / download this repo, then:
cd Hiver-support-agent
pip install -r requirements.txt
```

### 2. Set Your API Key

```bash
# Windows
set GEMINI_API_KEY=your-gemini-api-key-here

# Linux/Mac
export GEMINI_API_KEY=your-gemini-api-key-here
```

> Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com/apikey)

### 3. Run the Pipeline

There are two ways to run this, and they trade off time for coverage. **Use
the first one to reproduce results in under 15 minutes.**

```bash
# FAST PATH (~10 min on the free Gemini tier) — reproduces the pipeline
# end-to-end on a 30-example stratified slice of the golden set.
# --sample-size 30 is the default, so this is just:
python run_pipeline.py

# FULL PATH (~1-1.5 hours on the free tier; a few minutes on a paid tier) —
# runs all 200 golden-set examples for the exact headline numbers in REPORT.md.
python run_pipeline.py --sample-size 200 --api-delay 0.3

# Evaluation only (skips the data download + threading step, reuses
# data/processed/pairs.json if present; otherwise rebuilds it first —
# NOTE: the pairs.json shipped in this repo is a 12-pair demo fixture, not
# a full subsample; see data/processed/README.md for why and how to
# regenerate a real one)
python run_pipeline.py --eval-only --skip-judge --sample-size 30

# With LLM judge scoring (adds ~1 more LLM call per example)
python run_pipeline.py --eval-only --sample-size 30
```

**Why 30 examples by default:** the free Gemini tier is rate-limited to
roughly one request every 4.5s, and each example needs 3 LLM calls (intent,
reply, escalation) plus 1 more if the judge is enabled. At 200 examples that
adds up to over an hour even without a bug — so the fast path evaluates a
stratified subsample (still balanced across all 10 intents) to give you a
directionally-correct, reproducible result in the assignment's 15-minute
budget. `--sample-size 200` (or a paid-tier API key with `--api-delay 0.3`
or lower) gets you the exact numbers quoted in `REPORT.md`.

### 4. View Results

Results are saved to `outputs/`:
- `comparison.md` — Side-by-side metric comparison
- `main_results.json` — Detailed main system metrics
- `trivial_results.json` — Trivial baseline metrics
- `simple_results.json` — Simple baseline metrics

---

## Architecture

```
Customer Tweet
      |
      v
+-------------------+
| Intent Classifier  |  LLM few-shot (Gemini) with 10-intent taxonomy
+-------------------+
      |
      v
+-------------------+
| Reply Drafter      |  TF-IDF retrieval of similar historical pairs + LLM generation
+-------------------+
      |
      v
+-------------------+
| Escalation Decider |  Rule-based layer (keywords) + LLM judgment for grey zone
+-------------------+
      |
      v
  {intent, reply, escalation_decision + reason}
```

### Components

| File | Purpose |
|------|---------|
| `config.py` | Central configuration — API keys, intent taxonomy, escalation rules |
| `src/data_loader.py` | Download, filter (AppleSupport), clean, subsample the dataset |
| `src/thread_builder.py` | Reconstruct conversation threads, extract (customer, reply) pairs |
| `src/intent_classifier.py` | LLM-based intent classification with few-shot prompting |
| `src/reply_drafter.py` | RAG: TF-IDF retrieval + LLM reply generation |
| `src/escalation_decider.py` | Hybrid rules + LLM escalation routing |
| `src/evaluation/eval_harness.py` | Metrics: accuracy, F1, ROUGE-L, confusion matrices |
| `src/evaluation/llm_judge.py` | 4-dimension reply scoring rubric (1-5 scale) |
| `src/evaluation/baselines.py` | Trivial (random/template) and simple (TF-IDF/keyword) baselines |

---

## Golden Evaluation Set

**200 hand-labelled examples** in `data/golden_set.json`. These are
author-written (not pasted verbatim from the Kaggle CSV) but modelled closely
on ~500 real AppleSupport tweets I read while building the taxonomy — see
`data/golden_set_methodology.md` for exactly how and why, including the
trade-off that choice makes.

| Dimension | Distribution |
|-----------|-------------|
| Intents | 20 examples × 10 intents (balanced) |
| Difficulty | Easy: 81, Medium: 82, Hard: 37 |
| Escalation | Auto-handle: 151, Escalate: 49 |
| Reference replies | 47 examples (for ROUGE-L scoring) |

Sampling methodology documented in `data/golden_set_methodology.md`.

---

## Evaluation Approach

### Three Tasks Evaluated

1. **Intent Classification**: Accuracy, Macro-F1, per-class P/R/F1
2. **Reply Quality**: ROUGE-L (automated) + LLM-as-Judge (4 dimensions: relevance, brand voice, helpfulness, grounding)
3. **Escalation Decision**: Binary F1 (escalate vs. auto-handle)

### Three Systems Compared

| System | Intent | Reply | Escalation |
|--------|--------|-------|------------|
| **RAG+LLM Agent** | Gemini few-shot | TF-IDF retrieval + Gemini generation | Rules + Gemini |
| **Simple Baseline** | TF-IDF + LogReg | Nearest-neighbor retrieval | Keyword rules only |
| **Trivial Baseline** | Random | Fixed template | Random 50/50 |

### LLM-as-Judge

The judge scores each reply on a 1-5 scale across four dimensions:
- **Relevance**: Does the reply address the customer's actual issue?
- **Brand Voice**: Does it sound like Apple Support?
- **Helpfulness**: Does it provide concrete next steps?
- **Grounding**: Is it factually accurate, not hallucinated?

---

## Project Structure

```
.
├── README.md                          # This file
├── REPORT.md                          # Detailed 6-page report
├── requirements.txt                   # Python dependencies
├── config.py                          # Central configuration
├── run_pipeline.py                    # Single entry point
├── data/
│   ├── golden_set.json               # 200 hand-labelled examples
│   ├── golden_set_methodology.md     # Sampling documentation
│   └── processed/
│       ├── pairs.json                # DEMO fixture (12 pairs) — see below
│       └── README.md                 # What pairs.json is/isn't, how to regenerate
├── scripts/
│   ├── generate_golden_set.py        # Golden set generation script
│   └── add_reference_replies.py      # Adds reference replies for ROUGE-L
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Dataset download and preparation
│   ├── thread_builder.py             # Conversation thread reconstruction
│   ├── intent_classifier.py          # LLM intent classification
│   ├── reply_drafter.py              # RAG reply generation
│   ├── escalation_decider.py         # Hybrid escalation routing
│   └── evaluation/
│       ├── __init__.py
│       ├── eval_harness.py           # Metrics computation
│       ├── llm_judge.py              # Reply quality scoring
│       └── baselines.py              # Trivial and simple baselines
└── outputs/                           # Generated results
```

---

## Tech Stack

- **Python 3.11** — Core language
- **Google Gemini** (gemini-2.0-flash) — LLM for classification, generation, judging
- **scikit-learn** — TF-IDF vectorization, logistic regression, metrics
- **pandas** — Data processing
- **rouge-score** — Automated reply quality metric
- **kagglehub** — Dataset download

---

## Attribution

- **Dataset**: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) by Thought Vector (Kaggle)
- **LLM**: Google Gemini API (gemini-3.6-flash)

