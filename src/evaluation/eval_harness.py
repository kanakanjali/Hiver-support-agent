"""
Evaluation Harness — orchestrates metrics computation across all three tasks.

Computes:
  • Intent classification: Accuracy, Macro-F1, per-intent P/R/F1, confusion matrix
  • Reply quality: ROUGE-L, LLM-Judge scores (4 dimensions)
  • Escalation: Precision, Recall, F1 for the 'escalate' class
  • Comparison tables across baselines and the main system
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

import config

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  INTENT CLASSIFICATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_intent_classification(
    predictions: List[str],
    gold: List[str],
    label_names: Optional[List[str]] = None,
) -> dict:
    """
    Compute intent classification metrics.

    Returns accuracy, macro-F1, weighted-F1, and per-class report.
    """
    label_names = label_names or config.INTENT_LABELS

    accuracy = accuracy_score(gold, predictions)
    macro_f1 = f1_score(gold, predictions, average="macro", zero_division=0)
    weighted_f1 = f1_score(gold, predictions, average="weighted", zero_division=0)

    report_dict = classification_report(
        gold, predictions, labels=label_names, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        gold, predictions, labels=label_names, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(gold, predictions, labels=label_names)

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": report_dict,
        "report_str": report_str,
        "confusion_matrix": cm.tolist(),
        "label_names": label_names,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  REPLY QUALITY METRICS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_reply_quality_automated(
    generated_replies: List[str],
    reference_replies: List[str],
) -> dict:
    """
    Compute automated reply quality metrics: ROUGE-L and average length.

    Skips ROUGE-L when no non-empty reference replies are available
    (e.g., golden set without reference_reply fields).
    """
    # Filter to pairs where the reference is non-empty
    valid_pairs = [
        (gen, ref)
        for gen, ref in zip(generated_replies, reference_replies)
        if ref.strip()
    ]

    # ROUGE-L
    avg_rouge_l = None
    rouge_scores = []
    if not valid_pairs:
        logger.warning(
            "No non-empty reference replies found — skipping ROUGE-L. "
            "Add 'reference_reply' to the golden set for automated reply metrics."
        )
    else:
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            rouge_scores = [
                scorer.score(ref, gen)["rougeL"].fmeasure
                for gen, ref in valid_pairs
            ]
            avg_rouge_l = float(np.mean(rouge_scores)) if rouge_scores else 0.0
        except ImportError:
            logger.warning("rouge-score not installed; skipping ROUGE-L")

    # Length stats
    gen_lengths = [len(r.split()) for r in generated_replies]
    ref_lengths = [len(r.split()) for r in reference_replies if r.strip()]

    return {
        "rouge_l_mean": round(avg_rouge_l, 4) if avg_rouge_l is not None else None,
        "rouge_l_scores": [round(s, 4) for s in rouge_scores],
        "n_references": len(valid_pairs),
        "avg_gen_length_words": round(float(np.mean(gen_lengths)), 1) if gen_lengths else 0.0,
        "avg_ref_length_words": round(float(np.mean(ref_lengths)), 1) if ref_lengths else 0.0,
    }


def aggregate_judge_scores(scores: List[dict]) -> dict:
    """Aggregate LLM judge scores across items."""
    dims = ["relevance", "brand_voice", "helpfulness", "grounding", "overall"]
    agg = {}
    for dim in dims:
        vals = [s[dim] for s in scores if dim in s]
        if vals:
            agg[f"{dim}_mean"] = round(float(np.mean(vals)), 2)
            agg[f"{dim}_std"] = round(float(np.std(vals)), 2)
            agg[f"{dim}_median"] = float(np.median(vals))
    agg["n_scored"] = len(scores)
    return agg


# ══════════════════════════════════════════════════════════════════════════════
#  ESCALATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_escalation(
    predictions: List[str],
    gold: List[str],
) -> dict:
    """
    Compute escalation decision metrics.

    Binary task: 'escalate' is the positive class.
    """
    # Convert to binary
    pred_bin = [1 if p == "escalate" else 0 for p in predictions]
    gold_bin = [1 if g == "escalate" else 0 for g in gold]

    accuracy = accuracy_score(gold_bin, pred_bin)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold_bin, pred_bin, average="binary", zero_division=0
    )

    # Per-class report
    report_str = classification_report(
        gold, predictions, labels=["auto_handle", "escalate"], zero_division=0
    )

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "report_str": report_str,
        "confusion_matrix": confusion_matrix(
            gold, predictions, labels=["auto_handle", "escalate"]
        ).tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FULL EVALUATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_full_evaluation(
    golden_set: List[dict],
    intent_preds: List[dict],
    reply_results: List[dict],
    escalation_preds: List[dict],
    judge_scores: Optional[List[dict]] = None,
    system_name: str = "main",
) -> dict:
    """
    Run the full evaluation pipeline on one system's outputs.

    Parameters
    ----------
    golden_set : list of dict
        Gold-standard examples with ``gold_intent``, ``gold_escalation``.
    intent_preds : list of dict
        Each has ``intent``.
    reply_results : list of dict
        Each has ``reply``.
    escalation_preds : list of dict
        Each has ``decision``.
    judge_scores : optional list of dict
        Pre-computed LLM judge scores.

    Returns
    -------
    dict with all metrics.
    """
    gold_intents = [g["gold_intent"] for g in golden_set]
    pred_intents = [p["intent"] for p in intent_preds]

    gold_escalations = [g["gold_escalation"] for g in golden_set]
    pred_escalations = [p["decision"] for p in escalation_preds]

    generated_replies = [r.get("reply", "") for r in reply_results]
    reference_replies = [g.get("reference_reply", g.get("brand_reply", "")) for g in golden_set]

    results = {
        "system": system_name,
        "n_examples": len(golden_set),
        "intent": evaluate_intent_classification(pred_intents, gold_intents),
        "escalation": evaluate_escalation(pred_escalations, gold_escalations),
        "reply_automated": evaluate_reply_quality_automated(
            generated_replies, reference_replies
        ),
    }

    if judge_scores:
        results["reply_judge"] = aggregate_judge_scores(judge_scores)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════

def format_comparison_table(results_list: List[dict]) -> str:
    """
    Format a comparison table across multiple systems.

    Returns a markdown table string.
    """
    header = (
        "| System | Intent Acc | Intent F1 | Reply ROUGE-L | "
        "Reply Judge Avg | Escalation F1 |\n"
        "|--------|-----------|-----------|---------------|"
        "----------------|---------------|\n"
    )
    rows = []
    for r in results_list:
        intent_acc = r.get("intent", {}).get("accuracy", "—")
        intent_f1 = r.get("intent", {}).get("macro_f1", "—")
        rouge_l = r.get("reply_automated", {}).get("rouge_l_mean", "—")
        judge_avg = r.get("reply_judge", {}).get("overall_mean", "—")
        esc_f1 = r.get("escalation", {}).get("f1", "—")
        rows.append(
            f"| {r['system']:20s} | {intent_acc} | {intent_f1} | "
            f"{rouge_l} | {judge_avg} | {esc_f1} |"
        )
    return header + "\n".join(rows)


def save_results(results: dict, output_path: Path):
    """Save evaluation results to JSON."""

    def _convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_convert)
    logger.info("Results saved to %s", output_path)
