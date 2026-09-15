"""
run_pipeline.py — Single entry point for the AI Support Agent pipeline.

Usage:
    python run_pipeline.py                          # Full pipeline
    python run_pipeline.py --eval-only              # Evaluation only (uses golden set)
    python run_pipeline.py --brand AmazonHelp       # Different brand
    python run_pipeline.py --max-conversations 1000 # Smaller subsample
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def parse_args():
    p = argparse.ArgumentParser(description="AI Support Agent Pipeline")
    p.add_argument("--brand", default=config.BRAND, help="Brand to build agent for")
    p.add_argument("--max-conversations", type=int, default=config.MAX_CONVERSATIONS)
    p.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    p.add_argument("--skip-judge", action="store_true", help="Skip LLM judge (faster)")
    p.add_argument("--data-path", type=str, default=None, help="Path to local CSV")
    p.add_argument("--golden-set", type=str, default=str(config.GOLDEN_SET_PATH))
    p.add_argument("--output-dir", type=str, default=str(config.OUTPUTS_DIR))
    p.add_argument("--sample-size", type=int, default=30,
                   help="Subsample golden set to N examples (0 = use all 200). "
                        "Default 30 keeps a full run (classify+draft+escalate, "
                        "no judge) under ~15 minutes on the free Gemini tier. "
                        "Use 200 for the full headline numbers (~1-2 hours on "
                        "free tier; a few minutes on a paid tier).")
    p.add_argument("--api-delay", type=float, default=4.5,
                   help="Minimum seconds between LLM calls (sets the single "
                        "shared throttle in src/llm_helper.py — this does NOT "
                        "stack with an additional per-batch delay). Default "
                        "4.5s is safe for the free Gemini tier (~13 req/min). "
                        "Use 0.1-0.5 for a paid tier for a much faster run.")
    return p.parse_args()


def load_golden_set(path: str, sample_size: int = 0) -> list:
    """Load the golden evaluation set, optionally subsampling."""
    import random
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded golden set: %d examples from %s", len(data), path)

    if sample_size and sample_size < len(data):
        # Stratified sampling: pick evenly across intents
        from collections import defaultdict
        by_intent = defaultdict(list)
        for ex in data:
            by_intent[ex["gold_intent"]].append(ex)
        per_intent = max(1, sample_size // len(by_intent))
        sampled = []
        rng = random.Random(config.RANDOM_SEED)
        for intent, examples in by_intent.items():
            sampled.extend(rng.sample(examples, min(per_intent, len(examples))))
        # Fill remainder if needed
        remaining = sample_size - len(sampled)
        if remaining > 0:
            pool = [e for e in data if e not in sampled]
            sampled.extend(rng.sample(pool, min(remaining, len(pool))))
        data = sampled[:sample_size]
        logger.info("Subsampled golden set to %d examples (stratified)", len(data))

    return data


def step_1_load_data(args):
    """Step 1: Download and prepare brand data."""
    logger.info("=" * 60)
    logger.info("STEP 1: Loading and preparing data for %s", args.brand)
    logger.info("=" * 60)

    from src.data_loader import prepare_brand_data
    data_path = Path(args.data_path) if args.data_path else None
    df = prepare_brand_data(
        brand=args.brand,
        max_rows=args.max_conversations,
        data_path=data_path,
    )
    logger.info("Prepared %d rows for brand %s", len(df), args.brand)
    return df


def step_2_build_threads(df):
    """Step 2: Reconstruct conversation threads and extract pairs."""
    logger.info("=" * 60)
    logger.info("STEP 2: Building conversation threads")
    logger.info("=" * 60)

    from src.thread_builder import build_pairs_from_df
    threads, pairs = build_pairs_from_df(df)
    logger.info("Built %d threads, extracted %d pairs", len(threads), len(pairs))

    # Save pairs for later use
    pairs_path = config.PROCESSED_DATA_DIR / "pairs.json"
    with open(pairs_path, "w", encoding="utf-8") as f:
        json.dump(pairs[:5000], f, indent=2)  # Cap for file size
    logger.info("Saved pairs to %s", pairs_path)

    return threads, pairs


def step_3_run_agent(golden_set, pairs, api_delay=4.5):
    """Step 3: Run the AI agent on the golden set."""
    logger.info("=" * 60)
    logger.info("STEP 3: Running AI agent on golden set (%d examples, throttle=%.1fs)",
                len(golden_set), api_delay)
    logger.info("=" * 60)

    from src.intent_classifier import classify_batch
    from src.reply_drafter import ReplyDrafter
    from src.escalation_decider import decide_batch

    # 3a. Intent Classification
    logger.info("--- 3a. Classifying intents ---")
    intent_results = classify_batch(golden_set)  # pacing handled by the shared LLM throttle
    for i, r in enumerate(intent_results):
        golden_set[i]["pred_intent"] = r["intent"]
        golden_set[i]["intent_confidence"] = r["confidence"]

    # 3b. Reply Drafting
    logger.info("--- 3b. Drafting replies ---")
    drafter = ReplyDrafter(pairs)
    # Merge predicted intents into messages for reply drafting
    draft_inputs = [
        {**g, "intent": g.get("pred_intent", g.get("gold_intent", "other"))}
        for g in golden_set
    ]
    reply_results = drafter.draft_batch(draft_inputs)  # pacing handled by the shared LLM throttle
    for i, r in enumerate(reply_results):
        golden_set[i]["generated_reply"] = r["reply"]

    # 3c. Escalation Decision
    logger.info("--- 3c. Making escalation decisions ---")
    esc_inputs = [
        {**g, "intent": g.get("pred_intent", g.get("gold_intent", "other"))}
        for g in golden_set
    ]
    esc_results = decide_batch(esc_inputs)  # pacing handled by the shared LLM throttle
    for i, r in enumerate(esc_results):
        golden_set[i]["pred_escalation"] = r["decision"]
        golden_set[i]["escalation_reason"] = r["reason"]

    return intent_results, reply_results, esc_results


def step_4_run_baselines(golden_set, pairs):
    """Step 4: Run baselines on the golden set."""
    logger.info("=" * 60)
    logger.info("STEP 4: Running baselines")
    logger.info("=" * 60)

    from src.evaluation.baselines import (
        RandomIntentBaseline,
        TfidfIntentBaseline,
        TemplateReplyBaseline,
        NearestNeighborReplyBaseline,
        RandomEscalationBaseline,
        KeywordEscalationBaseline,
    )

    # --- Trivial Baselines ---
    logger.info("--- Trivial baselines ---")
    rand_intent = RandomIntentBaseline()
    trivial_intent = rand_intent.classify_batch(golden_set)

    template_reply = TemplateReplyBaseline(pairs)
    trivial_reply = template_reply.draft_batch(golden_set)

    rand_esc = RandomEscalationBaseline()
    trivial_esc = rand_esc.decide_batch(golden_set)

    # --- Simple Baselines ---
    logger.info("--- Simple baselines ---")
    from sklearn.model_selection import train_test_split
    tfidf_intent = TfidfIntentBaseline()
    messages = [g["customer_message"] for g in golden_set]
    labels = [g["gold_intent"] for g in golden_set]
    
    # 75/25 stratified split to prevent data leakage in baseline training
    # Note: We still classify the full set so the output arrays align with the main system
    try:
        train_msgs, _, train_lbls, _ = train_test_split(
            messages, labels, test_size=0.25, stratify=labels, random_state=config.RANDOM_SEED
        )
    except ValueError:
        train_msgs, _, train_lbls, _ = train_test_split(
            messages, labels, test_size=0.25, random_state=config.RANDOM_SEED
        )
        
    tfidf_intent.fit(train_msgs, train_lbls)
    simple_intent = tfidf_intent.classify_batch(golden_set)

    nn_reply = NearestNeighborReplyBaseline(pairs)
    simple_reply = nn_reply.draft_batch(golden_set)

    kw_esc = KeywordEscalationBaseline()
    simple_esc = kw_esc.decide_batch(golden_set)

    return {
        "trivial": {
            "intent": trivial_intent,
            "reply": trivial_reply,
            "escalation": trivial_esc,
        },
        "simple": {
            "intent": simple_intent,
            "reply": simple_reply,
            "escalation": simple_esc,
        },
    }


def step_5_evaluate(golden_set, intent_preds, reply_results, esc_preds,
                     baselines, output_dir, skip_judge=False, api_delay=4.5):
    """Step 5: Run full evaluation and generate comparison."""
    logger.info("=" * 60)
    logger.info("STEP 5: Evaluating all systems")
    logger.info("=" * 60)

    from src.evaluation.eval_harness import (
        run_full_evaluation,
        format_comparison_table,
        save_results,
    )
    from src.evaluation.llm_judge import score_batch, compute_judge_human_agreement

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # LLM Judge for main system
    judge_scores = None
    if not skip_judge:
        logger.info("--- Running LLM Judge on main system replies ---")
        judge_items = [
            {
                "customer_message": g["customer_message"],
                "generated_reply": g.get("generated_reply", ""),
                "intent": g.get("pred_intent", g.get("gold_intent", "other")),
                "conversation_context": g.get("conversation_context"),
            }
            for g in golden_set
        ]
        judge_scores = score_batch(judge_items)  # pacing handled by the shared LLM throttle

    # Main system evaluation
    main_results = run_full_evaluation(
        golden_set, intent_preds, reply_results, esc_preds,
        judge_scores=judge_scores, system_name="RAG+LLM Agent",
    )

    # Trivial baseline evaluation
    trivial_results = run_full_evaluation(
        golden_set,
        baselines["trivial"]["intent"],
        baselines["trivial"]["reply"],
        baselines["trivial"]["escalation"],
        system_name="Trivial Baseline",
    )

    # Simple baseline evaluation
    simple_results = run_full_evaluation(
        golden_set,
        baselines["simple"]["intent"],
        baselines["simple"]["reply"],
        baselines["simple"]["escalation"],
        system_name="Simple Baseline",
    )

    # Judge-human agreement calibration is disabled because human scores 
    # must be based on reviewing *actual generated replies*, not just expected 
    # quality based on the input message. 
    # If you review the actual outputs, you can re-enable this.

    # Save all results
    save_results(main_results, output_dir / "main_results.json")
    save_results(trivial_results, output_dir / "trivial_results.json")
    save_results(simple_results, output_dir / "simple_results.json")

    # Comparison table
    comparison = format_comparison_table([main_results, simple_results, trivial_results])
    logger.info("\n\n%s\n", comparison)

    with open(output_dir / "comparison.md", "w", encoding="utf-8") as f:
        f.write("# Evaluation Results Comparison\n\n")
        f.write(comparison)
        f.write("\n\n## Detailed Results\n\n")
        f.write("### Main System — Intent Classification\n")
        f.write(f"```\n{main_results['intent']['report_str']}\n```\n\n")
        f.write("### Main System — Escalation\n")
        f.write(f"```\n{main_results['escalation']['report_str']}\n```\n\n")
        if judge_scores:
            f.write("### Main System — LLM Judge Scores\n")
            for k, v in main_results.get("reply_judge", {}).items():
                f.write(f"  - {k}: {v}\n")
        if main_results.get("judge_human_agreement"):
            f.write("\n### Judge-Human Agreement (Calibration)\n")
            for k, v in main_results["judge_human_agreement"].items():
                f.write(f"  - {k}: {v}\n")

    logger.info("All results saved to %s", output_dir)
    return main_results, simple_results, trivial_results


def main():
    args = parse_args()
    start = time.time()

    # Override config with CLI args
    config.BRAND = args.brand
    config.MAX_CONVERSATIONS = args.max_conversations

    # Single source of truth for LLM call pacing (see src/llm_helper.py).
    from src.llm_helper import set_min_call_interval
    set_min_call_interval(args.api_delay)

    logger.info(">> AI Support Agent Pipeline -- brand: %s", args.brand)
    logger.info("   GEMINI_API_KEY set: %s", bool(config.GEMINI_API_KEY))
    logger.info("   LLM call throttle: %.1fs between calls", args.api_delay)

    if not config.GEMINI_API_KEY:
        logger.error(
            "[ERROR] GEMINI_API_KEY not set! Export it:\n"
            "   set GEMINI_API_KEY=your-key-here  (Windows)\n"
            "   export GEMINI_API_KEY=your-key-here  (Linux/Mac)"
        )
        sys.exit(1)

    # Load golden set
    golden_set = load_golden_set(args.golden_set, sample_size=args.sample_size)

    if args.eval_only:
        # In eval-only mode, we need pre-computed pairs
        pairs_path = config.PROCESSED_DATA_DIR / "pairs.json"
        MIN_HEALTHY_PAIRS = 200  # below this, retrieval quality degrades sharply
        if pairs_path.exists():
            with open(pairs_path, "r", encoding="utf-8") as f:
                pairs = json.load(f)
            logger.info("Loaded %d pre-computed pairs", len(pairs))
            if len(pairs) < MIN_HEALTHY_PAIRS:
                logger.warning(
                    "[DEMO DATA] Only %d historical (customer, brand_reply) pairs "
                    "are cached in %s. This repo ships a small illustrative "
                    "fixture, NOT the full ~2-3k pair subsample described in the "
                    "report -- Kaggle downloads aren't available in every "
                    "environment, so a full-size cache isn't bundled. The "
                    "reply-drafting RAG step will retrieve weak/thin matches "
                    "and quality will look worse than the report's headline "
                    "numbers. For a real run, delete this file and use the "
                    "full pipeline (drop --eval-only) with a Kaggle API key "
                    "configured (~/.kaggle/kaggle.json) so it downloads and "
                    "threads the full dataset first.",
                    len(pairs), pairs_path,
                )
        else:
            logger.warning("No pre-computed pairs found. Running data pipeline first...")
            df = step_1_load_data(args)
            _, pairs = step_2_build_threads(df)
    else:
        # Full pipeline
        df = step_1_load_data(args)
        _, pairs = step_2_build_threads(df)

    # Run main agent
    intent_preds, reply_results, esc_preds = step_3_run_agent(
        golden_set, pairs, api_delay=args.api_delay
    )

    # Run baselines
    baselines = step_4_run_baselines(golden_set, pairs)

    # Evaluate
    main_r, simple_r, trivial_r = step_5_evaluate(
        golden_set, intent_preds, reply_results, esc_preds,
        baselines, args.output_dir, skip_judge=args.skip_judge,
        api_delay=args.api_delay,
    )

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("[DONE] Pipeline complete in %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)
    logger.info("=" * 60)

    # Print headline numbers
    print("\n" + "=" * 60)
    print("HEADLINE RESULTS")
    print("=" * 60)
    print(f"  Intent Macro-F1:    {main_r['intent']['macro_f1']:.3f}")
    print(f"  Escalation F1:      {main_r['escalation']['f1']:.3f}")
    if main_r.get("reply_judge"):
        print(f"  Reply Quality (Judge Avg): {main_r['reply_judge'].get('overall_mean', 'N/A')}")
    if main_r.get("reply_automated"):
        print(f"  Reply ROUGE-L:      {main_r['reply_automated'].get('rouge_l_mean', 'N/A')}")
    print(f"\n  vs Simple Baseline:")
    print(f"    Intent F1: {simple_r['intent']['macro_f1']:.3f}")
    print(f"    Escalation F1: {simple_r['escalation']['f1']:.3f}")
    print(f"\n  vs Trivial Baseline:")
    print(f"    Intent F1: {trivial_r['intent']['macro_f1']:.3f}")
    print(f"    Escalation F1: {trivial_r['escalation']['f1']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
