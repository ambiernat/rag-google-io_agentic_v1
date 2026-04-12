#!/usr/bin/env python
# coding: utf-8
"""
evaluate_agentic.py
Evaluate agentic retrieval (LangGraph multi-strategy) using YAML configuration.
"""
import sys
import os
import logging
import yaml
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from agent.graph import run_agent
from evaluation.metrics import recall_at_k, precision_at_k, mrr

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)  # suppress retriever INFO logs during eval

# -----------------------------
# Load config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "evaluation" / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

GT_DIR = PROJECT_ROOT / config["evaluation"]["ground_truth_dir"]
TOP_K = config["evaluation"].get("top_k", 5)
GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / config["evaluation"]["ground_truth_dir"]
    / config["evaluation"]["ground_truth_file"]
)

# Prepare timestamped output file
timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
OUTPUT_PATH = (
    PROJECT_ROOT
    / config["evaluation"]["output_dir"]
    / f"agentic_eval_{timestamp}.json"
)

logger.info(f"[INFO] Using ground truth file: {GROUND_TRUTH_PATH.name}")
logger.info(f"[INFO] Results will be saved to: {OUTPUT_PATH}")

# -----------------------------
# Load ground truth
# -----------------------------
with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

# -----------------------------
# Evaluation loop
# -----------------------------
results_all = []
print(f"[INFO] Evaluating AGENTIC retriever for {len(ground_truth)} queries...")

for item in tqdm(ground_truth, desc="Queries"):
    query = item["query"]
    relevant_ids = item.get("relevant_doc_ids", [])

    # Run agentic graph
    final_state = run_agent(query)

    # Extract retrieved doc IDs from best results
    retrieved_ids = [
        hit.payload.get("doc_id", str(hit.id))
        for hit in final_state["results"]
    ]

    results_all.append({
        "query": query,
        "rewritten_query": final_state.get("rewritten_query", ""),
        "strategy_used": final_state.get("strategy", ""),
        "attempts": final_state.get("attempts", 0),
        "agent_score": final_state.get("score", 0.0),
        "relevant_doc_ids": relevant_ids,
        "retrieved_doc_ids": retrieved_ids,
        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k=TOP_K),
        "mrr": mrr(retrieved_ids, relevant_ids),
        "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k=TOP_K),
    })

# -----------------------------
# Save results
# -----------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results_all, f, indent=2)

print(f"[OK] Agentic evaluation complete. Results saved to {OUTPUT_PATH}")

# -----------------------------
# Summary table
# -----------------------------
n = len(results_all)
avg_recall    = sum(r["recall_at_k"]    for r in results_all) / n
avg_mrr       = sum(r["mrr"]            for r in results_all) / n
avg_precision = sum(r["precision_at_k"] for r in results_all) / n

# Strategy breakdown
from collections import Counter
strategy_counts = Counter(r["strategy_used"] for r in results_all)
avg_attempts = sum(r["attempts"] for r in results_all) / n

print()
print("=" * 44)
print(f"{'Metric':<20} {'Score':>10}")
print("-" * 44)
print(f"{'Recall@K':<20} {avg_recall:>10.4f}")
print(f"{'MRR':<20} {avg_mrr:>10.4f}")
print(f"{'Precision@K':<20} {avg_precision:>10.4f}")
print(f"{'Avg Attempts':<20} {avg_attempts:>10.2f}")
print("=" * 44)
print(f"\nQueries evaluated: {n}")
print(f"\nStrategy breakdown:")
for strategy, count in strategy_counts.most_common():
    print(f"  {strategy:<10} {count:>4} queries ({count/n*100:.1f}%)")