"""
Experiment D: Baseline Decision-Policy Comparison

Evaluates 4 deployment policies on the identical held-out test suite:
- Policy 1: Always Automate
- Policy 2: Confidence Only (threshold = 0.85)
- Policy 3: Confidence + OOD Detection (threshold = 0.85)
- Policy 4: Full Risk-Aware System (actual DecisionEngine with 3-way taxonomy)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from research.experiments.common import (
    load_dataset_and_split,
    load_vectorizer_and_models,
    save_csv_dual,
    get_ood_benchmark_queries,
    RESEARCH_RESULTS_DIR,
    RESEARCH_GRAPHS_DIR,
    EVAL_DIR
)
from src.intent_risk_map import get_intent_risk_level, INTENT_RISK_MAP
from src.sentiment_analyzer import SentimentAnalyzer
from src.ood_detector import OODDetector
from src.risk_engine import RiskEngine
from src.decision_engine import DecisionEngine


def run_experiment_d():
    print("=" * 75)
    print("EXPERIMENT D: BASELINE DECISION-POLICY COMPARISON")
    print("=" * 75)

    # 1. Load data and models
    X_train, X_test, y_train, y_test, _ = load_dataset_and_split()
    vectorizer, models = load_vectorizer_and_models()
    calibrated_model = models["Calibrated Linear SVM"]

    # 2. Prepare Reference Matrix for OOD TF-IDF Cosine Similarity
    print("Transforming training data for OOD reference matrix...")
    ref_matrix = vectorizer.transform(X_train)

    # Initialize live modules
    sentiment_analyzer = SentimentAnalyzer()
    ood_detector = OODDetector(vectorizer=vectorizer, reference_matrix=ref_matrix)
    risk_engine = RiskEngine()
    decision_engine = DecisionEngine(confidence_threshold=0.85)

    # 3. Build Unified Test Pool: Held-out In-Domain (8,977) + OOD Benchmark (100)
    ood_queries = get_ood_benchmark_queries()
    
    in_domain_records = []
    for query, true_intent in zip(X_test, y_test):
        is_hr = (INTENT_RISK_MAP.get(true_intent, "LOW") == "HIGH")
        in_domain_records.append({
            "query": query,
            "true_intent": true_intent,
            "is_ood": False,
            "is_high_risk": is_hr
        })

    ood_records = []
    for query in ood_queries:
        ood_records.append({
            "query": query,
            "true_intent": "OUT_OF_DOMAIN",
            "is_ood": True,
            "is_high_risk": False
        })

    test_pool = pd.DataFrame(in_domain_records + ood_records)
    print(f"Total Test Pool: {len(test_pool)} queries (In-Domain: {len(in_domain_records)}, OOD: {len(ood_records)})")
    print(f"High-Risk In-Domain Queries: {test_pool['is_high_risk'].sum()}")

    # 4. Batch Vectorize and Predict using Calibrated Model
    print("\nRunning inference through Calibrated Linear SVM...")
    X_pool_vec = vectorizer.transform(test_pool["query"])
    pred_intents = calibrated_model.predict(X_pool_vec)
    prob_matrix = calibrated_model.predict_proba(X_pool_vec)
    conf_scores = prob_matrix.max(axis=1)

    # 5. Extract Features for Decision Policies
    print("Evaluating OOD, Sentiment, and Risk for all test queries...")
    pool_records = []
    total_samples = len(test_pool)

    for i in range(total_samples):
        query = test_pool.iloc[i]["query"]
        true_intent = test_pool.iloc[i]["true_intent"]
        is_ood_true = test_pool.iloc[i]["is_ood"]
        is_hr_true = test_pool.iloc[i]["is_high_risk"]
        pred_intent = pred_intents[i]
        conf = float(conf_scores[i])
        query_vec = X_pool_vec[i]

        # Actual system component evaluation
        sentiment_data = sentiment_analyzer.analyze(query)
        ood_data = ood_detector.detect(query, conf, query_vector=query_vec)
        risk_data = risk_engine.evaluate_risk(pred_intent, conf, sentiment_data, ood_data)
        full_dec_data = decision_engine.make_decision(query, pred_intent, conf, sentiment_data, ood_data, risk_data)

        # Policy Decisions:
        # Policy 1: Always Automate
        p1_decision = "AUTOMATE"

        # Policy 2: Confidence Only (threshold = 0.85, else Human Escalation)
        p2_decision = "AUTOMATE" if conf >= 0.85 else "HUMAN_ESCALATION"

        # Policy 3: Confidence + OOD (conf >= 0.85 and domain_status == 'IN_DOMAIN', else Human Escalation)
        p3_decision = "AUTOMATE" if (conf >= 0.85 and not ood_data["is_ood"]) else "HUMAN_ESCALATION"

        # Policy 4: Full Risk-Aware System (DecisionEngine output: AUTOMATE, CLARIFY, HUMAN_ESCALATION)
        p4_decision = full_dec_data["decision"]

        pool_records.append({
            "query": query,
            "true_intent": true_intent,
            "pred_intent": pred_intent,
            "confidence": conf,
            "is_ood_true": is_ood_true,
            "is_high_risk_true": is_hr_true,
            "domain_status_pred": ood_data["domain_status"],
            "is_ood_pred": ood_data["is_ood"],
            "sentiment_pred": sentiment_data["sentiment"],
            "risk_level_pred": risk_data["risk_level"],
            "risk_score_pred": risk_data["risk_score"],
            "p1_decision": p1_decision,
            "p2_decision": p2_decision,
            "p3_decision": p3_decision,
            "p4_decision": p4_decision
        })

    eval_df = pd.DataFrame(pool_records)

    # 6. Compute Comparative Policy Metrics
    policy_names = [
        "Policy 1: Always Automate",
        "Policy 2: Confidence Only (tau=0.85)",
        "Policy 3: Confidence + OOD",
        "Policy 4: Full Risk-Aware System"
    ]
    policy_cols = ["p1_decision", "p2_decision", "p3_decision", "p4_decision"]

    num_total = len(eval_df)
    num_high_risk = eval_df["is_high_risk_true"].sum()
    num_ood = eval_df["is_ood_true"].sum()

    comparison_results = []

    for name, col in zip(policy_names, policy_cols):
        decisions = eval_df[col]
        
        # Coverage
        num_automated = (decisions == "AUTOMATE").sum()
        num_escalated = (decisions == "HUMAN_ESCALATION").sum()
        num_clarified = (decisions == "CLARIFY").sum()
        
        coverage = num_automated / num_total
        escalation_rate = num_escalated / num_total
        clarification_rate = num_clarified / num_total

        # Selective Accuracy among Automated Queries
        automated_mask = (decisions == "AUTOMATE")
        if num_automated > 0:
            # Correct prediction requires in-domain AND true_intent == pred_intent
            correct_automated = ((eval_df["pred_intent"] == eval_df["true_intent"]) & 
                                 (~eval_df["is_ood_true"]) & 
                                 automated_mask).sum()
            selective_accuracy = correct_automated / num_automated
            incorrect_automation_rate = 1.0 - selective_accuracy
        else:
            selective_accuracy = np.nan
            incorrect_automation_rate = np.nan

        # High-Risk Automation Rate
        hr_mask = eval_df["is_high_risk_true"]
        hr_automated = (automated_mask & hr_mask).sum()
        hr_auto_rate = hr_automated / num_high_risk if num_high_risk > 0 else 0.0

        # OOD False-Acceptance Rate
        ood_mask = eval_df["is_ood_true"]
        ood_automated = (automated_mask & ood_mask).sum()
        ood_false_accept_rate = ood_automated / num_ood if num_ood > 0 else 0.0

        comparison_results.append({
            "Policy": name,
            "Total Queries": num_total,
            "Automated Count": num_automated,
            "Escalated Count": num_escalated,
            "Clarified Count": num_clarified,
            "Automation Coverage (%)": round(coverage * 100, 2),
            "Selective Accuracy (%)": round(selective_accuracy * 100, 2),
            "Incorrect Automation Rate (%)": round(incorrect_automation_rate * 100, 2),
            "Human Escalation Rate (%)": round(escalation_rate * 100, 2),
            "Clarification Rate (%)": round(clarification_rate * 100, 2),
            "High-Risk Automation Rate (%)": round(hr_auto_rate * 100, 2),
            "High-Risk Automated Count": hr_automated,
            "OOD False-Acceptance Rate (%)": round(ood_false_accept_rate * 100, 2),
            "OOD False-Accepted Count": ood_automated
        })

    comp_df = pd.DataFrame(comparison_results)

    print("\n" + "=" * 75)
    print("BASELINE DECISION-POLICY COMPARISON RESULTS")
    print("=" * 75)
    print(comp_df.to_string(index=False))

    # Save to dual locations
    save_csv_dual(comp_df, "research_baseline_comparison.csv")
    save_csv_dual(eval_df, "research_baseline_evaluation_trace.csv")

    # 7. Generate Publication-Quality Figure
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    policies_short = ["Always Automate", "Confidence Only", "Conf + OOD", "Full Risk-Aware"]
    x = np.arange(len(policies_short))
    width = 0.35

    # Subplot 1: Automation Coverage vs. Selective Accuracy
    ax1.bar(x - width/2, comp_df["Automation Coverage (%)"], width, label="Automation Coverage (%)", color="#2563EB", alpha=0.85)
    ax1.bar(x + width/2, comp_df["Selective Accuracy (%)"], width, label="Selective Accuracy (%)", color="#10B981", alpha=0.85)
    ax1.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Operational Throughput & Accuracy by Policy", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies_short, fontsize=10, rotation=15)
    ax1.set_ylim(0, 105)
    ax1.legend(loc="lower left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    for i in range(len(x)):
        ax1.text(x[i] - width/2, comp_df["Automation Coverage (%)"].iloc[i] + 1.5, f"{comp_df['Automation Coverage (%)'].iloc[i]}%", ha='center', fontsize=9, fontweight='bold')
        ax1.text(x[i] + width/2, comp_df["Selective Accuracy (%)"].iloc[i] + 1.5, f"{comp_df['Selective Accuracy (%)'].iloc[i]}%", ha='center', fontsize=9, fontweight='bold')

    # Subplot 2: Critical Risk Rates (High-Risk Automation & OOD Acceptance)
    ax2.bar(x - width/2, comp_df["High-Risk Automation Rate (%)"], width, label="High-Risk Automation Rate (%)", color="#EF4444", alpha=0.85)
    ax2.bar(x + width/2, comp_df["OOD False-Acceptance Rate (%)"], width, label="OOD False-Acceptance Rate (%)", color="#F59E0B", alpha=0.85)
    ax2.set_ylabel("Failure / Exposure Rate (%)", fontsize=12, fontweight="bold")
    ax2.set_title("Safety Vulnerability & Failure Exposure by Policy", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(policies_short, fontsize=10, rotation=15)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    for i in range(len(x)):
        ax2.text(x[i] - width/2, comp_df["High-Risk Automation Rate (%)"].iloc[i] + 1.5, f"{comp_df['High-Risk Automation Rate (%)'].iloc[i]}%", ha='center', fontsize=9, fontweight='bold')
        ax2.text(x[i] + width/2, comp_df["OOD False-Acceptance Rate (%)"].iloc[i] + 1.5, f"{comp_df['OOD False-Acceptance Rate (%)'].iloc[i]}%", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()

    plot_path1 = os.path.join(RESEARCH_GRAPHS_DIR, "research_baseline_comparison.png")
    plot_path2 = os.path.join(EVAL_DIR, "research_baseline_comparison.png")
    plt.savefig(plot_path1, dpi=300, bbox_inches="tight")
    plt.savefig(plot_path2, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[Figure] Comparative graph saved: {plot_path1}")
    print(f"[Figure] Comparative graph saved: {plot_path2}")


if __name__ == "__main__":
    run_experiment_d()
