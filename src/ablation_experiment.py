"""
Ablation Experiment Script for E-Commerce Customer Service Decision Engine.

Compares system performance across 5 architecture variants:
  Variant A: Raw Intent Classifier Only (Uncalibrated)
  Variant B: Calibrated Confidence Thresholding
  Variant C: Calibrated Confidence + OOD Detection
  Variant D: Calibrated Confidence + OOD + Sentiment Analysis
  Variant E: Full System (Risk Engine + Hard High-Risk Policy + 3-Way Decision Engine)

Outputs results to evaluation/ablation_results.csv.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.decision_pipeline import CustomerServiceDecisionEngine

def run_ablation_experiment():
    print("=" * 70)
    print("RUNNING ABLATION EXPERIMENT")
    print("=" * 70)

    os.makedirs("evaluation", exist_ok=True)
    engine = CustomerServiceDecisionEngine()

    eval_queries = [
        # In-domain safe queries (LOW risk)
        {"query": "Where can I track my package?", "is_ood": False, "is_high_risk": False},
        {"query": "What are your physical store opening hours?", "is_ood": False, "is_high_risk": False},
        {"query": "How do I check my order history?", "is_ood": False, "is_high_risk": False},
        
        # High risk / sensitive queries
        {"query": "I want to request a refund for my order. I am very disappointed!", "is_ood": False, "is_high_risk": True},
        {"query": "I need to recover my password immediately.", "is_ood": False, "is_high_risk": True},
        {"query": "Please close my account permanently.", "is_ood": False, "is_high_risk": True},
        
        # Out-of-Domain queries
        {"query": "Who won the cricket match yesterday?", "is_ood": True, "is_high_risk": False},
        {"query": "What is the weather like today?", "is_ood": True, "is_high_risk": False},
        {"query": "Write a Python program to sort a list.", "is_ood": True, "is_high_risk": False},
        {"query": "Explain quantum physics in simple words.", "is_ood": True, "is_high_risk": False}
    ]

    ablation_summary = []

    for variant_name, use_calib, use_ood, use_sent, use_risk in [
        ("Variant A: Raw Intent Classifier Only", False, False, False, False),
        ("Variant B: + Confidence Calibration", True, False, False, False),
        ("Variant C: + OOD Detection", True, True, False, False),
        ("Variant D: + Sentiment Analysis", True, True, True, False),
        ("Variant E: Full System (Hard High-Risk Policy + Decision Engine)", True, True, True, True),
    ]:
        automated_count = 0
        escalated_count = 0
        false_automation_count = 0 # Automating OOD or High-Risk requests

        for test in eval_queries:
            res = engine.process(test["query"])

            if not use_calib:
                dec = "AUTOMATE"
            elif not use_ood and not use_sent and not use_risk:
                dec = "AUTOMATE" if res["confidence"] >= 0.85 else "HUMAN_ESCALATION"
            elif use_ood and not use_sent and not use_risk:
                if res["domain_status"] == "OUT_OF_DOMAIN" or res["confidence"] < 0.85:
                    dec = "HUMAN_ESCALATION"
                else:
                    dec = "AUTOMATE"
            elif use_ood and use_sent and not use_risk:
                if res["domain_status"] == "OUT_OF_DOMAIN" or res["confidence"] < 0.85 or res["sentiment"] == "NEGATIVE":
                    dec = "HUMAN_ESCALATION"
                else:
                    dec = "AUTOMATE"
            else:
                dec = res["decision"]

            if dec == "AUTOMATE":
                automated_count += 1
                if test["is_ood"] or test["is_high_risk"]:
                    false_automation_count += 1
            else:
                escalated_count += 1

        total = len(eval_queries)
        safety_rate = round((1.0 - (false_automation_count / total)) * 100, 2)
        ablation_summary.append({
            "System Variant": variant_name,
            "Total Test Queries": total,
            "Automated Count": automated_count,
            "Escalated Count": escalated_count,
            "False Automation (OOD/High Risk)": false_automation_count,
            "Automation Safety Rate (%)": safety_rate
        })

    summary_df = pd.DataFrame(ablation_summary)
    output_path = "evaluation/ablation_results.csv"
    summary_df.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    run_ablation_experiment()
