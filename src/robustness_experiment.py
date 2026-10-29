"""
Robustness Experiment Script for E-Commerce Customer Service Decision Engine.

Evaluates system stability under query perturbations:
  - Standard in-domain queries
  - Rephrased queries
  - Typo / noisy queries
  - Emotionally negative / angry queries
  - Ambiguous queries
  - High-risk queries (Refunds, Password Recovery)
  - Out-of-Domain (OOD) queries

Outputs results to evaluation/robustness_results.csv.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.decision_pipeline import CustomerServiceDecisionEngine

def run_robustness_experiment():
    print("=" * 70)
    print("RUNNING ROBUSTNESS EXPERIMENT")
    print("=" * 70)

    os.makedirs("evaluation", exist_ok=True)
    engine = CustomerServiceDecisionEngine()

    test_cases = [
        {
            "category": "Standard In-Domain",
            "query": "Where can I track my package?",
            "expected_decision": "AUTOMATE"
        },
        {
            "category": "Rephrased Query",
            "query": "Could you please tell me where my parcel is right now?",
            "expected_decision": "CLARIFY"
        },
        {
            "category": "Typo / Noisy Query",
            "query": "wheres my ordr pkg track status?",
            "expected_decision": "HUMAN_ESCALATION"
        },
        {
            "category": "Angry / Negative Sentiment (Low Risk)",
            "query": "WHERE IS MY PACKAGE?! I HAVE BEEN WAITING FOREVER AND THIS IS RIDICULOUS!",
            "expected_decision": "AUTOMATE"
        },
        {
            "category": "Ambiguous Query",
            "query": "I need help with my order.",
            "expected_decision": "CLARIFY"
        },
        {
            "category": "High-Risk Refund Request",
            "query": "I want to request a refund for my order. I am very disappointed!",
            "expected_decision": "HUMAN_ESCALATION"
        },
        {
            "category": "Account Security / Password (High Risk)",
            "query": "I need to recover my password immediately.",
            "expected_decision": "HUMAN_ESCALATION"
        },
        {
            "category": "Out-Of-Domain (Sports)",
            "query": "Who will win tomorrow's cricket match?",
            "expected_decision": "HUMAN_ESCALATION"
        },
        {
            "category": "Out-Of-Domain (Weather)",
            "query": "What is the weather forecast for tomorrow?",
            "expected_decision": "HUMAN_ESCALATION"
        },
        {
            "category": "Out-Of-Domain (Programming)",
            "query": "Write a Python script to sort a list of numbers.",
            "expected_decision": "HUMAN_ESCALATION"
        }
    ]

    results = []

    for test in test_cases:
        res = engine.process(test["query"])
        
        results.append({
            "Test Category": test["category"],
            "Query": test["query"],
            "Predicted Intent": res["predicted_intent"],
            "Confidence (%)": res["confidence_percentage"],
            "Sentiment": res["sentiment"],
            "Domain Status": res["domain_status"],
            "Risk Level": res["risk_level"],
            "Risk Score": res["risk_score"],
            "Actual Decision": res["decision"],
            "Reason": res["reason"]
        })

        print(f"\nCategory       : {test['category']}")
        print(f"Query          : '{test['query']}'")
        print(f"Predicted Intent: {res['predicted_intent']}")
        print(f"Confidence     : {res['confidence_percentage']:.2f}%")
        print(f"Sentiment      : {res['sentiment']}")
        print(f"Domain Status  : {res['domain_status']}")
        print(f"Risk Level     : {res['risk_level']} (Score: {res['risk_score']:.2f})")
        print(f"Actual Decision: {res['decision']}")

    results_df = pd.DataFrame(results)
    output_path = "evaluation/robustness_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("ROBUSTNESS EXPERIMENT COMPLETE")
    print(f"Results saved to: {output_path}")
    print("=" * 70)

if __name__ == "__main__":
    run_robustness_experiment()
