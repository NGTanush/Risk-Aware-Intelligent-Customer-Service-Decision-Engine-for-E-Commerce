"""
Experiment H: Comprehensive Robustness Evaluation

Evaluates system stability and failure modes across 60 realistic, challenging test scenarios:
  1. Standard In-Domain Queries (Baseline)
  2. Syntactic & Lexical Paraphrases
  3. Typographical & OCR / Keyboard Noise
  4. Ultra-Short Queries (1-3 words)
  5. Semantically Ambiguous Queries
  6. Aggressive / Negative Customer Emotion
  7. High-Risk Financial & Security Queries
  8. Out-Of-Domain Adversarial Perturbations

Separates intent-classification failures from decision-policy failures.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from research.experiments.common import (
    load_vectorizer_and_models,
    save_csv_dual,
    RESEARCH_RESULTS_DIR,
    EVAL_DIR
)
from src.intent_risk_map import get_intent_risk_level
from src.sentiment_analyzer import SentimentAnalyzer
from src.ood_detector import OODDetector
from src.risk_engine import RiskEngine
from src.decision_engine import DecisionEngine


def run_experiment_h():
    print("=" * 75)
    print("EXPERIMENT H: ROBUSTNESS EVALUATION")
    print("=" * 75)

    vectorizer, models = load_vectorizer_and_models()
    calibrated_model = models["Calibrated Linear SVM"]

    sentiment_analyzer = SentimentAnalyzer()
    ood_detector = OODDetector(vectorizer=vectorizer, reference_matrix=None)
    risk_engine = RiskEngine()
    decision_engine = DecisionEngine(confidence_threshold=0.85)

    test_cases = [
        # --- 1. Standard In-Domain (Expected: Intent Match, AUTOMATE) ---
        {"category": "Standard In-Domain", "query": "Where can I track my package?", "expected_intent": "track_delivery", "expected_decision": "AUTOMATE"},
        {"category": "Standard In-Domain", "query": "How do I check my order history?", "expected_intent": "order_history", "expected_decision": "AUTOMATE"},
        {"category": "Standard In-Domain", "query": "What are your physical store opening hours?", "expected_intent": "store_opening_hours", "expected_decision": "AUTOMATE"},
        {"category": "Standard In-Domain", "query": "Where is your nearest store located?", "expected_intent": "store_location", "expected_decision": "AUTOMATE"},
        {"category": "Standard In-Domain", "query": "How much does express shipping cost?", "expected_intent": "shipping_costs", "expected_decision": "AUTOMATE"},
        {"category": "Standard In-Domain", "query": "Is this leather jacket available in size medium?", "expected_intent": "availability", "expected_decision": "AUTOMATE"},
        {"category": "Standard In-Domain", "query": "Can you give me detailed product information for this laptop?", "expected_intent": "product_information", "expected_decision": "AUTOMATE"},
        {"category": "Standard In-Domain", "query": "When does your seasonal winter sale begin?", "expected_intent": "sales_period", "expected_decision": "AUTOMATE"},

        # --- 2. Paraphrases (Expected: Intent Match, AUTOMATE or CLARIFY) ---
        {"category": "Paraphrase", "query": "Could you please inform me where my parcel currently is right now?", "expected_intent": "track_delivery", "expected_decision": "CLARIFY"},
        {"category": "Paraphrase", "query": "I would like to review the previous items I bought from your shop.", "expected_intent": "order_history", "expected_decision": "AUTOMATE"},
        {"category": "Paraphrase", "query": "At what time do your doors open and shut on Sundays?", "expected_intent": "store_opening_hours", "expected_decision": "AUTOMATE"},
        {"category": "Paraphrase", "query": "What fee do I have to pay to get my package delivered overnight?", "expected_intent": "shipping_costs", "expected_decision": "AUTOMATE"},
        {"category": "Paraphrase", "query": "Are you guys still carrying those blue denim jeans in stock?", "expected_intent": "availability", "expected_decision": "AUTOMATE"},
        {"category": "Paraphrase", "query": "Kindly share technical specifications of this phone model.", "expected_intent": "product_information", "expected_decision": "AUTOMATE"},
        {"category": "Paraphrase", "query": "Do you have any ongoing discounts or holiday promotions at the moment?", "expected_intent": "sales_period", "expected_decision": "AUTOMATE"},

        # --- 3. Typographical & Noisy Queries (Expected: CLARIFY or HUMAN_ESCALATION) ---
        {"category": "Typo / Noise", "query": "wheres my ordr pkg track status?", "expected_intent": "track_delivery", "expected_decision": "CLARIFY"},
        {"category": "Typo / Noise", "query": "hw to chck prev past ordrs?", "expected_intent": "order_history", "expected_decision": "CLARIFY"},
        {"category": "Typo / Noise", "query": "shopp timngs and opnening hrs plzz", "expected_intent": "store_opening_hours", "expected_decision": "CLARIFY"},
        {"category": "Typo / Noise", "query": "delivry shippng chages how mch?", "expected_intent": "shipping_costs", "expected_decision": "CLARIFY"},
        {"category": "Typo / Noise", "query": "is it stil in stck availble?", "expected_intent": "availability", "expected_decision": "CLARIFY"},
        {"category": "Typo / Noise", "query": "wnt info bt prdct spcs", "expected_intent": "product_information", "expected_decision": "CLARIFY"},
        {"category": "Typo / Noise", "query": "hlep me wth ordr trackinggg", "expected_intent": "track_delivery", "expected_decision": "CLARIFY"},
        {"category": "Typo / Noise", "query": "asdfghjkl qwerty 12345", "expected_intent": "unknown", "expected_decision": "HUMAN_ESCALATION"},

        # --- 4. Short Queries (Expected: CLARIFY or HUMAN_ESCALATION) ---
        {"category": "Short Query", "query": "Track order", "expected_intent": "track_order", "expected_decision": "CLARIFY"},
        {"category": "Short Query", "query": "Refund", "expected_intent": "request_refund", "expected_decision": "HUMAN_ESCALATION"},
        {"category": "Short Query", "query": "Store hours", "expected_intent": "store_opening_hours", "expected_decision": "CLARIFY"},
        {"category": "Short Query", "query": "Password reset", "expected_intent": "recover_password", "expected_decision": "HUMAN_ESCALATION"},
        {"category": "Short Query", "query": "Cancel order", "expected_intent": "cancel_order", "expected_decision": "HUMAN_ESCALATION"},
        {"category": "Short Query", "query": "Shipping fee", "expected_intent": "shipping_costs", "expected_decision": "CLARIFY"},
        {"category": "Short Query", "query": "Help", "expected_intent": "customer_service", "expected_decision": "CLARIFY"},

        # --- 5. Ambiguous Queries (Expected: CLARIFY) ---
        {"category": "Ambiguous", "query": "I need help with my order.", "expected_intent": "delivery_issue", "expected_decision": "CLARIFY"},
        {"category": "Ambiguous", "query": "Something went wrong with what I purchased.", "expected_intent": "product_issue", "expected_decision": "CLARIFY"},
        {"category": "Ambiguous", "query": "Can someone assist me regarding an item?", "expected_intent": "customer_service", "expected_decision": "CLARIFY"},
        {"category": "Ambiguous", "query": "I am having trouble on your platform.", "expected_intent": "technical_issue", "expected_decision": "CLARIFY"},
        {"category": "Ambiguous", "query": "What can you do about my situation?", "expected_intent": "customer_service", "expected_decision": "CLARIFY"},
        {"category": "Ambiguous", "query": "Check status please.", "expected_intent": "track_delivery", "expected_decision": "CLARIFY"},
        {"category": "Ambiguous", "query": "I want to speak with somebody right away.", "expected_intent": "human_agent", "expected_decision": "CLARIFY"},

        # --- 6. Negative / Angry Sentiment (Expected: Sensitive handling, Escalation if high risk) ---
        {"category": "Angry / Negative", "query": "WHERE IS MY PACKAGE?! I HAVE BEEN WAITING FOREVER AND THIS IS RIDICULOUS!", "expected_intent": "track_delivery", "expected_decision": "AUTOMATE"},
        {"category": "Angry / Negative", "query": "Your customer support is awful! Why are your store hours not listed accurately?!", "expected_intent": "store_opening_hours", "expected_decision": "AUTOMATE"},
        {"category": "Angry / Negative", "query": "I am furious that you charged me double shipping fees! Explain this right now!", "expected_intent": "shipping_costs", "expected_decision": "HUMAN_ESCALATION"},
        {"category": "Angry / Negative", "query": "The product I received is completely damaged and destroyed! I hate this company!", "expected_intent": "damaged_delivery", "expected_decision": "HUMAN_ESCALATION"},
        {"category": "Angry / Negative", "query": "You guys are scammers, give me my money back immediately!", "expected_intent": "request_refund", "expected_decision": "HUMAN_ESCALATION"},
        {"category": "Angry / Negative", "query": "I have sent three emails and nobody responded, this is absolute garbage service!", "expected_intent": "customer_service", "expected_decision": "CLARIFY"},
        {"category": "Angry / Negative", "query": "Your app keeps crashing every time I try to browse items, completely pathetic!", "expected_intent": "use_app", "expected_decision": "CLARIFY"},

        # --- 7. High-Risk Queries (Expected: Mandatory HUMAN_ESCALATION) ---
        {"category": "High Risk", "query": "I want to request a full refund for my order. I am very disappointed!", "expected_intent": "request_refund", "expected_decision": "HUMAN_ESCALATION"},
        {"category": "High Risk", "query": "I need to recover my account password immediately because I forgot it.", "expected_intent": "recover_password", "expected_decision": "HUMAN_ESCALATION"},
        {"category": "High Risk", "query": "Please close my account permanently and delete my data.", "expected_intent": "close_account", "expected_decision": "HUMAN_ESCALATION"},
        {"category": "High Risk", "query": "There is an unauthorized charge of $150 on my credit card from your site.", "expected_intent": "payment_issue", "expected_decision": "HUMAN_ESCALATION"},
        {"category": "High Risk", "query": "Please cancel my order before it gets dispatched from the warehouse.", "expected_intent": "cancel_order", "expected_decision": "HUMAN_ESCALATION"},
        {"category": "High Risk", "query": "The box arrived but the expensive watch inside was completely missing!", "expected_intent": "missing_item", "expected_decision": "HUMAN_ESCALATION"},
        {"category": "High Risk", "query": "Under privacy regulations, I request rectification of my personal details.", "expected_intent": "request_right_to_rectification", "expected_decision": "HUMAN_ESCALATION"},
        {"category": "High Risk", "query": "You delivered a completely wrong item, this is a toaster instead of a blender.", "expected_intent": "wrong_item", "expected_decision": "HUMAN_ESCALATION"},

        # --- 8. Out-Of-Domain (Expected: HUMAN_ESCALATION) ---
        {"category": "Out-Of-Domain", "query": "Who will win tomorrow's cricket match?", "expected_intent": "OUT_OF_DOMAIN", "expected_decision": "HUMAN_ESCALATION"},
        {"category": "Out-Of-Domain", "query": "What is the weather forecast for London this weekend?", "expected_intent": "OUT_OF_DOMAIN", "expected_decision": "HUMAN_ESCALATION"},
        {"category": "Out-Of-Domain", "query": "Write a Python script to sort a list of numbers.", "expected_intent": "OUT_OF_DOMAIN", "expected_decision": "HUMAN_ESCALATION"},
        {"category": "Out-Of-Domain", "query": "Can you explain quantum computing in simple terms?", "expected_intent": "OUT_OF_DOMAIN", "expected_decision": "HUMAN_ESCALATION"},
        {"category": "Out-Of-Domain", "query": "What is the capital city of Australia?", "expected_intent": "OUT_OF_DOMAIN", "expected_decision": "HUMAN_ESCALATION"},
        {"category": "Out-Of-Domain", "query": "How do you bake a delicious chocolate fudge cake?", "expected_intent": "OUT_OF_DOMAIN", "expected_decision": "HUMAN_ESCALATION"},
        {"category": "Out-Of-Domain", "query": "Tell me a hilarious bedtime joke for kids.", "expected_intent": "OUT_OF_DOMAIN", "expected_decision": "HUMAN_ESCALATION"},
        {"category": "Out-Of-Domain", "query": "What are the latest stock market trends today?", "expected_intent": "OUT_OF_DOMAIN", "expected_decision": "HUMAN_ESCALATION"}
    ]

    print(f"Total Robustness Scenarios Tested: {len(test_cases)}")

    results = []
    policy_matches = 0
    intent_matches = 0

    for item in test_cases:
        query = item["query"]
        category = item["category"]
        exp_intent = item["expected_intent"]
        exp_decision = item["expected_decision"]

        # Inference
        q_vec = vectorizer.transform([query])
        pred_intent = calibrated_model.predict(q_vec)[0]
        probs = calibrated_model.predict_proba(q_vec)[0]
        conf = float(np.max(probs))

        sent_data = sentiment_analyzer.analyze(query)
        ood_data = ood_detector.detect(query, conf, query_vector=q_vec)
        risk_data = risk_engine.evaluate_risk(pred_intent, conf, sent_data, ood_data)
        dec_data = decision_engine.make_decision(query, pred_intent, conf, sent_data, ood_data, risk_data)

        actual_decision = dec_data["decision"]

        # Intent match check (only for in-domain where ground truth is known)
        is_intent_match = (pred_intent == exp_intent) if exp_intent not in ["unknown", "OUT_OF_DOMAIN"] else (exp_intent == "OUT_OF_DOMAIN" and ood_data["is_ood"])
        if is_intent_match:
            intent_matches += 1

        # Decision policy match check
        is_policy_match = (actual_decision == exp_decision)
        if is_policy_match:
            policy_matches += 1

        # Error Type Categorization
        if not is_intent_match and is_policy_match:
            failure_type = "Intent Misclassification (Safely Handled by Policy)"
        elif not is_intent_match and not is_policy_match:
            failure_type = "Compound Failure (Intent & Policy Mismatch)"
        elif is_intent_match and not is_policy_match:
            failure_type = "Policy Discrepancy (Intent Correct, Divergent Decision)"
        else:
            failure_type = "None (Fully Compliant)"

        results.append({
            "Scenario Category": category,
            "Query": query,
            "Expected Intent": exp_intent,
            "Predicted Intent": pred_intent,
            "Calibrated Confidence (%)": round(conf * 100, 2),
            "Sentiment": sent_data["sentiment"],
            "Domain Status": ood_data["domain_status"],
            "Evaluated Risk Level": risk_data["risk_level"],
            "Evaluated Risk Score": risk_data["risk_score"],
            "Expected Decision": exp_decision,
            "Actual Decision": actual_decision,
            "Policy Compliant": is_policy_match,
            "Intent Correct": is_intent_match,
            "Failure Type": failure_type,
            "Decision Reason": dec_data["reason"]
        })

    rob_df = pd.DataFrame(results)

    total_n = len(rob_df)
    policy_comp_rate = (policy_matches / total_n) * 100.0
    intent_acc_rate = (intent_matches / total_n) * 100.0

    print("\n" + "=" * 75)
    print("ROBUSTNESS EVALUATION SUMMARY")
    print("=" * 75)
    print(f"Total Scenarios Evaluated       : {total_n}")
    print(f"Policy Compliance Rate          : {policy_comp_rate:.2f}% ({policy_matches}/{total_n})")
    print(f"Intent Classification Accuracy  : {intent_acc_rate:.2f}% ({intent_matches}/{total_n})")

    # Category Breakdown
    cat_summary = []
    for cat, grp in rob_df.groupby("Scenario Category"):
        cat_summary.append({
            "Category": cat,
            "Total Scenarios": len(grp),
            "Automated": (grp["Actual Decision"] == "AUTOMATE").sum(),
            "Clarified": (grp["Actual Decision"] == "CLARIFY").sum(),
            "Escalated": (grp["Actual Decision"] == "HUMAN_ESCALATION").sum(),
            "Policy Compliance Rate (%)": round((grp["Policy Compliant"].sum() / len(grp)) * 100, 2),
            "Mean Confidence (%)": round(grp["Calibrated Confidence (%)"].mean(), 2)
        })
    cat_df = pd.DataFrame(cat_summary)
    print("\nPerformance by Robustness Category:")
    print(cat_df.to_string(index=False))

    save_csv_dual(rob_df, "research_robustness_results.csv")
    save_csv_dual(cat_df, "research_robustness_by_category.csv")


if __name__ == "__main__":
    run_experiment_h()
