"""
Risk Engine Module for E-Commerce Customer Service Decision Engine.

Calculates a transparent, explainable risk score (0.0 to 1.0) and risk level (LOW, MEDIUM, HIGH)
by combining:
  1. Base Intent Risk (LOW, MEDIUM, HIGH)
  2. Customer Sentiment (Negative tone penalty)
  3. Model Confidence Uncertainty
  4. Out-Of-Domain (OOD) Signal

Research Rule:
  ML Confidence measures prediction certainty, NOT authorization to perform risky actions.
  Therefore, any intent with intent_risk_level == "HIGH" ALWAYS results in risk_level = "HIGH".
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.intent_risk_map import get_intent_risk_level, get_intent_risk_score

class RiskEngine:
    def __init__(self, w_intent=0.45, w_sentiment=0.25, w_confidence=0.15, w_ood=0.15):
        # Configurable scoring weights
        self.w_intent = w_intent
        self.w_sentiment = w_sentiment
        self.w_confidence = w_confidence
        self.w_ood = w_ood

    def evaluate_risk(self, intent: str, confidence: float, sentiment_data: dict, ood_data: dict) -> dict:
        """
        Evaluate composite risk for a customer query.

        Args:
            intent: Predicted customer intent label
            confidence: Calibrated model confidence (0.0 to 1.0)
            sentiment_data: Output dict from SentimentAnalyzer
            ood_data: Output dict from OODDetector

        Returns:
            dict containing risk_level ('LOW', 'MEDIUM', 'HIGH'),
            risk_score (float 0.0 to 1.0), and detailed component breakdown.
        """
        # 1. Base Intent Risk Score
        intent_risk_level = get_intent_risk_level(intent)
        base_intent_score = get_intent_risk_score(intent)

        # 2. Sentiment Penalty
        is_negative = sentiment_data.get("is_negative", False)
        sentiment_score = sentiment_data.get("sentiment_score", 0.50)
        
        if is_negative:
            sentiment_risk_component = min(1.0, sentiment_score * 0.90)
        else:
            sentiment_risk_component = 0.10

        # 3. Model Uncertainty Component
        confidence_risk_component = max(0.0, 1.0 - confidence)

        # 4. Out-Of-Domain Component
        ood_score = ood_data.get("ood_score", 0.10)
        is_ood = ood_data.get("is_ood", False)
        if is_ood:
            ood_risk_component = max(0.80, ood_score)
        else:
            ood_risk_component = ood_score

        # Composite Risk Calculation
        raw_risk = (
            (self.w_intent * base_intent_score) +
            (self.w_sentiment * sentiment_risk_component) +
            (self.w_confidence * confidence_risk_component) +
            (self.w_ood * ood_risk_component)
        )

        # Safety override 1: If intent is inherently HIGH risk, enforce minimum risk score 0.70
        if intent_risk_level == "HIGH":
            raw_risk = max(raw_risk, 0.70)

        # Safety override 2: If OOD is detected, ensure at least 0.70 risk score
        if is_ood:
            raw_risk = max(raw_risk, 0.70)

        risk_score = round(float(min(1.0, max(0.0, raw_risk))), 4)

        # Explicit Hard Risk Level Classification:
        # HIGH-risk intent ALWAYS enforces risk_level = "HIGH" regardless of raw score.
        if intent_risk_level == "HIGH" or risk_score >= 0.65 or is_ood:
            risk_level = "HIGH"
        elif risk_score >= 0.35:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "intent_risk_level": intent_risk_level,
            "base_intent_score": base_intent_score,
            "sentiment_risk_component": round(float(sentiment_risk_component), 4),
            "confidence_risk_component": round(float(confidence_risk_component), 4),
            "ood_risk_component": round(float(ood_risk_component), 4),
            "weights": {
                "w_intent": self.w_intent,
                "w_sentiment": self.w_sentiment,
                "w_confidence": self.w_confidence,
                "w_ood": self.w_ood
            }
        }
