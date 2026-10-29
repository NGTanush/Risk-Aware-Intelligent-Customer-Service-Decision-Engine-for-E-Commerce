"""
Three-Way Decision Engine Module for E-Commerce Customer Service.

Makes selective automation decisions:
  - AUTOMATE
  - CLARIFY
  - HUMAN_ESCALATION

Research Policy Rules:
  1. ML Confidence measures prediction certainty, NOT authorization to perform risky actions.
  2. Any query with intent_risk_level == "HIGH" or risk_level == "HIGH" is MANDATORY HUMAN_ESCALATION.
  3. Out-Of-Domain queries are MANDATORY HUMAN_ESCALATION.
  4. LOW / MEDIUM risk queries with confidence < 55% route to HUMAN_ESCALATION.
  5. LOW / MEDIUM risk queries with 55% <= confidence < threshold route to CLARIFY.
  6. LOW / MEDIUM risk queries with confidence >= threshold and IN_DOMAIN route to AUTOMATE.
"""

class DecisionEngine:
    def __init__(self, confidence_threshold=0.85, clarify_threshold=0.55):
        self.confidence_threshold = confidence_threshold
        self.clarify_threshold = clarify_threshold

    def make_decision(self, query: str, intent: str, confidence: float, sentiment_data: dict, ood_data: dict, risk_data: dict) -> dict:
        """
        Evaluate customer query metadata and determine three-way action.

        Returns:
            dict containing decision ('AUTOMATE', 'CLARIFY', 'HUMAN_ESCALATION'),
            badge, color_code, dynamically generated reason, and policy details.
        """
        domain_status = ood_data.get("domain_status", "IN_DOMAIN")
        is_ood = ood_data.get("is_ood", False)
        sentiment = sentiment_data.get("sentiment", "NEUTRAL")
        risk_level = risk_data.get("risk_level", "LOW")
        risk_score = risk_data.get("risk_score", 0.0)
        intent_risk_level = risk_data.get("intent_risk_level", "LOW")

        words = query.strip().split()
        is_very_short = len(words) <= 3
        intent_display = intent.replace("_", " ").title()

        # ------------------------------------------------------------------
        # HARD SAFETY RULE SET 1: HUMAN ESCALATION (Mandatory High Risk & OOD Safety)
        # ------------------------------------------------------------------
        if is_ood:
            decision = "HUMAN_ESCALATION"
            reason = (
                f"Out-of-Domain Query: Domain status is {domain_status} (similarity: {ood_data.get('similarity_score', 0):.1%}). "
                f"Predicted intent '{intent_display}' (confidence: {confidence:.1%}) is unsupported. Escalated to human support."
            )
        elif intent_risk_level == "HIGH" or risk_level == "HIGH":
            decision = "HUMAN_ESCALATION"
            reason = (
                f"High-Risk Policy Enforcement: Predicted intent '{intent_display}' is classified as HIGH intent risk "
                f"(composite risk score: {risk_score:.2f}, sentiment: {sentiment}). High-confidence ML predictions do not override "
                f"safety policies for financial, account security, or dispute intents. Escalated to human support."
            )
        elif confidence < self.clarify_threshold:
            decision = "HUMAN_ESCALATION"
            reason = (
                f"Low Confidence Uncertainty: Predicted intent '{intent_display}' has critically low calibrated confidence "
                f"({confidence:.1%}, below minimum {self.clarify_threshold:.0%} threshold). Escalated to prevent misinterpretation."
            )

        # ------------------------------------------------------------------
        # RULE SET 2: CLARIFICATION (Ambiguous / Moderate Confidence for Non-High Risk)
        # ------------------------------------------------------------------
        elif confidence < self.confidence_threshold:
            decision = "CLARIFY"
            reason = (
                f"Moderate Confidence Clarification: Predicted intent '{intent_display}' ({intent_risk_level} risk) has calibrated confidence "
                f"of {confidence:.1%}, which is below the automation threshold ({self.confidence_threshold:.0%}). Clarification requested."
            )
        elif is_very_short and confidence < 0.90:
            decision = "CLARIFY"
            reason = (
                f"Ambiguous Query Clarification: Query is very concise ({len(words)} words) with intent '{intent_display}' "
                f"({confidence:.1%} confidence). Clarification requested to confirm exact customer intent."
            )

        # ------------------------------------------------------------------
        # RULE SET 3: AUTOMATE (High Confidence + In-Domain + Non-HIGH Risk)
        # ------------------------------------------------------------------
        else:
            decision = "AUTOMATE"
            reason = (
                f"Automation Approved: High-confidence ({confidence:.1%}) in-domain '{intent_display}' request with "
                f"{risk_level}-risk status (risk score: {risk_score:.2f}) and {sentiment} sentiment. Safe automation criteria met."
            )

        # Format decision indicator badge/color
        if decision == "AUTOMATE":
            badge = "🟢 AUTOMATION APPROVED"
            color_code = "green"
        elif decision == "CLARIFY":
            badge = "🟡 CLARIFICATION REQUIRED"
            color_code = "orange"
        else:
            badge = "🔴 HUMAN ESCALATION"
            color_code = "red"

        return {
            "decision": decision,
            "badge": badge,
            "color_code": color_code,
            "reason": reason,
            "confidence_threshold": self.confidence_threshold,
            "clarify_threshold": self.clarify_threshold,
            "evaluated_risk_level": risk_level,
            "evaluated_domain_status": domain_status
        }
