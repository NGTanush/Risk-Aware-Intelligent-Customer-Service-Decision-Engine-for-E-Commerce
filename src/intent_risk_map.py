"""
Intent Risk Mapping Module for E-Commerce Customer Service Decision Engine.

Categorizes all 46 intent classes from the Bitext dataset into LOW, MEDIUM, or HIGH risk.
Documentation of risk levels:
- LOW: Informational, tracking, store info, catalog, general policies.
- MEDIUM: Order changes, product returns, account updates, technical issues.
- HIGH: Financial refunds, payment disputes, account closure, password recovery, damaged/missing items, legal requests.
"""

INTENT_RISK_MAP = {
    # ------------------------------------------------------------------
    # HIGH RISK INTENTS (Financial impact, security, dispute, account closure)
    # ------------------------------------------------------------------
    "request_refund": "HIGH",
    "refund_status": "HIGH",
    "payment_issue": "HIGH",
    "close_account": "HIGH",
    "cancel_order": "HIGH",
    "damaged_delivery": "HIGH",
    "missing_item": "HIGH",
    "wrong_item": "HIGH",
    "request_right_to_rectification": "HIGH",
    "recover_password": "HIGH",
    "change_account": "HIGH",

    # ------------------------------------------------------------------
    # MEDIUM RISK INTENTS (Order changes, returns, exchanges, technical bugs)
    # ------------------------------------------------------------------
    "change_order": "MEDIUM",
    "return_product": "MEDIUM",
    "return_product_in_store": "MEDIUM",
    "return_product_online": "MEDIUM",
    "exchange_product": "MEDIUM",
    "exchange_product_in_store": "MEDIUM",
    "pay": "MEDIUM",
    "delivery_issue": "MEDIUM",
    "open_account": "MEDIUM",
    "technical_issue": "MEDIUM",
    "human_agent": "MEDIUM",
    "request_invoice": "MEDIUM",

    # ------------------------------------------------------------------
    # LOW RISK INTENTS (Read-only, informational, tracking, store info)
    # ------------------------------------------------------------------
    "track_order": "LOW",
    "track_delivery": "LOW",
    "delivery_time": "LOW",
    "product_information": "LOW",
    "product_issue": "LOW",
    "availability": "LOW",
    "availability_in_store": "LOW",
    "availability_online": "LOW",
    "order_history": "LOW",
    "refund_policy": "LOW",
    "return_policy": "LOW",
    "sales_period": "LOW",
    "shipping_costs": "LOW",
    "store_location": "LOW",
    "store_opening_hours": "LOW",
    "add_product": "LOW",
    "remove_product": "LOW",
    "customer_service": "LOW",
    "payment_methods": "LOW",
    "submit_feedback": "LOW",
    "submit_product_feedback": "LOW",
    "submit_product_idea": "LOW",
    "use_app": "LOW"
}

# Numerical risk weight assigned to intent risk category
RISK_WEIGHTS = {
    "LOW": 0.10,
    "MEDIUM": 0.40,
    "HIGH": 0.80
}

def get_intent_risk_level(intent: str) -> str:
    """Return risk level string ('LOW', 'MEDIUM', 'HIGH') for a given intent."""
    return INTENT_RISK_MAP.get(intent, "MEDIUM")

def get_intent_risk_score(intent: str) -> float:
    """Return baseline numerical risk score (0.0 - 1.0) for a given intent."""
    level = get_intent_risk_level(intent)
    return RISK_WEIGHTS.get(level, 0.40)
