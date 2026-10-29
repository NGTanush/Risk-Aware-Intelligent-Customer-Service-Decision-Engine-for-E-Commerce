"""
Test Suite for Audited ResponseGenerator Module.

Covers:
  1. Configurable OpenAI model (env var & init parameter)
  2. LLM failure fallback to nearest-neighbor retrieval
  3. Query-to-query nearest-neighbor retrieval within intent
  4. Safe fallback for low-similarity retrieval results
  5. Context-aware clarification templates
  6. Human escalation dialogue
  7. Secondary defense-in-depth high-risk automation prevention
  8. Unverified transactional claim guardrails
  9. Graceful dataset path failure handling
  10. Secure API error logging / secret redaction
"""

import os
import sys
import pytest
import joblib

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.response_generator import ResponseGenerator, sanitize_error_message, CLARIFICATION_TEMPLATES


def test_openai_model_config():
    """Verify OpenAI model is configurable via init argument and environment variable."""
    # Test default
    rg_default = ResponseGenerator(data_path="data/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv")
    assert rg_default.model_name in ["gpt-4o-mini", "gpt-3.5-turbo"]

    # Test explicit argument
    rg_custom = ResponseGenerator(model_name="gpt-4o")
    assert rg_custom.model_name == "gpt-4o"

    # Test environment variable override
    os.environ["OPENAI_MODEL"] = "gpt-4-turbo"
    rg_env = ResponseGenerator()
    assert rg_env.model_name == "gpt-4-turbo"
    del os.environ["OPENAI_MODEL"]
    print("[PASS] test_openai_model_config")


def test_llm_failure_fallback():
    """Verify that when LLM fails or is unconfigured, fallback retrieval is seamlessly engaged."""
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    rg = ResponseGenerator(vectorizer=vectorizer)
    rg.client = None  # Ensure client is unconfigured

    res = rg.get_response(
        query="Where can I track my package?",
        intent="track_delivery",
        confidence=0.98,
        risk_level="LOW",
        sentiment="NEUTRAL",
        decision="AUTOMATE"
    )

    assert "response" in res
    assert len(res["response"]) > 0
    assert res["generation_method"] == "Similarity-Based Retrieval (Dataset Fallback)"
    print("[PASS] test_llm_failure_fallback")


def test_fallback_retrieval_query_to_query():
    """Verify fallback retrieval matches customer query against training instructions within the predicted intent."""
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    rg = ResponseGenerator(vectorizer=vectorizer)

    # In track_delivery, test retrieval with clear shipping inquiry
    res = rg.generate_fallback_response(
        query="I need to track the delivery of my shipment",
        intent="track_delivery"
    )

    assert isinstance(res, str)
    assert len(res) > 20
    # Placeholders should be cleaned
    assert "{{Order Number}}" not in res
    assert "{{Email Address}}" not in res
    print("[PASS] test_fallback_retrieval_query_to_query")


def test_low_similarity_fallback():
    """Verify that low similarity retrieval results trigger a safe intent template rather than wrong candidate."""
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    # Set a high threshold to guarantee low similarity triggers fallback
    rg = ResponseGenerator(vectorizer=vectorizer, min_retrieval_similarity=0.95)

    res = rg.generate_fallback_response(
        query="something totally arbitrary and unrelated xyz 999",
        intent="store_opening_hours"
    )

    assert "Store Opening Hours" in res
    assert "Order ID or additional details" in res
    print("[PASS] test_low_similarity_fallback")


def test_contextual_clarification_templates():
    """Verify that CLARIFY decision produces context-specific clarification dialogues."""
    rg = ResponseGenerator()

    # Tracking intent should ask for Order ID / tracking number
    res_track = rg.get_response(
        query="check package",
        intent="track_delivery",
        confidence=0.65,
        risk_level="LOW",
        sentiment="NEUTRAL",
        decision="CLARIFY"
    )
    assert res_track["generation_method"] == "Clarification Template"
    assert "tracking number" in res_track["response"].lower() or "order id" in res_track["response"].lower()

    # Store hours intent should ask for location
    res_hours = rg.get_response(
        query="what time open",
        intent="store_opening_hours",
        confidence=0.60,
        risk_level="LOW",
        sentiment="NEUTRAL",
        decision="CLARIFY"
    )
    assert "store location" in res_hours["response"].lower() or "city" in res_hours["response"].lower()
    print("[PASS] test_contextual_clarification_templates")


def test_human_escalation_dialogue():
    """Verify HUMAN_ESCALATION decision produces clear and polite escalation dialogue."""
    rg = ResponseGenerator()

    res = rg.get_response(
        query="I am having an issue",
        intent="delivery_issue",
        confidence=0.40,
        risk_level="MEDIUM",
        sentiment="NEUTRAL",
        decision="HUMAN_ESCALATION"
    )

    assert res["generation_method"] == "Human Escalation Dialogue"
    assert "specialists" in res["response"].lower() or "representative" in res["response"].lower()
    print("[PASS] test_human_escalation_dialogue")


def test_high_risk_automation_prevention():
    """
    CRITICAL DEFENSE-IN-DEPTH TEST:
    Verify that an erroneous AUTOMATE decision CANNOT automate a HIGH-risk intent or risk level.
    """
    rg = ResponseGenerator()

    # Case 1: High-risk intent (request_refund) erroneously called with decision='AUTOMATE'
    res_refund = rg.get_response(
        query="I want a refund right now",
        intent="request_refund",
        confidence=0.99,
        risk_level="LOW", # simulated upstream error
        sentiment="NEUTRAL",
        decision="AUTOMATE"
    )
    assert res_refund["generation_method"] == "Safety Policy Override (High-Risk Escalation Enforcement)"
    assert "security policy" in res_refund["response"].lower() or "sensitive" in res_refund["response"].lower()

    # Case 2: High-risk intent (recover_password) erroneously called with decision='AUTOMATE'
    res_pw = rg.get_response(
        query="reset my password",
        intent="recover_password",
        confidence=0.95,
        risk_level="LOW",
        sentiment="NEUTRAL",
        decision="AUTOMATE"
    )
    assert res_pw["generation_method"] == "Safety Policy Override (High-Risk Escalation Enforcement)"

    # Case 3: Any intent marked with risk_level='HIGH'
    res_high = rg.get_response(
        query="some query",
        intent="availability",
        confidence=0.90,
        risk_level="HIGH",
        sentiment="NEGATIVE",
        decision="AUTOMATE"
    )
    assert res_high["generation_method"] == "Safety Policy Override (High-Risk Escalation Enforcement)"
    print("[PASS] test_high_risk_automation_prevention")


def test_prevent_unverified_transactional_claims():
    """Verify that automated responses claiming transactional actions receive disclaimer."""
    rg = ResponseGenerator()

    sample_claim = "Hello! I have processed your refund of $50 back to your card."
    guarded = rg._guard_unverified_claims(sample_claim)
    assert "no order cancellations, refunds, or account modifications have been finalized" in guarded.lower()

    safe_response = "Here are the instructions to request a return through your account."
    guarded_safe = rg._guard_unverified_claims(safe_response)
    assert guarded_safe == safe_response
    print("[PASS] test_prevent_unverified_transactional_claims")


def test_dataset_path_failure_graceful_handling():
    """Verify ResponseGenerator handles missing or corrupt dataset paths without crashing."""
    rg_missing = ResponseGenerator(data_path="data/non_existent_dataset_file.csv")
    assert rg_missing.is_dataset_loaded is False

    # Should still generate safe fallback responses
    res = rg_missing.generate_fallback_response("Where is my package?", "track_delivery")
    assert "Track Delivery" in res
    assert isinstance(res, str)
    print("[PASS] test_dataset_path_failure_graceful_handling")


def test_error_logging_redaction():
    """Verify error sanitization function redacts OpenAI API keys and Bearer tokens."""
    dirty_error = "AuthenticationError: Invalid API key provided: sk-proj-1234567890abcdef1234567890. Use Bearer secret_token_xyz"
    clean = sanitize_error_message(dirty_error)

    assert "sk-proj-1234567890abcdef1234567890" not in clean
    assert "[REDACTED_API_KEY]" in clean
    assert "Bearer [REDACTED_TOKEN]" in clean
    print("[PASS] test_error_logging_redaction")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING RESPONSE GENERATOR TEST SUITE")
    print("=" * 70)
    test_openai_model_config()
    test_llm_failure_fallback()
    test_fallback_retrieval_query_to_query()
    test_low_similarity_fallback()
    test_contextual_clarification_templates()
    test_human_escalation_dialogue()
    test_high_risk_automation_prevention()
    test_prevent_unverified_transactional_claims()
    test_dataset_path_failure_graceful_handling()
    test_error_logging_redaction()
    print("\nALL 10 RESPONSE GENERATOR TESTS PASSED SUCCESSFULLY!")
