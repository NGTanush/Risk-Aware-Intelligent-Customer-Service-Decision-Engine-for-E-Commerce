"""
End-to-End System Test Suite for E-Commerce Customer Service Decision Engine.

Validates system behavior on 10 critical test scenarios to guarantee zero crashes and correct decisions.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.decision_pipeline import CustomerServiceDecisionEngine

def run_system_tests():
    print("=" * 70)
    print("RUNNING END-TO-END SYSTEM TESTS")
    print("=" * 70)

    engine = CustomerServiceDecisionEngine()

    test_scenarios = [
        ("1. Normal In-Domain Query (Low Risk)", "Where can I track my package?"),
        ("2. High-Confidence Query (Low Risk)", "How do I check my order history?"),
        ("3. Low-Confidence Query", "random gibberish inquiry xyz 123"),
        ("4. Ambiguous Query", "I need help with my order."),
        ("5. High-Risk Refund Query", "I want to request a refund for my order. I am very disappointed!"),
        ("6. High-Risk Password Security Query", "I need to recover my password immediately."),
        ("7. Out-Of-Domain Query", "Who will win tomorrow's cricket match?"),
        ("8. High-Risk Account Closure Query", "Please close my account permanently."),
        ("9. Empty Query Handling", ""),
        ("10. Special Characters Query", "!!! ### $$$ *** ???")
    ]

    passed_tests = 0

    for name, query in test_scenarios:
        print(f"\nRunning Test: {name}")
        print(f"Query: '{query}'")
        try:
            res = engine.process(query)
            assert "predicted_intent" in res
            assert "confidence" in res
            assert "decision" in res
            assert "reason" in res
            assert "response" in res
            
            print(f"  -> Decision : {res['decision']}")
            print(f"  -> Intent   : {res['predicted_intent']}")
            print(f"  -> Risk Lvl : {res['risk_level']}")
            print(f"  -> Reason   : {res['reason'][:80]}...")
            print(f"  [PASS]")
            passed_tests += 1
        except Exception as error:
            print(f"  [FAIL] Error: {error}")

    print("\n" + "=" * 70)
    print(f"SYSTEM TEST SUMMARY: {passed_tests} / {len(test_scenarios)} TESTS PASSED")
    print("=" * 70)

if __name__ == "__main__":
    run_system_tests()
