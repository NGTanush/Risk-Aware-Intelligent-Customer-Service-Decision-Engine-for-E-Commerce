"""
FastAPI Server Test Suite.
"""

from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    print("Root API test passed!")

def test_stats():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["intents_count"] == 46
    print("Stats API test passed:", data)

def test_intents():
    response = client.get("/intents")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 46
    print("Intents API test passed! Total intents:", data["total"])

def test_predict_automate():
    payload = {"query": "Where is my package?", "threshold": 0.85}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_intent"] in ["track_order", "track_delivery"]
    assert data["decision"] == "AUTOMATE"
    assert "response" in data
    print("Predict AUTOMATE test passed!")

def test_predict_human_escalation_refund():
    payload = {"query": "I want to request a refund for my order. I am very disappointed!", "threshold": 0.85}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "HUMAN_ESCALATION"
    assert data["risk_level"] == "HIGH"
    print("Predict HUMAN ESCALATION (Refund) test passed!")

def test_predict_human_escalation_password():
    payload = {"query": "I need to recover my password immediately.", "threshold": 0.85}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_intent"] == "recover_password"
    assert data["decision"] == "HUMAN_ESCALATION"
    assert data["risk_level"] == "HIGH"
    print("Predict HUMAN ESCALATION (Password Security) test passed!")

def test_predict_ood():
    payload = {"query": "What is the weather like in Paris today?", "threshold": 0.85}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["domain_status"] == "OUT_OF_DOMAIN"
    assert data["decision"] == "HUMAN_ESCALATION"
    print("Predict OOD test passed!")

if __name__ == "__main__":
    test_home()
    test_stats()
    test_intents()
    test_predict_automate()
    test_predict_human_escalation_refund()
    test_predict_human_escalation_password()
    test_predict_ood()
    print("\nALL SERVER TESTS PASSED SUCCESSFULLY!")
