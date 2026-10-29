"""
FastAPI Backend Server for Risk-Aware Intelligent Customer Service Decision Engine.

Exposes REST API endpoints:
  - GET  /         : Health check & configuration
  - GET  /stats    : Model accuracy, baseline comparison & calibration metrics
  - GET  /intents  : List of supported 46 intent classes
  - POST /predict  : Core prediction & three-way decision engine endpoint
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from src.decision_pipeline import CustomerServiceDecisionEngine

app = FastAPI(
    title="Risk-Aware Customer Service AI Decision Engine API",
    description="Selective Automation Decision Engine using Intent Classification, Calibration, OOD Detection, Sentiment & Risk Assessment",
    version="2.0.0"
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize master decision engine instance
engine = CustomerServiceDecisionEngine()

class QueryRequest(BaseModel):
    query: str = Field(..., example="Where can I track my package?")
    threshold: Optional[float] = Field(default=0.85, ge=0.50, le=0.99, description="Configurable confidence threshold")

class QueryResponse(BaseModel):
    query: str
    predicted_intent: str
    confidence: float
    confidence_percentage: float
    confidence_threshold: float
    sentiment: str
    sentiment_score: float
    domain_status: str
    ood_score: float
    similarity_score: float
    risk_level: str
    risk_score: float
    decision: str
    badge: str
    reason: str
    response: str
    generation_method: str

@app.get("/")
def home():
    return {
        "system": "Risk-Aware Intelligent Customer Service Decision Engine",
        "status": "running",
        "default_confidence_threshold": engine.confidence_threshold,
        "supported_intents_count": 46,
        "calibration_method": "Platt Scaling (CalibratedClassifierCV w/ Sigmoid)"
    }

@app.get("/stats")
def get_stats():
    return {
        "model_name": "Calibrated Linear SVM (LinearSVC)",
        "baseline_accuracy": 0.9909,
        "macro_f1": 0.9904,
        "brier_score": 0.0072,
        "intents_count": 46,
        "categories_count": 13,
        "dataset_rows": 44884
    }

@app.get("/intents")
def list_intents():
    from src.intent_risk_map import INTENT_RISK_MAP
    return {
        "total": len(INTENT_RISK_MAP),
        "intents": INTENT_RISK_MAP
    }

@app.post("/predict", response_model=QueryResponse)
def predict(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    result = engine.process(query, custom_threshold=request.threshold)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)