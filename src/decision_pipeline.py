"""
Master Decision Pipeline for E-Commerce Customer Service Decision Engine.

Integrates:
  1. NLP Preprocessing & TF-IDF Vectorization
  2. Calibrated Intent Classification (Linear SVM + Platt Scaling)
  3. Customer Sentiment Analysis
  4. Out-Of-Domain (OOD) Detection
  5. Explainable Risk Engine
  6. Three-Way Decision Engine (AUTOMATE / CLARIFY / HUMAN_ESCALATION)
  7. Response Generation (OpenAI LLM + Cosine-Similarity Fallback)
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sentiment_analyzer import SentimentAnalyzer
from src.ood_detector import OODDetector
from src.risk_engine import RiskEngine
from src.decision_engine import DecisionEngine
from src.response_generator import ResponseGenerator

MODELS_DIR = "models"
DATA_PATH = "data/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv"

class CustomerServiceDecisionEngine:
    def __init__(self, models_dir=MODELS_DIR, data_path=DATA_PATH, confidence_threshold=0.85):
        self.models_dir = models_dir
        self.data_path = data_path
        self.confidence_threshold = confidence_threshold

        self.vectorizer = None
        self.model = None
        self.reference_matrix = None

        self._load_models_and_data()

        # Initialize sub-engines
        self.sentiment_analyzer = SentimentAnalyzer()
        self.ood_detector = OODDetector(
            vectorizer=self.vectorizer,
            reference_matrix=self.reference_matrix,
            confidence_threshold=self.confidence_threshold
        )
        self.risk_engine = RiskEngine()
        self.decision_engine = DecisionEngine(confidence_threshold=self.confidence_threshold)
        self.response_generator = ResponseGenerator(data_path=self.data_path, vectorizer=self.vectorizer)

    def _load_models_and_data(self):
        """Load TF-IDF vectorizer, calibrated model, and prepare reference matrix for similarity."""
        vec_path = os.path.join(self.models_dir, "tfidf_vectorizer.pkl")
        calib_path = os.path.join(self.models_dir, "calibrated_svm.pkl")
        svm_path = os.path.join(self.models_dir, "calibrated_linear_svm.pkl")
        best_path = os.path.join(self.models_dir, "best_model.pkl")

        if os.path.exists(vec_path):
            self.vectorizer = joblib.load(vec_path)
        else:
            raise FileNotFoundError(f"TF-IDF vectorizer not found at {vec_path}")

        if os.path.exists(calib_path):
            self.model = joblib.load(calib_path)
        elif os.path.exists(svm_path):
            self.model = joblib.load(svm_path)
        elif os.path.exists(best_path):
            self.model = joblib.load(best_path)
        else:
            raise FileNotFoundError("No trained model file found in models/")

        # Load reference training queries for TF-IDF cosine similarity calculations
        if os.path.exists(self.data_path):
            try:
                df = pd.read_csv(self.data_path).dropna(subset=["instruction", "intent"])
                df = df.drop_duplicates(subset=["instruction"])
                X_train, _, _, _ = train_test_split(
                    df["instruction"], df["intent"], test_size=0.20, random_state=42, stratify=df["intent"]
                )
                self.reference_matrix = self.vectorizer.transform(X_train)
            except Exception as e:
                print(f"[Pipeline] Reference matrix build warning: {e}")
                self.reference_matrix = None

    def process(self, query: str, custom_threshold: float = None) -> dict:
        """
        Run end-to-end Decision Engine pipeline on a customer query.

        Args:
            query: Input customer message
            custom_threshold: Optional override for confidence threshold

        Returns:
            dict containing full execution trace, prediction, risk, decision, and response.
        """
        threshold = custom_threshold if custom_threshold is not None else self.confidence_threshold
        query_str = query.strip() if query else ""

        if not query_str:
            return {
                "error": "Query cannot be empty",
                "query": "",
                "predicted_intent": "none",
                "confidence": 0.0,
                "sentiment": "NEUTRAL",
                "risk_level": "HIGH",
                "domain_status": "OUT_OF_DOMAIN",
                "decision": "HUMAN_ESCALATION",
                "reason": "Empty query provided.",
                "response": "Please enter a valid customer query."
            }

        # Step 1: TF-IDF Vectorization
        query_vector = self.vectorizer.transform([query_str])

        # Step 2: Intent Classification & Confidence Calibration
        predicted_intent = self.model.predict(query_vector)[0]

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(query_vector)[0]
            confidence = float(np.max(probs))
        else:
            confidence = 0.85

        # Step 3: Sentiment Analysis
        sentiment_data = self.sentiment_analyzer.analyze(query_str)

        # Step 4: OOD Detection
        ood_data = self.ood_detector.detect(query_str, confidence, query_vector=query_vector)

        # Step 5: Explainable Risk Engine
        risk_data = self.risk_engine.evaluate_risk(predicted_intent, confidence, sentiment_data, ood_data)

        # Step 6: Three-Way Decision Engine
        self.decision_engine.confidence_threshold = threshold
        decision_data = self.decision_engine.make_decision(
            query_str, predicted_intent, confidence, sentiment_data, ood_data, risk_data
        )

        # Step 7: Response Generation
        response_data = self.response_generator.get_response(
            query_str,
            predicted_intent,
            confidence,
            risk_data["risk_level"],
            sentiment_data["sentiment"],
            decision_data["decision"]
        )

        return {
            "query": query_str,
            "predicted_intent": predicted_intent,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 2),
            "confidence_threshold": threshold,
            "sentiment": sentiment_data["sentiment"],
            "sentiment_score": sentiment_data["sentiment_score"],
            "domain_status": ood_data["domain_status"],
            "ood_score": ood_data["ood_score"],
            "similarity_score": ood_data["similarity_score"],
            "risk_level": risk_data["risk_level"],
            "risk_score": risk_data["risk_score"],
            "decision": decision_data["decision"],
            "badge": decision_data["badge"],
            "reason": decision_data["reason"],
            "response": response_data["response"],
            "generation_method": response_data["generation_method"],
            "details": {
                "sentiment_details": sentiment_data,
                "ood_details": ood_data,
                "risk_details": risk_data,
                "decision_details": decision_data
            }
        }
