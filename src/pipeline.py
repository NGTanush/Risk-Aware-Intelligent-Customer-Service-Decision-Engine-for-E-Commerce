import os
import re
import random
import joblib
import pandas as pd
import numpy as np

MODELS_DIR = "models"
DATA_PATH = "data/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv"

class CustomerServicePipeline:
    def __init__(self, models_dir=MODELS_DIR, data_path=DATA_PATH):
        self.models_dir = models_dir
        self.data_path = data_path
        self.vectorizer = None
        self.svm_model = None
        self.calibrated_model = None
        self.response_database = {}
        self.intents_list = []
        self._load_artifacts()

    def _load_artifacts(self):
        # Load models
        vec_path = os.path.join(self.models_dir, "tfidf_vectorizer.pkl")
        svm_path = os.path.join(self.models_dir, "linear_svm.pkl")
        calib_path = os.path.join(self.models_dir, "calibrated_linear_svm.pkl")
        best_path = os.path.join(self.models_dir, "best_model.pkl")

        if os.path.exists(vec_path):
            self.vectorizer = joblib.load(vec_path)
        else:
            raise FileNotFoundError(f"TF-IDF vectorizer not found at {vec_path}")

        if os.path.exists(svm_path):
            self.svm_model = joblib.load(svm_path)
        elif os.path.exists(best_path):
            self.svm_model = joblib.load(best_path)

        if os.path.exists(calib_path):
            self.calibrated_model = joblib.load(calib_path)

        # Load response dataset
        if os.path.exists(self.data_path):
            df = pd.read_csv(self.data_path)
            df = df.dropna(subset=["intent", "response"])
            for intent, group in df.groupby("intent"):
                self.response_database[intent] = group["response"].tolist()
            self.intents_list = sorted(list(self.response_database.keys()))
        else:
            self.intents_list = getattr(self.svm_model, "classes_", []).tolist()

    def nlp_preprocess(self, query: str) -> dict:
        """Step 1: NLP Preprocessing"""
        cleaned_text = re.sub(r'[^\w\s#]', ' ', query.lower())
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        tokens = [t for t in cleaned_text.split() if len(t) > 0]
        
        return {
            "original_query": query,
            "cleaned_text": cleaned_text,
            "tokens": tokens,
            "word_count": len(tokens),
            "char_count": len(query)
        }

    def tfidf_vectorize(self, cleaned_text: str):
        """Step 2: TF-IDF Vectorization"""
        vector = self.vectorizer.transform([cleaned_text])
        
        # Extract top non-zero features for visualization
        feature_names = self.vectorizer.get_feature_names_out()
        nonzero_indices = vector.nonzero()[1]
        feature_weights = []
        for idx in nonzero_indices:
            feature_weights.append({
                "feature": feature_names[idx],
                "weight": round(float(vector[0, idx]), 4)
            })
        feature_weights = sorted(feature_weights, key=lambda x: x["weight"], reverse=True)[:10]

        return vector, {
            "vector_shape": [1, vector.shape[1]],
            "nonzero_features_count": len(nonzero_indices),
            "top_features": feature_weights
        }

    def svm_classify(self, vector):
        """Step 3 & 4: Linear SVM Classification & Intent Prediction"""
        predicted_intent = self.svm_model.predict(vector)[0]
        
        # Get decision function scores (margins)
        if hasattr(self.svm_model, "decision_function"):
            decision_scores = self.svm_model.decision_function(vector)[0]
            classes = self.svm_model.classes_
            
            # Sort top 5 decision scores
            top_indices = np.argsort(decision_scores)[::-1][:5]
            top_classes = []
            for idx in top_indices:
                top_classes.append({
                    "intent": classes[idx],
                    "margin": round(float(decision_scores[idx]), 4)
                })
            max_score = float(np.max(decision_scores))
        else:
            decision_scores = []
            top_classes = [{"intent": predicted_intent, "margin": 1.0}]
            max_score = 1.0

        return predicted_intent, max_score, {
            "predicted_intent": predicted_intent,
            "max_decision_score": round(max_score, 4),
            "top_margins": top_classes
        }

    def calibrate_confidence(self, vector, predicted_intent: str, max_decision_score: float) -> tuple[float, dict]:
        """Step 5: Confidence Calibration"""
        if self.calibrated_model is not None and hasattr(self.calibrated_model, "predict_proba"):
            probs = self.calibrated_model.predict_proba(vector)[0]
            classes = list(self.calibrated_model.classes_)
            if predicted_intent in classes:
                intent_idx = classes.index(predicted_intent)
                calibrated_confidence = float(probs[intent_idx])
            else:
                calibrated_confidence = float(np.max(probs))
            method = "Platt Scaling (CalibratedClassifierCV)"
        else:
            # Fallback mathematical calibration using sigmoid curve on decision margin
            calibrated_confidence = 1.0 / (1.0 + np.exp(-max_decision_score * 0.75))
            method = "Sigmoidal Margin Calibration"

        # Ensure bounds [0.05, 0.99]
        calibrated_confidence = min(0.99, max(0.05, float(calibrated_confidence)))

        return calibrated_confidence, {
            "confidence_score": round(calibrated_confidence, 4),
            "confidence_percentage": f"{round(calibrated_confidence * 100, 2)}%",
            "calibration_method": method
        }

    def decision_rule(self, confidence: float, threshold: float = 0.70) -> tuple[bool, dict]:
        """Step 6: Decision Rule based on confidence threshold"""
        is_high_confidence = confidence >= threshold
        status = "HIGH_CONFIDENCE" if is_high_confidence else "LOW_CONFIDENCE"
        
        return is_high_confidence, {
            "status": status,
            "confidence": round(confidence, 4),
            "threshold": round(threshold, 4),
            "accepted": is_high_confidence
        }

    def extract_missing_info_requests(self, query: str, intent: str) -> str | None:
        """Detect missing information required for specific intent categories."""
        query_lower = query.lower()
        
        # Check if an order ID is present (e.g., #12345, order 9876, 5+ digits)
        has_order_id = bool(re.search(r'#?\b\d{4,10}\b', query_lower) or 'order #' in query_lower)
        # Check if email present
        has_email = bool(re.search(r'[\w\.-]+@[\w\.-]+', query_lower))

        order_intents = {
            "track_order": "To track your shipment, please provide your **Order ID** (e.g. #12345).",
            "track_delivery": "To check your delivery status, please provide your **Order ID**.",
            "cancel_order": "To assist with canceling your order, please share your **Order ID**.",
            "change_order": "Please specify your **Order ID** and the exact changes you'd like to make.",
            "refund_status": "To check your refund status, please reply with your **Order ID** or refund reference number.",
            "request_refund": "To process your refund request, please provide your **Order ID** and reason for the refund.",
            "request_invoice": "To send your invoice, please share your **Order ID** and your account email.",
            "damaged_delivery": "We're very sorry your delivery was damaged! Please reply with your **Order ID** and a brief description/photo of the item.",
            "missing_item": "We apologize for the missing item. Please reply with your **Order ID** and which item was missing.",
            "wrong_item": "We apologize for sending the wrong item! Please share your **Order ID** so we can arrange a correct replacement."
        }

        account_intents = {
            "change_account": "To update your account details, please provide your registered **account email**.",
            "close_account": "To proceed with closing your account, please confirm your registered **account email**.",
            "recover_password": "To reset your password, please provide your registered **email address**."
        }

        if intent in order_intents and not has_order_id:
            return order_intents[intent]
        
        if intent in account_intents and not has_email:
            return account_intents[intent]

        return None

    def generate_response(self, query: str, intent: str, confidence: float, is_high_confidence: bool) -> str:
        """
        Response Generation Layer.
        Strictly follows constraints:
        - Never mentions internal ML terms to customer.
        - High confidence: Uses predicted intent context, polite, concise, natural.
        - Low confidence: Recommends human customer service escalation without guessing intent.
        - Never invents fake order details, prices, or policies.
        """
        if not is_high_confidence:
            return (
                "I'm sorry, I want to make sure you get the right help with your request. "
                "Could you please provide a little more detail, or would you like me to connect you "
                "with a human customer service representative?"
            )

        # Check if specific required info (e.g. Order ID) is missing
        missing_info_prompt = self.extract_missing_info_requests(query, intent)
        
        # Get dataset templates for predicted intent
        possible_responses = self.response_database.get(intent, [])

        if possible_responses:
            raw_response = random.choice(possible_responses)
            # Clean up bitext dataset template tokens like {{Order Number}}, {{Email Address}}, {{Company Website URL}}, etc.
            cleaned_resp = re.sub(r'\{\{\s*Order Number\s*\}\}', 'Order ID', raw_response, flags=re.IGNORECASE)
            cleaned_resp = re.sub(r'\{\{\s*Email Address\s*\}\}', 'account email', cleaned_resp, flags=re.IGNORECASE)
            cleaned_resp = re.sub(r'\{\{\s*Customer Support Working Hours\s*\}\}', 'standard business hours', cleaned_resp, flags=re.IGNORECASE)
            cleaned_resp = re.sub(r'\{\{\s*Customer Support Phone Number\s*\}\}', 'our support helpline', cleaned_resp, flags=re.IGNORECASE)
            cleaned_resp = re.sub(r'\{\{\s*Company Website URL\s*\}\}', 'our website', cleaned_resp, flags=re.IGNORECASE)
            cleaned_resp = re.sub(r'\{\{\s*Order History\s*\}\}', 'Order History', cleaned_resp, flags=re.IGNORECASE)
            cleaned_resp = re.sub(r'\{\{\s*Track Orders\s*\}\}', 'Track Orders', cleaned_resp, flags=re.IGNORECASE)
            cleaned_resp = re.sub(r'\{\{\s*([^}]+)\s*\}\}', r'\1', cleaned_resp)

            if missing_info_prompt and ("order" in missing_info_prompt.lower() or "email" in missing_info_prompt.lower()):
                response = f"{cleaned_resp}\n\n{missing_info_prompt}"
            else:
                response = cleaned_resp
        else:
            # Fallback natural responses mapped to intent name
            formatted_intent_name = intent.replace("_", " ").title()
            if missing_info_prompt:
                response = f"I'd be happy to assist you with your request regarding {formatted_intent_name}.\n\n{missing_info_prompt}"
            else:
                response = f"I'd be glad to help you with {formatted_intent_name}. Please let me know how I can further assist you."

        return response

    def process_query(self, query: str, threshold: float = 0.70) -> dict:
        """Run complete end-to-end customer service AI pipeline."""
        # 1. Preprocess
        step1 = self.nlp_preprocess(query)
        
        # 2. Vectorize
        vector, step2 = self.tfidf_vectorize(step1["cleaned_text"])
        
        # 3 & 4. SVM Classify & Predict Intent
        predicted_intent, max_margin, step3 = self.svm_classify(vector)
        
        # 5. Confidence Calibration
        confidence, step5 = self.calibrate_confidence(vector, predicted_intent, max_margin)
        
        # 6. Decision Rule
        is_high_conf, step6 = self.decision_rule(confidence, threshold)
        
        # 7. Response Generation Layer
        customer_response = self.generate_response(query, predicted_intent, confidence, is_high_conf)
        
        return {
            "customer_query": query,
            "predicted_intent": predicted_intent,
            "confidence": round(confidence, 4),
            "confidence_percentage": f"{round(confidence * 100, 2)}%",
            "threshold": round(threshold, 4),
            "is_high_confidence": is_high_conf,
            "customer_response": customer_response,
            "pipeline_trace": {
                "step1_nlp": step1,
                "step2_tfidf": step2,
                "step3_linear_svm": step3,
                "step5_calibration": step5,
                "step6_decision": step6
            }
        }
