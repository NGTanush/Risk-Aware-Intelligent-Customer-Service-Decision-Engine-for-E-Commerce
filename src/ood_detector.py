"""
Out-Of-Domain (OOD) Detector Module for E-Commerce Customer Service Decision Engine.

Uses calibrated confidence, TF-IDF cosine similarity against reference training data,
and heuristic domain verification to detect queries outside the supported e-commerce domain.
"""

import re
import numpy as np

class OODDetector:
    def __init__(self, vectorizer=None, reference_matrix=None, similarity_threshold=0.15, confidence_threshold=0.75):
        self.vectorizer = vectorizer
        self.reference_matrix = reference_matrix
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold

        # Common non-e-commerce / general knowledge keywords for heuristic check
        self.unrelated_keywords = {
            "weather", "cricket", "football", "match", "movie", "movies", "joke", "python",
            "code", "program", "capital of", "quantum", "physics", "solar system", "spanish",
            "translate", "recipe", "cake", "cook", "stock market", "population", "weight loss",
            "europe", "travel", "exam", "study", "president", "prime minister", "song", "poem"
        }

        # Common e-commerce terms for domain relevance boost
        self.ecommerce_keywords = {
            "order", "package", "item", "product", "delivery", "shipping", "track", "tracking",
            "refund", "return", "exchange", "cancel", "account", "password", "invoice", "pay",
            "payment", "card", "store", "buy", "price", "discount", "cart", "basket", "address",
            "damaged", "missing", "wrong", "hours", "opening", "location", "help", "support"
        }

    def compute_max_similarity(self, query_vector) -> float:
        """Calculate max cosine similarity of query against reference training data."""
        if self.reference_matrix is None or query_vector is None:
            return 0.50

        try:
            # TF-IDF vectors are normalized, so dot product equals cosine similarity
            similarity_scores = query_vector @ self.reference_matrix.T
            max_sim = float(similarity_scores.max())
            return round(max_sim, 4)
        except Exception:
            return 0.50

    def detect(self, query: str, confidence: float, query_vector=None) -> dict:
        """
        Detect whether a query is Out-Of-Domain.
        
        Returns:
            dict containing domain_status ('IN_DOMAIN' / 'OUT_OF_DOMAIN'),
            ood_score (0.0 to 1.0), similarity_score, and explanation.
        """
        query_lower = query.lower()
        words = set(re.findall(r'\b\w+\b', query_lower))

        # Signal 1: Max TF-IDF Cosine Similarity
        similarity = self.compute_max_similarity(query_vector) if query_vector is not None else 0.50

        # Signal 2: Keyword domain presence
        unrelated_matches = [w for w in self.unrelated_keywords if w in query_lower]
        ecom_matches = [w for w in self.ecommerce_keywords if w in words]

        # Calculate composite OOD score (higher = more likely out of domain)
        # OOD score components: low confidence contribution, low similarity contribution, unrelated keywords
        conf_ood_component = max(0.0, (1.0 - confidence))
        sim_ood_component = max(0.0, (1.0 - similarity)) if self.reference_matrix is not None else (1.0 - confidence)

        base_ood_score = (0.5 * conf_ood_component) + (0.5 * sim_ood_component)

        if len(unrelated_matches) > 0 and len(ecom_matches) == 0:
            base_ood_score = max(base_ood_score, 0.85)

        if len(ecom_matches) > 0:
            base_ood_score = min(base_ood_score, 0.40)

        ood_score = round(float(np.clip(base_ood_score, 0.0, 1.0)), 4)

        # Decision threshold for OOD
        is_ood = (ood_score >= 0.60) or (confidence < 0.45 and similarity < self.similarity_threshold)

        domain_status = "OUT_OF_DOMAIN" if is_ood else "IN_DOMAIN"

        if is_ood:
            if len(unrelated_matches) > 0:
                reason = f"Query contains general knowledge / non-ecommerce keywords ({', '.join(unrelated_matches[:2])})."
            elif similarity < self.similarity_threshold:
                reason = f"Low vocabulary similarity ({similarity:.1%}) to supported e-commerce topics."
            else:
                reason = f"Uncertain intent prediction (confidence {confidence:.1%}) outside known patterns."
        else:
            reason = f"Query matches e-commerce domain topics (similarity {similarity:.1%}, confidence {confidence:.1%})."

        return {
            "domain_status": domain_status,
            "is_ood": is_ood,
            "ood_score": ood_score,
            "similarity_score": similarity,
            "confidence": round(confidence, 4),
            "unrelated_keyword_matches": unrelated_matches,
            "reason": reason
        }
