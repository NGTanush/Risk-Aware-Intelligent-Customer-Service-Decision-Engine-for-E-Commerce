"""
Sentiment Analyzer Component for E-Commerce Customer Service Decision Engine.

Provides lightweight, robust sentiment analysis without requiring heavy external model downloads.
Categorizes queries into POSITIVE, NEUTRAL, or NEGATIVE sentiment with a normalized sentiment score.
"""

import re

class SentimentAnalyzer:
    def __init__(self):
        # Lexicons tailored for e-commerce customer interactions
        self.negative_words = {
            "angry", "terrible", "worst", "ridiculous", "scam", "broken", "hate",
            "awful", "horrible", "upset", "disappointed", "useless", "fraud",
            "waste", "unacceptable", "disaster", "furious", "annoyed", "pathetic",
            "stolen", "late", "delay", "delayed", "wrong", "damaged", "defective",
            "never", "fail", "failed", "cheated", "lie", "liar", "garbage",
            "poor", "unhelpful", "rip-off", "rip off", "scammer",
            "waiting forever", "where is my", "no response", "still haven't",
            "zero stars", "don't buy", "do not buy", "do not trust", "sucks"
        }
        
        self.positive_words = {
            "great", "love", "thanks", "thank you", "good", "excellent", "awesome",
            "amazing", "perfect", "wonderful", "happy", "satisfied", "fantastic",
            "helpful", "fast", "superb", "brilliant", "pleased", "appreciate",
            "best", "smooth", "nice", "kind"
        }

        self.intensifiers = {
            "very", "extremely", "really", "so", "totally", "absolutely",
            "completely", "utterly", "incredibly", "highly"
        }

    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment of a customer query.
        
        Returns:
            dict containing sentiment ('POSITIVE', 'NEUTRAL', 'NEGATIVE'),
            sentiment_score (float 0.0 to 1.0), and details.
        """
        if not text or not isinstance(text, str):
            return {
                "sentiment": "NEUTRAL",
                "sentiment_score": 0.5,
                "negative_confidence": 0.0,
                "is_negative": False,
                "details": "Empty input"
            }

        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        neg_count = 0.0
        pos_count = 0.0

        # Check for phrase matches
        for phrase in self.negative_words:
            if " " in phrase and phrase in text_lower:
                neg_count += 2.0

        for phrase in self.positive_words:
            if " " in phrase and phrase in text_lower:
                pos_count += 2.0

        # Check single words
        for idx, word in enumerate(words):
            if word in self.negative_words:
                if idx > 0 and words[idx - 1] in self.intensifiers:
                    neg_count += 2.0
                else:
                    neg_count += 1.0
            elif word in self.positive_words:
                if idx > 0 and words[idx - 1] in self.intensifiers:
                    pos_count += 2.0
                else:
                    pos_count += 1.0

        # Check for ALL CAPS (emotional intensity indicator)
        raw_words = text.split()
        caps_words = [w for w in raw_words if w.isupper() and len(w) > 1 and w.isalpha()]
        if len(caps_words) >= 2:
            neg_count += 1.5

        # Check for multiple exclamation marks
        exclamation_count = text.count("!")
        if exclamation_count >= 2:
            neg_count += 1.0

        # Calculate sentiment category and score
        if neg_count > pos_count and neg_count >= 1.0:
            sentiment = "NEGATIVE"
            sentiment_score = min(0.99, 0.50 + (neg_count * 0.12))
            is_negative = True
        elif pos_count > neg_count and pos_count >= 1.0:
            sentiment = "POSITIVE"
            sentiment_score = min(0.99, 0.50 + (pos_count * 0.12))
            is_negative = False
        else:
            sentiment = "NEUTRAL"
            sentiment_score = 0.50
            is_negative = False

        return {
            "sentiment": sentiment,
            "sentiment_score": round(float(sentiment_score), 2),
            "negative_score": round(float(neg_count), 2),
            "positive_score": round(float(pos_count), 2),
            "is_negative": is_negative,
            "exclamation_count": exclamation_count,
            "caps_count": len(caps_words)
        }
