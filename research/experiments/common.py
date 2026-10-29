"""
Common Utilities for Academic Research Experiments.

Ensures strict reproducibility, consistent held-out splits,
pre-trained model loading, and efficient evaluation.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.intent_risk_map import get_intent_risk_level, get_intent_risk_score, INTENT_RISK_MAP
from src.sentiment_analyzer import SentimentAnalyzer
from src.ood_detector import OODDetector
from src.risk_engine import RiskEngine
from src.decision_engine import DecisionEngine

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "bitext-retail-ecommerce-llm-chatbot-training-dataset.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
RESEARCH_RESULTS_DIR = os.path.join(PROJECT_ROOT, "research", "results")
RESEARCH_GRAPHS_DIR = os.path.join(PROJECT_ROOT, "research", "graphs")

os.makedirs(RESEARCH_RESULTS_DIR, exist_ok=True)
os.makedirs(RESEARCH_GRAPHS_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


def load_dataset_and_split(random_state=42, test_size=0.20):
    """
    Load Bitext dataset and return exact stratified split matching train_model.py.
    """
    df = pd.read_csv(DATA_PATH)
    df = df[["instruction", "intent"]].dropna()
    df = df.drop_duplicates(subset=["instruction"])
    
    X = df["instruction"]
    y = df["intent"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, df


def load_vectorizer_and_models():
    """
    Load pre-trained TF-IDF vectorizer and all candidate models.
    """
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    vectorizer = joblib.load(vec_path)
    
    models = {
        "Linear SVM": joblib.load(os.path.join(MODELS_DIR, "linear_svm.pkl")),
        "Logistic Regression": joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl")),
        "Naive Bayes": joblib.load(os.path.join(MODELS_DIR, "naive_bayes.pkl")),
        "Calibrated Linear SVM": joblib.load(os.path.join(MODELS_DIR, "calibrated_svm.pkl"))
    }
    return vectorizer, models


def save_csv_dual(df, filename):
    """
    Save DataFrame to both research/results/ and evaluation/ for dual delivery.
    """
    path1 = os.path.join(RESEARCH_RESULTS_DIR, filename)
    path2 = os.path.join(EVAL_DIR, filename)
    df.to_csv(path1, index=False)
    df.to_csv(path2, index=False)
    print(f"[Export] Saved: {path1}")
    print(f"[Export] Saved: {path2}")


def get_ood_benchmark_queries():
    """
    Curated set of 100 diverse out-of-domain queries across multiple domains:
    Weather, Sports, Programming, General Science, Cooking/Recipes, Entertainment,
    World Geography, Philosophy/Literature, Math, and Medical/Health.
    """
    queries = [
        # Weather
        "What is the weather like today?",
        "What is the weather forecast for tomorrow?",
        "Is it going to rain in London this weekend?",
        "What is the temperature in Tokyo right now?",
        "Will there be snow in New York next week?",
        "How do meteorologists predict hurricanes?",
        "What causes a tornado to form?",
        "Tell me the current UV index in Sydney.",
        "Why is the sky blue during the day?",
        "What is the humidity level outside?",

        # Sports
        "Who won the cricket match yesterday?",
        "What are today's football scores?",
        "Who will win tomorrow's cricket match?",
        "Who won the FIFA World Cup in 2022?",
        "What is the score in the NBA finals game?",
        "How many Grand Slam titles does Rafael Nadal have?",
        "Explain the offside rule in soccer.",
        "Who is the fastest runner in Olympic history?",
        "What are the rules of badminton doubles?",
        "When is the next Formula 1 Grand Prix?",

        # Programming & Computer Science
        "Write a Python program to sort a list.",
        "Write a Python script to sort a list of numbers.",
        "How do I fix a NullPointerException in Java?",
        "Explain how asynchronous programming works in JavaScript.",
        "Write a SQL query to find second highest salary.",
        "What is the time complexity of quicksort?",
        "How do I dockerize a React application?",
        "Explain the difference between TCP and UDP.",
        "How do I implement a binary search tree in C++?",
        "Write a bash script to monitor CPU usage.",

        # General Science & Physics
        "Explain quantum physics in simple words.",
        "What is artificial intelligence?",
        "How does a car engine work?",
        "Tell me about the solar system.",
        "What is the theory of general relativity?",
        "How does photosynthesis convert sunlight to energy?",
        "What is dark matter made of?",
        "Why do objects fall at the same rate in a vacuum?",
        "How do airplanes generate aerodynamic lift?",
        "Explain CRISPR gene editing technology.",

        # Cooking & Recipes
        "How do I make chocolate cake?",
        "What is the best recipe for Italian carbonara?",
        "How long do you bake chicken breast at 400 degrees?",
        "Give me a recipe for homemade pizza dough.",
        "How do I make French onion soup from scratch?",
        "What can I substitute for buttermilk in pancakes?",
        "How do you caramelize onions properly?",
        "What is the secret to fluffy scrambled eggs?",
        "How do I temper chocolate for dipping?",
        "Give me an authentic recipe for Mexican guacamole.",

        # Entertainment & Pop Culture
        "Tell me a funny joke.",
        "What are the best movies to watch?",
        "Who won the Oscar for Best Picture in 2023?",
        "Can you recommend a good sci-fi TV series?",
        "Who directed the movie Inception?",
        "What is the plot of Hamlet by William Shakespeare?",
        "Can you write a poem for me?",
        "Who played Iron Man in the Marvel Cinematic Universe?",
        "What are some popular anime series to start with?",
        "Tell me an interesting trivia fact about Hollywood.",

        # World Geography & History
        "What is the capital of France?",
        "What is the population of India?",
        "Give me a travel itinerary for Europe.",
        "What is the longest river in the world?",
        "In what year did World War II end?",
        "What were the primary causes of the French Revolution?",
        "Which country has the most natural lakes?",
        "What is the deepest point in the world's oceans?",
        "How many continents are there on Earth?",
        "What was the capital of the Byzantine Empire?",

        # Language & Education
        "Help me learn Spanish.",
        "How do I prepare for an exam?",
        "How do you conjugate the verb avoir in French?",
        "What is the difference between affect and effect?",
        "Translate 'Good morning' into Japanese.",
        "What is a metaphor versus a simile?",
        "How can I improve my English vocabulary quickly?",
        "Give me tips for writing a college admissions essay.",
        "How do memory palaces work for memorization?",
        "What are the grammatical cases in German?",

        # Health & Fitness
        "How can I lose weight quickly?",
        "What are the benefits of intermittent fasting?",
        "How much water should an adult drink daily?",
        "What exercises build lower back strength?",
        "Why is REM sleep important for cognitive health?",
        "How do vaccines create immunity in the body?",
        "What are the symptoms of vitamin D deficiency?",
        "How does meditation reduce cortisol and stress?",
        "What is the Mediterranean diet food pyramid?",
        "How do muscles repair and hypertrophy after lifting?",

        # Economy & Finance (General Macro)
        "What is the current stock market doing?",
        "How do central banks control interest rates?",
        "What is the difference between inflation and stagflation?",
        "Explain how blockchain cryptography works.",
        "What was the cause of the 2008 financial crisis?",
        "How does compound interest grow savings over decades?",
        "What is gross domestic product and how is it measured?",
        "How do currency exchange rates fluctuate?",
        "What is an index fund and how does it track markets?",
        "Explain the law of supply and demand."
    ]
    return queries
