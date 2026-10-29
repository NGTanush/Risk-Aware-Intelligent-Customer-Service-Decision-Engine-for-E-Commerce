import os
import joblib
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

MODEL_PATH = "models/calibrated_svm.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

os.makedirs("evaluation", exist_ok=True)


# ============================================================
# 2. OUT-OF-DOMAIN TEST QUERIES
# ============================================================

ood_queries = [
    "What is the weather like today?",
    "Who won the cricket match yesterday?",
    "Write a Python program to sort a list.",
    "Explain quantum physics in simple words.",
    "Tell me a funny joke.",
    "What is the capital of France?",
    "How do I make chocolate cake?",
    "What are the best movies to watch?",
    "Help me learn Spanish.",
    "How does a car engine work?",
    "What is the current stock market doing?",
    "Can you write a poem for me?",
    "What is artificial intelligence?",
    "How do I prepare for an exam?",
    "Tell me about the solar system.",
    "What is the population of India?",
    "How can I lose weight quickly?",
    "Give me a travel itinerary for Europe.",
    "How do I fix my laptop?",
    "What are today's football scores?"
]


# ============================================================
# 3. LOAD MODEL
# ============================================================

print("=" * 65)
print("OUT-OF-DISTRIBUTION (OOD) EXPERIMENT")
print("=" * 65)

print("\nLoading calibrated SVM...")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("Model loaded successfully.")


# ============================================================
# 4. TRANSFORM QUERIES
# ============================================================

X_ood = vectorizer.transform(
    ood_queries
)


# ============================================================
# 5. PREDICTIONS
# ============================================================

predicted_intents = model.predict(
    X_ood
)

probabilities = model.predict_proba(
    X_ood
)

confidence_scores = probabilities.max(
    axis=1
)


# ============================================================
# 6. CREATE RESULTS TABLE
# ============================================================

results = []

for query, intent, confidence in zip(
    ood_queries,
    predicted_intents,
    confidence_scores
):

    results.append({
        "Query": query,
        "Predicted Intent": intent,
        "Confidence": confidence,
        "Confidence (%)": confidence * 100
    })


results_df = pd.DataFrame(results)


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("OOD PREDICTIONS")
print("=" * 65)

for _, row in results_df.iterrows():

    print("\nQuery:")
    print(row["Query"])

    print("Predicted Intent:")
    print(row["Predicted Intent"])

    print(
        f"Confidence: "
        f"{row['Confidence (%)']:.2f}%"
    )


# ============================================================
# 8. SUMMARY STATISTICS
# ============================================================

average_confidence = (
    results_df["Confidence"].mean()
)

minimum_confidence = (
    results_df["Confidence"].min()
)

maximum_confidence = (
    results_df["Confidence"].max()
)


print("\n" + "=" * 65)
print("OOD SUMMARY")
print("=" * 65)

print(
    f"Number of OOD queries: {len(results_df)}"
)

print(
    f"Average confidence: "
    f"{average_confidence:.2%}"
)

print(
    f"Minimum confidence: "
    f"{minimum_confidence:.2%}"
)

print(
    f"Maximum confidence: "
    f"{maximum_confidence:.2%}"
)


# ============================================================
# 9. TEST OUR 90% THRESHOLD
# ============================================================

threshold = 0.90

rejected = (
    results_df["Confidence"] < threshold
)

accepted = (
    results_df["Confidence"] >= threshold
)

rejection_rate = rejected.mean()
acceptance_rate = accepted.mean()


print("\n" + "=" * 65)
print("OOD REJECTION AT 90% THRESHOLD")
print("=" * 65)

print(
    f"Queries rejected/escalated: "
    f"{rejected.sum()} / {len(results_df)}"
)

print(
    f"Rejection rate: "
    f"{rejection_rate:.2%}"
)

print(
    f"Queries incorrectly accepted: "
    f"{accepted.sum()} / {len(results_df)}"
)

print(
    f"Acceptance rate: "
    f"{acceptance_rate:.2%}"
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

output_path = (
    "evaluation/ood_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\nResults saved to:")
print(output_path)


# ============================================================
# 11. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 65)
print("OOD EXPERIMENT COMPLETE")
print("=" * 65)