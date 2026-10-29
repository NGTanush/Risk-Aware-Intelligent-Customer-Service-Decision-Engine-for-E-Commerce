import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "data/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv"
MODEL_PATH = "models/calibrated_svm.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

os.makedirs("evaluation", exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 65)
print("SELECTIVE AUTOMATION / CONFIDENCE THRESHOLD EXPERIMENT")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

df = df[["instruction", "intent"]].dropna()

df = df.drop_duplicates(
    subset=["instruction"]
)

print("\nDataset loaded.")
print("Rows after cleaning:", len(df))
print("Number of intents:", df["intent"].nunique())


# ============================================================
# 3. CREATE THE SAME TEST SPLIT
# ============================================================

X = df["instruction"]
y = df["intent"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Test samples:", len(X_test))


# ============================================================
# 4. LOAD CALIBRATED MODEL AND VECTORIZER
# ============================================================

print("\nLoading calibrated SVM...")

model = joblib.load(
    MODEL_PATH
)

vectorizer = joblib.load(
    VECTORIZER_PATH
)

print("Model loaded successfully.")


# ============================================================
# 5. TRANSFORM TEST DATA
# ============================================================

X_test_tfidf = vectorizer.transform(
    X_test
)


# ============================================================
# 6. GET PREDICTIONS AND PROBABILITIES
# ============================================================

predictions = model.predict(
    X_test_tfidf
)

probabilities = model.predict_proba(
    X_test_tfidf
)

# Highest probability for each query
confidence_scores = probabilities.max(
    axis=1
)

# Whether prediction is correct
correct = (
    predictions == y_test.to_numpy()
)


# ============================================================
# 7. TEST MULTIPLE CONFIDENCE THRESHOLDS
# ============================================================

thresholds = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]

results = []


print("\n" + "=" * 65)
print("THRESHOLD RESULTS")
print("=" * 65)

for threshold in thresholds:

    # Queries that the AI is confident enough to answer
    accepted = confidence_scores >= threshold

    answered_count = accepted.sum()

    total_count = len(confidence_scores)

    # Coverage = percentage automatically answered
    coverage = answered_count / total_count

    # Escalation = percentage sent to human
    escalation_rate = 1 - coverage

    if answered_count > 0:

        # Correct predictions among answered queries
        answered_correct = correct[accepted].sum()

        # Accuracy among answered queries
        selective_accuracy = (
            answered_correct / answered_count
        )

        # Error rate among answered queries
        error_rate = (
            1 - selective_accuracy
        )

    else:

        selective_accuracy = np.nan
        error_rate = np.nan

    results.append({
        "Threshold": threshold,
        "Coverage": coverage,
        "Selective Accuracy": selective_accuracy,
        "Error Rate": error_rate,
        "Escalation Rate": escalation_rate,
        "Answered Queries": int(answered_count),
        "Escalated Queries": int(total_count - answered_count)
    })

    print(
        f"\nThreshold: {threshold:.0%}"
    )

    print(
        f"Coverage:           {coverage:.2%}"
    )

    print(
        f"Selective Accuracy: {selective_accuracy:.2%}"
    )

    print(
        f"Error Rate:         {error_rate:.2%}"
    )

    print(
        f"Escalation Rate:    {escalation_rate:.2%}"
    )


# ============================================================
# 8. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_path = (
    "evaluation/"
    "confidence_threshold_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\n" + "=" * 65)

print(
    "Results saved to:"
)

print(
    results_path
)


# ============================================================
# 9. PLOT: THRESHOLD VS COVERAGE
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    results_df["Threshold"] * 100,
    results_df["Coverage"] * 100,
    marker="o"
)

plt.xlabel(
    "Confidence Threshold (%)"
)

plt.ylabel(
    "Automatic Answer Coverage (%)"
)

plt.title(
    "Confidence Threshold vs Automatic Answer Coverage"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "evaluation/"
    "threshold_vs_coverage.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. PLOT: THRESHOLD VS SELECTIVE ACCURACY
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    results_df["Threshold"] * 100,
    results_df["Selective Accuracy"] * 100,
    marker="o"
)

plt.xlabel(
    "Confidence Threshold (%)"
)

plt.ylabel(
    "Accuracy of Automatically Answered Queries (%)"
)

plt.title(
    "Confidence Threshold vs Selective Accuracy"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "evaluation/"
    "threshold_vs_selective_accuracy.png",
    dpi=300
)

plt.close()


# ============================================================
# 11. PLOT: COVERAGE VS ACCURACY
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    results_df["Coverage"] * 100,
    results_df["Selective Accuracy"] * 100,
    marker="o"
)

plt.xlabel(
    "Automatic Answer Coverage (%)"
)

plt.ylabel(
    "Selective Accuracy (%)"
)

plt.title(
    "Coverage vs Accuracy Trade-off"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "evaluation/"
    "coverage_vs_accuracy.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("EXPERIMENT COMPLETE")
print("=" * 65)

print("\nGenerated files:")

print(
    "evaluation/confidence_threshold_results.csv"
)

print(
    "evaluation/threshold_vs_coverage.png"
)

print(
    "evaluation/threshold_vs_selective_accuracy.png"
)

print(
    "evaluation/coverage_vs_accuracy.png"
)

print("\nThese results can be used in the research paper.")