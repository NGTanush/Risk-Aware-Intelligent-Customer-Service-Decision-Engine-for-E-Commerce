import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = (
    "data/"
    "bitext-retail-ecommerce-llm-chatbot-training-dataset.csv"
)

MODEL_PATH = "models/calibrated_svm.pkl"

VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

OUTPUT_PATH = (
    "evaluation/"
    "ood_similarity_results.csv"
)

# Our previously selected confidence threshold
CONFIDENCE_THRESHOLD = 0.90

# Similarity thresholds to investigate
SIMILARITY_THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70
]

os.makedirs("evaluation", exist_ok=True)


# ============================================================
# 2. OOD QUERIES
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
# 3. LOAD DATASET
# ============================================================

print("=" * 70)
print("SIMILARITY-ENHANCED OOD DETECTION EXPERIMENT")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

df = df[["instruction", "intent"]].dropna()

df = df.drop_duplicates(
    subset=["instruction"]
)

print("Rows after cleaning:", len(df))
print("Number of intents:", df["intent"].nunique())


# ============================================================
# 4. RECREATE THE SAME TRAIN / TEST SPLIT
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

print("Training queries:", len(X_train))
print("Testing queries :", len(X_test))


# ============================================================
# 5. LOAD MODEL + TF-IDF
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
# 6. TRANSFORM TRAINING AND TEST DATA
# ============================================================

print("\nCreating TF-IDF representations...")

X_train_tfidf = vectorizer.transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)

X_ood_tfidf = vectorizer.transform(
    ood_queries
)

print("TF-IDF transformation completed.")


# ============================================================
# 7. FUNCTION TO CALCULATE MAXIMUM SIMILARITY
# ============================================================

def calculate_max_similarity(
    query_matrix,
    reference_matrix,
    batch_size=100
):
    """
    Calculates the maximum cosine similarity of each query
    against the reference training queries.

    TF-IDF vectors are L2-normalized by default, so the
    dot product corresponds to cosine similarity.
    """

    maximum_similarities = []

    total_queries = query_matrix.shape[0]

    for start in range(
        0,
        total_queries,
        batch_size
    ):

        end = min(
            start + batch_size,
            total_queries
        )

        batch = query_matrix[start:end]

        similarity_matrix = (
            batch @ reference_matrix.T
        )

        batch_max = (
            similarity_matrix.max(axis=1)
            .toarray()
            .ravel()
        )

        maximum_similarities.extend(
            batch_max
        )

    return np.array(
        maximum_similarities
    )


# ============================================================
# 8. PREDICT KNOWN TEST QUERIES
# ============================================================

print("\nPredicting known test queries...")

known_predictions = model.predict(
    X_test_tfidf
)

known_probabilities = model.predict_proba(
    X_test_tfidf
)

known_confidence = (
    known_probabilities.max(axis=1)
)

known_correct = (
    known_predictions == y_test.to_numpy()
)


# ============================================================
# 9. PREDICT OOD QUERIES
# ============================================================

print("Predicting OOD queries...")

ood_predictions = model.predict(
    X_ood_tfidf
)

ood_probabilities = model.predict_proba(
    X_ood_tfidf
)

ood_confidence = (
    ood_probabilities.max(axis=1)
)


# ============================================================
# 10. CALCULATE SIMILARITY
# ============================================================

print("\nCalculating similarity to training queries...")

known_similarity = calculate_max_similarity(
    X_test_tfidf,
    X_train_tfidf
)

ood_similarity = calculate_max_similarity(
    X_ood_tfidf,
    X_train_tfidf
)

print("Similarity calculation completed.")


# ============================================================
# 11. BASELINE: CONFIDENCE ONLY
# ============================================================

print("\n" + "=" * 70)
print("BASELINE: CONFIDENCE-ONLY OOD DETECTION")
print("=" * 70)

known_baseline_accepted = (
    known_confidence >= CONFIDENCE_THRESHOLD
)

ood_baseline_accepted = (
    ood_confidence >= CONFIDENCE_THRESHOLD
)

known_baseline_coverage = (
    known_baseline_accepted.mean()
)

known_baseline_accuracy = (
    known_correct[known_baseline_accepted].mean()
)

ood_baseline_rejection = (
    (~ood_baseline_accepted).mean()
)

print(
    f"Confidence threshold: "
    f"{CONFIDENCE_THRESHOLD:.0%}"
)

print(
    f"Known-data coverage: "
    f"{known_baseline_coverage:.2%}"
)

print(
    f"Known-data selective accuracy: "
    f"{known_baseline_accuracy:.2%}"
)

print(
    f"OOD rejection rate: "
    f"{ood_baseline_rejection:.2%}"
)


# ============================================================
# 12. SIMILARITY-ENHANCED DETECTOR
# ============================================================

print("\n" + "=" * 70)
print("SIMILARITY-ENHANCED DETECTOR")
print("=" * 70)

results = []


for similarity_threshold in SIMILARITY_THRESHOLDS:

    # --------------------------------------------------------
    # Known / in-domain queries
    # --------------------------------------------------------

    known_accepted = (
        (known_confidence >= CONFIDENCE_THRESHOLD)
        &
        (known_similarity >= similarity_threshold)
    )

    known_coverage = (
        known_accepted.mean()
    )

    if known_accepted.sum() > 0:

        known_accuracy = (
            known_correct[known_accepted].mean()
        )

    else:

        known_accuracy = np.nan


    # --------------------------------------------------------
    # OOD queries
    # --------------------------------------------------------

    ood_accepted = (
        (ood_confidence >= CONFIDENCE_THRESHOLD)
        &
        (ood_similarity >= similarity_threshold)
    )

    ood_rejected = ~ood_accepted

    ood_rejection = (
        ood_rejected.mean()
    )

    ood_acceptance = (
        ood_accepted.mean()
    )


    results.append({
        "Confidence Threshold":
            CONFIDENCE_THRESHOLD,

        "Similarity Threshold":
            similarity_threshold,

        "Known Coverage":
            known_coverage,

        "Known Selective Accuracy":
            known_accuracy,

        "OOD Rejection Rate":
            ood_rejection,

        "OOD Acceptance Rate":
            ood_acceptance,

        "Known Accepted":
            int(known_accepted.sum()),

        "Known Total":
            len(known_confidence),

        "OOD Accepted":
            int(ood_accepted.sum()),

        "OOD Total":
            len(ood_confidence)
    })


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"\nSimilarity threshold: "
        f"{similarity_threshold:.0%}"
    )

    print(
        f"Known coverage: "
        f"{known_coverage:.2%}"
    )

    print(
        f"Known selective accuracy: "
        f"{known_accuracy:.2%}"
    )

    print(
        f"OOD rejection rate: "
        f"{ood_rejection:.2%}"
    )

    print(
        f"OOD acceptance rate: "
        f"{ood_acceptance:.2%}"
    )


# ============================================================
# 13. SAVE EXPERIMENT RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(
    OUTPUT_PATH
)


# ============================================================
# 14. SAVE QUERY-LEVEL OOD RESULTS
# ============================================================

ood_details = pd.DataFrame({
    "Query": ood_queries,
    "Predicted Intent": ood_predictions,
    "Confidence": ood_confidence,
    "Confidence (%)":
        ood_confidence * 100,
    "Max Training Similarity":
        ood_similarity,
    "Max Similarity (%)":
        ood_similarity * 100
})

ood_details.to_csv(
    "evaluation/ood_similarity_details.csv",
    index=False
)

print(
    "evaluation/ood_similarity_details.csv"
)


# ============================================================
# 15. PRINT MOST DANGEROUS OOD CASES
# ============================================================

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE OOD CASES")
print("=" * 70)

dangerous = pd.DataFrame({
    "Query": ood_queries,
    "Intent": ood_predictions,
    "Confidence": ood_confidence,
    "Similarity": ood_similarity
})

dangerous = dangerous.sort_values(
    by="Confidence",
    ascending=False
)

for _, row in dangerous.head(10).iterrows():

    print("\nQuery:")
    print(row["Query"])

    print(
        f"Predicted intent: "
        f"{row['Intent']}"
    )

    print(
        f"Confidence: "
        f"{row['Confidence']:.2%}"
    )

    print(
        f"Similarity: "
        f"{row['Similarity']:.2%}"
    )


# ============================================================
# 16. FIND POTENTIALLY GOOD OPERATING POINTS
# ============================================================

print("\n" + "=" * 70)
print("POSSIBLE OPERATING POINTS")
print("=" * 70)

# Example research criterion:
# Maintain at least 95% OOD rejection while keeping
# at least 90% known-data coverage.

candidate_points = results_df[
    (results_df["OOD Rejection Rate"] >= 0.95)
    &
    (results_df["Known Coverage"] >= 0.90)
]

if len(candidate_points) > 0:

    print(
        "\nThresholds achieving "
        ">=95% OOD rejection and "
        ">=90% known-data coverage:"
    )

    print(
        candidate_points.to_string(
            index=False
        )
    )

else:

    print(
        "\nNo threshold achieved both "
        "criteria simultaneously."
    )

    print(
        "This is itself a useful research finding."
    )


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)

print("\nWe compared:")

print(
    "1. Confidence-only OOD detection"
)

print(
    "2. Confidence + TF-IDF similarity"
)

print(
    "\nGenerated files:"
)

print(
    "evaluation/ood_similarity_results.csv"
)

print(
    "evaluation/ood_similarity_details.csv"
)

print(
    "\nThese results can be used for the "
    "OOD robustness section of the research paper."
)