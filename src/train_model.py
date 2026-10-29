import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    brier_score_loss
)


# ============================================================
# 1. CREATE REQUIRED FOLDERS
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("evaluation", exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

DATA_PATH = "data/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv"

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Total rows:", len(df))
print("Columns:", list(df.columns))


# Keep only the columns needed for intent classification
df = df[["instruction", "intent"]].dropna()

# Remove duplicate customer queries
df = df.drop_duplicates(subset=["instruction"])

print("Rows after cleaning:", len(df))
print("Number of intents:", df["intent"].nunique())


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

X = df["instruction"]
y = df["intent"]


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 5. TF-IDF VECTORIZATION
# ============================================================

print("\n" + "=" * 60)
print("TF-IDF VECTORIZATION")
print("=" * 60)

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=20000,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF completed.")
print("Number of features:", len(vectorizer.get_feature_names_out()))


# ============================================================
# 6. TRAIN THREE MACHINE LEARNING MODELS
# ============================================================

print("\n" + "=" * 60)
print("TRAINING MODELS")
print("=" * 60)

models = {
    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Linear SVM": LinearSVC(
        C=1.0,
        random_state=42
    )
}


# ============================================================
# 7. TRAIN AND EVALUATE MODELS
# ============================================================

results = []
trained_models = {}

for model_name, model in models.items():

    print("\nTraining:", model_name)

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    print(
        f"{model_name}: "
        f"Accuracy={accuracy:.4f}, "
        f"Macro F1={f1:.4f}"
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1
    })

    trained_models[model_name] = model

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    report_file = (
        "evaluation/"
        + model_name.lower().replace(" ", "_")
        + "_classification_report.txt"
    )

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)


# ============================================================
# 8. MODEL COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

results_df.to_csv(
    "evaluation/model_comparison.csv",
    index=False
)


# ============================================================
# 9. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\nBest model:", best_model_name)

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("Best model saved to:")
print("models/best_model.pkl")


# ============================================================
# 10. SAVE INDIVIDUAL MODELS
# ============================================================

joblib.dump(
    trained_models["Naive Bayes"],
    "models/naive_bayes.pkl"
)

joblib.dump(
    trained_models["Logistic Regression"],
    "models/logistic_regression.pkl"
)

joblib.dump(
    trained_models["Linear SVM"],
    "models/linear_svm.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

print("\nIndividual models saved.")


# ============================================================
# 11. CONFUSION MATRIX FOR BEST MODEL
# ============================================================

best_predictions = best_model.predict(X_test_tfidf)

cm = confusion_matrix(
    y_test,
    best_predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=best_model.classes_
)

fig, ax = plt.subplots(figsize=(18, 18))

disp.plot(
    ax=ax,
    xticks_rotation="vertical",
    cmap="Blues",
    colorbar=False
)

plt.title(
    f"{best_model_name} - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "evaluation/best_model_confusion_matrix.png",
    dpi=300
)

plt.close()

print("\nConfusion matrix saved.")


# ============================================================
# 12. TRAIN CALIBRATED LINEAR SVM
# ============================================================

print("\n" + "=" * 60)
print("TRAINING CALIBRATED LINEAR SVM")
print("=" * 60)

# Create a fresh Linear SVM for calibration
base_svm = LinearSVC(
    C=1.0,
    random_state=42
)

calibrated_svm = CalibratedClassifierCV(
    estimator=base_svm,
    method="sigmoid",
    cv=5
)

calibrated_svm.fit(
    X_train_tfidf,
    y_train
)

print("Calibrated SVM trained successfully.")


# ============================================================
# 13. EVALUATE CALIBRATED SVM
# ============================================================

calibrated_predictions = calibrated_svm.predict(
    X_test_tfidf
)

calibrated_probabilities = calibrated_svm.predict_proba(
    X_test_tfidf
)

calibrated_accuracy = accuracy_score(
    y_test,
    calibrated_predictions
)

calibrated_precision = precision_score(
    y_test,
    calibrated_predictions,
    average="macro",
    zero_division=0
)

calibrated_recall = recall_score(
    y_test,
    calibrated_predictions,
    average="macro",
    zero_division=0
)

calibrated_f1 = f1_score(
    y_test,
    calibrated_predictions,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 60)
print("CALIBRATED LINEAR SVM RESULTS")
print("=" * 60)

print(f"Accuracy        : {calibrated_accuracy:.4f}")
print(f"Macro Precision : {calibrated_precision:.4f}")
print(f"Macro Recall    : {calibrated_recall:.4f}")
print(f"Macro F1        : {calibrated_f1:.4f}")


# ============================================================
# 14. CALIBRATED CONFIDENCE
# ============================================================

# Highest probability assigned to each prediction
confidence_scores = calibrated_probabilities.max(axis=1)

# Whether each prediction was correct
correct_predictions = (
    calibrated_predictions == y_test.to_numpy()
).astype(int)

average_confidence = confidence_scores.mean()

print("\n" + "=" * 60)
print("CONFIDENCE INFORMATION")
print("=" * 60)

print(
    f"Average predicted confidence: "
    f"{average_confidence:.4f}"
)


# ============================================================
# 15. BRIER SCORE
# ============================================================

# We treat "prediction correct or incorrect"
# as a binary outcome and compare it with
# the model's confidence.

brier = brier_score_loss(
    correct_predictions,
    confidence_scores
)

print(
    f"Brier score: {brier:.4f}"
)


# ============================================================
# 16. CALIBRATION CURVE
# ============================================================

prob_true, prob_pred = calibration_curve(
    correct_predictions,
    confidence_scores,
    n_bins=10,
    strategy="uniform"
)

plt.figure(figsize=(8, 6))

plt.plot(
    prob_pred,
    prob_true,
    marker="o",
    label="Calibrated Linear SVM"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.xlabel("Mean Predicted Confidence")
plt.ylabel("Actual Accuracy")

plt.title(
    "Confidence Calibration Curve"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "evaluation/calibration_curve.png",
    dpi=300
)

plt.close()

print("\nCalibration curve saved to:")
print("evaluation/calibration_curve.png")


# ============================================================
# 17. CALIBRATED CONFUSION MATRIX
# ============================================================

calibrated_cm = confusion_matrix(
    y_test,
    calibrated_predictions
)

calibrated_disp = ConfusionMatrixDisplay(
    confusion_matrix=calibrated_cm,
    display_labels=calibrated_svm.classes_
)

fig, ax = plt.subplots(figsize=(18, 18))

calibrated_disp.plot(
    ax=ax,
    xticks_rotation="vertical",
    cmap="Blues",
    colorbar=False
)

plt.title(
    "Calibrated Linear SVM - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "evaluation/calibrated_svm_confusion_matrix.png",
    dpi=300
)

plt.close()

print("Calibrated confusion matrix saved.")


# ============================================================
# 18. SAVE CALIBRATED MODEL
# ============================================================

joblib.dump(
    calibrated_svm,
    "models/calibrated_svm.pkl"
)

print("\nCalibrated model saved to:")
print("models/calibrated_svm.pkl")


# ============================================================
# 19. SAVE CALIBRATION RESULTS
# ============================================================

calibration_results = pd.DataFrame([
    {
        "Model": "Calibrated Linear SVM",
        "Accuracy": calibrated_accuracy,
        "Macro Precision": calibrated_precision,
        "Macro Recall": calibrated_recall,
        "Macro F1": calibrated_f1,
        "Average Confidence": average_confidence,
        "Brier Score": brier
    }
])

calibration_results.to_csv(
    "evaluation/calibration_results.csv",
    index=False
)

print("Calibration results saved to:")
print("evaluation/calibration_results.csv")


# ============================================================
# 20. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("TRAINING AND CALIBRATION COMPLETE")
print("=" * 60)

print("\nFiles created/updated:")

print("\nModels:")
print("  models/naive_bayes.pkl")
print("  models/logistic_regression.pkl")
print("  models/linear_svm.pkl")
print("  models/best_model.pkl")
print("  models/calibrated_svm.pkl")
print("  models/tfidf_vectorizer.pkl")

print("\nEvaluation:")
print("  evaluation/model_comparison.csv")
print("  evaluation/calibration_results.csv")
print("  evaluation/calibration_curve.png")
print("  evaluation/best_model_confusion_matrix.png")
print("  evaluation/calibrated_svm_confusion_matrix.png")

print("\nEverything completed successfully.")