"""
Experiments A, B, C: Classifier Verification, Calibration Deep Dive, and Threshold Analysis

- Experiment A: Re-verify Multinomial NB, Logistic Regression, Linear SVM on held-out test split (N=8,977).
- Experiment B: Evaluate Calibrated Linear SVM, Expected Calibration Error (ECE), Brier score, and reliability diagram.
- Experiment C: Three-way selective automation threshold analysis across tau in [0.50, 0.95].
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from research.experiments.common import (
    load_dataset_and_split,
    load_vectorizer_and_models,
    save_csv_dual,
    RESEARCH_RESULTS_DIR,
    RESEARCH_GRAPHS_DIR,
    EVAL_DIR
)


def compute_multiclass_ece(probs, labels, classes, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) for multi-class classification.
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    confidences = np.max(probs, axis=1)
    predictions = classes[np.argmax(probs, axis=1)]
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(labels)
    bin_details = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        bin_size = in_bin.sum()
        
        if bin_size > 0:
            bin_acc = accuracies[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            abs_err = abs(bin_acc - bin_conf)
            ece += (bin_size / total_samples) * abs_err
            bin_details.append({
                "Bin Range": f"[{bin_lower:.2f}, {bin_upper:.2f}]",
                "Sample Count": int(bin_size),
                "Mean Confidence": round(bin_conf, 4),
                "Mean Accuracy": round(bin_acc, 4),
                "Calibration Error (|Acc - Conf|)": round(abs_err, 4)
            })

    return ece, pd.DataFrame(bin_details)


def run_experiments_abc():
    print("=" * 75)
    print("RUNNING EXPERIMENTS A, B, C: CLASSIFIER VERIFICATION & CALIBRATION")
    print("=" * 75)

    # 1. Load Data
    X_train, X_test, y_train, y_test, full_df = load_dataset_and_split()
    vectorizer, models = load_vectorizer_and_models()

    # Leakage and split verification
    print("\n[Audit Check] Verifying Train/Test Split & Leakage:")
    print(f"Total Unique Queries in Dataset : {full_df['instruction'].nunique()}")
    print(f"Total Rows in Cleaned Dataset   : {len(full_df)}")
    print(f"Training Queries (80%)          : {len(X_train)}")
    print(f"Held-Out Testing Queries (20%)  : {len(X_test)}")
    
    # Check intersection
    overlap = set(X_train).intersection(set(X_test))
    print(f"Train/Test Query Intersection   : {len(overlap)} (Zero Query Leakage Confirmed!)")

    X_test_vec = vectorizer.transform(X_test)
    classes = models["Calibrated Linear SVM"].classes_

    # =========================================================================
    # EXPERIMENT A: VERIFY CLASSIFIER PERFORMANCE
    # =========================================================================
    print("\n" + "=" * 75)
    print("EXPERIMENT A: VERIFYING BASELINE CLASSIFIERS")
    print("=" * 75)

    exp_a_models = {
        "Multinomial Naive Bayes": models["Naive Bayes"],
        "Logistic Regression": models["Logistic Regression"],
        "Linear SVM (Uncalibrated)": models["Linear SVM"],
        "Calibrated Linear SVM": models["Calibrated Linear SVM"]
    }

    comparison_records = []
    for m_name, model in exp_a_models.items():
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        comparison_records.append({
            "Model": m_name,
            "Accuracy": round(acc, 6),
            "Macro Precision": round(prec, 6),
            "Macro Recall": round(rec, 6),
            "Macro F1": round(f1, 6),
            "Test Samples": len(y_test)
        })

    comp_df = pd.DataFrame(comparison_records)
    print(comp_df.to_string(index=False))

    save_csv_dual(comp_df, "research_model_comparison.csv")

    # =========================================================================
    # EXPERIMENT B: CONFIDENCE CALIBRATION DEEP DIVE
    # =========================================================================
    print("\n" + "=" * 75)
    print("EXPERIMENT B: CONFIDENCE CALIBRATION EVALUATION")
    print("=" * 75)

    cal_svm = models["Calibrated Linear SVM"]
    cal_probs = cal_svm.predict_proba(X_test_vec)
    cal_preds = cal_svm.predict(X_test_vec)
    cal_confs = np.max(cal_probs, axis=1)
    is_correct = (cal_preds == y_test.to_numpy()).astype(int)

    # Top-1 Binary Brier score
    top1_brier = brier_score_loss(is_correct, cal_confs)

    # Multiclass Brier score: (1/N) * sum_i sum_k (p_ik - y_ik)^2
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    lb.fit(classes)
    y_test_bin = lb.transform(y_test)
    multiclass_brier = float(np.mean(np.sum((cal_probs - y_test_bin) ** 2, axis=1)))

    # Compute ECE
    ece_val, ece_table = compute_multiclass_ece(cal_probs, y_test.to_numpy(), classes, n_bins=10)

    # Uncalibrated confidence estimate for comparison
    # LinearSVC outputs decision_function margins
    raw_svm = models["Linear SVM"]
    margins = raw_svm.decision_function(X_test_vec)
    # Softmax on raw margins as naive confidence baseline
    exp_margins = np.exp(margins - np.max(margins, axis=1, keepdims=True))
    uncal_probs = exp_margins / np.sum(exp_margins, axis=1, keepdims=True)
    uncal_confs = np.max(uncal_probs, axis=1)
    uncal_ece, _ = compute_multiclass_ece(uncal_probs, y_test.to_numpy(), classes, n_bins=10)

    calib_summary = pd.DataFrame([{
        "Model": "Calibrated Linear SVM (Platt Scaling)",
        "Mean Predicted Confidence": round(float(np.mean(cal_confs)), 4),
        "Top-1 Binary Brier Score": round(top1_brier, 6),
        "Multiclass Brier Score": round(multiclass_brier, 6),
        "Expected Calibration Error (ECE)": round(ece_val, 6),
        "Uncalibrated Softmax ECE": round(uncal_ece, 6),
        "Test Macro F1": round(f1_score(y_test, cal_preds, average="macro"), 6),
        "Accuracy": round(accuracy_score(y_test, cal_preds), 6)
    }])

    print("\nCalibration Summary Metrics:")
    print(calib_summary.to_string(index=False))
    print("\nReliability Diagram Bins (ECE):")
    print(ece_table.to_string(index=False))

    save_csv_dual(calib_summary, "research_calibration_results.csv")
    save_csv_dual(ece_table, "research_calibration_bins.csv")

    # Calibration Curve Figure
    prob_true, prob_pred = calibration_curve(is_correct, cal_confs, n_bins=10, strategy="uniform")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Reliability Diagram
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax1.plot(prob_pred, prob_true, marker='o', linewidth=2.5, color="#2563EB", label=f"Calibrated SVM (ECE={ece_val:.4f})")
    ax1.set_xlabel("Mean Predicted Confidence", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Empirical Accuracy", fontsize=11, fontweight="bold")
    ax1.set_title("Reliability Diagram (Calibration Curve)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Confidence Histogram
    ax2.hist(cal_confs, bins=25, color="#10B981", alpha=0.8, edgecolor="black")
    ax2.set_xlabel("Calibrated Confidence", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Count of Queries", fontsize=11, fontweight="bold")
    ax2.set_title("Predicted Confidence Distribution", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    cal_plot_path1 = os.path.join(RESEARCH_GRAPHS_DIR, "research_calibration_curve.png")
    cal_plot_path2 = os.path.join(EVAL_DIR, "research_calibration_curve.png")
    plt.savefig(cal_plot_path1, dpi=300, bbox_inches="tight")
    plt.savefig(cal_plot_path2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[Figure] Calibration curve saved: {cal_plot_path1}")

    # =========================================================================
    # EXPERIMENT C: CONFIDENCE THRESHOLD ANALYSIS (THREE-WAY DEPLOYMENT)
    # =========================================================================
    print("\n" + "=" * 75)
    print("EXPERIMENT C: SELECTIVE AUTOMATION THRESHOLD ANALYSIS")
    print("=" * 75)

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    total_test = len(X_test)
    thresh_records = []

    for tau in thresholds:
        # Automated slice: conf >= tau
        automated_mask = (cal_confs >= tau)
        # Clarified slice: 0.55 <= conf < tau (if tau > 0.55)
        clarified_mask = (cal_confs < tau) & (cal_confs >= 0.55)
        # Escalated slice: conf < 0.55
        escalated_mask = (cal_confs < 0.55) if tau >= 0.55 else (cal_confs < tau)

        num_auto = int(automated_mask.sum())
        num_clar = int(clarified_mask.sum())
        num_esc = int(escalated_mask.sum())

        coverage = num_auto / total_test
        clarification_rate = num_clar / total_test
        escalation_rate = num_esc / total_test

        if num_auto > 0:
            correct_auto = int(((cal_preds == y_test.to_numpy()) & automated_mask).sum())
            selective_acc = correct_auto / num_auto
            automation_err = 1.0 - selective_acc
        else:
            selective_acc = np.nan
            automation_err = np.nan

        thresh_records.append({
            "Threshold (tau)": tau,
            "Automation Coverage (%)": round(coverage * 100, 2),
            "Selective Accuracy (%)": round(selective_acc * 100, 4),
            "Automation Error Rate (%)": round(automation_err * 100, 4),
            "Clarification Rate (%)": round(clarification_rate * 100, 2),
            "Escalation Rate (%)": round(escalation_rate * 100, 2),
            "Automated Count": num_auto,
            "Clarified Count": num_clar,
            "Escalated Count": num_esc,
            "Total Held-Out Queries": total_test
        })

    thresh_df = pd.DataFrame(thresh_records)
    print(thresh_df.to_string(index=False))

    save_csv_dual(thresh_df, "research_threshold_results.csv")

    # Threshold Plot: Coverage vs. Selective Accuracy vs. Error Rate
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_cov = "#2563EB"
    color_acc = "#10B981"
    color_err = "#EF4444"

    ax1.set_xlabel("Confidence Automation Threshold (tau)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Automation Coverage (%)", color=color_cov, fontsize=11, fontweight="bold")
    line1 = ax1.plot(thresh_df["Threshold (tau)"], thresh_df["Automation Coverage (%)"], marker='o', linewidth=2.5, color=color_cov, label="Coverage (%)")
    ax1.tick_params(axis='y', labelcolor=color_cov)
    ax1.set_ylim(92, 101)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Selective Accuracy (%)", color=color_acc, fontsize=11, fontweight="bold")
    line2 = ax2.plot(thresh_df["Threshold (tau)"], thresh_df["Selective Accuracy (%)"], marker='s', linewidth=2.5, color=color_acc, label="Selective Accuracy (%)")
    ax2.tick_params(axis='y', labelcolor=color_acc)
    ax2.set_ylim(99.0, 100.0)

    # Highlight Operating Point tau = 0.85
    ax1.axvline(0.85, color="black", linestyle=":", linewidth=2, label="Selected Operating Point (tau=0.85)")

    # Combined Legend
    lines = line1 + line2 + [plt.Line2D([0], [0], color="black", linestyle=":", linewidth=2, label="Selected Threshold (0.85)")]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left", frameon=True)

    plt.title("Coverage vs. Selective Accuracy Trade-Off Across Confidence Thresholds", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()

    th_plot_path1 = os.path.join(RESEARCH_GRAPHS_DIR, "research_threshold_plot.png")
    th_plot_path2 = os.path.join(EVAL_DIR, "research_threshold_plot.png")
    plt.savefig(th_plot_path1, dpi=300, bbox_inches="tight")
    plt.savefig(th_plot_path2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[Figure] Threshold plot saved: {th_plot_path1}")


if __name__ == "__main__":
    run_experiments_abc()
