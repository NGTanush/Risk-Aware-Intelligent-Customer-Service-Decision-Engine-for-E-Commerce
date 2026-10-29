# Comprehensive Research Audit: Risk-Aware Intelligence Customer Service Decision Engine

**Date:** October 2026  
**Auditor:** ML Research Engineer  
**Project:** Risk-Aware Intelligent Customer Service Decision Engine for E-Commerce  
**Paper Topic:** *"Confidence and Risk-Aware Selective Automation for E-Commerce Customer Service Using Intent Classification and LLM-Based Response Generation"*

---

## 1. Executive Summary

This audit assesses the state of experimental validation, model artifacts, evaluation datasets, and decision pipeline implementations in the repository prior to conducting new research paper experiments.

### Key Takeaways
1. **Strong Baseline Models & Datasets:** The 44,884-sample Bitext retail dataset is properly deduplicated and split (80/20 train/test, stratified, seed 42) into 35,907 training queries and 8,977 test queries across 46 intents. Trained classifiers (MultinomialNB, Logistic Regression, Linear SVM, and Calibrated Linear SVM) are preserved in `models/`.
2. **Critical Paper Premise Validated:** Out-of-domain (OOD) tests clearly demonstrate that standard classifiers assign extremely high confidence (>95%) to completely unsupported, out-of-domain queries (e.g., *"How do I make chocolate cake?"* predicted as `pay` with 97.3% confidence; *"What are today's football scores?"* predicted as `product_information` with 93.2% confidence). This empirically verifies the central research thesis: **High confidence does not imply safety or domain validity.**
3. **Severe Evaluation Weaknesses in Preliminary Scripts:** While baseline classifier metrics and confidence threshold tables exist across all 8,977 test samples, the preliminary ablation study (`src/ablation_experiment.py`) and robustness script (`src/robustness_experiment.py`) only evaluated **10 hardcoded queries**. The OOD experiment only evaluated **20 queries**.
4. **Missing Priority Experiments for Academic Publication:**
   - Systematic Policy Comparison (Always Automate vs. Confidence Only vs. Confidence+OOD vs. Full Risk-Aware System) across the complete held-out test suite and OOD sets.
   - Large-scale Ablation Study across all held-out queries.
   - Dedicated High-Risk Safety Evaluation verifying zero false automations for financial/account security queries.
   - Comprehensive Error Analysis categorizing model and policy failure modes.
   - Truly Independent Test Set (200–300 curated queries) reflecting realistic distribution shifts.

---

## 2. Experiments Already Completed & Existing Results

| Experiment / Artifact | Status | File Location | Sample Size / Scope | Key Findings |
| :--- | :---: | :--- | :--- | :--- |
| **Model Comparison (Uncalibrated)** | ✅ Valid | `evaluation/model_comparison.csv`<br>`evaluation/*_classification_report.txt`<br>`evaluation/*_confusion_matrix.png` | 8,977 held-out test queries | Linear SVM achieved Macro F1 = 99.10%, Logistic Regression = 98.63%, Naive Bayes = 97.52%. |
| **Linear SVM Calibration** | ✅ Valid | `evaluation/calibration_results.csv`<br>`evaluation/calibration_curve.png`<br>`models/calibrated_svm.pkl` | 8,977 held-out test queries | 5-fold cross-validation sigmoid calibration (Platt scaling). Accuracy = 99.03%, Macro F1 = 99.04%, Avg Confidence = 97.43%, Binary Brier Score = 0.0072. |
| **Confidence Threshold Analysis** | ⚠️ Partial | `evaluation/confidence_threshold_results.csv`<br>`evaluation/coverage_vs_accuracy.png`<br>`evaluation/threshold_vs_coverage.png` | 8,977 test queries | Tested $\tau \in [0.50, 0.95]$ in 0.05 steps. Coverage ranges from 99.65% ($\tau=0.50$) to 94.71% ($\tau=0.95$). Selective accuracy ranges from 99.26% to 99.89%. |
| **OOD Detection (Confidence Only)** | ⚠️ Partial | `evaluation/ood_results.csv`<br>`evaluation/ood_full_output.txt` | 20 OOD queries | At 90% threshold, confidence-only rejected 85% (17/20), but falsely accepted 15% (3/20) with high confidence. |
| **OOD Similarity Enhancement** | ⚠️ Partial | `evaluation/ood_similarity_results.csv`<br>`evaluation/ood_similarity_details.csv` | 20 OOD queries + 8,977 known test queries | Explored combining TF-IDF cosine similarity ($\tau_{sim} \in [0.1, 0.7]$) with confidence $\tau=0.90$. |
| **Toy Ablation Study** | ❌ Inadequate | `evaluation/ablation_results.csv` | **10 queries only** | Showed preliminary progression from 30% safety to 100% safety, but sample size is far too small for peer-reviewed claims. |
| **Toy Robustness Test** | ❌ Inadequate | `evaluation/robustness_results.csv` | **10 queries only** | Handcrafted qualitative sanity checks across 10 categories. |

---

## 3. Implementation Bugs & Evaluation Weaknesses Identified

1. **Ablation & Robustness Sample Size:**
   - *Problem:* 10 queries cannot establish statistical rigor or empirical credibility in an academic paper.
   - *Fix:* Run systematic baseline comparisons and ablation studies on the full held-out test set ($N=8,977$) supplemented with controlled OOD sets and high-risk query slices.
2. **Binary vs. Multiclass Brier Score & Missing ECE:**
   - *Problem:* `src/train_model.py` evaluated Brier score via `brier_score_loss(correct_predictions, confidence_scores)`. While this measures top-1 confidence-correctness discrepancy, Expected Calibration Error (ECE) and full multi-class calibration metrics should be computed and reported with mathematical precision.
3. **Threshold Analysis Missed the Three-Way Decision Structure:**
   - *Problem:* `src/threshold_experiment.py` evaluated only binary accept/reject (coverage vs. escalation). The project’s actual decision engine has a **three-way output**: `AUTOMATE`, `CLARIFY`, and `HUMAN_ESCALATION`. The threshold experiment did not quantify clarification rates or separate clarification from escalation.
4. **Reference Matrix Memory & Latency in Pipeline:**
   - *Note:* Building `self.reference_matrix = self.vectorizer.transform(X_train)` at startup loads 35,907 queries. While fine for batch execution and API server runtime (~8–10s startup), new research experiment scripts should reuse a cached reference matrix or pre-computed features to optimize runtime.
5. **High-Risk Safety Must Be Formally Audited:**
   - *Observation:* The project defines 11 intents as `HIGH` risk in `src/intent_risk_map.py` (e.g. `request_refund`, `refund_status`, `payment_issue`, `close_account`, `cancel_order`, `damaged_delivery`, `missing_item`, `wrong_item`, `request_right_to_rectification`, `recover_password`, `change_account`).
   - *Question to verify:* Does the live decision engine *always* route every high-risk query to `HUMAN_ESCALATION` without exception? In `src/decision_engine.py`:
     ```python
     elif intent_risk_level == "HIGH" or risk_level == "HIGH":
         decision = "HUMAN_ESCALATION"
     ```
     This rule is implemented at line 51 of `src/decision_engine.py`. We must rigorously test it on hundreds of real high-risk test queries.

---

## 4. Required Next Experiments & Execution Roadmap

To satisfy all requirements of the research paper and maintain strict reproducibility:

1. **Experiment A (Verification of Classifier Performance):**
   - Re-verify MultinomialNB, LogisticRegression, and LinearSVC on the stratified 80/20 test split ($N=8,977$).
   - Confirm accuracy, macro precision, recall, and macro F1; check for any train-test leakage.
   - Output: `research/results/research_model_comparison.csv` (and `evaluation/research_model_comparison.csv`).

2. **Experiment B (Confidence Calibration Deep Dive):**
   - Evaluate Calibrated Linear SVM; compute Expected Calibration Error (ECE), top-1 Brier score, multiclass Brier score, and reliability diagram bins.
   - Compare uncalibrated margin-derived confidence with Platt calibrated confidence.
   - Output: `research/results/research_calibration_results.csv`, `research/graphs/research_calibration_curve.png`.

3. **Experiment C (Three-Way Confidence Threshold Analysis):**
   - Evaluate $\tau \in [0.50, 0.95]$ incorporating the 3-way decision taxonomy (`AUTOMATE`, `CLARIFY`, `HUMAN_ESCALATION`).
   - Plot Coverage vs. Selective Accuracy vs. Escalation / Clarification.
   - Output: `research/results/research_threshold_results.csv`, `research/graphs/research_threshold_plot.png`.

4. **Experiment D (Baseline Decision-Policy Comparison — PRIORITY):**
   - Compare Policy 1 (Always Automate), Policy 2 (Confidence Only), Policy 3 (Confidence + OOD), and Policy 4 (Full Risk-Aware System) on identical held-out test data.
   - Output: `research/results/research_baseline_comparison.csv`, `research/graphs/research_baseline_comparison.png`.

5. **Experiment E (Ablation Study on Full Test Set):**
   - Disentangle the marginal contribution of: Confidence $\to$ Intent Risk $\to$ Sentiment $\to$ OOD Detection $\to$ Combined System.
   - Output: `research/results/research_ablation_results.csv`, `research/graphs/research_ablation_plot.png`.

6. **Experiment F (High-Risk Safety Verification):**
   - Extract all high-risk test queries from the Bitext dataset (all 11 high-risk intents, totaling ~2,100+ instances) + adversarial high-risk queries.
   - Validate whether high-risk automated handling is strictly 0.00%.
   - Output: `research/results/research_safety_results.csv`.

7. **Experiment G & H (OOD & Robustness Expansion):**
   - Expand beyond the 20 toy OOD queries with an expanded benchmark and evaluate false-acceptance vs. rejection.
   - Robustness evaluation across typos, paraphrasing, aggressive/negative tone, and ambiguity.
   - Output: `research/results/research_ood_results.csv`, `research/results/research_robustness_results.csv`.

8. **Experiment I & J (Independent Test Set & Error Analysis):**
   - Construct a curated, independent benchmark of 250 diverse queries with documented ground truth.
   - Detailed failure mode categorization (confusion matrix analysis, boundary cases).
   - Output: `data/independent_test_set.csv`, `research/results/research_independent_eval.csv`, `research/results/research_error_analysis.csv`.

9. **Consolidated Deliverable:**
   - `research/RESEARCH_RESULTS_SUMMARY.md` synthesizes all tables, metrics, paper-ready figures, and honest disclosures of limitations.
