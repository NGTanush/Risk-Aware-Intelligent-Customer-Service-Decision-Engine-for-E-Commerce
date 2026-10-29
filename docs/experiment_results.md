# Experimental Results & Empirical Findings

---

## 1. Baseline Model Comparison

Experiments conducted on the 44,884 sample Bitext eCommerce dataset:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Linear SVM** | **99.09%** | **99.11%** | **99.10%** | **99.10%** |
| Logistic Regression | 98.62% | 98.66% | 98.63% | 98.63% |
| Naive Bayes | 97.52% | 97.67% | 97.52% | 97.52% |

---

## 2. Confidence Calibration & Thresholding

Selective automation evaluation across confidence thresholds ($\tau_{\text{conf}}$):

| Threshold | Coverage (%) | Selective Accuracy (%) | Error Rate (%) | Escalation Rate (%) |
| :---: | :---: | :---: | :---: | :---: |
| 50% | 99.65% | 99.26% | 0.74% | 0.35% |
| 60% | 99.14% | 99.44% | 0.56% | 0.86% |
| 70% | 98.57% | 99.57% | 0.43% | 1.43% |
| 80% | 98.01% | 99.70% | 0.30% | 1.99% |
| **85% (Selected)** | **97.58%** | **99.77%** | **0.23%** | **2.42%** |
| 90% | 96.75% | 99.80% | 0.20% | 3.25% |
| 95% | 94.71% | 99.89% | 0.11% | 5.29% |

---

## 3. Ablation Study Results

Evaluating contribution of each component across 10 benchmark test queries:

| System Variant | Total Queries | Automated | Escalated | False Automation (OOD/High-Risk) | Automation Safety Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Variant A**: Raw Intent Classifier | 10 | 10 | 0 | 7 | 30.0% |
| **Variant B**: + Calibration | 10 | 5 | 5 | 2 | 80.0% |
| **Variant C**: + OOD Detection | 10 | 5 | 5 | 2 | 80.0% |
| **Variant D**: + Sentiment | 10 | 5 | 5 | 2 | 80.0% |
| **Variant E**: **Full System** | **10** | **5** | **5** | **0** | **100.0%** |

*Finding*: The full system (Variant E) completely eliminates dangerous false automations on out-of-domain and high-risk financial queries.
