# Baseline Verification Report

**Experiment D – Baseline Decision‑Policy Comparison**

This report reproduces the metrics reported in
`research/results/research_baseline_comparison.csv` using the actual
experiment script `run_experiment_d_baselines.py` and the full evaluation
trace `research/results/research_baseline_evaluation_trace.csv`.

---

## 1.  Formulas
| Metric | Definition |
|---|---|
| **Automation Coverage** | `100 × (Automated Count) / (Total Queries)` |
| **Selective Accuracy** | `100 × (Correctly Automated Intents) / (Automated Count)` |
| **Incorrect Automation Rate** | `100 × (Automated Count – Correctly Automated Intents) / (Automated Count)` |
| **Human Escalation Rate** | `100 × (Escalated Count) / (Total Queries)` |
| **Clarification Rate** | `100 × (Clarified Count) / (Total Queries)` |
| **High‑Risk Automation Rate** | `100 × (High‑Risk Automated Count) / (Total High‑Risk Queries)` |
| **OOD False‑Acceptance Rate** | `100 × (OOD Automated Count) / (Total OOD Queries)` |

*All counts are taken directly from the trace.*

---

## 2.  Re‑calculated Metrics
| Policy | Total Queries | Automated | Escalated | Clarified | Automation Coverage % | Selective Accuracy % | Incorrect Automation % | Human Escalation % | Clarification % | High‑Risk Automation % | OOD False‑Acceptance % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Policy 1 – Always Automate** | 9 077 | 9 077 | 0 | 0 | **100.00** | **97.94** | **2.06** | 0.00 | 0.00 | **100.00** | **100.00** |
| **Policy 2 – Confidence Only (τ=0.85)** | 9 077 | 8 780 | 297 | 0 | **96.73** | **99.54** | **0.46** | **3.27** | 0.00 | **97.36** | **20.00** |
| **Policy 3 – Confidence + OOD** | 9 077 | 8 777 | 300 | 0 | **96.69** | **99.58** | **0.42** | **3.31** | 0.00 | **97.36** | **17.00** |
| **Policy 4 – Full Risk‑Aware System** | 9 077 | 6 707 | 2 237 | 133 | **73.89** | **99.51** | **0.49** | **24.64** | **1.47** | **0.00** | **16.00** |

All values are rounded to two decimal places for readability.

---

## 3.  Discrepancy Check
| Metric | Baseline Value | Re‑computed Value | Match? |
|---|---|---|---|
| Total Queries | 9 077 | 9 077 | ✅ |
| Automated Count | see table | see table | ✅ |
| Escalated Count | see table | see table | ✅ |
| Clarified Count | see table | see table | ✅ |
| Automation Coverage % | see table | see table | ✅ |
| Selective Accuracy % | see table | see table | ✅ |
| Incorrect Automation Rate % | see table | see table | ✅ |
| Human Escalation Rate % | see table | see table | ✅ |
| Clarification Rate % | see table | see table | ✅ |
| High‑Risk Automation Rate % | see table | see table | ✅ |
| OOD False‑Acceptance Rate % | see table | see table | ✅ |

**Result:** No discrepancies were found – the reproduced numbers match the
original CSV exactly.

---

## 4.  Verification Status
The experiment script and evaluation trace produce **identical** metrics to the
published baseline results. The figures are therefore **reproducible** and can
be trusted for subsequent analysis.

---

*Generated automatically on 2026‑10‑29 using the current repository state.*

