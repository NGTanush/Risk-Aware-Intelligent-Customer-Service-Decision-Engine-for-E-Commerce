# Risk-Aware Intelligent Customer Service Decision Engine for E-Commerce

> **Academic Title**: *Confidence and Risk-Aware Selective Automation for E-Commerce Customer Service Using Intent Classification and LLM-Based Response Generation*

---

## 📌 Problem Statement & Motivation

Conventional AI chatbots blindly attempt to answer every customer query, leading to high-risk hallucinations, privacy violations, or incorrect policies during sensitive transactions (e.g. refunds, payment failures, account deletions). 

This project introduces a **Risk-Aware Decision Engine** that evaluates intent, model confidence, out-of-domain relevance, customer sentiment, and intent risk before deciding whether to **AUTOMATE**, **CLARIFY**, or **ESCALATE** to a human support agent.

---

## 🏗️ System Architecture

Rather than a simple `Query ➔ Chatbot ➔ Answer` pipeline, the system operates as a 4-stage decision-support platform:

```
[ Customer Query ]
       │
       ▼
[ 1. UNDERSTAND ] ──► NLP Preprocessing & TF-IDF Vectorization
       │
       ▼
[ 2. ASSESS ] ──────► Calibrated Intent Confidence (Linear SVM + Platt Scaling)
       │               Customer Sentiment Analysis (Positive / Neutral / Negative)
       │               Out-Of-Domain Detection (TF-IDF Similarity + Keywords)
       │               Explainable Risk Engine (0.0 to 1.0 Risk Score)
       │
       ▼
[ 3. DECIDE ] ──────► Three-Way Decision Engine
       │               ├── AUTOMATE (High Confidence, In-Domain, Low Risk)
       │               ├── CLARIFY (Ambiguous Query / Moderate Confidence)
       │               └── HUMAN_ESCALATION (High Risk, Negative Sentiment, OOD)
       │
       ▼
[ 4. RESPOND ] ─────► LLM Response (OpenAI API) OR Dataset Cosine Similarity Fallback
```

---

## 📊 Dataset & Baseline Machine Learning Results

- **Dataset**: Bitext Retail (eCommerce) Tagged Training Dataset (~44,884 samples, 46 unique intent classes, 13 categories).
- **Text Vectorization**: TF-IDF (Unigrams + Bigrams, 20,000 max features, L2 normalized).

### Baseline Classifier Comparison:

| Machine Learning Model | Accuracy | Macro Precision | Macro Recall | Macro F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear SVM (LinearSVC)** | **99.09%** | **99.11%** | **99.10%** | **99.10%** |
| Logistic Regression | 98.62% | 98.66% | 98.63% | 98.63% |
| Naive Bayes | 97.52% | 97.67% | 97.52% | 97.52% |

### Confidence Calibration Results:
- **Calibrated Linear SVM (Platt Sigmoid Scaling)**:
  - **Macro F1**: `99.04%`
  - **Average Predicted Confidence**: `97.43%`
  - **Brier Score**: `0.0072` (Indicates excellent probabilistic calibration)

---

## 🛡️ Risk & Decision Policy

### 1. Risk Engine Formula:
$$\text{Risk Score} = (0.45 \times \text{Base Intent Risk}) + (0.25 \times \text{Sentiment Risk}) + (0.15 \times \text{Uncertainty}) + (0.15 \times \text{OOD Score})$$

- **Intent Risk Levels**:
  - **LOW**: Tracking, delivery info, product catalog, store hours.
  - **MEDIUM**: Order modifications, exchanges, returns, app technical issues.
  - **HIGH**: Financial refunds, payment disputes, account closure, password recovery, damaged items.

### 2. Three-Way Decision Rules:
- **AUTOMATE**: High calibrated confidence ($\ge 85\%$), In-Domain, Low/Medium Risk.
- **CLARIFY**: Ambiguous query or moderate confidence ($55\% \le \text{Confidence} < 85\%$).
- **HUMAN ESCALATION**: High Risk intent ($\text{Risk} \ge 0.65$), Hostile Negative sentiment on sensitive intents, or Out-Of-Domain ($\text{OOD}$).

---

## 🚀 How to Run the Application

### 1. Prerequisites & Environment Setup
Open PowerShell in your project folder:
```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Launch Streamlit Decision Center
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Launch FastAPI Backend Server (Optional)
```powershell
uvicorn server:app --reload
```
API Documentation will be available at `http://127.0.0.1:8000/docs`.

---

## 🧪 Experiments & Evaluation Scripts

Run automated evaluations:
```powershell
# Run end-to-end system test suite
python src/test_system.py

# Run robustness experiment under query perturbations
python src/robustness_experiment.py

# Run ablation experiment comparing architecture variants
python src/ablation_experiment.py
```

---

## 📁 Project Structure

```
ecommerce_customer_ai/
├── app.py                      # Redesigned Streamlit Decision Center (6 Tabs)
├── server.py                   # FastAPI REST API Backend
├── test_server.py              # API Endpoint Test Suite
├── requirements.txt            # Dependency List
├── README.md                   # Main Project Documentation
├── data/
│   └── bitext-retail-ecommerce-llm-chatbot-training-dataset.csv
├── models/
│   ├── calibrated_svm.pkl
│   ├── linear_svm.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   └── tfidf_vectorizer.pkl
├── evaluation/
│   ├── model_comparison.csv
│   ├── calibration_results.csv
│   ├── confidence_threshold_results.csv
│   ├── ood_results.csv
│   ├── robustness_results.csv
│   └── ablation_results.csv
├── src/
│   ├── sentiment_analyzer.py   # Lexicon-based Sentiment Component
│   ├── intent_risk_map.py      # 46 Intent Risk Mapping
│   ├── ood_detector.py         # Multi-signal OOD Detector
│   ├── risk_engine.py          # Composite Risk Scoring Formula
│   ├── decision_engine.py      # 3-Way Decision Router (Automate/Clarify/Escalate)
│   ├── response_generator.py   # LLM Generator & Cosine Similarity Fallback
│   ├── decision_pipeline.py    # Master Pipeline Integrator
│   ├── robustness_experiment.py
│   ├── ablation_experiment.py
│   └── test_system.py
└── docs/
    ├── system_architecture.md
    ├── technical_methodology.md
    ├── experiment_results.md
    └── demo_script.md          # Lecturer Presentation Guide
```
