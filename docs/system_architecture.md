# System Architecture & Technical Specifications

> **Project Title**: Risk-Aware Intelligent Customer Service Decision Engine for E-Commerce

---

## 1. System Overview

The system is designed to provide selective automation for e-commerce customer support. Unlike conventional chatbots that attempt to answer every input query regardless of risk or uncertainty, this system incorporates a multi-layer evaluation pipeline:

```
[ Customer Query Input ]
          │
          ▼
   [ Preprocessing ] ────► Lowercasing, punctuation stripping, tokenization
          │
          ▼
 [ TF-IDF Vectorization ] ─► 20,000 max features, 1-2 ngrams, sublinear TF
          │
          ├───► [ Calibrated Linear SVM ] ─────► Intent Label & Calibrated Confidence
          │
          ├───► [ Sentiment Analyzer ] ────────► Sentiment Label (POS/NEU/NEG) & Score
          │
          └───► [ OOD Detector ] ──────────────► Domain Status (IN_DOMAIN/OUT_OF_DOMAIN)
                  │
                  ▼
       [ Composite Risk Engine ] ──────────────► Risk Level (LOW/MEDIUM/HIGH) & Risk Score (0-1)
                  │
                  ▼
     [ 3-Way Decision Router ] ───────────────► AUTOMATE / CLARIFY / HUMAN_ESCALATION
                  │
                  ▼
     [ Response Generation Layer ] ───────────► OpenAI LLM OR Dataset Cosine Similarity Fallback
```

---

## 2. Component Specifications

### 2.1 Intent Classifier & Calibration
- **Base Classifier**: Linear Support Vector Classifier (`LinearSVC`).
- **Calibration Method**: Platt Scaling using 5-fold cross-validated sigmoid calibrator (`CalibratedClassifierCV`).
- **Output**: Calibrated probability vector across 46 dataset intent classes.

### 2.2 Sentiment Analyzer
- Lexicon-based analyzer capturing e-commerce emotional cues, intensifiers, ALL CAPS, and exclamations.
- Outputs `POSITIVE`, `NEUTRAL`, or `NEGATIVE` classification and `sentiment_score`.

### 2.3 Out-Of-Domain (OOD) Detector
- Multi-signal detector combining:
  1. Calibrated intent probability.
  2. Max cosine similarity against 35,000+ reference dataset vectors.
  3. Domain keyword presence vs unrelated topic indicators (e.g. sports, weather, programming).

### 2.4 Risk Engine
- Composite scoring formula:
  $$\text{Risk} = (0.45 \times \text{IntentRisk}) + (0.25 \times \text{SentimentRisk}) + (0.15 \times \text{Uncertainty}) + (0.15 \times \text{OODRisk})$$
- Evaluates risk score (0.0 to 1.0) and maps to `LOW`, `MEDIUM`, or `HIGH`.

### 2.5 3-Way Decision Engine
- Evaluates policy rules to select:
  - **AUTOMATE**: High confidence ($\ge 85\%$), in-domain, low/medium risk.
  - **CLARIFY**: Ambiguous query or moderate confidence ($55\% - 85\%$).
  - **HUMAN_ESCALATION**: High risk, hostile sentiment on sensitive intent, out-of-domain, or confidence $< 55\%$.

### 2.6 Response Generator
- Triggers OpenAI LLM response when `AUTOMATE` is approved.
- Provides TF-IDF cosine-similarity response retrieval fallback from the Bitext dataset if the OpenAI API is unconfigured or fails.
