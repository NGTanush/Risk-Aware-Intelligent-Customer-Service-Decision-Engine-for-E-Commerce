# Technical Methodology & Algorithmic Formulation

---

## 1. Feature Representation (TF-IDF Vectorization)

Text queries are converted into continuous numerical feature vectors using Term Frequency-Inverse Document Frequency (TF-IDF) with sublinear term frequency scaling:

$$\text{tf-idf}(t, d, D) = (1 + \log(\text{tf}(t, d))) \times \log\left(\frac{1 + |D|}{1 + |\text{df}(D, t)|}\right) + 1$$

- **N-gram Range**: $(1, 2)$ (captures single words and word pairs like "track order", "refund status").
- **Vocabulary Size**: 20,000 top features.
- **Normalization**: L2 normalization ($\|v\|_2 = 1$).

---

## 2. Classifier Calibration (Platt Sigmoidal Scaling)

Standard Linear Support Vector Machines output uncalibrated decision margins ($f(x) \in (-\infty, +\infty)$). To convert decision margins into calibrated posterior probabilities ($P(y=k | x)$), Platt Scaling is applied:

$$P(y=1 | x) = \frac{1}{1 + \exp(A \cdot f(x) + B)}$$

where parameters $A$ and $B$ are fit via maximum likelihood estimation over 5-fold cross-validation.

- **Brier Score Metric**:
  $$\text{BS} = \frac{1}{N} \sum_{i=1}^{N} (f_i - o_i)^2 = 0.0072$$
  where $f_i$ is predicted probability and $o_i \in \{0, 1\}$ is actual correctness.

---

## 3. Out-of-Domain Similarity Detection

Cosine similarity between a query vector $q$ and reference training vectors $r_j$ is defined as:

$$\text{Sim}(q, r_j) = \frac{q \cdot r_j}{\|q\|_2 \|r_j\|_2} = q \cdot r_j \quad (\text{since } \|q\|_2 = \|r_j\|_2 = 1)$$

Maximum similarity $\mathcal{S}_{\max}(q) = \max_{j} (q \cdot r_j)$ is computed against all training samples. If $\mathcal{S}_{\max}(q) < \tau_{\text{sim}}$, the query is flagged as Out-Of-Domain.

---

## 4. Risk Formulation

Composite risk score $R \in [0, 1]$ is computed as:

$$R = \text{clamp}\Big( w_1 R_{\text{intent}} + w_2 R_{\text{sentiment}} + w_3 (1 - C) + w_4 R_{\text{OOD}}, \, 0, \, 1 \Big)$$

where:
- $w_1 = 0.45, \, w_2 = 0.25, \, w_3 = 0.15, \, w_4 = 0.15$
- $R_{\text{intent}} \in \{0.10, 0.40, 0.80\}$ based on intent risk classification.
- $C \in [0, 1]$ is calibrated model confidence.
- $R_{\text{sentiment}} \in [0.10, 0.99]$ based on negative tone score.
- $R_{\text{OOD}} \in [0.10, 1.00]$ based on OOD detector score.
