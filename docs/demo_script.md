# Lecturer Demonstration Script & Presentation Guide

> **Project Title**: Risk-Aware Intelligent Customer Service Decision Engine for E-Commerce  
> **Academic Title**: *Confidence and Risk-Aware Selective Automation for E-Commerce Customer Service Using Intent Classification and LLM-Based Response Generation*

---

## 🎤 Opening Statement (What to say first)

> *"Good morning / afternoon Professor,*
>
> *Today I am presenting our Final Year Project titled **Risk-Aware Intelligent Customer Service Decision Engine for E-Commerce**.*
>
> *Traditional AI chatbots have a major flaw: they blindly attempt to answer every single query, which leads to hallucinated answers, privacy risks, or incorrect policy promises during sensitive transactions like refunds or payment errors.*
>
> *Our system is **not a normal chatbot**. Instead, it acts as an intelligent **Decision Center**. When a customer submits a query, the system first understands the intent, calibrates the ML model confidence, inspects out-of-domain relevance, analyzes customer sentiment, and calculates an overall risk score.*
>
> *It then makes a transparent, explainable **three-way decision**: **AUTOMATE**, **CLARIFY**, or **ESCALATE TO A HUMAN**."*

---

## 💻 Live Software Demonstration (Step-by-Step Cases)

Open your browser to the Streamlit app (`streamlit run app.py`). Navigate to **Tab 1: Decision Center** and demonstrate these 5 test cases live:

---

### CASE 1: Standard Low-Risk Query (Automated Response)
- **Input Query**: `"Where can I track my package?"`
- **What to say**:
  > *"Here, the customer is asking a standard package tracking question. The Linear SVM model predicts `track_delivery` with **97.3% calibrated confidence**. Because the intent risk is **LOW** and sentiment is **NEUTRAL**, the Decision Engine approves **🟢 AUTOMATION APPROVED** and generates a response."*

---

### CASE 2: Angry Customer Query
- **Input Query**: `"WHERE IS MY PACKAGE?! I HAVE BEEN WAITING FOREVER AND THIS IS RIDICULOUS!"`
- **What to say**:
  > *"Notice what happens when an angry customer submits a query. The Sentiment Analyzer detects **NEGATIVE** emotional tone and uppercase intensity. The Risk Engine factors this emotional distress into the risk score, demonstrating how sentiment directly influences automation safety."*

---

### CASE 3: Ambiguous Query (Clarification Required)
- **Input Query**: `"I need help with my order."`
- **What to say**:
  > *"In this case, the customer query is very short and ambiguous. The calibrated confidence drops to **61.4%**. Rather than guessing, the system triggers **🟡 CLARIFICATION REQUIRED** and asks the customer for specific details like their Order ID."*

---

### CASE 4: Sensitive Financial / High-Risk Request (Human Escalation)
- **Input Query**: `"I want to request a refund for my order. I am very disappointed!"`
- **What to say**:
  > *"Here, the customer is asking for a financial refund. Refund requests carry a **HIGH** baseline intent risk. Combined with negative sentiment, the composite risk score rises to **0.75**. The Decision Engine immediately triggers **🔴 HUMAN ESCALATION** and routes the query to a human support agent with a clear explanation."*

---

### CASE 5: Out-Of-Domain Query (Rejection)
- **Input Query**: `"Who will win tomorrow's cricket match?"`
- **What to say**:
  > *"Finally, if a user enters an out-of-domain question about sports or weather, our TF-IDF similarity detector and OOD component detect that the query is **OUT_OF_DOMAIN**. The system safely rejects automation and escalates."*

---

## 📊 Dashboard Walkthrough (Showing your research results)

Walk your lecturer through the remaining tabs:

1. **Tab 2 — Model Performance**:
   > *"We benchmarked Naive Bayes (97.52%), Logistic Regression (98.63%), and Linear SVM (99.10%). Linear SVM performed best. We calibrated it using Platt Sigmoid scaling, achieving a **Brier Score of 0.0072**."*

2. **Tab 3 — Confidence Analysis**:
   > *"This tab displays our selective automation experiment across confidence thresholds from 50% to 95%. At our selected **85% threshold**, the system achieves **97.58% coverage** with **99.77% selective accuracy**."*

3. **Tab 4 — Risk Monitor**:
   > *"Provides live audit telemetry, tracking the distribution of Low, Medium, and High risk queries in real-time."*

4. **Tab 5 — Robustness Lab**:
   > *"Demonstrates 1-click comparative testing across 7 query categories (noisy queries, typos, rephrased queries, and angry tone)."*

5. **Tab 6 — System Architecture**:
   > *"Shows the full visual data flow diagram from NLP vectorization to risk evaluation and decision routing."*

---

## 🎯 Summary Conclusion (What to say at the end)

> *"In conclusion, our Risk-Aware Decision Engine proves that combining calibrated confidence, sentiment analysis, and explainable risk management enables safe, reliable AI automation for e-commerce customer support.*
>
> *Thank you, Professor! I am happy to answer any questions."*
