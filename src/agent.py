import os
import joblib
from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

CONFIDENCE_THRESHOLD = 0.85

MODEL_PATH = "models/calibrated_svm.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


# ============================================================
# LOAD ML MODEL
# ============================================================

print("Loading machine learning model...")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("Calibrated Linear SVM loaded successfully.")


# ============================================================
# OPENAI CLIENT
# ============================================================

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY was not detected. "
        "Please check your environment variable."
    )

client = OpenAI(
    api_key=api_key
)

print("OpenAI API connection configured.")


# ============================================================
# INTENT PREDICTION
# ============================================================

def predict_intent(query):
    """
    Predict the customer's intent and calibrated confidence.
    """

    query_vector = vectorizer.transform([query])

    predicted_intent = model.predict(
        query_vector
    )[0]

    probabilities = model.predict_proba(
        query_vector
    )[0]

    confidence = probabilities.max()

    return predicted_intent, confidence


# ============================================================
# API RESPONSE GENERATION
# ============================================================

def generate_api_response(
    query,
    predicted_intent,
    confidence
):
    """
    Generate a natural customer-service response
    using the OpenAI Responses API.
    """

    prompt = f"""
You are a professional e-commerce customer service assistant.

Customer query:
{query}

Machine-learning predicted intent:
{predicted_intent}

Calibrated confidence:
{confidence:.2%}

Generate a helpful response based ONLY on the customer's
query and the predicted intent.

Rules:
- Be polite and concise.
- Stay within the predicted intent.
- Do not invent order numbers, delivery dates, refund status,
  prices, policies, or account information.
- Do not claim that you accessed the customer's account.
- Ask for necessary information when required.
- Do not mention the machine-learning model, confidence,
  calibration, API, or internal system.
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    return response.output_text.strip()


# ============================================================
# HUMAN ESCALATION
# ============================================================

def human_escalation(
    query,
    predicted_intent,
    confidence
):
    """
    Response shown when confidence is below threshold.
    """

    return (
        "I'm sorry, but I want to make sure you receive "
        "the correct assistance. Your request needs further "
        "review by a customer-service representative. "
        "Please provide a little more information so we can "
        "help you appropriately."
    )


# ============================================================
# MAIN CUSTOMER-SERVICE FUNCTION
# ============================================================

def process_query(query):
    """
    Complete customer-service pipeline.
    """

    predicted_intent, confidence = predict_intent(
        query
    )

    print(
        f"\nPredicted Intent: {predicted_intent}"
    )

    print(
        f"Confidence: {confidence:.2%}"
    )

    # --------------------------------------------------------
    # HIGH CONFIDENCE
    # --------------------------------------------------------

    if confidence >= CONFIDENCE_THRESHOLD:

        print(
            "\nDecision: HIGH CONFIDENCE"
        )

        print(
            "Generating response using OpenAI API..."
        )

        try:

            response = generate_api_response(
                query,
                predicted_intent,
                confidence
            )

            return response

        except Exception as error:

            print(
                "\nAPI error occurred."
            )

            print(
                "Using safe fallback response."
            )

            return (
                "I understand your request regarding "
                f"{predicted_intent.replace('_', ' ')}. "
                "Please provide any relevant order or "
                "account details so we can assist you."
            )

    # --------------------------------------------------------
    # LOW CONFIDENCE
    # --------------------------------------------------------

    else:

        print(
            "\nDecision: LOW CONFIDENCE"
        )

        print(
            "Escalating to human customer service."
        )

        return human_escalation(
            query,
            predicted_intent,
            confidence
        )


# ============================================================
# INTERACTIVE TERMINAL
# ============================================================

print("\n" + "=" * 60)
print("E-COMMERCE CUSTOMER SERVICE AI")
print("=" * 60)

print(
    f"Confidence threshold: "
    f"{CONFIDENCE_THRESHOLD:.0%}"
)

print(
    "Type 'exit' to stop."
)


while True:

    query = input(
        "\nCustomer: "
    ).strip()

    if query.lower() == "exit":
        print(
            "\nThank you. Goodbye!"
        )
        break

    if not query:
        print(
            "Please enter a customer query."
        )
        continue

    try:

        response = process_query(
            query
        )

        print(
            "\nAssistant:"
        )

        print(
            response
        )

    except Exception as error:

        print(
            "\nSystem error:"
        )

        print(
            error
        )