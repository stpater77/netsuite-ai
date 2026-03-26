import os
import psycopg2
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATABASE_URL = os.getenv("DATABASE_URL")


def get_db_connection():
    return psycopg2.connect(DATABASE_URL)


def detect_intent(message):
    message = message.lower()

    if "invoice" in message and "sales order" in message:
        return "ar__invoice__invoice_a_sales_order"

    if "create" in message and "invoice" in message:
        return "ar__invoice__create_an_invoice"

    if "payment" in message:
        return "ar__payment__accept_customer_payments"

    if "credit" in message:
        return "ar__credit__create_credit_memo"

    return None


def handle_user_message(message):
    intent = detect_intent(message)

    # -----------------------------
    # TRY DATABASE MATCH
    # -----------------------------
    if intent:
        try:
            conn = get_db_connection()
            cur = conn.cursor()

            cur.execute(
                "SELECT metadata FROM documents WHERE metadata->>'intent' = %s",
                (intent,)
            )

            result = cur.fetchone()

            if result:
                metadata = result[0]

                return f"\n{metadata['title']}\n\nSteps:\n{metadata['steps']}"

        except Exception as e:
            print("DB error:", e)

    # -----------------------------
    # GPT FALLBACK (THIS IS THE FIX)
    # -----------------------------
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a NetSuite expert. Provide clear step-by-step instructions."
                },
                {"role": "user", "content": message}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        print("GPT error:", e)
        return "Sorry, GPT fallback failed."
