import os
import psycopg2
from openai import OpenAI

# -----------------------------
# CONFIG
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

client = OpenAI(api_key=OPENAI_API_KEY)


# -----------------------------
# DATABASE CONNECTION
# -----------------------------
def get_db_connection():
    return psycopg2.connect(DATABASE_URL)


# -----------------------------
# EMBEDDING MATCH FUNCTION
# -----------------------------
def find_best_match(message):
    try:
        # Create embedding for user input
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=message
        ).data[0].embedding

        conn = get_db_connection()
        cur = conn.cursor()

        # Find closest match in DB
        cur.execute("""
            SELECT metadata,
                   embedding <-> %s AS distance
            FROM documents
            ORDER BY distance ASC
            LIMIT 1
        """, (embedding,))

        result = cur.fetchone()

        if result:
            metadata = result[0]
            distance = result[1]

            print(f"DEBUG: Embedding distance → {distance}")

            # Threshold tuning (0.3–0.6 typical)
            if distance < 0.5:
                return metadata

        return None

    except Exception as e:
        print("Embedding error:", e)
        return None


# -----------------------------
# MAIN HANDLER
# -----------------------------
def handle_user_message(message):

    # 🔥 STEP 1 — EMBEDDING MATCH
    match = find_best_match(message)

    if match:
        print("DEBUG: Using embedding match")

        return f"""
{match.get('title', 'Workflow')}

Steps:
{match.get('steps', '')}
"""

    # 🔥 STEP 2 — GPT FALLBACK
    try:
        print("DEBUG: Using GPT fallback")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a NetSuite expert. Provide clear, step-by-step instructions."
                },
                {
                    "role": "user",
                    "content": message
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        print("GPT error:", e)
        return "Sorry, I couldn't process your request."


# -----------------------------
# OPTIONAL: HEALTH CHECK
# -----------------------------
def test_connection():
    try:
        conn = get_db_connection()
        print("✅ DB connected")
    except Exception as e:
        print("❌ DB connection failed:", e)
