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
# EMBEDDING MATCH
# -----------------------------
def find_best_match(message):
    try:
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=message
        ).data[0].embedding

        conn = get_db_connection()
        cur = conn.cursor()

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

            if distance < 0.5:
                return metadata

        return None

    except Exception as e:
        print("Embedding error:", e)
        return None


# -----------------------------
# FORMAT WORKFLOW OUTPUT
# -----------------------------
def format_workflow(metadata):
    title = metadata.get("title", "Workflow")
    steps = metadata.get("steps", "")

    return f"""{title}

Steps:
{steps}
"""


# -----------------------------
# GPT FALLBACK (STRICT + CLEAN)
# -----------------------------
def gpt_fallback(message):
    try:
        print("DEBUG: Using GPT fallback")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": """You are a NetSuite assistant.

You MUST follow this exact format:

Title

Steps:
1. ...
2. ...
3. ...

STRICT RULES:
- NO explanations
- NO paragraphs
- NO tips
- NO extra sections
- ONLY return steps
- Maximum 5 steps
- Each step must be ONE line
- Keep it short and direct

If you break format, your answer is wrong.
"""
                },
                {
                    "role": "user",
                    "content": message
                }
            ]
        )

        raw = response.choices[0].message.content.strip()

        # -----------------------------
        # HARD CLEANUP (FORCE FORMAT)
        # -----------------------------
        lines = raw.split("\n")

        cleaned = []
        step_count = 1

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # First line = title
            if len(cleaned) == 0:
                cleaned.append(line)
                cleaned.append("\nSteps:")
                continue

            # Force numbered steps
            if step_count <= 5:
                cleaned.append(f"{step_count}. {line}")
                step_count += 1

        return "\n".join(cleaned)

    except Exception as e:
        print("GPT error:", e)
        return "Error processing request."


# -----------------------------
# MAIN HANDLER
# -----------------------------
def handle_user_message(message):

    # 🔥 STEP 1 — EMBEDDING MATCH
    match = find_best_match(message)

    if match:
        print("DEBUG: Using embedding match")
        return format_workflow(match)

    # 🔥 STEP 2 — GPT FALLBACK
    return gpt_fallback(message)


# -----------------------------
# OPTIONAL TEST FUNCTION
# -----------------------------
def test_connection():
    try:
        conn = get_db_connection()
        print("✅ DB connected")
    except Exception as e:
        print("❌ DB connection failed:", e)
