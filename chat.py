import os
import psycopg2
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

client = OpenAI(api_key=OPENAI_API_KEY)


def get_db_connection():
    return psycopg2.connect(DATABASE_URL)


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

        cur.close()
        conn.close()

        if result:
            metadata = result[0]
            distance = result[1]

            print(f"DEBUG: Embedding distance -> {distance}")

            if distance < 0.5:
                return metadata

        return None

    except Exception as e:
        print("Embedding error:", e)
        return None


def format_workflow(metadata):
    title = metadata.get("title", "Workflow")
    steps = metadata.get("steps", "")

    lines = steps.split("\n")
    cleaned_steps = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if "step" in line.lower():
            continue

        if line and line[0].isdigit():
            line = line.split(".", 1)[-1].strip()

        cleaned_steps.append(line)

    result = [title, "", "Steps:"]

    for i, step in enumerate(cleaned_steps[:5], start=1):
        result.append(f"{i}. {step}")

    return "\n".join(result)


def gpt_fallback(message):
    try:
        print("DEBUG: Using GPT fallback")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": "You are a NetSuite assistant. Return only short action steps. No explanations."
                },
                {
                    "role": "user",
                    "content": message
                }
            ]
        )

        raw = response.choices[0].message.content.strip()

        lines = raw.split("\n")
        cleaned_steps = []

        for line in lines:
            line = line.strip()

            if not line:
                continue

            if "step" in line.lower():
                continue

            if line and line[0].isdigit():
                line = line.split(".", 1)[-1].strip()

            cleaned_steps.append(line)

        title = message.strip().capitalize()

        result = [title, "", "Steps:"]

        for i, step in enumerate(cleaned_steps[:5], start=1):
            result.append(f"{i}. {step}")

        return "\n".join(result)

    except Exception as e:
        print("GPT error:", e)
        return "Error processing request."


def handle_user_message(message):
    match = find_best_match(message)

    if match:
        print("DEBUG: Using embedding match")
        return format_workflow(match)

    return gpt_fallback(message)


def test_connection():
    try:
        conn = get_db_connection()
        conn.close()
        print("DB connected")
    except Exception as e:
        print("DB connection failed:", e)
