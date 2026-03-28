import os
import psycopg2
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

client = OpenAI(api_key=OPENAI_API_KEY)


def get_db_connection():
    return psycopg2.connect(DATABASE_URL)


def clean_step_text(step):
    if not step:
        return ""

    step = str(step).strip()

    # Remove leading numeric prefixes like "1. "
    if step and step[0].isdigit() and "." in step[:5]:
        step = step.split(".", 1)[-1].strip()

    # Light cleanup of extra whitespace
    step = " ".join(step.split())

    return step


def find_best_match(message):
    try:
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=message
        ).data[0].embedding

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                content,
                metadata,
                embedding <-> %s::vector AS distance
            FROM documents
            ORDER BY distance ASC
            LIMIT 3
            """,
            (embedding,)
        )

        results = cur.fetchall()

        cur.close()
        conn.close()

        if not results:
            return None

        best_content, best_metadata, best_distance = results[0]

        print(f"DEBUG: Best embedding distance -> {best_distance}")

        # Slightly more forgiving than the old 0.5 threshold
        if best_distance < 1.0:
            return {
                "content": best_content,
                "metadata": best_metadata,
                "distance": best_distance
            }

        return None

    except Exception as e:
        print("Embedding error:", e)
        return None


def format_workflow(match):
    metadata = match.get("metadata", {}) or {}

    title = metadata.get("title") or "NetSuite Procedure"
    module = metadata.get("module") or ""
    navigation = metadata.get("navigation") or ""
    steps = metadata.get("steps") or []

    # Handle either list or string just in case
    if isinstance(steps, str):
        raw_steps = [s.strip() for s in steps.split("\n") if s.strip()]
    elif isinstance(steps, list):
        raw_steps = [str(s).strip() for s in steps if str(s).strip()]
    else:
        raw_steps = []

    cleaned_steps = []
    for step in raw_steps:
        cleaned = clean_step_text(step)
        if cleaned:
            cleaned_steps.append(cleaned)

    result = [title]

    if module:
        result.append(f"Module: {module}")

    if navigation:
        result.append(f"Navigation: {navigation}")

    result.append("")
    result.append("Steps:")

    for i, step in enumerate(cleaned_steps[:5], start=1):
        result.append(f"{i}. {step}")

    if not cleaned_steps:
        result.append("1. No detailed steps were found in the matched procedure.")

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
                    "content": (
                        "You are a NetSuite assistant. "
                        "If the question is about NetSuite, answer with short, direct action steps. "
                        "If the exact internal procedure is not available, give a safe general answer. "
                        "Keep the answer concise."
                    )
                },
                {
                    "role": "user",
                    "content": message
                }
            ]
        )

        raw = response.choices[0].message.content.strip()

        lines = [line.strip() for line in raw.split("\n") if line.strip()]

        title = message.strip().capitalize()
        result = [title, "", "Steps:"]

        step_num = 1
        for line in lines:
            cleaned = clean_step_text(line)
            if not cleaned:
                continue
            result.append(f"{step_num}. {cleaned}")
            step_num += 1
            if step_num > 5:
                break

        if step_num == 1:
            result.append("1. No fallback steps were generated.")

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
