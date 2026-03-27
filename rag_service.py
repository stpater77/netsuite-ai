import os
import requests
import psycopg2

DB_URL = os.getenv("DATABASE_URL")

OLLAMA_URL = "http://localhost:11434/api"
OLLAMA_MODEL = "gpt-oss:120b"
EMBED_MODEL = "nomic-embed-text"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_conn():
    return psycopg2.connect(DB_URL)


def embed(text):
    res = requests.post(
        f"{OLLAMA_URL}/embeddings",
        json={
            "model": EMBED_MODEL,
            "prompt": text
        }
    )
    res.raise_for_status()
    return res.json()["embedding"]


def detect_section(query):
    q = query.lower()

    if any(x in q for x in ["invoice", "payment", "cash"]):
        return "order_to_cash"

    if any(x in q for x in ["return", "credit", "refund"]):
        return "return_to_credit"

    if any(x in q for x in ["pricing", "price", "rate"]):
        return "set_up_pricing"

    return None


def search_procedures(query_embedding, section=None):
    conn = get_conn()
    cur = conn.cursor()

    if section:
        cur.execute("""
            SELECT content, metadata,
                   1 - (embedding <=> %s) AS similarity
            FROM netsuite_procedures
            WHERE section = %s
            ORDER BY embedding <=> %s
            LIMIT 3
        """, (query_embedding, section, query_embedding))
    else:
        cur.execute("""
            SELECT content, metadata,
                   1 - (embedding <=> %s) AS similarity
            FROM netsuite_procedures
            ORDER BY embedding <=> %s
            LIMIT 3
        """, (query_embedding, query_embedding))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows


def build_context(rows):
    context = ""

    for i, row in enumerate(rows):
        content, metadata, similarity = row

        context += f"\n--- Procedure {i+1} (score={round(similarity,2)}) ---\n"
        context += content

    return context


def call_ollama(query, context):
    prompt = f"""
You are a NetSuite AI Consultant.

Rules:
- Provide step-by-step instructions
- Use numbered steps
- Include exact navigation paths
- Include field values when available
- Do NOT guess
- Combine procedures logically if needed

User question:
{query}

Relevant procedures:
{context}

Answer:
"""

    res = requests.post(
        f"{OLLAMA_URL}/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    res.raise_for_status()
    return res.json()["response"]


def call_openai(query):
    res = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": query}
            ]
        }
    )

    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]


def handle_query(query):
    return f"RAG WORKING: {query}"

    try:
        section = detect_section(query)

        query_embedding = embed(query)

        results = search_procedures(query_embedding, section)

        if not results:
            return call_openai(query)

        top_score = results[0][2]

        if top_score < 0.70:
            return call_openai(query)

        context = build_context(results)

        return call_ollama(query, context)

    except Exception as e:
        print("ERROR:", str(e))
        return call_openai(query)
