import psycopg2
import os

# Try importing ollama (only works locally)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except:
    OLLAMA_AVAILABLE = False

# =========================
# DATABASE CONFIG (DUAL)
# =========================
if os.getenv("PGHOST"):
    # Railway config
    DB_CONFIG = {
        "host": os.getenv("PGHOST"),
        "port": os.getenv("PGPORT"),
        "database": os.getenv("PGDATABASE"),
        "user": os.getenv("PGUSER"),
        "password": os.getenv("PGPASSWORD")
    }
else:
    # Local config
    DB_CONFIG = {
        "host": "localhost",
        "port": 5433,
        "database": "postgres",
        "user": "postgres",
        "password": ""
    }

LLM_MODEL = "llama3"

# =========================
# CONNECT
# =========================
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# =========================
# INTENT MAP
# =========================
INTENT_MAP = {
    "ar__payment__accept_customer_payments": [
        "accept payment",
        "receive payment",
        "customer payment",
        "apply payment"
    ],
    "ar__invoice__create_an_invoice": [
        "create invoice",
        "make invoice",
        "invoice customer"
    ],
    "ar__invoice__invoice_a_sales_order": [
        "invoice sales order",
        "bill sales order"
    ],
    "ar__credit__create_credit_memo": [
        "credit memo",
        "create credit memo",
        "issue credit",
        "customer credit"
    ]
}

# =========================
# INTENT DETECTION
# =========================
def detect_intent(query):
    query = query.lower()

    for intent, keywords in INTENT_MAP.items():
        for keyword in keywords:
            words = keyword.split()
            if all(word in query for word in words):
                return intent

    return None

# =========================
# SEARCH
# =========================
def search(query):
    intent = detect_intent(query)

    print(f"DEBUG: Detected intent → {intent}")

    try:
        if intent:
            print(f"DEBUG: Intent matched → {intent}")

            cur.execute("""
                SELECT metadata
                FROM documents
                WHERE metadata->>'intent' = %s
                LIMIT 1
            """, (intent,))

            result = cur.fetchone()

            print(f"DEBUG: DB result → {result}")

            if result:
                return result[0]

        return None

    except Exception as e:
        print(f"DEBUG: DB error → {e}")
        conn.rollback()
        return None

# =========================
# PROMPT BUILDER
# =========================
def build_prompt(question, meta):
    return f"""
You are a NetSuite AI consultant.

Rules:
- Only use provided workflow
- Be concise
- Use numbered steps

QUESTION:
{question}

WORKFLOW:
Title: {meta.get('title')}
Navigation: {meta.get('navigation')}
Steps: {meta.get('steps')}

ANSWER:
"""

# =========================
# MAIN HANDLER
# =========================
def handle_user_message(q: str) -> str:
    meta = search(q)

    if not meta:
        return "No workflow found"

    prompt = build_prompt(q, meta)

    # =========================
    # TRY OLLAMA (LOCAL ONLY)
    # =========================
    if OLLAMA_AVAILABLE:
        try:
            response = ollama.generate(
                model=LLM_MODEL,
                prompt=prompt
            )
            return response["response"]
        except Exception as e:
            print(f"DEBUG: Ollama error → {e}")

    # =========================
    # FALLBACK (CLOUD SAFE)
    # =========================
    return f"""
{meta.get('title')}

Steps:
{meta.get('steps')}
"""
