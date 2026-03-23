import psycopg2
import ollama
import json

# =========================
# CONFIG
# =========================
DB_CONFIG = {
    "host": "localhost",
    "port": 5433,          # Docker container port
    "database": "postgres",
    "user": "postgres",
    "password": ""         # trust auth = no password
}

EMBED_MODEL = "nomic-embed-text"

# =========================
# CONNECT
# =========================
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# =========================
# EMBEDDING FUNCTION
# =========================
def embed_text(text):
    response = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )
    return response["embedding"]

# =========================
# INSERT FUNCTION
# =========================
def insert_document(content, metadata, source=None):
    embedding = embed_text(content)

    cur.execute("""
        INSERT INTO documents (content, embedding, metadata, source)
        VALUES (%s, %s, %s, %s)
    """, (content, embedding, json.dumps(metadata), source))

    conn.commit()

# =========================
# TEST INGEST
# =========================
def ingest_sample():
    sample = {
        "content": "Create a customer invoice in NetSuite",
        "metadata": {
            "intent": "ar__invoice__create_an_invoice",
            "title": "Create Invoice",
            "navigation": "Transactions > Sales > Create Invoices",
            "steps": "1. Go to Transactions > Sales > Create Invoices\n2. Select customer\n3. Enter details\n4. Save"
        },
        "source": "manual_test"
    }

    insert_document(
        content=sample["content"],
        metadata=sample["metadata"],
        source=sample["source"]
    )

    print("✅ Inserted sample document")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    ingest_sample()
