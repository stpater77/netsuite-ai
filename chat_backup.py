import chromadb
import ollama

CHROMA_COLLECTION = "netsuite_knowledge"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"


# ---------- CONNECT ----------
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(CHROMA_COLLECTION)


# ---------- INTENT MAP ----------
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
    ]
}


def detect_intent(query):
    query = query.lower()

    for intent, keywords in INTENT_MAP.items():
        for keyword in keywords:
            if keyword in query:
                return intent

    return None


# ---------- EMBED ----------
def embed_query(query):
    return ollama.embeddings(
        model=EMBED_MODEL,
        prompt=query
    )["embedding"]


# ---------- SEARCH ----------
def search(query):
    intent = detect_intent(query)

    # 🔥 FIX: MANUAL INTENT MATCH (100% reliable)
    if intent:
        print(f"DEBUG: Intent matched → {intent}")

        all_data = collection.get()

        for meta in all_data.get("metadatas", []):
            if meta.get("intent") == intent:
                return meta

    # 🔁 FALLBACK: EMBEDDING SEARCH
    embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=2
    )

    print("\nDEBUG (fallback):", results)

    if not results["metadatas"] or not results["metadatas"][0]:
        return None

    return results["metadatas"][0][0]


# ---------- PROMPT ----------
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
Title: {meta['title']}
Navigation: {meta['navigation']}
Steps: {meta['steps']}

ANSWER:
"""


# ---------- CHAT ----------
def chat():
    print("\n🚀 Ready (llama3)\n")

    while True:
        q = input("You: ")

        if q.lower() == "exit":
            break

        meta = search(q)

        if not meta:
            print("\nAI: No workflow found\n")
            continue

        prompt = build_prompt(q, meta)

        response = ollama.generate(
            model=LLM_MODEL,
            prompt=prompt
        )

        print("\nAI:\n")
        print(response["response"])
        print("\n----------------------\n")


if __name__ == "__main__":
    chat()
