import os
import psycopg2
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATABASE_URL = os.getenv("DATABASE_URL")


conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

cur.execute("SELECT id, metadata FROM documents")
rows = cur.fetchall()

for row in rows:
    doc_id = row[0]
    metadata = row[1]

    text = metadata["steps"]

    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

    cur.execute(
        "UPDATE documents SET embedding = %s WHERE id = %s",
        (embedding, doc_id)
    )

conn.commit()
print("✅ Embeddings generated!")
