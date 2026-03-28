import json
import os
import sys
from typing import Any, Dict, List

import psycopg2
from psycopg2.extras import Json
from openai import OpenAI

DATASET_PATH = "knowledge/netsuite_master_dataset.json"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
TRUNCATE_FIRST = os.getenv("TRUNCATE_FIRST", "false").lower() == "true"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        print(f"ERROR: missing environment variable: {name}")
        sys.exit(1)
    return value


def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    procedures = data.get("procedures", [])
    if not procedures:
        print("ERROR: no procedures found in master dataset")
        sys.exit(1)

    return procedures


def make_content(proc: Dict[str, Any]) -> str:
    if proc.get("embedding_text"):
        return proc["embedding_text"]

    parts = [
        f"Title: {proc.get('title', '')}",
        f"Intent: {proc.get('intent', '')}",
        f"Module: {proc.get('module', '')}",
        f"Section: {proc.get('section', '')}",
        f"Navigation: {proc.get('navigation', '')}",
        f"Scenario: {proc.get('scenario', '')}",
        "Fields:",
    ]

    fields = proc.get("fields", [])
    if fields:
        for field in fields:
            parts.append(f"- {field}")
    else:
        parts.append("")

    parts.append("Steps:")
    for step in proc.get("steps", []):
        parts.append(step)

    keywords = proc.get("keywords", [])
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")

    return "\n".join(parts).strip()


def main() -> None:
    database_url = require_env("DATABASE_URL")
    require_env("OPENAI_API_KEY")

    procedures = load_dataset(DATASET_PATH)
    client = OpenAI()

    conn = psycopg2.connect(database_url)
    conn.autocommit = False
    cur = conn.cursor()

    print("Ensuring pgvector and documents table exist...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding vector({EMBED_DIM}),
            metadata JSONB
        )
        """
    )

    if TRUNCATE_FIRST:
        print("Truncating documents table...")
        cur.execute("TRUNCATE TABLE documents RESTART IDENTITY")

    processed = 0

    for idx, proc in enumerate(procedures, start=1):
        content = make_content(proc)

        res = client.embeddings.create(model=EMBED_MODEL, input=content)
        embedding = res.data[0].embedding

        metadata = dict(proc)
        metadata["ingested_from"] = DATASET_PATH
        metadata["external_id"] = proc.get("id")

        cur.execute(
            """
            INSERT INTO documents (content, embedding, metadata)
            VALUES (%s, %s::vector, %s)
            """,
            (content, json.dumps(embedding), Json(metadata)),
        )

        processed += 1

        if idx % 10 == 0:
            conn.commit()
            print(f"Processed {idx}/{len(procedures)} procedures...")

    conn.commit()

    print("\nDone.")
    print(f"Procedures processed: {processed}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
