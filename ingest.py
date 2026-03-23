import os
import re
import json
import psycopg2
import ollama
from concurrent.futures import ProcessPoolExecutor
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract

# =========================
# CONFIG
# =========================
OUTPUT_DIR = "knowledge_output"
TESSERACT_CONFIG = r'--oem 3 --psm 6'

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "postgres",
    "user": "postgres",
    "password": ""
}

EMBED_MODEL = "nomic-embed-text"

# =========================
# DB CONNECTION
# =========================
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# =========================
# EMBEDDING
# =========================
def embed_text(text):
    return ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )["embedding"]

# =========================
# INSERT INTO DB
# =========================
def insert_document(content, metadata):
    embedding = embed_text(content)

    cur.execute("""
        INSERT INTO documents (content, embedding, metadata, source)
        VALUES (%s, %s, %s, %s)
    """, (content, embedding, json.dumps(metadata), metadata.get("source_file")))

    conn.commit()

# =========================
# EXTRACTION
# =========================
def extract_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        pages = [p.extract_text() or "" for p in reader.pages]
        if len(" ".join(pages)) < 100:
            raise Exception("Low text, fallback to OCR")
        return pages
    except:
        images = convert_from_path(pdf_path, dpi=300)
        return [pytesseract.image_to_string(img, config=TESSERACT_CONFIG) for img in images]

# =========================
# PARSE INTO PROCEDURES
# =========================
def extract_procedures(pages):
    procedures = []
    current = None

    for page_num, text in enumerate(pages, start=1):
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        for line in lines:

            if re.match(r'^(Invoice|Create|Accept|Record|Process)', line):
                if current:
                    procedures.append(current)

                current = {
                    "title": line,
                    "steps": [],
                    "fields": [],
                    "navigation": None,
                    "scenario": None,
                    "page": page_num
                }
                continue

            if not current:
                continue

            if "Scenario" in line:
                current["scenario"] = line
                continue

            if "Navigate to" in line:
                current["navigation"] = line
                continue

            if re.match(r'^\d+[\.\)]', line) or line.lower().startswith(("click", "enter", "verify", "check")):
                current["steps"].append(line)
                continue

            if re.match(r'^[A-Za-z #/]+\s{2,}.+', line):
                parts = re.split(r'\s{2,}', line)
                if len(parts) >= 2:
                    current["fields"].append({
                        "name": parts[0],
                        "value": parts[1]
                    })

    if current:
        procedures.append(current)

    return procedures

# =========================
# BUILD CHUNKS (IMPORTANT)
# =========================
def build_chunks(procedure):
    chunks = []

    # Main workflow chunk
    content = f"""
Title: {procedure['title']}
Navigation: {procedure.get('navigation', '')}
Scenario: {procedure.get('scenario', '')}
Steps:
{chr(10).join(procedure['steps'])}
"""

    chunks.append((content.strip(), procedure))

    # Optional: field chunks (advanced retrieval)
    for field in procedure.get("fields", []):
        field_text = f"{procedure['title']} - Field: {field['name']} = {field['value']}"
        chunks.append((field_text, procedure))

    return chunks

# =========================
# PROCESS PDF
# =========================
def process_pdf(pdf_path):
    filename = os.path.basename(pdf_path)

    print(f"Processing: {filename}")

    pages = extract_text(pdf_path)
    procedures = extract_procedures(pages)

    for p in procedures:
        p["source_file"] = filename
        p["keywords"] = list(set(
            p["title"].lower().split() +
            (p["scenario"] or "").lower().split()
        ))

        chunks = build_chunks(p)

        for content, metadata in chunks:
            insert_document(content, metadata)

    print(f"✅ Stored in DB: {filename}")

# =========================
# MAIN
# =========================
def main():
    pdfs = [f for f in os.listdir() if f.lower().endswith(".pdf")]

    for pdf in pdfs:
        process_pdf(pdf)

if __name__ == "__main__":
    main()
