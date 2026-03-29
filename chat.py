import os
import re
import psycopg2
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

client = OpenAI(api_key=OPENAI_API_KEY)

EMBED_MODEL = "text-embedding-3-small"
FALLBACK_MODEL = "gpt-4o-mini"

# Keep the existing proven exact-match threshold
STRICT_MATCH_DISTANCE = 1.0

# Slightly looser threshold only for overview synthesis mode
OVERVIEW_MATCH_DISTANCE = 1.15

# Pull a few candidates so overview prompts can synthesize top matches
CANDIDATE_LIMIT = 6

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with",
    "how", "what", "is", "are", "do", "does", "i", "me", "my", "we", "our",
    "you", "your", "this", "that", "these", "those", "it", "its", "as",
    "at", "by", "from", "be", "been", "being", "into", "about", "through",
    "main", "overview", "high", "level", "walk", "involved", "fit", "together",
    "should", "know", "give"
}


def get_db_connection():
    return psycopg2.connect(DATABASE_URL)


def clean_step_text(step):
    if not step:
        return ""

    step = str(step).strip()
    step = step.replace("**", "")

    if step and step[0].isdigit() and "." in step[:5]:
        step = step.split(".", 1)[-1].strip()

    step = re.sub(r"NetSuite:.*?\|\s*.*$", "", step, flags=re.IGNORECASE).strip()
    step = re.sub(r"ERP:\s*Fundamentals.*?\|\s*.*$", "", step, flags=re.IGNORECASE).strip()
    step = re.sub(r"ERP Fundamentals.*?\|\s*.*$", "", step, flags=re.IGNORECASE).strip()
    step = re.sub(r"SuiteFlow:.*?\|\s*.*$", "", step, flags=re.IGNORECASE).strip()
    step = re.sub(r"\|\s*\d+\s*$", "", step).strip()

    step = " ".join(step.split())
    step = step.strip(" -|")
    return step


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def is_overview_prompt(message):
    text = message.lower()

    overview_signals = [
        "overview",
        "what is involved in",
        "walk me through the main",
        "main processes",
        "main tasks",
        "main responsibilities",
        "high-level",
        "high level",
        "what should i know",
        "capabilities",
        "fit together",
        "main areas",
        "consultant should know",
        "give me an overview",
    ]

    return any(signal in text for signal in overview_signals)


def get_embedding(text):
    return client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    ).data[0].embedding


def fetch_candidate_matches(message, limit=CANDIDATE_LIMIT):
    try:
        embedding = get_embedding(message)

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
            LIMIT %s
            """,
            (embedding, limit)
        )

        rows = cur.fetchall()
        cur.close()
        conn.close()

        matches = []
        for content, metadata, distance in rows:
            matches.append({
                "content": content,
                "metadata": metadata or {},
                "distance": float(distance)
            })

        if matches:
            print("DEBUG: Top candidate distances ->", [round(m["distance"], 4) for m in matches])

        return matches

    except Exception as e:
        print("Embedding error:", e)
        return []


def get_match_text_blob(match):
    metadata = match.get("metadata", {}) or {}
    parts = [
        normalize_text(metadata.get("title")),
        normalize_text(metadata.get("module")),
        normalize_text(metadata.get("section")),
        normalize_text(metadata.get("navigation")),
        normalize_text(metadata.get("intent")),
        normalize_text(metadata.get("scenario")),
        normalize_text(match.get("content", "")),
    ]

    keywords = metadata.get("keywords") or []
    if isinstance(keywords, list):
        parts.extend([normalize_text(k) for k in keywords])

    return " ".join([p for p in parts if p]).lower()


def rerank_matches(message, matches):
    query_tokens = set(tokenize(message))
    if not query_tokens:
        return matches

    reranked = []
    for match in matches:
        blob = get_match_text_blob(match)
        overlap = sum(1 for token in query_tokens if token in blob)

        metadata = match.get("metadata", {}) or {}
        title = normalize_text(metadata.get("title")).lower()
        module = normalize_text(metadata.get("module")).lower()
        navigation = normalize_text(metadata.get("navigation")).lower()

        exact_phrase_bonus = 0
        if "accounts receivable" in message.lower() and "accounts receivable" in module:
            exact_phrase_bonus += 3
        if "manufacturing" in message.lower() and "manufacturing" in (title + " " + module):
            exact_phrase_bonus += 3
        if "administrator" in message.lower() and "administrator" in (title + " " + module):
            exact_phrase_bonus += 3
        if "suiteanalytics" in message.lower() and "suiteanalytics" in (title + " " + module):
            exact_phrase_bonus += 3
        if "report" in message.lower() and "report" in (title + " " + navigation):
            exact_phrase_bonus += 1
        if "search" in message.lower() and "search" in (title + " " + navigation):
            exact_phrase_bonus += 1

        # Lower is better. Distance remains primary, overlap improves ranking.
        blended_score = match["distance"] - (overlap * 0.03) - (exact_phrase_bonus * 0.05)

        enriched = dict(match)
        enriched["keyword_overlap"] = overlap
        enriched["blended_score"] = blended_score
        reranked.append(enriched)

    reranked.sort(key=lambda x: (x["blended_score"], x["distance"]))
    print("DEBUG: Reranked titles ->", [
        (normalize_text(m.get("metadata", {}).get("title", "Untitled")), round(m["distance"], 4), m.get("keyword_overlap", 0))
        for m in reranked[:5]
    ])
    return reranked


def dedupe_and_select_overview_matches(matches, max_matches=3):
    selected = []
    seen_titles = set()

    for match in matches:
        if match["distance"] > OVERVIEW_MATCH_DISTANCE:
            continue

        metadata = match.get("metadata", {}) or {}
        title = normalize_text(metadata.get("title")) or "Untitled"
        title_key = title.lower()

        if title_key in seen_titles:
            continue

        selected.append(match)
        seen_titles.add(title_key)

        if len(selected) >= max_matches:
            break

    return selected


def format_workflow(match):
    metadata = match.get("metadata", {}) or {}

    title = metadata.get("title") or "NetSuite Procedure"
    module = metadata.get("module") or ""
    navigation = metadata.get("navigation") or ""
    steps = metadata.get("steps") or []

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


def build_match_context(match):
    metadata = match.get("metadata", {}) or {}

    title = normalize_text(metadata.get("title")) or "Untitled"
    module = normalize_text(metadata.get("module"))
    navigation = normalize_text(metadata.get("navigation"))
    section = normalize_text(metadata.get("section"))
    scenario = normalize_text(metadata.get("scenario"))

    steps = metadata.get("steps") or []
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
        if len(cleaned_steps) >= 3:
            break

    lines = [f"Title: {title}"]
    if module:
        lines.append(f"Module: {module}")
    if section:
        lines.append(f"Section: {section}")
    if navigation:
        lines.append(f"Navigation: {navigation}")
    if scenario:
        lines.append(f"Scenario: {scenario}")

    if cleaned_steps:
        lines.append("Representative Steps:")
        for i, step in enumerate(cleaned_steps, start=1):
            lines.append(f"{i}. {step}")

    return "\n".join(lines)


def deterministic_overview(message, matches):
    result = [message.strip(), "", "Summary:"]

    for idx, match in enumerate(matches[:3], start=1):
        metadata = match.get("metadata", {}) or {}
        title = normalize_text(metadata.get("title")) or "NetSuite Procedure"
        module = normalize_text(metadata.get("module"))
        navigation = normalize_text(metadata.get("navigation"))

        line = f"{idx}. {title}"
        extras = []
        if module:
            extras.append(f"Module: {module}")
        if navigation:
            extras.append(f"Navigation: {navigation}")
        if extras:
            line += " — " + " | ".join(extras)

        result.append(line)

    result.append("")
    result.append("Use these procedures as the starting points for this topic.")
    return "\n".join(result)


def synthesize_overview(message, matches):
    try:
        print("DEBUG: Using overview synthesis")

        context_blocks = []
        for i, match in enumerate(matches[:3], start=1):
            context_blocks.append(f"Procedure {i}\n{build_match_context(match)}")

        context_text = "\n\n".join(context_blocks)

        response = client.chat.completions.create(
            model=FALLBACK_MODEL,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a NetSuite consultant assistant. "
                        "The user asked a broad NetSuite question. "
                        "You must answer ONLY from the retrieved procedure context provided. "
                        "Do not invent features or workflows not supported by the retrieved context. "
                        "Write a concise consultant-style summary. "
                        "Start with a short title line. "
                        "Then include 'Main Areas:' and list 3 to 5 bullets. "
                        "Then include 'Representative Procedures:' and list up to 3 concrete procedures or navigation paths from the context. "
                        "If the context is too thin or obviously not relevant, reply exactly with: INSUFFICIENT_CONTEXT"
                    )
                },
                {
                    "role": "user",
                    "content": f"Question:\n{message}\n\nRetrieved NetSuite procedure context:\n\n{context_text}"
                }
            ]
        )

        text = response.choices[0].message.content.strip()

        if not text or text == "INSUFFICIENT_CONTEXT":
            return None

        return text

    except Exception as e:
        print("Overview synthesis error:", e)
        return None


def gpt_fallback(message):
    try:
        print("DEBUG: Using GPT fallback")

        response = client.chat.completions.create(
            model=FALLBACK_MODEL,
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
    matches = fetch_candidate_matches(message, limit=CANDIDATE_LIMIT)

    if not matches:
        return gpt_fallback(message)

    ranked_matches = rerank_matches(message, matches)
    best_match = ranked_matches[0]

    # New middle path: broad/overview prompts synthesize top 2-3 matches
    if is_overview_prompt(message):
        overview_matches = dedupe_and_select_overview_matches(ranked_matches, max_matches=3)

        print("DEBUG: Overview mode ->", [
            (normalize_text(m.get("metadata", {}).get("title", "Untitled")), round(m["distance"], 4))
            for m in overview_matches
        ])

        if len(overview_matches) >= 2:
            synthesized = synthesize_overview(message, overview_matches)
            if synthesized:
                return synthesized

            return deterministic_overview(message, overview_matches)

        # If only one strong match exists, fall through to exact match formatting
        if best_match["distance"] < STRICT_MATCH_DISTANCE:
            print("DEBUG: Overview mode fell back to single retrieval match")
            return format_workflow(best_match)

        return gpt_fallback(message)

    # Existing proven path for narrow procedure-style prompts
    if best_match["distance"] < STRICT_MATCH_DISTANCE:
        print("DEBUG: Using embedding match")
        return format_workflow(best_match)

    return gpt_fallback(message)


def test_connection():
    try:
        conn = get_db_connection()
        conn.close()
        print("DB connected")
    except Exception as e:
        print("DB connection failed:", e)
