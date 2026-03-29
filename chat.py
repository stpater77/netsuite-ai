import os
import re
import psycopg2
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

client = OpenAI(api_key=OPENAI_API_KEY)

EMBED_MODEL = "text-embedding-3-small"
FALLBACK_MODEL = "gpt-4o-mini"

STRICT_MATCH_DISTANCE = 1.0
OVERVIEW_MATCH_DISTANCE = 1.05
CANDIDATE_LIMIT = 12

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with",
    "how", "what", "is", "are", "do", "does", "i", "me", "my", "we", "our",
    "you", "your", "this", "that", "these", "those", "it", "its", "as",
    "at", "by", "from", "be", "been", "being", "into", "about", "through",
    "main", "overview", "high", "level", "walk", "involved", "fit", "together",
    "should", "know", "give", "tell", "explain", "help", "understand"
}

TOPIC_KEYWORDS = {
    "accounts_receivable": [
        "accounts receivable", "receivable", "invoice", "invoices", "billing",
        "payment", "payments", "customer payments", "cash sale", "ar"
    ],
    "manufacturing": [
        "manufacturing", "work order", "work orders", "assembly",
        "bill of materials", "bom", "build", "production", "components"
    ],
    "administrator": [
        "administrator", "admin", "role", "roles", "permissions",
        "general preferences", "employees", "setup", "company"
    ],
    "suiteanalytics": [
        "suiteanalytics", "report", "reports", "reporting", "saved search",
        "saved searches", "search", "searches", "dashboard", "dashboards"
    ],
    "financial_management": [
        "financial management", "financial", "budget", "forecast",
        "cash flow", "finance"
    ]
}

OFF_TOPIC_PATTERNS = {
    "manufacturing": [
        "general preferences",
        "inline editing",
        "saved search",
        "accept customer payments",
        "accounts receivable"
    ],
    "accounts_receivable": [
        "work order",
        "assembly",
        "bill of materials",
        "manufacturing"
    ],
    "administrator": [
        "accept customer payments",
        "invoice sales orders",
        "work order",
        "assembly"
    ],
    "suiteanalytics": [
        "accept customer payments",
        "invoice sales orders",
        "general preferences",
        "work order",
        "assembly"
    ]
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
        "what are the main"
    ]
    return any(signal in text for signal in overview_signals)


def detect_topic(message):
    text = message.lower()
    scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in text:
                score += 1
        if score:
            scores[topic] = score

    if not scores:
        return None

    return max(scores, key=scores.get)


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

        print("DEBUG: top distances ->", [round(m["distance"], 4) for m in matches[:8]])
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


def topic_alignment_score(topic, blob):
    if not topic:
        return 0

    score = 0
    for kw in TOPIC_KEYWORDS.get(topic, []):
        if kw in blob:
            score += 1
    return score


def off_topic_penalty(topic, blob):
    if not topic:
        return 0

    penalty = 0
    for pattern in OFF_TOPIC_PATTERNS.get(topic, []):
        if pattern in blob:
            penalty += 1
    return penalty


def get_title_and_navigation_text(match):
    metadata = match.get("metadata", {}) or {}
    title = normalize_text(metadata.get("title")).lower()
    navigation = normalize_text(metadata.get("navigation")).lower()
    module = normalize_text(metadata.get("module")).lower()
    return f"{title} {navigation} {module}".strip()


def exact_intent_score(message, match):
    text = message.lower()
    title_nav = get_title_and_navigation_text(match)
    score = 0

    if "invoice" in text and "invoice" in title_nav:
        score += 5
    if "payment" in text and "payment" in title_nav:
        score += 5
    if "customer" in text and "customer" in title_nav:
        score += 2
    if "manufacturing" in text and "manufacturing" in title_nav:
        score += 5
    if "work order" in text and "work order" in title_nav:
        score += 5
    if "administrator" in text and "administrator" in title_nav:
        score += 5
    if "report" in text and "report" in title_nav:
        score += 4
    if "search" in text and "search" in title_nav:
        score += 4
    if "create" in text and "create" in title_nav:
        score += 2
    if "accept" in text and "accept" in title_nav:
        score += 2

    if "create an invoice" in text and "create invoice" in title_nav:
        score += 8
    if "accept customer payments" in text and "accept customer payments" in title_nav:
        score += 8

    return score


def rerank_narrow_matches(message, matches):
    query_tokens = set(tokenize(message))
    ranked = []

    for match in matches:
        blob = get_match_text_blob(match)
        overlap = sum(1 for token in query_tokens if token in blob)
        intent_score = exact_intent_score(message, match)

        narrow_score = (
            match["distance"]
            - (overlap * 0.03)
            - (intent_score * 0.07)
        )

        enriched = dict(match)
        enriched["overlap"] = overlap
        enriched["intent_score"] = intent_score
        enriched["narrow_score"] = narrow_score
        ranked.append(enriched)

    ranked.sort(key=lambda x: (x["narrow_score"], x["distance"]))

    print("DEBUG: narrow rerank ->", [
        {
            "title": normalize_text(m.get("metadata", {}).get("title", "Untitled")),
            "distance": round(m["distance"], 4),
            "intent_score": m.get("intent_score", 0),
            "narrow_score": round(m["narrow_score"], 4)
        }
        for m in ranked[:6]
    ])

    return ranked


def rerank_overview_matches(message, matches):
    query_tokens = set(tokenize(message))
    topic = detect_topic(message)
    ranked = []

    for match in matches:
        blob = get_match_text_blob(match)
        overlap = sum(1 for token in query_tokens if token in blob)
        topic_score = topic_alignment_score(topic, blob)
        penalty = off_topic_penalty(topic, blob)

        overview_score = (
            match["distance"]
            - (overlap * 0.02)
            - (topic_score * 0.10)
            + (penalty * 0.15)
        )

        enriched = dict(match)
        enriched["overlap"] = overlap
        enriched["topic_score"] = topic_score
        enriched["penalty"] = penalty
        enriched["overview_score"] = overview_score
        ranked.append(enriched)

    ranked.sort(key=lambda x: (x["overview_score"], x["distance"]))

    print("DEBUG: overview rerank ->", [
        {
            "title": normalize_text(m.get("metadata", {}).get("title", "Untitled")),
            "distance": round(m["distance"], 4),
            "topic_score": m.get("topic_score", 0),
            "penalty": m.get("penalty", 0),
            "overview_score": round(m["overview_score"], 4)
        }
        for m in ranked[:8]
    ])

    return ranked


def select_overview_matches(message, matches, max_matches=3):
    topic = detect_topic(message)
    selected = []
    seen_titles = set()

    for match in matches:
        metadata = match.get("metadata", {}) or {}
        title = normalize_text(metadata.get("title")) or "Untitled"
        title_key = title.lower()

        if title_key in seen_titles:
            continue
        if match["distance"] > OVERVIEW_MATCH_DISTANCE:
            continue
        if topic and match.get("topic_score", 0) <= 0:
            continue
        if match.get("penalty", 0) > 0:
            continue

        selected.append(match)
        seen_titles.add(title_key)

        if len(selected) >= max_matches:
            break

    return selected


def has_coherent_overview_cluster(message, matches):
    topic = detect_topic(message)
    if len(matches) < 2:
        return False
    if not topic:
        return True

    aligned = [m for m in matches if m.get("topic_score", 0) > 0 and m.get("penalty", 0) == 0]
    return len(aligned) >= 2


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
    if cleaned_steps:
        lines.append("Representative Steps:")
        for i, step in enumerate(cleaned_steps, start=1):
            lines.append(f"{i}. {step}")

    return "\n".join(lines)


def deterministic_overview(message, matches):
    topic_name = message.strip()
    result = [topic_name, "", "Main Areas:"]

    for match in matches[:3]:
        title = normalize_text(match.get("metadata", {}).get("title")) or "NetSuite Procedure"
        result.append(f"- {title}")

    result.append("")
    result.append("Representative Procedures:")

    for i, match in enumerate(matches[:3], start=1):
        metadata = match.get("metadata", {}) or {}
        title = normalize_text(metadata.get("title")) or "NetSuite Procedure"
        navigation = normalize_text(metadata.get("navigation"))
        if navigation:
            result.append(f"{i}. {title} — Navigation: {navigation}")
        else:
            result.append(f"{i}. {title}")

    return "\n".join(result)


def synthesize_overview(message, matches):
    try:
        print("DEBUG: using overview synthesis")

        context_blocks = []
        for i, match in enumerate(matches[:3], start=1):
            context_blocks.append(f"Procedure {i}\n{build_match_context(match)}")

        context_text = "\n\n".join(context_blocks)

        response = client.chat.completions.create(
            model=FALLBACK_MODEL,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a NetSuite consultant assistant. "
                        "Answer ONLY from the provided retrieved procedures. "
                        "Do not invent workflows. "
                        "Do not include off-topic material. "
                        "Use the actual topic from the user's question. "
                        "Format exactly like this:\n\n"
                        "Short title line\n\n"
                        "Main Areas:\n"
                        "- bullet\n"
                        "- bullet\n"
                        "- bullet\n\n"
                        "Representative Procedures:\n"
                        "1. procedure\n"
                        "2. procedure\n"
                        "3. procedure\n\n"
                        "If the retrieved procedures are weak or mixed, reply exactly: INSUFFICIENT_CONTEXT"
                    )
                },
                {
                    "role": "user",
                    "content": f"Question:\n{message}\n\nRetrieved procedures:\n\n{context_text}"
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
                {"role": "user", "content": message}
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

    if is_overview_prompt(message):
        ranked = rerank_overview_matches(message, matches)
        overview_matches = select_overview_matches(message, ranked, max_matches=3)

        print("DEBUG: selected overview ->", [
            {
                "title": normalize_text(m.get("metadata", {}).get("title", "Untitled")),
                "distance": round(m["distance"], 4),
                "topic_score": m.get("topic_score", 0),
                "penalty": m.get("penalty", 0)
            }
            for m in overview_matches
        ])

        if has_coherent_overview_cluster(message, overview_matches):
            synthesized = synthesize_overview(message, overview_matches)
            if synthesized:
                return synthesized
            return deterministic_overview(message, overview_matches)

        # If overview grounding is weak, do not force a bad synthetic answer.
        return gpt_fallback(message)

    ranked = rerank_narrow_matches(message, matches)
    best_match = ranked[0]

    print("DEBUG: chosen narrow ->", {
        "title": normalize_text(best_match.get("metadata", {}).get("title", "Untitled")),
        "distance": round(best_match["distance"], 4),
        "intent_score": best_match.get("intent_score", 0)
    })

    if best_match["distance"] < STRICT_MATCH_DISTANCE:
        return format_workflow(best_match)

    return gpt_fallback(message)


def test_connection():
    try:
        conn = get_db_connection()
        conn.close()
        print("DB connected")
    except Exception as e:
        print("DB connection failed:", e)
