"""
Ingestion pipeline: ServiceNow -> clean text -> embeddings -> pgvector.

  1. Fetch resolved incidents (only ones with a real resolution note)
  2. Fetch published KB articles, strip HTML, split long articles into chunks
  3. Embed everything with Gemini
  4. Upsert into Postgres (safe to re-run: existing rows are updated, not duplicated)

Run from the project root (venv active, Docker DB running):
    python -m scripts.ingest
"""
import html
import re

from app.db import count_documents, init_db, upsert_documents
from app.embeddings import embed_texts
from app.servicenow_client import ServiceNowClient

MIN_RESOLUTION_CHARS = 40   # skip junk like "This incident is resolved."
CHUNK_SIZE = 1500           # characters per KB chunk
CHUNK_OVERLAP = 200         # overlap so sentences cut at a boundary aren't lost


def clean_html(raw: str) -> str:
    text = re.sub(r"<(script|style).*?</\1>", " ", raw or "", flags=re.S | re.I)
    text = re.sub(r"<br\s*/?>|</p>|</li>|</h\d>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # try to cut at a sentence/line end instead of mid-word
            cut = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return chunks


def incident_to_doc(inc: dict) -> dict | None:
    resolution = (inc.get("close_notes") or "").strip()
    if len(resolution) < MIN_RESOLUTION_CHARS:
        return None
    content = (
        f"Incident: {inc['short_description']}\n"
        f"Details: {inc.get('description') or ''}\n"
        f"Category: {inc.get('category')} / {inc.get('subcategory')}\n"
        f"Resolution: {resolution}"
    )
    return {
        "source": "incident",
        "source_id": inc["number"],
        "title": inc["short_description"],
        "content": content,
        "metadata": {
            "category": inc.get("category"),
            "subcategory": inc.get("subcategory"),
            "priority": inc.get("priority"),
            "assignment_group": inc.get("assignment_group"),
            "resolution": resolution,
        },
    }


def kb_to_docs(article: dict) -> list[dict]:
    body = clean_html(article.get("text", ""))
    if not body:
        return []
    docs = []
    for i, chunk in enumerate(chunk_text(body)):
        docs.append({
            "source": "kb",
            "source_id": f"{article['number']}#{i}",
            "title": article["short_description"],
            "content": f"Knowledge article: {article['short_description']}\n{chunk}",
            "metadata": {"kb_number": article["number"], "chunk": i},
        })
    return docs


def main():
    print("Creating table/index if needed...")
    init_db()

    sn = ServiceNowClient()
    print("Fetching resolved incidents...")
    incidents = sn.get_incidents(limit=1000, only_resolved=True)
    incident_docs = [d for d in (incident_to_doc(i) for i in incidents) if d]
    print(f"  {len(incidents)} resolved, {len(incident_docs)} with useful resolution notes")

    print("Fetching published KB articles...")
    articles = sn.get_kb_articles(limit=1000)
    kb_docs = [d for a in articles for d in kb_to_docs(a)]
    print(f"  {len(articles)} articles -> {len(kb_docs)} chunks")

    docs = incident_docs + kb_docs
    if not docs:
        print("Nothing to ingest.")
        return

    print(f"Embedding {len(docs)} documents with Gemini...")
    vectors = embed_texts([d["content"] for d in docs])
    for d, v in zip(docs, vectors):
        d["embedding"] = v

    print("Saving to Postgres...")
    upsert_documents(docs)
    print(f"Done. Documents in DB: {count_documents()}")


if __name__ == "__main__":
    main()
