"""
Turns text into embeddings (lists of numbers that capture meaning) using Gemini.

task_type matters:
  - RETRIEVAL_DOCUMENT for the incidents/KB we store
  - RETRIEVAL_QUERY    for the user's search text
Using the right one noticeably improves search quality.
"""
import time

from google import genai
from google.genai import types

from app.config import settings

_client = None


def _get_client():
    global _client
    if _client is None:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is missing in .env")
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT",
                batch_size: int = 50) -> list[list[float]]:
    """Embed many texts, in batches, with a simple retry for rate limits (free tier)."""
    client = _get_client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(5):
            try:
                result = client.models.embed_content(
                    model=settings.EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=settings.EMBED_DIM,
                    ),
                )
                vectors.extend(e.values for e in result.embeddings)
                break
            except Exception as e:  # rate limit or temporary error
                if attempt == 4:
                    raise
                wait = 2 ** attempt * 5
                print(f"  embedding error ({e.__class__.__name__}), retrying in {wait}s...")
                time.sleep(wait)
    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]
