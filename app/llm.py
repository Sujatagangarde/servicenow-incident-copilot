"""
LLM calls (Gemini) with structured JSON output.

We pass a Pydantic model as response_schema, so Gemini must return JSON in exactly
that shape. No fragile string parsing, and FastAPI can return it directly.
Every call also records token usage, so we can track cost later (Phase 4).
"""
import time

from google import genai
from google.genai import types
from pydantic import BaseModel

from app.config import settings

_client = None


def _get_client():
    global _client
    if _client is None:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is missing in .env")
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def generate_structured(prompt: str, schema: type[BaseModel], system: str) -> tuple[BaseModel, dict]:
    """Call the LLM and return (parsed_object, usage_dict). Retries on rate limits."""
    client = _get_client()
    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=settings.LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.2,   # low = consistent, factual answers
                ),
            )
            break
        except Exception as e:
            if attempt == 3:
                raise
            wait = 2 ** attempt * 5
            print(f"  LLM error ({e.__class__.__name__}), retrying in {wait}s...")
            time.sleep(wait)

    parsed = response.parsed
    if parsed is None:  # fallback: validate raw JSON text ourselves
        parsed = schema.model_validate_json(response.text)

    meta = response.usage_metadata
    usage = {
        "model": settings.LLM_MODEL,
        "input_tokens": getattr(meta, "prompt_token_count", 0) or 0,
        "output_tokens": getattr(meta, "candidates_token_count", 0) or 0,
        "total_tokens": getattr(meta, "total_token_count", 0) or 0,
    }
    return parsed, usage
