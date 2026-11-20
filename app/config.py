"""Loads settings from the .env file so secrets never live in code."""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ServiceNow
    SN_INSTANCE: str = os.getenv("SN_INSTANCE", "").rstrip("/")
    SN_USER: str = os.getenv("SN_USER", "")
    SN_PASSWORD: str = os.getenv("SN_PASSWORD", "")

    # Gemini (embeddings now, LLM later)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "gemini-embedding-001")
    EMBED_DIM: int = int(os.getenv("EMBED_DIM", "768"))

    # Postgres + pgvector (matches docker-compose.yml)
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "postgresql://copilot:copilot@localhost:5432/copilot"
    )


settings = Settings()
