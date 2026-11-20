"""
Postgres + pgvector: one table, `documents`, holds both incidents and KB chunks.

Columns:
  source     'incident' or 'kb'
  source_id  e.g. 'INC0010012' or 'KB0000011#2' (KB articles are split into chunks)
  title      short description
  content    the text that was embedded
  metadata   extra info as JSON (category, priority, resolution...)
  embedding  vector(EMBED_DIM)
"""
import json

import psycopg
from pgvector.psycopg import register_vector

from app.config import settings


def get_conn() -> psycopg.Connection:
    conn = psycopg.connect(settings.DATABASE_URL, autocommit=True)
    register_vector(conn)
    return conn


def init_db():
    """Create the extension, table and index if they don't exist yet."""
    with psycopg.connect(settings.DATABASE_URL, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    with get_conn() as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id         SERIAL PRIMARY KEY,
                source     TEXT NOT NULL,
                source_id  TEXT NOT NULL,
                title      TEXT,
                content    TEXT NOT NULL,
                metadata   JSONB DEFAULT '{{}}',
                embedding  vector({settings.EMBED_DIM}) NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT now(),
                UNIQUE (source, source_id)
            )
        """)
        # HNSW index = fast approximate nearest-neighbour search using cosine distance
        conn.execute("""
            CREATE INDEX IF NOT EXISTS documents_embedding_idx
            ON documents USING hnsw (embedding vector_cosine_ops)
        """)


def upsert_documents(rows: list[dict]):
    """Insert or update rows: each dict has source, source_id, title, content, metadata, embedding."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany("""
                INSERT INTO documents (source, source_id, title, content, metadata, embedding)
                VALUES (%(source)s, %(source_id)s, %(title)s, %(content)s, %(metadata)s, %(embedding)s)
                ON CONFLICT (source, source_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    updated_at = now()
            """, [{**r, "metadata": json.dumps(r["metadata"])} for r in rows])


def search_similar(query_vector, k: int = 5, source: str | None = None) -> list[dict]:
    """Return the k most similar documents. similarity = 1 - cosine distance (1.0 = identical)."""
    sql = """
        SELECT source, source_id, title, content, metadata,
               1 - (embedding <=> %(qv)s::vector) AS similarity
        FROM documents
        {where}
        ORDER BY embedding <=> %(qv)s::vector
        LIMIT %(k)s
    """.format(where="WHERE source = %(source)s" if source else "")
    with get_conn() as conn:
        rows = conn.execute(sql, {"qv": query_vector, "k": k, "source": source}).fetchall()
    return [
        {"source": r[0], "source_id": r[1], "title": r[2], "content": r[3],
         "metadata": r[4], "similarity": round(float(r[5]), 4)}
        for r in rows
    ]


def count_documents() -> dict:
    with get_conn() as conn:
        rows = conn.execute("SELECT source, COUNT(*) FROM documents GROUP BY source").fetchall()
    return {r[0]: r[1] for r in rows}
