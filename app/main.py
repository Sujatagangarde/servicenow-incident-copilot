"""
FastAPI entry point.
Run:  uvicorn app.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import psycopg
import requests
from fastapi import FastAPI, HTTPException, Query

from app.db import count_documents, search_similar
from app.embeddings import embed_query
from app.servicenow_client import ServiceNowClient

app = FastAPI(title="ServiceNow Incident Copilot", version="0.2.0")
sn = ServiceNowClient()


def _call(fn, *args, **kwargs):
    """Turn ServiceNow errors into clean API errors instead of a crash."""
    try:
        return fn(*args, **kwargs)
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"ServiceNow error: {e}")
    except requests.ConnectionError:
        raise HTTPException(status_code=503, detail="Cannot reach ServiceNow (is the PDI awake?)")
    except requests.exceptions.InvalidURL as e:
        raise HTTPException(status_code=500, detail=f"Check SN_INSTANCE in your .env file: {e}")
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Request to ServiceNow failed: {e}")
    except ValueError:
        # ServiceNow returned a web page (e.g. login/hibernation page) instead of JSON
        raise HTTPException(status_code=502, detail="ServiceNow did not return JSON. Open your PDI in the browser to wake it up.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/incidents")
def list_incidents(limit: int = Query(10, le=100), only_resolved: bool = False):
    return _call(sn.get_incidents, limit=limit, only_resolved=only_resolved)


@app.get("/incidents/{number}")
def get_incident(number: str):
    incident = _call(sn.get_incident, number)
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {number} not found")
    return incident


@app.get("/kb")
def list_kb(limit: int = Query(10, le=100)):
    return _call(sn.get_kb_articles, limit=limit)


# ---------- Phase 2: semantic search over pgvector ----------

@app.get("/search")
def search(q: str = Query(..., min_length=3, description="Describe the problem in plain words"),
           k: int = Query(5, ge=1, le=20),
           source: str | None = Query(None, pattern="^(incident|kb)$")):
    """Find the most similar past incidents / KB chunks by meaning, not keywords."""
    try:
        query_vector = embed_query(q)
        return {"query": q, "results": search_similar(query_vector, k=k, source=source)}
    except RuntimeError as e:          # e.g. missing GEMINI_API_KEY
        raise HTTPException(status_code=500, detail=str(e))
    except psycopg.OperationalError:
        raise HTTPException(status_code=503, detail="Cannot reach Postgres. Is Docker running? (docker compose up -d)")


@app.get("/stats")
def stats():
    try:
        return {"documents": count_documents()}
    except psycopg.OperationalError:
        raise HTTPException(status_code=503, detail="Cannot reach Postgres. Is Docker running? (docker compose up -d)")
