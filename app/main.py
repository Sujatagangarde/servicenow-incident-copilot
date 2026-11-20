"""
FastAPI entry point.
Run:  uvicorn app.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import psycopg
import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.copilot_graph import analyze_incident
from app.db import count_documents, search_similar
from app.embeddings import embed_query
from app.servicenow_client import ServiceNowClient

app = FastAPI(title="ServiceNow Incident Copilot", version="0.3.0")
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


# ---------- Phase 3: the Copilot (LangGraph + LLM) ----------

class AnalyzeRequest(BaseModel):
    short_description: str = Field(..., min_length=5, examples=["VPN keeps disconnecting every few minutes"])
    description: str = Field("", examples=["Working from home, AnyConnect drops and I have to reconnect."])


def _run_copilot(short_description: str, description: str) -> dict:
    try:
        return analyze_incident(short_description, description)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except psycopg.OperationalError:
        raise HTTPException(status_code=503, detail="Cannot reach Postgres. Check DATABASE_URL in .env")
    except Exception as e:  # LLM/API errors (rate limit, bad key, model name...)
        raise HTTPException(status_code=502, detail=f"Copilot failed: {e.__class__.__name__}: {e}")


@app.post("/copilot/analyze")
def copilot_analyze(req: AnalyzeRequest):
    """Analyze any incident text: classify it and suggest a fix based on similar past cases."""
    return _run_copilot(req.short_description, req.description)


@app.get("/copilot/incident/{number}")
def copilot_for_incident(number: str):
    """Fetch a real incident from ServiceNow and run the Copilot on it."""
    incident = _call(sn.get_incident, number)
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {number} not found")
    analysis = _run_copilot(incident["short_description"], incident.get("description") or "")
    return {"incident": {"number": incident["number"], "short_description": incident["short_description"]},
            **analysis}


@app.get("/stats")
def stats():
    try:
        return {"documents": count_documents()}
    except psycopg.OperationalError:
        raise HTTPException(status_code=503, detail="Cannot reach Postgres. Is Docker running? (docker compose up -d)")
