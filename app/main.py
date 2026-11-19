"""
FastAPI entry point.
Run:  python -m app.main          (opens the docs in your default browser)
  or: uvicorn app.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import requests
from fastapi import FastAPI, HTTPException, Query

from app.servicenow_client import ServiceNowClient

app = FastAPI(title="ServiceNow Incident Copilot", version="0.1.0")
sn = ServiceNowClient()


def _call(fn, *args, **kwargs):
    """Turn ServiceNow errors into clean API errors instead of a crash."""
    try:
        return fn(*args, **kwargs)
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"ServiceNow error: {e}")
    except requests.ConnectionError:
        raise HTTPException(status_code=503, detail="Cannot reach ServiceNow (is the PDI awake?)")


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


if __name__ == "__main__":
    # python -m app.main  -> starts the server and opens the docs in the system browser
    import threading
    import webbrowser

    import uvicorn

    url = "http://127.0.0.1:8000/docs"
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
