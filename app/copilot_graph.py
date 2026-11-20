"""
The Copilot workflow, built with LangGraph.

            ┌──────────┐    ┌──────────┐   top match strong?   ┌────────────────────┐
  START ──► │ retrieve │──► │ classify │──► yes ─────────────► │ generate_grounded  │──► END
            └──────────┘    └──────────┘    no  ──┐            └────────────────────┘
                                                  └──────────► │ generate_general   │──► END

  retrieve           : embed the incident, find similar past incidents + KB chunks (pgvector)
  classify           : LLM picks category, impact, urgency, assignment group (using similar cases as examples)
  generate_grounded  : LLM writes a fix based ONLY on the retrieved cases, with citations
  generate_general   : no close match -> generic troubleshooting, low confidence, suggest escalation
"""
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.db import search_similar
from app.embeddings import embed_query
from app.llm import generate_structured

MIN_SIMILARITY = 0.60   # below this, past cases are not similar enough to trust

CATEGORIES = ["network", "software", "hardware", "database", "inquiry"]
GROUPS = ["Network", "Software", "Hardware", "Database", "Service Desk"]

# ServiceNow priority matrix: (impact, urgency) -> priority
PRIORITY_MATRIX = {
    (1, 1): 1, (1, 2): 2, (2, 1): 2,
    (1, 3): 3, (2, 2): 3, (3, 1): 3,
    (2, 3): 4, (3, 2): 4, (3, 3): 5,
}
PRIORITY_LABELS = {1: "1 - Critical", 2: "2 - High", 3: "3 - Moderate", 4: "4 - Low", 5: "5 - Planning"}


# ---------- Structured outputs the LLM must return ----------

class Classification(BaseModel):
    category: Literal["network", "software", "hardware", "database", "inquiry"]
    subcategory: str = Field(description="short lowercase subcategory, e.g. vpn, email, disk")
    impact: int = Field(ge=1, le=3, description="1=many users/business critical, 2=a team, 3=one user")
    urgency: int = Field(ge=1, le=3, description="1=work fully blocked, 2=degraded, 3=can wait")
    assignment_group: Literal["Network", "Software", "Hardware", "Database", "Service Desk"]
    confidence: float = Field(ge=0, le=1)
    reasoning: str = Field(description="one or two sentences explaining the choice")


class Suggestion(BaseModel):
    likely_root_cause: str
    suggested_steps: list[str] = Field(description="ordered, concrete troubleshooting/fix steps")
    sources: list[str] = Field(description="IDs of the past incidents/KB articles used, e.g. INC0010001, KB0000011")
    confidence: float = Field(ge=0, le=1)
    draft_resolution_note: str = Field(description="resolution note an engineer could paste after fixing")


# ---------- Graph state ----------

class CopilotState(TypedDict, total=False):
    short_description: str
    description: str
    similar_incidents: list[dict]
    kb_chunks: list[dict]
    top_similarity: float
    classification: dict
    suggestion: dict
    path: str
    usage: list[dict]


def _incident_text(state: CopilotState) -> str:
    return f"{state['short_description']}\n{state.get('description') or ''}".strip()


def _format_cases(state: CopilotState) -> str:
    lines = ["PAST RESOLVED INCIDENTS:"]
    for inc in state["similar_incidents"]:
        m = inc["metadata"]
        lines.append(
            f"- [{inc['source_id']}] (similarity {inc['similarity']}) {inc['title']}\n"
            f"  category={m.get('category')}/{m.get('subcategory')}, group={m.get('assignment_group')}, "
            f"priority={m.get('priority')}\n  resolution: {m.get('resolution')}"
        )
    lines.append("\nKNOWLEDGE ARTICLE EXCERPTS:")
    for kb in state["kb_chunks"]:
        lines.append(f"- [{kb['metadata'].get('kb_number')}] (similarity {kb['similarity']}) "
                     f"{kb['content'][:700]}")
    return "\n".join(lines)


# ---------- Nodes ----------

def retrieve(state: CopilotState) -> dict:
    vector = embed_query(_incident_text(state))          # embed once, search twice
    incidents = search_similar(vector, k=5, source="incident")
    kb = search_similar(vector, k=3, source="kb")
    top = max([r["similarity"] for r in incidents + kb], default=0.0)
    return {"similar_incidents": incidents, "kb_chunks": kb, "top_similarity": top}


def classify(state: CopilotState) -> dict:
    system = ("You are an experienced IT service desk analyst triaging ServiceNow incidents. "
              f"Allowed categories: {CATEGORIES}. Allowed assignment groups: {GROUPS}. "
              "Use the similar past incidents as examples of how this organisation classifies tickets.")
    prompt = f"NEW INCIDENT:\n{_incident_text(state)}\n\n{_format_cases(state)}\n\nClassify the NEW incident."
    result, usage = generate_structured(prompt, Classification, system)
    data = result.model_dump()
    p = PRIORITY_MATRIX[(data["impact"], data["urgency"])]
    data["priority"] = PRIORITY_LABELS[p]
    return {"classification": data, "usage": state.get("usage", []) + [{"step": "classify", **usage}]}


def _generate(state: CopilotState, grounded: bool) -> dict:
    if grounded:
        system = ("You are a senior IT support engineer. Suggest a fix for the new incident using ONLY the "
                  "past incidents and knowledge excerpts provided. Cite the IDs you used in 'sources'. "
                  "If the provided cases do not fully fit, say what to check first. Never invent IDs.")
    else:
        system = ("You are a senior IT support engineer. No sufficiently similar past case was found. "
                  "Give safe, generic first-line troubleshooting steps, set confidence at or below 0.4, "
                  "leave 'sources' empty unless a provided case is genuinely relevant, and recommend "
                  "escalation to the assignment group if the steps do not resolve it.")
    prompt = (f"NEW INCIDENT:\n{_incident_text(state)}\n\n"
              f"Classification: {state['classification']}\n\n{_format_cases(state)}")
    result, usage = generate_structured(prompt, Suggestion, system)
    data = result.model_dump()

    # Guardrail: keep only sources we actually retrieved (stops hallucinated ticket numbers)
    allowed = {r["source_id"] for r in state["similar_incidents"]}
    allowed |= {k["metadata"].get("kb_number") for k in state["kb_chunks"]}
    data["sources"] = [s for s in data["sources"] if s in allowed]

    return {"suggestion": data, "path": "grounded" if grounded else "low_match",
            "usage": state.get("usage", []) + [{"step": "generate", **usage}]}


def generate_grounded(state: CopilotState) -> dict:
    return _generate(state, grounded=True)


def generate_general(state: CopilotState) -> dict:
    return _generate(state, grounded=False)


def route_after_classify(state: CopilotState) -> str:
    return "generate_grounded" if state["top_similarity"] >= MIN_SIMILARITY else "generate_general"


# ---------- Build the graph ----------

def build_graph():
    g = StateGraph(CopilotState)
    g.add_node("retrieve", retrieve)
    g.add_node("classify", classify)
    g.add_node("generate_grounded", generate_grounded)
    g.add_node("generate_general", generate_general)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "classify")
    g.add_conditional_edges("classify", route_after_classify,
                            {"generate_grounded": "generate_grounded",
                             "generate_general": "generate_general"})
    g.add_edge("generate_grounded", END)
    g.add_edge("generate_general", END)
    return g.compile()


copilot = build_graph()


def analyze_incident(short_description: str, description: str = "") -> dict:
    final = copilot.invoke({"short_description": short_description, "description": description, "usage": []})
    return {
        "classification": final["classification"],
        "suggestion": final["suggestion"],
        "path": final["path"],
        "top_similarity": final["top_similarity"],
        "similar_incidents": [
            {"id": r["source_id"], "title": r["title"], "similarity": r["similarity"]}
            for r in final["similar_incidents"]
        ],
        "kb_articles": [
            {"id": k["metadata"].get("kb_number"), "title": k["title"], "similarity": k["similarity"]}
            for k in final["kb_chunks"]
        ],
        "usage": final["usage"],
    }
