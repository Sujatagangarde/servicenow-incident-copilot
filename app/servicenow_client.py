"""
All communication with ServiceNow lives here.
Keeping it in one class means the rest of the app never deals with URLs or auth.
"""
import requests
from app.config import settings

INCIDENT_FIELDS = [
    "sys_id", "number", "short_description", "description",
    "category", "subcategory", "priority", "state",
    "assignment_group", "close_code", "close_notes",
]
KB_FIELDS = ["sys_id", "number", "short_description", "text"]


class ServiceNowClient:
    def __init__(self):
        self.base_url = f"{settings.SN_INSTANCE}/api/now/table"
        self.session = requests.Session()
        self.session.auth = (settings.SN_USER, settings.SN_PASSWORD)
        self.session.headers.update({"Accept": "application/json",
                                     "Content-Type": "application/json"})

    def _get(self, table: str, fields: list[str], query: str = "", limit: int = 10) -> list[dict]:
        params = {
            "sysparm_fields": ",".join(fields),
            "sysparm_limit": limit,
            "sysparm_display_value": "true",  # human-readable values (e.g. group name, not sys_id)
            "sysparm_exclude_reference_link": "true",
        }
        if query:
            params["sysparm_query"] = query
        resp = self.session.get(f"{self.base_url}/{table}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()["result"]

    def get_incidents(self, limit: int = 10, only_resolved: bool = False) -> list[dict]:
        # state 6 = Resolved, 7 = Closed. Resolved incidents have close_notes = the "fix",
        # which is exactly what our copilot will learn from later.
        query = "stateIN6,7^ORDERBYDESCsys_updated_on" if only_resolved else "ORDERBYDESCsys_created_on"
        return self._get("incident", INCIDENT_FIELDS, query, limit)

    def get_incident(self, number: str) -> dict | None:
        result = self._get("incident", INCIDENT_FIELDS, f"number={number}", 1)
        return result[0] if result else None

    def get_kb_articles(self, limit: int = 10) -> list[dict]:
        return self._get("kb_knowledge", KB_FIELDS, "workflow_state=published", limit)
