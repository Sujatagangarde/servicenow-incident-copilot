"""
Load realistic resolved incidents into your ServiceNow instance.

Each incident gets correlation_id = "copilot-seed" so we can find (or delete) them later
and so running the script twice does not create duplicates.

Run from the project root (with venv active):
    python -m scripts.seed_incidents --dry-run   # show what would be created, sends nothing
    python -m scripts.seed_incidents             # actually create the incidents
"""
import argparse
import json
from pathlib import Path

import requests

from app.config import settings

SEED_TAG = "copilot-seed"
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "seed_incidents.json"

# Resolution code must match a choice in your instance.
# Check: open any incident > Resolution Information tab > "Resolution code" dropdown.
CLOSE_CODE = "Solution provided"


def main(dry_run: bool):
    incidents = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(incidents)} incidents from {DATA_FILE.name}")

    session = requests.Session()
    session.auth = (settings.SN_USER, settings.SN_PASSWORD)
    session.headers.update({"Accept": "application/json", "Content-Type": "application/json"})
    url = f"{settings.SN_INSTANCE}/api/now/table/incident"

    if dry_run:
        print("\nDRY RUN - nothing will be sent. First incident payload would be:")
        print(json.dumps({**incidents[0], "state": "6", "close_code": CLOSE_CODE,
                          "correlation_id": SEED_TAG}, indent=2))
        return

    # Skip if already seeded (makes the script safe to re-run)
    check = session.get(url, params={"sysparm_query": f"correlation_id={SEED_TAG}",
                                     "sysparm_fields": "number", "sysparm_limit": 1}, timeout=30)
    check.raise_for_status()
    if check.json()["result"]:
        print("Seed incidents already exist - nothing to do.")
        return

    created, failed = 0, 0
    for inc in incidents:
        payload = {
            **inc,
            "state": "6",                 # 6 = Resolved
            "close_code": CLOSE_CODE,
            "correlation_id": SEED_TAG,
        }
        # sysparm_input_display_value lets us send "Network" instead of a group sys_id
        resp = session.post(url, params={"sysparm_input_display_value": "true"},
                            data=json.dumps(payload), timeout=30)
        if resp.status_code == 201:
            number = resp.json()["result"]["number"]
            created += 1
            print(f"  created {number}: {inc['short_description']}")
        else:
            failed += 1
            print(f"  FAILED ({resp.status_code}) {inc['short_description']}: {resp.text[:200]}")

    print(f"\nDone. Created: {created}, Failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args().dry_run)
