"""Drive a running server through the full §42 scenario via the HTTP API.
Usage: SEED_PASSWORD=... python scripts/run_demo.py [http://127.0.0.1:8000] [admin_user operator_user]"""
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000") + "/api/v1"
PW = os.environ["SEED_PASSWORD"]


def login(user):
    r = httpx.post(f"{BASE}/auth/login", json={"username": user, "password": PW}, timeout=30)
    r.raise_for_status()
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


admin, op = login("admin"), login("operator")
zone = json.loads((ROOT / "simulations" / "zone_restricted_a.geojson").read_text())
if not any(z["name"] == "Restricted Zone A" for z in httpx.get(f"{BASE}/zones", headers=admin).json()):
    httpx.post(f"{BASE}/zones", headers=admin, json={"name": "Restricted Zone A", "zone_type": "restricted", "geojson": zone}).raise_for_status()
m = httpx.post(f"{BASE}/missions", headers=op, json={"name": "Demo patrol"}).json()
with open(ROOT / "datasets" / "demo_aerial.mp4", "rb") as f:
    v = httpx.post(f"{BASE}/videos/upload", headers=op, data={"mission_id": m["id"]}, files={"file": ("demo_aerial.mp4", f, "video/mp4")}, timeout=120).json()
httpx.post(f"{BASE}/videos/{v['id']}/process", headers=op).raise_for_status()
while True:
    st = httpx.get(f"{BASE}/videos/{v['id']}", headers=op).json()
    print(f"  {st['status']} {st['progress']}%")
    if st["status"] in ("done", "failed"):
        break
    time.sleep(1)
print(json.dumps(st["meta"].get("processing"), indent=1) if st["status"] == "done" else st["error"])
for i in httpx.get(f"{BASE}/incidents", headers=op, params={"mission_id": m["id"]}).json():
    print(f"{i['code']}  {i['severity']} {i['risk_score']}/100  {i['object_class']} #{i['track_id']}  {i['reasons']}")
print("Open the dashboard and review the incident.")
