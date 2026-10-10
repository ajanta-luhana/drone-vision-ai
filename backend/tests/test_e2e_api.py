"""Full pipeline through the HTTP API on the synthetic §42 scenario."""
import json

import pytest

from conftest import login

P = "/api/v1"
state = {}


def test_auth_required_and_bad_login(client):
    assert client.get(f"{P}/missions").status_code == 401
    assert client.post(f"{P}/auth/login", json={"username": "admin", "password": "wrong"}).status_code == 401


def test_rbac_enforced_server_side(client, demo_assets):
    _, zone = demo_assets
    body = {"name": "x", "geojson": json.loads(zone.read_text())}
    for role in ("operator", "analyst", "viewer"):
        assert client.post(f"{P}/zones", json=body, headers=login(client, role)).status_code == 403
    assert client.post(f"{P}/missions", json={"name": "m"}, headers=login(client, "viewer")).status_code == 403
    assert client.get(f"{P}/audit", headers=login(client, "operator")).status_code == 403
    assert client.post(f"{P}/assistant/query", json={"question": "hello"}, headers=login(client, "viewer")).status_code == 403


def test_admin_creates_zone_and_rejects_bad_geojson(client, demo_assets):
    _, zone = demo_assets
    h = login(client, "admin")
    bad = client.post(f"{P}/zones", json={"name": "bad", "geojson": {"type": "Point", "coordinates": [0, 0]}}, headers=h)
    assert bad.status_code == 422
    r = client.post(f"{P}/zones", json={"name": "Restricted Zone A", "zone_type": "restricted",
                                        "geojson": json.loads(zone.read_text())}, headers=h)
    assert r.status_code == 201, r.text
    state["zone_id"] = r.json()["id"]


def test_upload_validation(client):
    h = login(client, "operator")
    m = client.post(f"{P}/missions", json={"name": "Validation"}, headers=h).json()
    assert client.post(f"{P}/videos/upload", data={"mission_id": m["id"]}, files={"file": ("a.exe", b"MZ", "application/octet-stream")}, headers=h).status_code == 415
    assert client.post(f"{P}/videos/upload", data={"mission_id": m["id"]}, files={"file": ("fake.mp4", b"not a video at all", "video/mp4")}, headers=h).status_code == 422


def test_pipeline_end_to_end(client, demo_assets):
    video, _ = demo_assets
    h = login(client, "operator")
    mission = client.post(f"{P}/missions", json={"name": "Demo patrol", "altitude": 60, "heading": 0}, headers=h).json()
    state["mission_id"] = mission["id"]
    with open(video, "rb") as f:
        up = client.post(f"{P}/videos/upload", data={"mission_id": mission["id"]}, files={"file": ("demo.mp4", f, "video/mp4")}, headers=h)
    assert up.status_code == 201, up.text
    vid = up.json()["id"]
    assert client.post(f"{P}/videos/{vid}/process", headers=h).status_code == 202
    v = client.get(f"{P}/videos/{vid}", headers=h).json()          # background task finished with TestClient
    assert v["status"] == "done", v
    state["video_id"] = vid
    assert v["meta"]["processing"]["tracks"] >= 5
    assert client.get(f"{P}/videos/{vid}/stream", headers=h).status_code == 200      # annotated video produced


def test_tracking_persistent_ids_and_trajectories(client):
    h = login(client, "analyst")
    tracks = client.get(f"{P}/tracks", params={"mission_id": state["mission_id"]}, headers=h).json()
    cars = [t for t in tracks if t["class"] == "car"]
    people = [t for t in tracks if t["class"] == "person"]
    assert len(cars) == 3 and len(people) == 2        # exactly the scene's 3 vehicles + 2 people: no ID fragmentation
    long_car = max(cars, key=lambda t: t["n_points"])
    assert long_car["n_points"] > 100 and long_car["stats"]["distance_m"] > 20


def test_zone_violation_dwell_and_risk(client):
    h = login(client, "analyst")
    incs = client.get(f"{P}/incidents", params={"mission_id": state["mission_id"]}, headers=h).json()
    crit = [i for i in incs if i["object_class"] == "car"]
    assert len(crit) == 1, incs                              # only the parked car is an incident; passing cars are not
    i = crit[0]
    assert i["severity"] == "CRITICAL" and i["risk_score"] >= 76
    assert {"Restricted zone intrusion", "Extended stationary period", "Person-vehicle proximity"} <= set(i["reasons"])
    assert i["requires_review"] and "ESCALATE" in i["actions"]
    state["incident_id"] = i["id"]
    detail = client.get(f"{P}/incidents/{i['id']}", headers=h).json()
    assert {e["event_type"] for e in detail["events"]} >= {"ZONE_INTRUSION", "STATIONARY", "PERSON_VEHICLE_PROXIMITY"}
    stat = next(e for e in detail["events"] if e["event_type"] == "STATIONARY")
    assert stat["details"]["duration_s"] >= 4                # dwell time calculated
    assert client.get(f"{P}/incidents/{i['id']}/frame", headers=h).status_code == 200   # evidence frame
    assert detail["vlm"]["provider"] == "stub" and detail["model_versions"]["detector"].startswith("blob")


def test_bystander_and_passing_traffic_not_flagged(client):
    h = login(client, "analyst")
    tracks = {t["track_id"]: t for t in client.get(f"{P}/tracks", params={"mission_id": state["mission_id"]}, headers=h).json()}
    flagged = {i["track_id"] for i in client.get(f"{P}/incidents", params={"mission_id": state["mission_id"]}, headers=h).json()}
    for tid, t in tracks.items():
        if t["class"] == "car" and t["stats"]["stationary_s"] == 0:
            assert tid not in flagged                       # moving cars never become incidents


def test_map_geojson_layer(client):
    h = login(client, "viewer")
    fc = client.get(f"{P}/missions/{state['mission_id']}/geojson", headers=h).json()
    kinds = {f["properties"]["kind"] for f in fc["features"]}
    assert kinds == {"zone", "trajectory", "incident"}


def test_human_review_and_audit_trail(client):
    op, adm = login(client, "operator"), login(client, "admin")
    iid = state["incident_id"]
    assert client.post(f"{P}/incidents/{iid}/review", json={"decision": "dismissed", "comment": ""}, headers=op).status_code == 422
    assert client.post(f"{P}/incidents/{iid}/review", json={"decision": "confirmed", "comment": "Confirmed restricted-zone violation"}, headers=login(client, "viewer")).status_code == 403
    r = client.post(f"{P}/incidents/{iid}/review", json={"decision": "confirmed", "comment": "Confirmed restricted-zone violation"}, headers=op)
    assert r.status_code == 200 and r.json()["status"] == "confirmed"
    audit = client.get(f"{P}/audit", headers=adm).json()
    entry = next(a for a in audit if a["action"] == "incident_reviewed")
    assert entry["user"] == "operator" and entry["previous_state"] == {"status": "pending"} and entry["new_state"]["status"] == "confirmed"
    assert {"login", "login_failed", "zone_created", "video_uploaded"} <= {a["action"] for a in audit}


def test_reports_pdf_json_csv(client):
    h = login(client, "analyst")
    r = client.post(f"{P}/reports", json={"mission_id": state["mission_id"]}, headers=h)
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    pdf = client.get(f"{P}/reports/{rid}/download", params={"format": "pdf"}, headers=h)
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    csv = client.get(f"{P}/reports/{rid}/download", params={"format": "csv"}, headers=h).text
    assert csv.splitlines()[0].startswith("event_id,type") and "ZONE_INTRUSION" in csv
    data = client.get(f"{P}/reports/{rid}", headers=h).json()
    assert data["incidents"][0]["operator_decisions"][0]["decision"] == "confirmed"   # decision flows into report
    assert data["incidents"][0]["model_versions"]


def test_analytics_overview(client):
    o = client.get(f"{P}/analytics/overview", headers=login(client, "viewer")).json()
    assert o["high_risk_events"] >= 1 and o["zone_violations"] >= 2 and o["object_counts"]["car"] >= 3


def test_assistant_is_tool_grounded(client):
    h = login(client, "analyst")
    ask = lambda q: client.post(f"{P}/assistant/query", json={"question": q}, headers=h).json()
    a = ask("Show all high-risk events today.")
    assert a["tool_calls"][0]["tool"] == "search_events" and "CRITICAL" in a["answer"]
    a = ask("What is the procedure for a vehicle entering Restricted Zone A?")
    assert a["tool_calls"][0]["tool"] == "search_policy" and a["sources"][0]["source"] == "vehicle_intrusion_sop.md" and "Verify the evidence frame" in a["answer"]
    a = ask("Why was vehicle #99 classified as high risk?")                     # nonexistent: must not invent
    assert "No matching incident" in a["answer"]
    a = ask("Which vehicles entered restricted zones?")
    assert a["tool_calls"][0]["tool"] == "search_events" and "car #" in a["answer"]
    a = ask("Generate a report for today's incidents")
    assert a["tool_calls"][0]["tool"] == "generate_report" and "Report #" in a["answer"]
    a = ask("what is the airspeed velocity of a swallow")
    assert not a["tool_calls"]


def test_system_metrics_and_health(client):
    assert client.get(f"{P}/system/health").json()["status"] == "ok"
    m = client.get(f"{P}/system/metrics", headers=login(client, "operator")).json()
    assert m["videos_processed"] >= 1 and m["memory_rss_mb"] > 0 and m["avg_processing_fps"]


def test_reprocessing_is_idempotent(client):
    h = login(client, "operator")
    client.post(f"{P}/videos/{state['video_id']}/process", headers=h)
    incs = client.get(f"{P}/incidents", params={"mission_id": state["mission_id"]}, headers=h).json()
    assert len([i for i in incs if i["object_class"] == "car"]) == 1       # no duplicates after rerun
