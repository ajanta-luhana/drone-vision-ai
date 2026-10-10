import json

import cv2
import numpy as np
import pytest

from conftest import login

P = "/api/v1"
ZONE_PX = [[560, 250], [860, 250], [860, 90], [560, 90]]      # same rectangle the demo zone was generated from


# ---------------- obstacle detection ----------------
def _track(tid, cls, path, dt=0.25):
    from app.geospatial.geo import offset_latlon
    from app.services.types import Point, TrackData
    t = TrackData(tid, cls)
    for i, (e, n) in enumerate(path):
        lat, lon = offset_latlon(24.8607, 67.0011, e, n)
        t.points.append(Point(i * dt, i, lat, lon, 0, 0))
    return t


def test_obstacle_on_road_zone_and_risk():
    from app.core.config import Settings
    from app.geospatial.geo import make_box
    from app.risk.engine import assess
    from app.services.events import detect_events
    from app.services.types import ZoneDef
    road = ZoneDef(7, "Main road", "road", make_box(24.8607, 67.0011, 60, 12))
    cfg = Settings()
    parked = _track(1, "car", [(i * 4 - 20, 0) for i in range(8)] + [(8, 0)] * 30)          # drives in, stops
    passing = _track(2, "car", [(i * 4 - 28, 1) for i in range(30)])                         # just passes through
    debris = _track(3, "debris", [(0, 2)] * 20)
    ev = detect_events([parked, passing, debris], [road], cfg)
    obs = {e.track_id for e in ev if e.event_type == "OBSTACLE"}
    assert obs == {1, 3}                                      # passing traffic is not an obstacle
    r = assess([e for e in ev if e.track_id == 1 and e.event_type == "OBSTACLE"])
    assert "Obstruction on road" in r["reasons"] and r["risk_score"] == 25


def test_stationary_car_outside_road_zone_is_not_obstacle():
    from app.core.config import Settings
    from app.services.events import detect_events
    ev = detect_events([_track(1, "car", [(0, 0)] * 30)], [], Settings())
    assert not any(e.event_type == "OBSTACLE" for e in ev)


# ---------------- privacy blur ----------------
def test_blur_people_blurs_only_person_regions():
    from app.services.pipeline import blur_people
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (100, 200, 3), dtype=np.uint8)
    ref = img.copy()
    blur_people(img, [(1, "person", 0.9, (20, 20, 50, 60)), (2, "car", 0.9, (100, 20, 160, 60))])
    assert img[20:60, 20:50].std() < ref[20:60, 20:50].std() * 0.5       # person area smoothed
    assert np.array_equal(img[20:60, 100:160], ref[20:60, 100:160])      # car untouched


# ---------------- Anthropic VLM provider logic (HTTP faked) ----------------
def test_anthropic_vlm_parses_json_and_falls_back_on_error(monkeypatch, tmp_path):
    import httpx
    from app.agents import vlm
    from app.core.config import settings
    img = tmp_path / "f.jpg"
    cv2.imwrite(str(img), np.zeros((10, 10, 3), np.uint8))
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    seen = {}

    class R:
        def __init__(s, ok=True): s.ok = ok
        def raise_for_status(s):
            if not s.ok: raise httpx.HTTPStatusError("boom", request=None, response=None)
        def json(s): return {"content": [{"type": "text", "text": 'Sure! {"summary": "Red car parked in a fenced area", "concern_level": "high", "rationale": "inside zone"} done'}]}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.update(url=url, headers=headers, body=json)
        return R()
    monkeypatch.setattr(httpx, "post", fake_post)
    out = vlm.analyze_with_fallback(str(img), {"track_id": 1, "risk_score": 90}, vlm.AnthropicVLM())
    assert out["provider"] == "anthropic" and out["concern_level"] == "high" and "parked" in out["summary"]
    assert seen["headers"]["x-api-key"] == "test-key" and seen["body"]["messages"][0]["content"][0]["type"] == "image"
    assert vlm.risk_adjustment(out) == 5

    monkeypatch.setattr(httpx, "post", lambda *a, **k: R(ok=False))
    out = vlm.analyze_with_fallback(str(img), {"track_id": 1, "risk_score": 90}, vlm.AnthropicVLM())
    assert out["provider"] == "stub" and "fallback" in out["rationale"]            # never breaks the pipeline
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    assert vlm.analyze_with_fallback(str(img), {}, vlm.AnthropicVLM())["provider"] == "stub"


# ---------------- zone drawing on the video frame (end to end) ----------------
state = {}


def test_draw_zone_on_video_then_pipeline_flags_incident(client, demo_assets, monkeypatch):
    from app.core.config import settings
    from app.geospatial.geo import haversine_m
    video, zone_file = demo_assets
    adm, op = login(client, "admin"), login(client, "operator")
    for z in client.get(f"{P}/zones", headers=adm).json():                     # isolate from zones other tests made
        client.delete(f"{P}/zones/{z['id']}", headers=adm)
    m = client.post(f"{P}/missions", json={"name": "Draw-zone mission"}, headers=op).json()
    with open(video, "rb") as f:
        v = client.post(f"{P}/videos/upload", data={"mission_id": m["id"]}, files={"file": ("d.mp4", f, "video/mp4")}, headers=op).json()
    assert v["proc_size"] == [960, 540] and v["width"] == 960
    prev = client.get(f"{P}/videos/{v['id']}/preview", headers=op)
    assert prev.status_code == 200 and prev.headers["content-type"] == "image/jpeg" and prev.headers["x-frame-size"] == "960x540"

    # validation + RBAC
    body = {"name": "Drawn Zone", "zone_type": "restricted", "points": ZONE_PX}
    assert client.post(f"{P}/videos/{v['id']}/zone-from-pixels", json=body, headers=op).status_code == 403
    assert client.post(f"{P}/videos/{v['id']}/zone-from-pixels", json={**body, "points": [[0, 0], [5, 5]]}, headers=adm).status_code == 422
    assert client.post(f"{P}/videos/{v['id']}/zone-from-pixels", json={**body, "points": [[0, 0], [5000, 5], [0, 9]]}, headers=adm).status_code == 422
    assert client.post(f"{P}/videos/{v['id']}/zone-from-pixels", json={**body, "points": [[10, 10], [11, 10], [12, 10]]}, headers=adm).status_code == 422

    z = client.post(f"{P}/videos/{v['id']}/zone-from-pixels", json=body, headers=adm)
    assert z.status_code == 201, z.text
    drawn = z.json()["geojson"]["coordinates"][0]
    ref = json.loads(zone_file.read_text())["geometry"]["coordinates"][0]
    assert len(drawn) == 5 and drawn[0] == drawn[-1]
    assert max(haversine_m(a[1], a[0], b[1], b[0]) for a, b in zip(drawn, ref)) < 0.05      # matches the generator's zone to 5 cm

    monkeypatch.setattr(settings, "blur_persons", True)                          # also exercise privacy mode end to end
    assert client.post(f"{P}/videos/{v['id']}/process", headers=op).status_code == 202
    assert client.get(f"{P}/videos/{v['id']}", headers=op).json()["status"] == "done"
    incs = client.get(f"{P}/incidents", params={"mission_id": m["id"]}, headers=op).json()
    car = next(i for i in incs if i["object_class"] == "car")
    assert car["severity"] == "CRITICAL" and "Restricted zone intrusion" in car["reasons"]
    assert any("Drawn Zone" in e for e in client.get(f"{P}/incidents/{car['id']}", headers=op).json()["evidence"])
    assert client.get(f"{P}/incidents/{car['id']}/frame", headers=op).status_code == 200
    state.update(video_id=v["id"], mission_id=m["id"])


# ---------------- websocket live feed ----------------
def test_websocket_requires_auth_and_streams_pipeline_events(client):
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/events") as ws:
            ws.send_json({"token": "garbage"})
            ws.receive_json()
    token = login(client, "viewer")["Authorization"].split()[1]
    op = login(client, "operator")
    with client.websocket_connect("/ws/events") as ws:
        ws.send_json({"token": token})
        assert ws.receive_json()["type"] == "hello"
        assert client.post(f"{P}/videos/{state['video_id']}/process", headers=op).status_code == 202
        types, frames = [], []
        for _ in range(2000):
            msg = ws.receive_json()
            types.append(msg["type"])
            if msg["type"] == "frame":
                frames.append(msg)
            if msg["type"] == "video_done":
                assert msg["incidents"] >= 1
                break
        assert "incident" in types and types[-1] == "video_done"
        first_inc = types.index("incident")
        assert all(t != "frame" for t in types[first_inc:])    # incidents are announced only after the DB commit, so clients can fetch them at once
        assert len(frames) >= 15                                         # throttled live view (<=15 fps when not live-paced)
        f = frames[len(frames) // 2]
        import base64
        jpg = cv2.imdecode(np.frombuffer(base64.b64decode(f["jpeg"]), np.uint8), cv2.IMREAD_COLOR)
        assert jpg.shape[1] == 640 and f["objects"] and {"id", "cls", "conf", "lat", "lon"} <= set(f["objects"][0])
        assert f["tel"]["alt"] > 50 and f["n"] <= f["of"] + 1


# ---------------- prometheus ----------------
def test_prometheus_metrics(client):
    assert client.get(f"{P}/system/prometheus", headers=login(client, "viewer")).status_code == 403
    r = client.get(f"{P}/system/prometheus", headers=login(client, "operator"))
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    assert 'dronevision_incidents_total{severity="CRITICAL"}' in r.text
    assert 'dronevision_videos_total{status="done"}' in r.text
    for line in r.text.splitlines():                                                  # well-formed exposition
        assert line.startswith("#") or len(line.split(" ")) == 2


def test_road_zone_type_accepted(client):
    adm = login(client, "admin")
    g = {"type": "Polygon", "coordinates": [[[67.001, 24.860], [67.002, 24.860], [67.002, 24.861], [67.001, 24.861], [67.001, 24.860]]]}
    assert client.post(f"{P}/zones", json={"name": "Road 1", "zone_type": "road", "geojson": g}, headers=adm).status_code == 201


def test_live_mode_runs_in_real_time_and_normal_mode_does_not(client):
    import time
    op = login(client, "operator")
    t0 = time.time()
    assert client.post(f"{P}/videos/{state['video_id']}/process", headers=op).status_code == 202
    fast = time.time() - t0
    t0 = time.time()
    assert client.post(f"{P}/videos/{state['video_id']}/process", params={"live": "true"}, headers=op).status_code == 202
    live = time.time() - t0
    assert client.get(f"{P}/videos/{state['video_id']}", headers=op).json()["status"] == "done"
    assert live >= 17 and fast < 12            # demo clip is 20 s long; live mode paces to ~video time
