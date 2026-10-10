import pytest
from app.agents.decision import decide
from app.core.config import Settings
from app.geospatial.geo import make_box, offset_latlon
from app.risk.engine import assess, severity_for
from app.services.events import detect_events
from app.services.types import EventDef, Point, TrackData, ZoneDef
from app.tracking.tracker import ByteLikeTracker, iou
from app.tracking.detector import Det

LAT, LON = 24.8607, 67.0011
CFG = Settings()
ZONE = ZoneDef(1, "Restricted Zone A", "restricted", make_box(LAT, LON, 40, 40))
FAR = (60, 0)  # metres east: outside the zone


def track(tid, cls, path, dt=0.25):
    """path: list of (east_m, north_m) offsets, one per sample."""
    t = TrackData(tid, cls)
    for i, (e, n) in enumerate(path):
        lat, lon = offset_latlon(LAT, LON, e, n)
        t.points.append(Point(i * dt, i, lat, lon, 0, 0))
    return t


def test_severity_bands_follow_spec():
    assert [severity_for(s) for s in (0, 25, 26, 50, 51, 75, 76, 100)] == \
        ["LOW", "LOW", "MEDIUM", "MEDIUM", "HIGH", "HIGH", "CRITICAL", "CRITICAL"]


def test_risk_is_bounded_and_explainable():
    evs = [EventDef("FIRE_SMOKE", 1, 0, 1, LAT, LON), EventDef("ZONE_INTRUSION", 1, 0, 1, LAT, LON, details={"zone_type": "restricted"})]
    r = assess(evs)
    assert r["risk_score"] == 100 and r["severity"] == "CRITICAL" and r["factors"]


def test_risk_zone_type_multiplier_and_info_events_ignored():
    assert assess([EventDef("ZONE_INTRUSION", 1, 0, 1, 0, 0, details={"zone_type": "monitoring"})])["risk_score"] == 12
    assert assess([EventDef("OBJECT_ENTRY", 1, 0, 0, 0, 0)])["risk_score"] == 0


def test_vlm_adjust_is_clamped():
    base = [EventDef("LOITERING", 1, 0, 1, 0, 0)]
    assert assess(base, vlm_adjust=999)["risk_score"] == 25
    assert assess(base, vlm_adjust=-999)["risk_score"] == 5


def test_stationary_vehicle_in_zone_generates_intrusion_and_stationary():
    path = [(-50 + i * 6, 0) for i in range(10)] + [(0, 0)] * 30          # drive in, then park for 7.5 s
    ev = detect_events([track(1, "car", path)], [ZONE], CFG)
    types = {e.event_type for e in ev}
    assert {"ZONE_INTRUSION", "STATIONARY"} <= types
    st = next(e for e in ev if e.event_type == "STATIONARY")
    assert st.details["in_alert_zone"] and st.details["duration_s"] >= CFG.stationary_min_s


def test_moving_vehicle_outside_zone_generates_no_incident_events():
    ev = detect_events([track(1, "car", [(FAR[0] + i * 6, 30) for i in range(30)])], [ZONE], CFG)
    assert {e.event_type for e in ev} <= {"OBJECT_ENTRY", "OBJECT_EXIT"}


def test_person_vehicle_proximity():
    car = track(1, "car", [(FAR[0], 0)] * 30)
    near = track(2, "person", [(FAR[0] + 3, 2)] * 30)
    far = track(3, "person", [(FAR[0] + 40, 2)] * 30)
    ev = detect_events([car, near, far], [], CFG)
    prox = [e for e in ev if e.event_type == "PERSON_VEHICLE_PROXIMITY"]
    assert len(prox) == 1 and prox[0].related_track_id == 2 and prox[0].details["vehicle_stationary"]


def test_forbidden_classes_filter():
    z = ZoneDef(2, "Vehicles only", "restricted", make_box(LAT, LON, 40, 40), forbidden_classes=["car"])
    person = track(5, "person", [(i * 0.3, 0) for i in range(20)])
    assert not [e for e in detect_events([person], [z], CFG) if e.event_type == "ZONE_INTRUSION"]


def test_zone_dwell_limit_loitering():
    z = ZoneDef(3, "Short stay", "inspection", make_box(LAT, LON, 40, 40), max_dwell_s=2.0)
    ev = detect_events([track(1, "person", [(i * 0.5, 0) for i in range(20)])], [z], CFG)
    assert any(e.event_type == "LOITERING" for e in ev)


def test_erratic_movement_is_unusual_activity():
    path = [(FAR[0] + (6 if i % 2 else 0), 0) for i in range(12)]
    ev = detect_events([track(1, "person", path)], [], CFG)
    assert any(e.event_type == "UNUSUAL_ACTIVITY" for e in ev)


def test_fire_smoke_event():
    ev = detect_events([track(9, "fire", [(0, 0)] * 4)], [], CFG)
    assert any(e.event_type == "FIRE_SMOKE" for e in ev)


@pytest.mark.parametrize("sev,review", [("CRITICAL", True), ("HIGH", True), ("MEDIUM", False), ("LOW", False)])
def test_decision_agent_enforces_human_review_and_no_physical_actions(sev, review):
    d = decide(sev, [], "car", set())
    assert d["requires_review"] is review
    assert d["autonomous_physical_action"] is False
    assert all(a in ("LOG", "ALERT", "REQUEST_REVIEW", "GENERATE_REPORT", "ESCALATE", "TRACK") for a in d["actions"])


def test_tracker_keeps_ids_and_separates_objects():
    tr = ByteLikeTracker()
    ids_a, ids_b = set(), set()
    for f in range(30):
        dets = [Det("car", 0.9, (10 + f * 5, 100, 50 + f * 5, 124)), Det("car", 0.9, (400 - f * 5, 200, 440 - f * 5, 224))]
        for t, d in tr.update(dets, f):
            (ids_a if d.bbox[1] == 100 else ids_b).add(t.id)
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b


def test_tracker_survives_short_occlusion():
    tr, seen = ByteLikeTracker(), set()
    for f in range(40):
        dets = [] if 15 <= f < 18 else [Det("car", 0.9, (10 + f * 4, 100, 50 + f * 4, 124))]
        seen |= {t.id for t, _ in tr.update(dets, f)}
    assert len(seen) == 1


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0


def test_blob_detector_separates_touching_objects_of_different_colour():
    import cv2, numpy as np
    from app.tracking.detector import BlobDetector
    f = np.zeros((200, 300, 3), np.uint8); f[:] = (60, 140, 60)
    cv2.rectangle(f, (50, 50), (94, 74), (200, 80, 30), -1)      # blue car
    cv2.rectangle(f, (50, 75), (94, 99), (20, 110, 230), -1)     # orange car, touching
    cv2.circle(f, (200, 100), 7, (180, 40, 200), -1)             # person
    d = BlobDetector().detect(f)
    assert sorted(x.cls for x in d) == ["car", "car", "person"]


@pytest.mark.parametrize("fps", ["5", "8", "10"])
def test_demo_scene_tracks_are_stable_across_sampling_rates(fps):
    import pathlib, cv2
    from app.tracking.detector import BlobDetector
    video = pathlib.Path(__file__).resolve().parents[2] / "datasets" / "demo_aerial.mp4"
    cap, step = cv2.VideoCapture(str(video)), round(15 / float(fps))
    det, tr, pts, cls, i = BlobDetector(), ByteLikeTracker(), {}, {}, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            for t, _ in tr.update(det.detect(f), i):
                pts[t.id] = pts.get(t.id, 0) + 1; cls[t.id] = t.cls
        i += 1
    real = [cls[k] for k, v in pts.items() if v >= 3]
    assert sorted(real) == ["car", "car", "car", "person", "person"]


def test_tracker_keeps_one_id_when_class_flickers_car_truck_bus():
    tr, ids, classes = ByteLikeTracker(), set(), set()
    flick = ["car", "truck", "car", "car", "bus", "car", "truck", "car"] * 4
    for f, c in enumerate(flick):
        for t, _ in tr.update([Det(c, 0.8, (10 + f * 4, 100, 50 + f * 4, 124))], f):
            ids.add(t.id); classes.add(t.cls)
    assert len(ids) == 1
    assert tr.tracks[0].cls == "car"                 # majority vote, not last-frame class


def test_tracker_does_not_merge_person_with_vehicle():
    tr = ByteLikeTracker()
    for f in range(6):
        tr.update([Det("car", 0.9, (100, 100, 140, 124)), Det("person", 0.9, (104, 104, 112, 112))], f)
    assert sorted(t.cls for t in tr.tracks) == ["car", "person"]


def test_yolo_detector_output_parsing_with_fake_model():
    """Verifies COCO mapping/bbox parsing against the ultralytics result shape (model itself is faked)."""
    from app.tracking.detector import YoloDetector

    class T:                      # mimics a 1-element tensor
        def __init__(s, v): s.v = v
        def __int__(s): return int(s.v)
        def __float__(s): return float(s.v)
    class Box:
        def __init__(s, c, p, xy): s.cls, s.conf = T(c), T(p); s.xyxy = [type("A", (), {"tolist": lambda self: xy})()]
    class Res: boxes = [Box(2, 0.9, [10, 20, 60, 50]), Box(0, 0.5, [5, 5, 15, 25]), Box(16, 0.9, [0, 0, 9, 9])]  # car, person, dog(ignored)
    class Model:
        def predict(s, *a, **k): return [Res()]
    d = YoloDetector(); d._model = Model()
    out = d.detect(None, 0.3)
    assert [(x.cls, x.bbox) for x in out] == [("car", (10, 20, 60, 50)), ("person", (5, 5, 15, 25))]
