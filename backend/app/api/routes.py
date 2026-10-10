import resource
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Literal

import cv2
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..agents.assistant import Tools, route
from ..core.config import settings
from ..core.security import create_token, hash_password, throttle, verify_password
from ..geospatial.geo import GeoMapper, GPSSimulator, extract_geometry
from ..models.db import (AuditLog, Detection, Event, Incident, Mission, ModelRecord, Report, Review, SessionLocal,
                         Track, User, Video, Zone)
from ..rag.store import PolicyStore
from ..reports import generator
from ..services.pipeline import process_video, processed_size
from .deps import audit, current_user, get_db, require

router = APIRouter(prefix="/api/v1")
ZONE_TYPES = {"restricted", "inspection", "emergency", "construction", "monitoring", "road", "custom"}
_proc_lock = threading.Semaphore(1)  # one heavy job at a time (3 GB RAM)
_policies = None
_started = time.time()


def policies() -> PolicyStore:
    global _policies
    if _policies is None:
        _policies = PolicyStore()
    return _policies


# ---------- serializers ----------
def s_inc(i: Incident):
    return {c: getattr(i, c) for c in ("id", "code", "mission_id", "video_id", "track_id", "object_class", "primary_type",
            "risk_score", "severity", "reasons", "recommendation", "actions", "requires_review", "status", "start_t",
            "end_t", "lat", "lon", "zone_id", "evidence", "vlm", "model_versions")} | {
        "has_frame": bool(i.key_frame), "created_at": i.created_at.isoformat()}


def s_event(e: Event):
    return {c: getattr(e, c) for c in ("id", "mission_id", "video_id", "incident_id", "event_type", "track_id",
            "related_track_id", "zone_id", "start_t", "end_t", "lat", "lon", "details")}


def s_zone(z: Zone):
    return {"id": z.id, "name": z.name, "zone_type": z.zone_type, "geojson": z.geojson,
            "forbidden_classes": z.forbidden_classes, "max_dwell_s": z.max_dwell_s, "active": z.active}


def s_video(v: Video):
    return {"id": v.id, "code": v.code, "mission_id": v.mission_id, "filename": v.filename, "status": v.status,
            "progress": v.progress, "error": v.error, "fps": v.fps, "width": v.width, "height": v.height,
            "frame_count": v.frame_count, "meta": v.meta, "has_annotated": bool(v.annotated_path),
            "proc_size": list(processed_size(v.width, v.height)[:2]) if v.width and v.height else None}


def s_mission(m: Mission, db: Session):
    return {"id": m.id, "code": m.code, "name": m.name, "description": m.description, "status": m.status,
            "start_lat": m.start_lat, "start_lon": m.start_lon, "altitude": m.altitude, "heading": m.heading,
            "path_mode": m.path_mode, "videos": [s_video(v) for v in m.videos],
            "incident_count": db.query(Incident).filter(Incident.mission_id == m.id).count(),
            "created_at": m.created_at.isoformat()}


# ---------- auth ----------
class LoginBody(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=256)


@router.post("/auth/login")
def login(body: LoginBody, request: Request, db: Session = Depends(get_db)):
    key = f"{request.client.host if request.client else '-'}:{body.username.lower()}"
    if throttle.blocked(key):
        raise HTTPException(429, "Too many failed attempts. Try again later.")
    user = db.query(User).filter(User.username == body.username).first()
    if not user or not user.active or not verify_password(body.password, user.password_hash):
        throttle.fail(key)
        audit(db, body.username, "login_failed", "auth", request=request)
        raise HTTPException(401, "Invalid credentials")
    throttle.reset(key)
    audit(db, user.username, "login", "auth", request=request)
    return {"access_token": create_token(user.username, user.role), "token_type": "bearer",
            "user": {"username": user.username, "role": user.role}}


@router.get("/auth/me")
def me(user: User = Depends(current_user)):
    return {"username": user.username, "role": user.role}


class NewUser(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=10, max_length=128)
    role: Literal["admin", "operator", "analyst", "viewer"]


@router.post("/users", status_code=201)
def create_user(body: NewUser, request: Request, db: Session = Depends(get_db), user: User = Depends(require("users:write"))):
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(409, "Username exists")
    db.add(User(username=body.username, password_hash=hash_password(body.password), role=body.role))
    db.commit()
    audit(db, user.username, "user_created", f"user:{body.username}", None, {"role": body.role}, request)
    return {"username": body.username, "role": body.role}


@router.get("/users")
def list_users(db: Session = Depends(get_db), user: User = Depends(require("users:write"))):
    return [{"username": u.username, "role": u.role, "active": u.active} for u in db.query(User).all()]


@router.get("/audit")
def audit_log(limit: int = 100, db: Session = Depends(get_db), user: User = Depends(require("audit:read"))):
    rows = db.query(AuditLog).order_by(AuditLog.id.desc()).limit(min(limit, 500)).all()
    return [{"id": r.id, "user": r.user, "action": r.action, "resource": r.resource, "previous_state": r.previous_state,
             "new_state": r.new_state, "ip": r.ip, "timestamp": r.created_at.isoformat()} for r in rows]


@router.get("/models")
def list_models(db: Session = Depends(get_db), user: User = Depends(require("models:read"))):
    return [{"name": m.name, "version": m.version, "task": m.task, "dataset": m.dataset, "metrics": m.metrics,
             "status": m.status, "created": m.created_at.isoformat()} for m in db.query(ModelRecord).all()]


# ---------- missions ----------
class MissionBody(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str = Field("", max_length=2000)
    start_lat: float = Field(24.8607, ge=-90, le=90)
    start_lon: float = Field(67.0011, ge=-180, le=180)
    altitude: float = Field(60.0, gt=1, le=500)
    heading: float = Field(0.0, ge=0, lt=360)
    path_mode: Literal["hover", "linear"] = "hover"
    hfov_deg: float = Field(84.0, gt=10, lt=170)
    seed: int = 42


@router.post("/missions", status_code=201)
def create_mission(body: MissionBody, request: Request, db: Session = Depends(get_db), user: User = Depends(require("missions:write"))):
    m = Mission(code="TMP", created_by=user.username, **body.model_dump())
    db.add(m)
    db.flush()
    m.code = f"MISSION-{m.id:03d}"
    db.commit()
    audit(db, user.username, "mission_created", m.code, None, body.model_dump(), request)
    return s_mission(m, db)


@router.get("/missions")
def list_missions(db: Session = Depends(get_db), user: User = Depends(require("missions:read"))):
    return [s_mission(m, db) for m in db.query(Mission).order_by(Mission.id.desc()).all()]


@router.get("/missions/{mid}")
def get_mission(mid: int, db: Session = Depends(get_db), user: User = Depends(require("missions:read"))):
    m = db.get(Mission, mid)
    if not m:
        raise HTTPException(404, "Mission not found")
    return s_mission(m, db)


@router.get("/missions/{mid}/geojson")
def mission_geojson(mid: int, db: Session = Depends(get_db), user: User = Depends(require("missions:read"))):
    """Map layer: zones, object trajectories and incident markers as one FeatureCollection."""
    if not db.get(Mission, mid):
        raise HTTPException(404, "Mission not found")
    feats = [{"type": "Feature", "properties": {"kind": "zone", "name": z.name, "zone_type": z.zone_type, "id": z.id},
              "geometry": z.geojson if z.geojson.get("type") in ("Polygon", "MultiPolygon") else extract_geometry(z.geojson)}
             for z in db.query(Zone).filter(Zone.active.is_(True))]
    for t in db.query(Track).filter(Track.mission_id == mid):
        feats.append({"type": "Feature", "properties": {"kind": "trajectory", "track_id": t.track_id, "class": t.object_class},
                      "geometry": {"type": "LineString", "coordinates": [[p[2], p[1]] for p in t.trajectory]}})
    for i in db.query(Incident).filter(Incident.mission_id == mid):
        feats.append({"type": "Feature", "properties": {"kind": "incident", "code": i.code, "severity": i.severity,
                      "risk_score": i.risk_score, "object": f"{i.object_class} #{i.track_id}", "status": i.status, "id": i.id},
                      "geometry": {"type": "Point", "coordinates": [i.lon, i.lat]}})
    return {"type": "FeatureCollection", "features": feats}


# ---------- videos ----------
@router.post("/videos/upload", status_code=201)
async def upload_video(request: Request, mission_id: int = Form(...), file: UploadFile = File(...),
                       db: Session = Depends(get_db), user: User = Depends(require("videos:upload"))):
    mission = db.get(Mission, mission_id)
    if not mission:
        raise HTTPException(404, "Mission not found")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.allowed_ext:
        raise HTTPException(415, f"Unsupported type. Allowed: {', '.join(settings.allowed_ext)}")
    code = f"VIDEO-{uuid.uuid4().hex[:8].upper()}"
    dest = settings.storage_dir / "videos" / f"{code}{ext}"  # server-chosen name: no path traversal
    limit, size = settings.max_upload_mb * 1024 * 1024, 0
    with open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB")
            out.write(chunk)
    cap = cv2.VideoCapture(str(dest))  # content validation, not just extension
    ok, _ = cap.read() if cap.isOpened() else (False, None)
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vfps, vframes = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not ok:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, "File is not a readable video")
    v = Video(code=code, mission_id=mission_id, filename=Path(file.filename).name[:200], path=str(dest),
              width=vw, height=vh, fps=vfps, frame_count=vframes)
    db.add(v)
    db.commit()
    audit(db, user.username, "video_uploaded", code, None, {"size": size, "mission": mission.code}, request)
    return s_video(v)


def _run(video_id: int, live: bool = False):
    with _proc_lock:
        try:
            process_video(video_id, live=live)
        except Exception:
            pass  # failure is recorded on the video row


@router.post("/videos/{vid}/process", status_code=202)
def process(vid: int, bg: BackgroundTasks, request: Request, live: bool = False, db: Session = Depends(get_db),
            user: User = Depends(require("videos:process"))):
    v = db.get(Video, vid)
    if not v:
        raise HTTPException(404, "Video not found")
    if v.status == "processing":
        raise HTTPException(409, "Already processing")
    v.status, v.progress = "processing", 0.0
    db.commit()
    audit(db, user.username, "video_process_started", v.code, None, {"detector": settings.detector}, request)
    bg.add_task(_run, vid, live)
    return s_video(v)


@router.get("/videos/{vid}")
def get_video(vid: int, db: Session = Depends(get_db), user: User = Depends(require("missions:read"))):
    v = db.get(Video, vid)
    if not v:
        raise HTTPException(404, "Video not found")
    return s_video(v)


@router.get("/videos/{vid}/stream")
def stream_video(vid: int, db: Session = Depends(get_db), user: User = Depends(require("missions:read"))):
    v = db.get(Video, vid)
    if not v or not v.annotated_path or not Path(v.annotated_path).exists():
        raise HTTPException(404, "Annotated video not available")
    return FileResponse(v.annotated_path, media_type="video/mp4")


@router.get("/videos/{vid}/preview")
def preview_frame(vid: int, db: Session = Depends(get_db), user: User = Depends(require("missions:read"))):
    """First frame at pipeline resolution - the canvas the zone-drawing UI works on."""
    v = db.get(Video, vid)
    if not v or not Path(v.path).exists():
        raise HTTPException(404, "Video not found")
    cap = cv2.VideoCapture(v.path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(422, "Cannot read video")
    pw, ph, scale = processed_size(frame.shape[1], frame.shape[0])
    if scale < 1.0:
        frame = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"X-Frame-Size": f"{frame.shape[1]}x{frame.shape[0]}"})


class PixelZoneBody(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    zone_type: str = "restricted"
    points: list[list[float]] = Field(min_length=3, max_length=200)   # [[x, y], ...] in preview-frame pixels
    forbidden_classes: list[str] = []
    max_dwell_s: float | None = Field(None, gt=0)


@router.post("/videos/{vid}/zone-from-pixels", status_code=201)
def zone_from_pixels(vid: int, body: PixelZoneBody, request: Request, db: Session = Depends(get_db),
                     user: User = Depends(require("zones:write"))):
    """Turn a polygon drawn on the video frame into a GeoJSON zone using the mission's (simulated) camera
    geometry at t=0. Zones are stored in lat/lon, so they work for any later video of the same area."""
    v = db.get(Video, vid)
    if not v or not v.width or not v.height:
        raise HTTPException(404, "Video not found")
    if body.zone_type not in ZONE_TYPES:
        raise HTTPException(422, f"zone_type must be one of {sorted(ZONE_TYPES)}")
    pw, ph, _ = processed_size(v.width, v.height)
    for x, y in body.points:
        if not (-0.05 * pw <= x <= 1.05 * pw and -0.05 * ph <= y <= 1.05 * ph):
            raise HTTPException(422, "Point outside the video frame")
    area = 0.5 * abs(sum(body.points[i][0] * body.points[(i + 1) % len(body.points)][1]
                         - body.points[(i + 1) % len(body.points)][0] * body.points[i][1] for i in range(len(body.points))))
    if area < 25:
        raise HTTPException(422, "Polygon is degenerate (area too small)")
    m = db.get(Mission, v.mission_id)
    tel = GPSSimulator(m.start_lat, m.start_lon, m.altitude, m.heading, m.path_mode, m.seed).at(0.0)
    mapper = GeoMapper(pw, ph, m.hfov_deg)
    ring = [[lon, lat] for lat, lon in (mapper.pixel_to_geo(x, y, tel) for x, y in body.points)]
    ring.append(ring[0])
    z = Zone(name=body.name.strip(), zone_type=body.zone_type, geojson={"type": "Polygon", "coordinates": [ring]},
             forbidden_classes=body.forbidden_classes, max_dwell_s=body.max_dwell_s, created_by=user.username)
    db.add(z)
    db.commit()
    audit(db, user.username, "zone_created_from_video", f"zone:{z.id}", None, {"name": z.name, "video": v.code}, request)
    return s_zone(z)


# ---------- detections / tracks ----------
@router.get("/detections")
def detections(video_id: int | None = None, track_id: int | None = None, limit: int = 500, db: Session = Depends(get_db),
               user: User = Depends(require("events:read"))):
    q = db.query(Detection)
    if video_id:
        q = q.filter(Detection.video_id == video_id)
    if track_id:
        q = q.filter(Detection.track_id == track_id)
    return [{"id": d.id, "frame_id": d.frame_id, "timestamp": d.timestamp, "track_id": d.track_id, "class": d.object_class,
             "confidence": d.confidence, "bbox": d.bbox, "lat": d.lat, "lon": d.lon, "mission_id": d.mission_id}
            for d in q.order_by(Detection.id).limit(min(limit, 5000))]


@router.get("/tracks")
def tracks(mission_id: int | None = None, db: Session = Depends(get_db), user: User = Depends(require("events:read"))):
    q = db.query(Track)
    if mission_id:
        q = q.filter(Track.mission_id == mission_id)
    return [{"id": t.id, "track_id": t.track_id, "class": t.object_class, "first_seen": t.first_seen, "last_seen": t.last_seen,
             "n_points": t.n_points, "avg_conf": t.avg_conf, "stats": t.stats, "mission_id": t.mission_id} for t in q]


# ---------- zones ----------
class ZoneBody(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    zone_type: str = "restricted"
    geojson: dict
    forbidden_classes: list[str] = []
    max_dwell_s: float | None = Field(None, gt=0)


@router.post("/zones", status_code=201)
def create_zone(body: ZoneBody, request: Request, db: Session = Depends(get_db), user: User = Depends(require("zones:write"))):
    if body.zone_type not in ZONE_TYPES:
        raise HTTPException(422, f"zone_type must be one of {sorted(ZONE_TYPES)}")
    try:
        geom = extract_geometry(body.geojson)
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise HTTPException(422, f"Invalid GeoJSON: {e}")
    z = Zone(name=body.name.strip(), zone_type=body.zone_type, geojson=geom, forbidden_classes=body.forbidden_classes,
             max_dwell_s=body.max_dwell_s, created_by=user.username)
    db.add(z)
    db.commit()
    audit(db, user.username, "zone_created", f"zone:{z.id}", None, {"name": z.name, "type": z.zone_type}, request)
    return s_zone(z)


@router.get("/zones")
def list_zones(db: Session = Depends(get_db), user: User = Depends(require("zones:read"))):
    return [s_zone(z) for z in db.query(Zone).all()]


@router.delete("/zones/{zid}")
def delete_zone(zid: int, request: Request, db: Session = Depends(get_db), user: User = Depends(require("zones:write"))):
    z = db.get(Zone, zid)
    if not z:
        raise HTTPException(404, "Zone not found")
    prev = s_zone(z)
    z.active = False  # soft delete keeps history intact
    db.commit()
    audit(db, user.username, "zone_deactivated", f"zone:{zid}", {"active": True}, {"active": False}, request)
    return {"ok": True, "previous": prev["name"]}


# ---------- events / incidents ----------
@router.get("/events")
def list_events(mission_id: int | None = None, event_type: str | None = None, limit: int = 500, db: Session = Depends(get_db),
                user: User = Depends(require("events:read"))):
    q = db.query(Event)
    if mission_id:
        q = q.filter(Event.mission_id == mission_id)
    if event_type:
        q = q.filter(Event.event_type == event_type)
    return [s_event(e) for e in q.order_by(Event.start_t).limit(min(limit, 2000))]


@router.get("/events/{eid}")
def get_event(eid: int, db: Session = Depends(get_db), user: User = Depends(require("events:read"))):
    e = db.get(Event, eid)
    if not e:
        raise HTTPException(404, "Event not found")
    return s_event(e)


@router.get("/incidents")
def list_incidents(severity: str | None = None, status: str | None = None, mission_id: int | None = None,
                   db: Session = Depends(get_db), user: User = Depends(require("incidents:read"))):
    q = db.query(Incident)
    if severity:
        q = q.filter(Incident.severity == severity.upper())
    if status:
        q = q.filter(Incident.status == status)
    if mission_id:
        q = q.filter(Incident.mission_id == mission_id)
    return [s_inc(i) for i in q.order_by(Incident.risk_score.desc())]


@router.get("/incidents/{iid}")
def get_incident(iid: int, db: Session = Depends(get_db), user: User = Depends(require("incidents:read"))):
    i = db.get(Incident, iid)
    if not i:
        raise HTTPException(404, "Incident not found")
    reviews = db.query(Review).filter(Review.incident_id == iid).order_by(Review.id).all()
    evs = db.query(Event).filter(Event.incident_id == iid).all()
    return s_inc(i) | {"events": [s_event(e) for e in evs],
                       "reviews": [{"operator": r.operator, "decision": r.decision, "comment": r.comment,
                                    "timestamp": r.created_at.isoformat()} for r in reviews]}


@router.get("/incidents/{iid}/frame")
def incident_frame(iid: int, db: Session = Depends(get_db), user: User = Depends(require("incidents:read"))):
    i = db.get(Incident, iid)
    if not i or not i.key_frame or not Path(i.key_frame).exists():
        raise HTTPException(404, "No evidence frame")
    return FileResponse(i.key_frame, media_type="image/jpeg")


class ReviewBody(BaseModel):
    decision: Literal["confirmed", "dismissed", "investigating"]
    comment: str = Field("", max_length=1000)


@router.post("/incidents/{iid}/review")
def review_incident(iid: int, body: ReviewBody, request: Request, db: Session = Depends(get_db),
                    user: User = Depends(require("incidents:review"))):
    i = db.get(Incident, iid)
    if not i:
        raise HTTPException(404, "Incident not found")
    if body.decision == "dismissed" and len(body.comment.strip()) < 3:
        raise HTTPException(422, "A short comment is required when dismissing an incident")
    prev = i.status
    i.status = body.decision
    db.add(Review(incident_id=iid, operator=user.username, decision=body.decision, comment=body.comment.strip()))
    db.commit()
    audit(db, user.username, "incident_reviewed", i.code, {"status": prev}, {"status": i.status, "comment": body.comment}, request)
    return {"incident": i.code, "status": i.status}


# ---------- reports ----------
class ReportBody(BaseModel):
    mission_id: int
    incident_id: int | None = None


@router.post("/reports", status_code=201)
def create_report(body: ReportBody, request: Request, db: Session = Depends(get_db), user: User = Depends(require("reports:create"))):
    m = db.get(Mission, body.mission_id)
    if not m:
        raise HTTPException(404, "Mission not found")
    rep = generator.generate(db, m, user.username, body.incident_id)
    audit(db, user.username, "report_generated", f"report:{rep.id}", None, {"mission": m.code}, request)
    return {"id": rep.id, "kind": rep.kind, "summary": rep.summary, "mission_id": rep.mission_id}


@router.get("/reports")
def list_reports(db: Session = Depends(get_db), user: User = Depends(require("reports:read"))):
    return [{"id": r.id, "kind": r.kind, "mission_id": r.mission_id, "summary": r.summary, "created_by": r.created_by,
             "created_at": r.created_at.isoformat()} for r in db.query(Report).order_by(Report.id.desc())]


@router.get("/reports/{rid}")
def get_report(rid: int, db: Session = Depends(get_db), user: User = Depends(require("reports:read"))):
    r = db.get(Report, rid)
    if not r:
        raise HTTPException(404, "Report not found")
    import json
    return json.loads(Path(r.path_json).read_text(encoding="utf8"))


@router.get("/reports/{rid}/download")
def download_report(rid: int, request: Request, format: Literal["pdf", "json", "csv"] = "pdf", db: Session = Depends(get_db),
                    user: User = Depends(require("reports:read"))):
    r = db.get(Report, rid)
    if not r:
        raise HTTPException(404, "Report not found")
    path = {"pdf": r.path_pdf, "json": r.path_json, "csv": r.path_csv}[format]
    audit(db, user.username, "report_downloaded", f"report:{rid}", None, {"format": format}, request)
    return FileResponse(path, filename=f"report_{rid}.{format}")


# ---------- assistant ----------
class AssistantBody(BaseModel):
    question: str = Field(min_length=2, max_length=500)


@router.post("/assistant/query")
def assistant(body: AssistantBody, request: Request, db: Session = Depends(get_db), user: User = Depends(require("assistant:use"))):
    res = route(body.question.strip(), Tools(db, policies(), user.username, user.role))
    audit(db, user.username, "assistant_query", "assistant", None, {"q": body.question[:200], "tools": [c["tool"] for c in res["tool_calls"]]}, request)
    return res


# ---------- analytics / system ----------
@router.get("/analytics/overview")
def overview(db: Session = Depends(get_db), user: User = Depends(require("analytics:read"))):
    inc = db.query(Incident).all()
    ev = db.query(Event).all()
    dets = db.query(Detection.object_class).all()
    return {
        "active_missions": db.query(Mission).filter(Mission.status == "processing").count(),
        "total_missions": db.query(Mission).count(),
        "total_objects": db.query(Track).count(),
        "active_incidents": sum(1 for i in inc if i.status in ("pending", "investigating")),
        "high_risk_events": sum(1 for i in inc if i.severity in ("HIGH", "CRITICAL")),
        "object_counts": dict(Counter(t[0] for t in db.query(Track.object_class))),
        "events_by_type": dict(Counter(e.event_type for e in ev)),
        "risk_distribution": dict(Counter(i.severity for i in inc)),
        "zone_violations": sum(1 for e in ev if e.event_type == "ZONE_INTRUSION"),
        "detections_total": len(dets),
        "events_over_time": sorted(Counter(int(e.start_t // 5) * 5 for e in ev if e.event_type not in ("OBJECT_ENTRY", "OBJECT_EXIT")).items()),
    }


@router.get("/system/health")
def health():
    return {"status": "ok", "uptime_s": round(time.time() - _started, 1), "detector": settings.detector,
            "vlm_provider": settings.vlm_provider, "privacy_blur": settings.blur_persons}


@router.get("/system/prometheus")
def prometheus(db: Session = Depends(get_db), user: User = Depends(require("system:read"))):
    """Prometheus text exposition (scrape with a bearer token)."""
    vids = db.query(Video).all()
    done = [(v.meta or {}).get("processing") for v in vids if v.status == "done" and (v.meta or {}).get("processing")]
    done = [p for p in done if not p.get("live_paced")]
    sev = Counter(i.severity for i in db.query(Incident))
    lines = [
        "# TYPE dronevision_uptime_seconds gauge", f"dronevision_uptime_seconds {time.time() - _started:.1f}",
        "# TYPE dronevision_memory_rss_mb gauge", f"dronevision_memory_rss_mb {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}",
        "# TYPE dronevision_videos_total gauge",
        *[f'dronevision_videos_total{{status="{s}"}} {sum(1 for v in vids if v.status == s)}' for s in ("uploaded", "processing", "done", "failed")],
        "# TYPE dronevision_incidents_total gauge",
        *[f'dronevision_incidents_total{{severity="{s}"}} {sev.get(s, 0)}' for s in ("LOW", "MEDIUM", "HIGH", "CRITICAL")],
        "# TYPE dronevision_processing_fps gauge",
        f"dronevision_processing_fps {(sum(p['effective_fps'] or 0 for p in done) / len(done)) if done else 0:.2f}",
        "# TYPE dronevision_worker_busy gauge", f"dronevision_worker_busy {1 if _proc_lock._value == 0 else 0}", ""]
    return Response("\n".join(lines), media_type="text/plain; version=0.0.4")


@router.get("/system/metrics")
def metrics(db: Session = Depends(get_db), user: User = Depends(require("system:read"))):
    vids = db.query(Video).filter(Video.status == "done").all()
    procs = [(v.meta or {}).get("processing") for v in vids if (v.meta or {}).get("processing")]
    n_done = len(procs)
    procs = [p for p in procs if not p.get("live_paced")]   # live mode is deliberately real-time paced; exclude from speed averages
    return {"memory_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "videos_processed": n_done,
            "avg_processing_fps": round(sum(p["effective_fps"] or 0 for p in procs) / len(procs), 2) if procs else None,
            "avg_processing_seconds": round(sum(p["seconds"] for p in procs) / len(procs), 2) if procs else None,
            "failed_jobs": db.query(Video).filter(Video.status == "failed").count(),
            "queue_busy": _proc_lock._value == 0}
