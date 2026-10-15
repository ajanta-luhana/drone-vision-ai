"""End-to-end pipeline:
video -> frame sampling -> detection -> tracking -> geo-projection -> temporal/event engine ->
risk engine -> (selected-frame) VLM -> decision agent -> incidents + annotated video.

Memory notes (3 GB machine): frames are streamed (never buffered), processing is down-sampled in time and
resolution, evidence frames are re-read on demand, and the VLM only sees a handful of key frames.
"""
import base64
import resource
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..agents.decision import decide
from ..agents.vlm import analyze_with_fallback, get_provider, risk_adjustment
from ..core.config import settings
from ..geospatial.geo import GeoMapper, GPSSimulator, extract_geometry
from ..models.db import (Detection, Event, Incident, Mission, Review, SessionLocal, Track, Video, Zone)
from ..risk.engine import INFO_EVENTS, assess
from ..tracking.detector import get_detector
from ..tracking.tracker import ByteLikeTracker
from .events import detect_events
from . import notify
from .hub import hub
from .temporal import track_stats
from .types import EventDef, Point, TrackData, ZoneDef

MAX_VLM_CALLS = 3


def processed_size(W: int, H: int):
    """Resolution the pipeline works at (and that zone-drawing pixel coordinates refer to)."""
    scale = min(1.0, settings.max_width / W)
    return int(W * scale) // 2 * 2, int(H * scale) // 2 * 2, scale

COLORS = {"car": (255, 160, 40), "truck": (255, 120, 0), "bus": (255, 120, 0), "person": (60, 220, 255),
          "motorcycle": (255, 200, 0), "bicycle": (255, 200, 0), "fire": (0, 0, 255), "smoke": (140, 140, 140)}
EVENT_PRIORITY = ["FIRE_SMOKE", "ZONE_INTRUSION", "PERSON_VEHICLE_PROXIMITY", "OBSTACLE", "STATIONARY", "LOITERING", "UNUSUAL_ACTIVITY"]


def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def to_h264(src: Path, dst: Path):
    """Browser-playable H.264; falls back to the raw mp4v file if ffmpeg is unavailable."""
    exe = _ffmpeg()
    if exe:
        r = subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], capture_output=True)
        if r.returncode == 0 and dst.exists():
            src.unlink(missing_ok=True)
            return
    shutil.move(str(src), str(dst))


def load_zone_defs(db) -> list[ZoneDef]:
    out = []
    for z in db.query(Zone).filter(Zone.active.is_(True)).all():
        try:
            out.append(ZoneDef(z.id, z.name, z.zone_type, extract_geometry(z.geojson), z.forbidden_classes or [], z.max_dwell_s))
        except ValueError:
            continue
    return out


def blur_people(img, dets_now, pad=4):
    """Privacy mode: irreversibly blur person regions in anything we store/show."""
    h, w = img.shape[:2]
    for _tid, cls, _conf, bbox in dets_now:
        if cls == "person":
            x1, y1, x2, y2 = max(0, bbox[0] - pad), max(0, bbox[1] - pad), min(w, bbox[2] + pad), min(h, bbox[3] + pad)
            if x2 > x1 and y2 > y1:
                img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (0, 0), max(3.0, (x2 - x1) / 3.0))
    return img


def draw_frame(frame, dets_now, trails, zones, mapper, tel, t, highlight=None, caption=None):
    out = frame.copy()
    if settings.blur_persons:
        blur_people(out, dets_now)
    for z in zones:
        ring = z.geometry["coordinates"][0] if z.geometry["type"] == "Polygon" else z.geometry["coordinates"][0][0]
        pts = np.array([mapper.geo_to_pixel(lat, lon, tel) for lon, lat in ring], np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255) if z.type == "restricted" else (0, 200, 255))
        out = cv2.addWeighted(overlay, 0.22, out, 0.78, 0)
        cv2.polylines(out, [pts], True, (0, 0, 255) if z.type == "restricted" else (0, 200, 255), 2)
        cv2.putText(out, z.name, (int(pts[:, 0].min()) + 4, int(pts[:, 1].min()) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    for tid, tr in trails.items():
        if len(tr) > 1:
            cv2.polylines(out, [np.array(tr[-40:], np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
    for tid, cls, conf, bbox in dets_now:
        x1, y1, x2, y2 = bbox
        hot = highlight is not None and tid == highlight
        col = (0, 0, 255) if hot else COLORS.get(cls, (0, 255, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 3 if hot else 2)
        label = f"{cls} #{tid} {conf * 100:.0f}%"
        cv2.putText(out, label, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    hud = f"t={t:6.2f}s  GPS {tel.lat:.5f},{tel.lon:.5f}  alt {tel.altitude:.0f}m"
    cv2.putText(out, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if caption:
        cv2.rectangle(out, (0, out.shape[0] - 26), (out.shape[1], out.shape[0]), (0, 0, 0), -1)
        cv2.putText(out, caption, (8, out.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def _evidence(events: list[EventDef]) -> list[str]:
    ev = []
    for e in events:
        d = e.details
        if e.event_type == "ZONE_INTRUSION":
            ev.append(f"Entered {d.get('zone', 'zone')} ({d.get('zone_type')})")
        elif e.event_type == "STATIONARY":
            ev.append(f"Stationary for {d['duration_s']:.0f}s")
        elif e.event_type == "PERSON_VEHICLE_PROXIMITY":
            ev.append(f"Person #{e.related_track_id} within {d['min_distance_m']} m for {d['duration_s']:.1f}s")
        elif e.event_type == "LOITERING":
            ev.append(f"Loitering ({d.get('reason', '')}, {d.get('dwell_s', d.get('duration_s', 0)):.0f}s)")
        elif e.event_type == "OBSTACLE":
            ev.append(f"{d.get('class', 'object')} obstructing {d.get('zone') or 'road'} for {d['duration_s']:.0f}s")
        elif e.event_type == "UNUSUAL_ACTIVITY":
            ev.append(f"Unusual activity: {d.get('reason')}")
        elif e.event_type == "FIRE_SMOKE":
            ev.append(f"{d.get('class', 'fire/smoke')} detected in {d.get('n_detections')} frames")
    return ev


def build_incidents(events: list[EventDef], tracks: dict[int, TrackData], cfg=settings, vlm_hook=None):
    """Pure-ish grouping + scoring. `vlm_hook(group_info) -> vlm dict | None` lets the caller attach VLM output."""
    groups = defaultdict(list)
    for e in events:
        if e.event_type not in INFO_EVENTS:
            groups[e.track_id].append(e)
    results = []
    for tid, evs in groups.items():
        base = assess(evs, cfg.stationary_min_s)
        results.append({"track_id": tid, "events": evs, "assessment": base})
    results.sort(key=lambda r: -r["assessment"]["risk_score"])
    calls = 0
    for r in results:
        if vlm_hook and r["assessment"]["risk_score"] >= 51 and calls < MAX_VLM_CALLS:
            vlm = vlm_hook(r)
            calls += 1
            if vlm:
                r["vlm"] = vlm
                r["assessment"] = assess(r["events"], cfg.stationary_min_s, risk_adjustment(vlm))
    return results


def process_video(video_id: int, session_factory=SessionLocal, live: bool = False) -> dict:
    db = session_factory()
    t_start = time.time()
    video = db.get(Video, video_id)
    try:
        mission = db.get(Mission, video.mission_id)
        video.status, video.progress, video.error = "processing", 0.0, None
        mission.status = "processing"
        db.commit()

        cap = cv2.VideoCapture(video.path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        pw, ph, scale = processed_size(W, H)
        step = max(1, round(fps / settings.process_fps))
        out_fps = fps / step
        video.fps, video.width, video.height, video.frame_count = fps, W, H, total
        db.commit()

        mapper = GeoMapper(pw, ph, mission.hfov_deg)
        sim = GPSSimulator(mission.start_lat, mission.start_lon, mission.altitude, mission.heading,
                           mission.path_mode, mission.seed)
        zones = load_zone_defs(db)
        detector = get_detector(settings.detector, settings.yolo_weights)
        tracker = ByteLikeTracker(high_thresh=max(0.5, settings.conf_threshold))

        tmp = settings.storage_dir / "tmp" / f"{video.code}_raw.mp4"
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (pw, ph))
        track_data: dict[int, TrackData] = {}
        trails: dict[int, list] = {}
        frame_dets: dict[int, list] = {}
        det_rows, processed, idx = [], 0, 0
        wall0 = time.time()
        last_pub = 0.0                       # throttle for the live view (fast offline processing would flood the browser)
        est_total = max(1, total // step)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                if scale < 1.0:
                    frame = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
                t = idx / fps
                tel = sim.at(t)
                dets = detector.detect(frame, settings.conf_threshold)
                matched = tracker.update(dets, idx)
                now, objs = [], []
                for tr, d in matched:
                    cx, cy = (d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2
                    lat, lon = mapper.pixel_to_geo(cx, cy, tel)
                    td = track_data.setdefault(tr.id, TrackData(tr.id, tr.cls))
                    td.points.append(Point(t, idx, lat, lon, cx, cy, d.conf, d.bbox))
                    trails.setdefault(tr.id, []).append((int(cx), int(cy)))
                    now.append((tr.id, tr.cls, d.conf, d.bbox))
                    objs.append({"id": tr.id, "cls": tr.cls, "conf": round(d.conf, 2), "lat": round(lat, 6), "lon": round(lon, 6)})
                    det_rows.append(dict(video_id=video.id, mission_id=mission.id, frame_id=idx, timestamp=round(t, 3),
                                         track_id=tr.id, object_class=tr.cls, confidence=d.conf, bbox=list(d.bbox),
                                         lat=lat, lon=lon))
                frame_dets[idx] = now
                annotated = draw_frame(frame, now, trails, zones, mapper, tel, t)
                writer.write(annotated)
                processed += 1
                if hub.has_clients and (live or time.time() - last_pub >= 0.066):   # <=15 fps unless paced live
                    last_pub = time.time()
                    small = annotated if pw <= 640 else cv2.resize(annotated, (640, int(ph * 640 / pw)), interpolation=cv2.INTER_AREA)
                    okj, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 62])
                    if okj:
                        hub.publish({"type": "frame", "video_id": video.id, "t": round(t, 2), "n": processed, "of": est_total,
                                     "jpeg": base64.b64encode(buf.tobytes()).decode(), "objects": objs,
                                     "tel": {"lat": round(tel.lat, 6), "lon": round(tel.lon, 6), "alt": round(tel.altitude, 1)}})
                if live:                     # live mode: play at real-time speed
                    lag = (wall0 + t) - time.time()
                    if lag > 0:
                        time.sleep(min(lag, 1.0))
                if processed % 20 == 0:
                    video.progress = round(100 * idx / max(total, 1), 1)
                    db.commit()
                    hub.publish({"type": "progress", "video_id": video.id, "progress": video.progress})
            idx += 1
        cap.release()
        writer.release()

        # ---- atomic swap: old results stay visible (Map, Incident Center) until the new ones are ready.
        #      Everything below is one transaction; it only commits at the very end.
        old_incidents = [i.id for i in db.query(Incident).filter(Incident.video_id == video.id)]
        if old_incidents:
            db.query(Review).filter(Review.incident_id.in_(old_incidents)).delete(synchronize_session=False)
        for m in (Event, Incident, Track, Detection):
            db.query(m).filter(m.video_id == video.id).delete(synchronize_session=False)

        # ---- tracks / events
        tracks = {tid: td for tid, td in track_data.items() if len(td.points) >= 3}
        for tid, td in tracks.items():
            db.add(Track(video_id=video.id, mission_id=mission.id, track_id=tid, object_class=td.cls,
                         first_seen=td.first_seen, last_seen=td.last_seen, n_points=len(td.points),
                         avg_conf=round(sum(p.conf for p in td.points) / len(td.points), 3),
                         trajectory=[[round(p.t, 2), p.lat, p.lon] for p in td.points],
                         stats=track_stats(td, zones, settings)))
        db.bulk_insert_mappings(Detection, [r for r in det_rows if r["track_id"] in tracks])

        events = detect_events(list(tracks.values()), zones, settings)
        zmap = {z.id: z for z in zones}
        provider = get_provider()
        cap2 = cv2.VideoCapture(video.path)
        proc_frames = sorted(frame_dets)

        def render_key_frame(group, code_hint):
            evs = group["events"]
            prio = sorted(evs, key=lambda e: EVENT_PRIORITY.index(e.event_type) if e.event_type in EVENT_PRIORITY else 99)
            t_key = (prio[0].start_t + prio[0].end_t) / 2
            fi = min(proc_frames, key=lambda f: abs(f / fps - t_key)) if proc_frames else 0
            cap2.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap2.read()
            if not ok:
                return None, 0.0
            if scale < 1.0:
                fr = cv2.resize(fr, (pw, ph), interpolation=cv2.INTER_AREA)
            tel = sim.at(fi / fps)
            img = draw_frame(fr, frame_dets.get(fi, []), trails, zones, mapper, tel, fi / fps, group["track_id"],
                             caption=f"{code_hint}  evidence frame")
            path = settings.storage_dir / "frames" / f"{video.code}_{code_hint}.jpg"
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return str(path), fi / fps

        def vlm_hook(group):
            path, _ = render_key_frame(group, f"track{group['track_id']}")
            if not path:
                return None
            evs = group["events"]
            st = max((e.details.get("duration_s", 0) for e in evs if e.event_type == "STATIONARY"), default=0)
            zone = next((e.details.get("zone") for e in evs if e.event_type == "ZONE_INTRUSION"), None)
            ctx = {"object_class": tracks[group["track_id"]].cls, "track_id": group["track_id"], "zone": zone,
                   "stationary_s": st, "person_nearby": any(e.event_type == "PERSON_VEHICLE_PROXIMITY" for e in evs),
                   "risk_score": group["assessment"]["risk_score"]}
            res = analyze_with_fallback(path, ctx, provider)
            res["frame"] = path
            return res

        groups = build_incidents(events, tracks, settings, vlm_hook)
        cap2.release()
        model_versions = {"detector": f"{detector.name}:{detector.version}", "tracker": "bytelike-v1",
                          "vlm": f"{provider.name}:{provider.model}", "risk": "rules-v1"}

        n_inc = 0
        inc_msgs = []                          # published only AFTER commit so clients can immediately fetch them
        saved_event_ids = set()
        for g in groups:
            a, evs = g["assessment"], g["events"]
            if a["risk_score"] < settings.incident_min_score:
                continue
            td = tracks[g["track_id"]]
            prim = min(evs, key=lambda e: EVENT_PRIORITY.index(e.event_type) if e.event_type in EVENT_PRIORITY else 99)
            etypes = {e.event_type for e in evs}
            dec = decide(a["severity"], a["reasons"], td.cls, etypes)
            key_frame = (g.get("vlm") or {}).get("frame")
            if not key_frame:
                key_frame, _ = render_key_frame(g, f"track{g['track_id']}")  # evidence for every incident
            inc = Incident(code="TMP", video_id=video.id, mission_id=mission.id, track_id=g["track_id"],
                           object_class=td.cls, primary_type=prim.event_type, risk_score=a["risk_score"],
                           severity=a["severity"], reasons=a["reasons"], recommendation=dec["recommendation"],
                           actions=dec["actions"], requires_review=dec["requires_review"],
                           start_t=min(e.start_t for e in evs), end_t=max(e.end_t for e in evs),
                           lat=prim.lat, lon=prim.lon, zone_id=next((e.zone_id for e in evs if e.zone_id), None),
                           evidence=_evidence(evs), key_frame=key_frame,
                           vlm={k: v for k, v in (g.get("vlm") or {}).items() if k != "frame"} or None,
                           model_versions=model_versions)
            db.add(inc)
            db.flush()
            inc.code = f"EVT-{10000 + inc.id}"
            n_inc += 1
            inc_msgs.append({"type": "incident", "code": inc.code, "severity": inc.severity, "risk_score": inc.risk_score,
                         "object": f"{inc.object_class} #{inc.track_id}", "video_id": video.id,
                         "id": inc.id, "lat": inc.lat, "lon": inc.lon, "type_": inc.primary_type,
                         "reasons": a["reasons"], "recommendation": dec["recommendation"]})
            for e in evs:
                db.add(Event(video_id=video.id, mission_id=mission.id, incident_id=inc.id, event_type=e.event_type,
                             track_id=e.track_id, related_track_id=e.related_track_id, zone_id=e.zone_id,
                             start_t=e.start_t, end_t=e.end_t, lat=e.lat, lon=e.lon, details=e.details))
                saved_event_ids.add(id(e))
        for e in events:  # remaining events (info + sub-threshold) without an incident
            if id(e) not in saved_event_ids:
                db.add(Event(video_id=video.id, mission_id=mission.id, event_type=e.event_type, track_id=e.track_id,
                             related_track_id=e.related_track_id, zone_id=e.zone_id, start_t=e.start_t, end_t=e.end_t,
                             lat=e.lat, lon=e.lon, details=e.details))

        # ---- finalize
        final = settings.storage_dir / "annotated" / f"{video.code}.mp4"
        to_h264(tmp, final)
        elapsed = time.time() - t_start
        video.annotated_path = str(final)
        video.status, video.progress = "done", 100.0
        video.meta = {**(video.meta or {}), "processing": {
            "seconds": round(elapsed, 2), "processed_frames": processed, "source_fps": fps,
            "effective_fps": round(processed / elapsed, 2) if elapsed else None, "frame_step": step,
            "resolution": [pw, ph], "live_paced": live, "tracks": len(tracks), "events": len(events), "incidents": n_inc,
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "model_versions": model_versions}}
        mission.status = "completed"
        db.commit()
        for msg in inc_msgs:
            hub.publish(msg)
            notify.send_alert(msg)               # Telegram / webhook (best effort, after commit)
        hub.publish({"type": "video_done", "video_id": video.id, "incidents": n_inc})
        return video.meta["processing"]
    except Exception as e:  # surface failures to the UI instead of dying silently
        db.rollback()
        video = db.get(Video, video_id)
        video.status, video.error = "failed", f"{type(e).__name__}: {e}"
        mission = db.get(Mission, video.mission_id)
        mission.status = "failed"
        db.commit()
        hub.publish({"type": "video_failed", "video_id": video.id, "error": video.error})
        raise
    finally:
        db.close()
