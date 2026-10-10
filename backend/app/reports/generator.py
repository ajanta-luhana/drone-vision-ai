"""Mission / incident report generation (JSON, CSV, PDF). Narrative is templated from structured facts, so it
cannot hallucinate; an LLM could later rewrite it, but the structured sections remain the source of truth."""
import csv
import json
from pathlib import Path

from sqlalchemy.orm import Session

from ..core.config import settings
from ..models.db import Event, Incident, Mission, Report, Review, Track, Video, Zone


def mmss(t):
    t = float(t or 0)
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def _incident_dict(db: Session, i: Incident) -> dict:
    reviews = db.query(Review).filter(Review.incident_id == i.id).order_by(Review.id).all()
    zone = db.get(Zone, i.zone_id) if i.zone_id else None
    return {
        "incident_id": i.code, "type": i.primary_type, "object": f"{i.object_class} #{i.track_id}",
        "severity": i.severity, "risk_score": i.risk_score, "reasons": i.reasons, "evidence": i.evidence,
        "time_window": [mmss(i.start_t), mmss(i.end_t)], "location": {"lat": i.lat, "lon": i.lon},
        "zone": zone.name if zone else None, "ai_recommendation": i.recommendation, "actions": i.actions,
        "vlm_analysis": i.vlm, "status": i.status, "requires_review": i.requires_review,
        "operator_decisions": [{"operator": r.operator, "decision": r.decision, "comment": r.comment,
                                "timestamp": r.created_at.isoformat()} for r in reviews],
        "model_versions": i.model_versions, "key_frame": i.key_frame,
    }


def narrative(mission: Mission, incidents: list[dict], n_tracks: int) -> str:
    if not incidents:
        return (f"Mission {mission.code} processed {n_tracks} tracked objects. No incident reached the reporting "
                f"threshold.")
    worst = max(incidents, key=lambda x: x["risk_score"])
    lines = [f"Mission {mission.code} tracked {n_tracks} objects and produced {len(incidents)} incident(s). "
             f"The highest-risk incident is {worst['incident_id']} ({worst['severity']}, {worst['risk_score']}/100): "
             f"{worst['object']} - {', '.join(worst['evidence'])}. "
             f"Recommended action: {worst['ai_recommendation']}"]
    pending = sum(1 for i in incidents if i["status"] == "pending" and i["requires_review"])
    if pending:
        lines.append(f"{pending} incident(s) still await operator review. This report is decision support only.")
    return " ".join(lines)


def mission_report_data(db: Session, mission: Mission, incident_id: int | None = None) -> dict:
    q = db.query(Incident).filter(Incident.mission_id == mission.id)
    if incident_id:
        q = q.filter(Incident.id == incident_id)
    incidents = [_incident_dict(db, i) for i in q.order_by(Incident.risk_score.desc()).all()]
    videos = db.query(Video).filter(Video.mission_id == mission.id).all()
    tracks = db.query(Track).filter(Track.mission_id == mission.id).all()
    events = db.query(Event).filter(Event.mission_id == mission.id).all()
    return {
        "mission": {"code": mission.code, "name": mission.name, "status": mission.status,
                    "start": {"lat": mission.start_lat, "lon": mission.start_lon, "altitude": mission.altitude},
                    "created_at": mission.created_at.isoformat()},
        "videos": [{"code": v.code, "filename": v.filename, "status": v.status,
                    "processing": (v.meta or {}).get("processing")} for v in videos],
        "detected_objects": [{"track": f"{t.object_class} #{t.track_id}", "first_seen": mmss(t.first_seen),
                              "last_seen": mmss(t.last_seen), "avg_confidence": t.avg_conf,
                              "stats": t.stats} for t in tracks],
        "summary": narrative(mission, incidents, len(tracks)),
        "incidents": incidents,
        "events": [{"event_id": e.id, "type": e.event_type, "track_id": e.track_id, "start": mmss(e.start_t),
                    "end": mmss(e.end_t), "lat": e.lat, "lon": e.lon, "incident_id": e.incident_id,
                    "details": e.details} for e in events],
    }


def write_csv(path: Path, data: dict):
    with open(path, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "type", "track_id", "start", "end", "lat", "lon", "incident_id", "details"])
        for e in data["events"]:
            w.writerow([e["event_id"], e["type"], e["track_id"], e["start"], e["end"], e["lat"], e["lon"],
                        e["incident_id"], json.dumps(e["details"])])


def write_pdf(path: Path, data: dict):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm, title=f"{data['mission']['code']} report")
    esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    el = [Paragraph("DroneVision AI - Mission Report", ss["Title"]),
          Paragraph(f"{esc(data['mission']['code'])} - {esc(data['mission']['name'])}", ss["Heading2"]),
          Paragraph(esc(data["summary"]), ss["BodyText"]), Spacer(1, 10)]
    m = data["mission"]
    el.append(Paragraph(f"Start position: {m['start']['lat']:.5f}, {m['start']['lon']:.5f} at "
                        f"{m['start']['altitude']:.0f} m. Status: {esc(m['status'])}.", ss["BodyText"]))
    sev_col = {"LOW": colors.green, "MEDIUM": colors.orange, "HIGH": colors.red, "CRITICAL": colors.darkred}
    for inc in data["incidents"]:
        el += [Spacer(1, 12), Paragraph(f"Incident {esc(inc['incident_id'])} - {esc(inc['type'])}", ss["Heading3"])]
        rows = [["Severity / risk", f"{inc['severity']} / {inc['risk_score']}"],
                ["Object", inc["object"]], ["Time window", " - ".join(inc["time_window"])],
                ["Location", f"{inc['location']['lat']:.5f}, {inc['location']['lon']:.5f}"],
                ["Zone", inc["zone"] or "-"], ["Evidence", "; ".join(inc["evidence"])],
                ["Reasons", "; ".join(inc["reasons"])], ["AI recommendation", inc["ai_recommendation"]],
                ["VLM analysis", (inc["vlm_analysis"] or {}).get("summary", "-")],
                ["Operator status", inc["status"]],
                ["Operator decisions", "; ".join(f"{d['operator']}: {d['decision']} ({d['comment']})"
                                                 for d in inc["operator_decisions"]) or "-"],
                ["Model versions", json.dumps(inc["model_versions"])]]
        t = Table([[Paragraph(esc(a), ss["BodyText"]), Paragraph(esc(b), ss["BodyText"])] for a, b in rows],
                  colWidths=[4 * cm, 13 * cm])
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                               ("LINEBEFORE", (0, 0), (0, 0), 4, sev_col.get(inc["severity"], colors.grey))]))
        el.append(t)
        kf = inc.get("key_frame")
        if kf and Path(kf).exists():
            el += [Spacer(1, 6), Image(kf, width=12 * cm, height=6.75 * cm)]
    el += [Spacer(1, 14), Paragraph("Detected objects", ss["Heading3"])]
    rows = [["Object", "First seen", "Last seen", "Avg conf.", "Stationary (s)"]] + [
        [o["track"], o["first_seen"], o["last_seen"], o["avg_confidence"], o["stats"].get("stationary_s", 0)]
        for o in data["detected_objects"]]
    t = Table(rows, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
    el.append(t)
    el += [Spacer(1, 10), Paragraph("AI-generated analysis is decision support. High-risk findings require human review.",
                                    ss["Italic"])]
    doc.build(el)


def generate(db: Session, mission: Mission, user: str, incident_id: int | None = None) -> Report:
    data = mission_report_data(db, mission, incident_id)
    rep = Report(mission_id=mission.id, incident_id=incident_id, kind="incident" if incident_id else "mission",
                 summary=data["summary"], created_by=user)
    db.add(rep)
    db.flush()
    base = settings.storage_dir / "reports" / f"report_{rep.id}"
    base.parent.mkdir(parents=True, exist_ok=True)
    rep.path_json, rep.path_csv, rep.path_pdf = f"{base}.json", f"{base}.csv", f"{base}.pdf"
    Path(rep.path_json).write_text(json.dumps(data, indent=2, default=str), encoding="utf8")
    write_csv(Path(rep.path_csv), data)
    write_pdf(Path(rep.path_pdf), data)
    db.commit()
    return rep
