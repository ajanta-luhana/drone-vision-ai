"""AI Investigation Assistant.

Design: an intent router selects *tools* (search_events, get_mission, get_object_track, search_zone,
get_incident, generate_report, search_policy) and the answer is composed only from tool output, so it cannot
invent incidents or policies (hallucination resistance). An LLM planner can replace `route()` later without
changing the tools or their structured outputs.
"""
import datetime as dt
import re

from sqlalchemy.orm import Session

from ..core.security import has_permission
from ..models.db import Event, Incident, Mission, Track, Zone
from ..rag.store import PolicyStore
from ..reports import generator

SEV_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
VEHICLES = {"car", "truck", "bus", "motorcycle", "construction_equipment"}


class Tools:
    def __init__(self, db: Session, policies: PolicyStore, user: str, role: str):
        self.db, self.policies, self.user, self.role = db, policies, user, role

    def search_events(self, min_severity=None, event_type=None, only_today=False, object_class=None, vehicles_only=False):
        q = self.db.query(Incident)
        if min_severity:
            q = q.filter(Incident.severity.in_([s for s, v in SEV_ORDER.items() if v >= SEV_ORDER[min_severity]]))
        rows = q.order_by(Incident.risk_score.desc()).all()
        if only_today:
            today = dt.datetime.now(dt.timezone.utc).date()
            rows = [r for r in rows if r.created_at.date() == today]
        if event_type:
            ids = {e.incident_id for e in self.db.query(Event).filter(Event.event_type == event_type, Event.incident_id.isnot(None))}
            rows = [r for r in rows if r.id in ids]
        if vehicles_only:
            rows = [r for r in rows if r.object_class in VEHICLES]
        return [{"code": r.code, "object": f"{r.object_class} #{r.track_id}", "severity": r.severity,
                 "risk_score": r.risk_score, "type": r.primary_type, "evidence": r.evidence, "status": r.status} for r in rows]

    def get_mission(self, code=None):
        q = self.db.query(Mission)
        m = q.filter(Mission.code == code).first() if code else q.order_by(Mission.id.desc()).first()
        if not m:
            return None
        n_inc = self.db.query(Incident).filter(Incident.mission_id == m.id).count()
        n_tr = self.db.query(Track).filter(Track.mission_id == m.id).count()
        return {"code": m.code, "name": m.name, "status": m.status, "tracks": n_tr, "incidents": n_inc, "id": m.id}

    def get_object_track(self, track_id: int, mission_id=None):
        q = self.db.query(Track).filter(Track.track_id == track_id)
        if mission_id:
            q = q.filter(Track.mission_id == mission_id)
        t = q.order_by(Track.id.desc()).first()
        if not t:
            return None
        return {"track_id": t.track_id, "class": t.object_class, "first_seen_s": t.first_seen, "last_seen_s": t.last_seen,
                "avg_confidence": t.avg_conf, "stats": t.stats, "mission_id": t.mission_id}

    def get_incident(self, code=None, track_id=None):
        q = self.db.query(Incident)
        i = q.filter(Incident.code == code).first() if code else (
            q.filter(Incident.track_id == track_id).order_by(Incident.risk_score.desc()).first())
        return generator._incident_dict(self.db, i) if i else None

    def search_zone(self, name_query: str):
        zones = [z for z in self.db.query(Zone).all() if name_query.lower() in z.name.lower()]
        out = []
        for z in zones:
            incs = self.db.query(Incident).filter(Incident.zone_id == z.id).order_by(Incident.risk_score.desc()).all()
            out.append({"zone": z.name, "type": z.zone_type, "incidents": [
                {"code": i.code, "object": f"{i.object_class} #{i.track_id}", "severity": i.severity} for i in incidents_unique(incs)]})
        return out

    def generate_report(self, mission_code=None):
        if not has_permission(self.role, "reports:create"):
            return {"error": "Your role is not allowed to generate reports."}
        m = self.get_mission(mission_code)
        if not m:
            return {"error": "No mission found."}
        rep = generator.generate(self.db, self.db.get(Mission, m["id"]), self.user)
        return {"report_id": rep.id, "mission": m["code"], "summary": rep.summary}

    def search_policy(self, query: str):
        return self.policies.search(query, 3)


def incidents_unique(rows):
    seen, out = set(), []
    for r in rows:
        if r.id not in seen:
            seen.add(r.id)
            out.append(r)
    return out


def fmt_inc(i):
    return f"{i['code']}: {i['object']} - {i['severity']} ({i['risk_score']}/100) - {'; '.join(i['evidence']) or i['type']}"


def route(question: str, tools: Tools) -> dict:
    q = question.lower()
    calls, lines, sources = [], [], []

    def call(name, fn, **kw):
        res = fn(**kw)
        calls.append({"tool": name, "args": kw, "ok": bool(res)})
        return res

    code = re.search(r"evt-\d+", q)
    track = re.search(r"(?:vehicle|car|truck|bus|person|object|track)\s*#?\s*(\d+)", q)
    mission_code = re.search(r"mission-\d+", question, re.I)

    if re.search(r"\breport\b", q) and re.search(r"generate|create|make|produce", q):
        res = call("generate_report", tools.generate_report, mission_code=mission_code.group(0).upper() if mission_code else None)
        lines.append(res.get("error") or f"Report #{res['report_id']} generated for {res['mission']}. {res['summary']}")
    elif re.search(r"procedure|policy|sop|rule|protocol|what should|how (do|to) (we|i)", q):
        res = call("search_policy", tools.search_policy, query=question)
        if res:
            lines.append(f"From {res[0]['source']} ({res[0]['heading']}):\n{res[0]['text']}")
            sources = [{"source": r["source"], "heading": r["heading"]} for r in res]
        else:
            lines.append("I could not find a matching policy. I will not guess; please check with your supervisor.")
    elif code or (track and re.search(r"why|risk|classif|explain", q)):
        inc = call("get_incident", tools.get_incident, code=code.group(0).upper() if code else None,
                   track_id=int(track.group(1)) if track and not code else None)
        if inc:
            lines.append(f"{inc['incident_id']} is {inc['severity']} ({inc['risk_score']}/100) because: "
                         f"{'; '.join(inc['reasons'])}. Evidence: {'; '.join(inc['evidence'])}. "
                         f"Recommended: {inc['ai_recommendation']}")
        else:
            lines.append("No matching incident was found in the database.")
    elif track:
        t = call("get_object_track", tools.get_object_track, track_id=int(track.group(1)))
        if t:
            s = t["stats"]
            lines.append(f"{t['class']} #{t['track_id']}: seen {t['first_seen_s']:.1f}s-{t['last_seen_s']:.1f}s, "
                         f"stationary for {s.get('stationary_s', 0):.1f}s, travelled {s.get('distance_m', 0):.0f} m, "
                         f"average speed {s.get('avg_speed_mps', 0):.1f} m/s.")
        else:
            lines.append("No such tracked object exists.")
    elif re.search(r"high[- ]risk|critical|severe", q):
        res = call("search_events", tools.search_events, min_severity="HIGH", only_today="today" in q)
        lines += [fmt_inc(i) for i in res] or ["No high-risk incidents found."]
    elif re.search(r"enter|restricted|intrusion|violation", q):
        res = call("search_events", tools.search_events, event_type="ZONE_INTRUSION", vehicles_only="vehicle" in q)
        lines += [fmt_inc(i) for i in res] or ["No zone intrusions found."]
    elif re.search(r"near|zone|building|area", q):
        words = [w for w in re.findall(r"[a-z0-9]+", q) if len(w) > 2]
        found = []
        for z in tools.db.query(Zone).all():
            if any(w in z.name.lower() for w in words if w not in ("the", "zone", "near", "incidents", "show")) or z.name.lower() in q:
                found += call("search_zone", tools.search_zone, name_query=z.name)
        for z in found:
            lines.append(f"{z['zone']} ({z['type']}): " + (", ".join(f"{i['code']} {i['object']} {i['severity']}" for i in z["incidents"]) or "no incidents"))
        if not found:
            lines.append("No matching zone was found.")
    elif re.search(r"mission|summary|status", q):
        m = call("get_mission", tools.get_mission, code=mission_code.group(0).upper() if mission_code else None)
        lines.append(f"{m['code']} '{m['name']}' is {m['status']}: {m['tracks']} tracked objects, {m['incidents']} incidents." if m else "No missions yet.")
    else:
        lines.append("I can search incidents (\"show all high-risk events\"), explain risk (\"why was vehicle #2 high risk?\"), "
                     "report object dwell time, look up zone incidents, generate reports, and look up policies "
                     "(\"what is the procedure for a vehicle entering Restricted Zone A?\").")
    return {"answer": "\n".join(lines), "tool_calls": calls, "sources": sources}
