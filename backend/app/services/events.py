"""Event engine: converts tracks + zones into semantic events using configurable rules."""
from ..geospatial.geo import haversine_m
from .temporal import direction_changes, max_speed_mps, stationary_runs, zone_visits
from .types import PERSON_CLASSES, VEHICLE_CLASSES, EventDef, TrackData, ZoneDef

ALERT_ZONE_TYPES = {"restricted", "emergency", "construction"}
SPEED_LIMIT_MPS = {"person": 7.0, "bicycle": 12.0}  # beyond this => abnormal movement
FIRE_CLASSES = {"fire", "smoke"}
OBSTACLE_CLASSES = {"obstacle", "debris"}


def _zone_applies(z: ZoneDef, cls: str) -> bool:
    return not z.forbidden_classes or cls in z.forbidden_classes


def detect_events(tracks: list[TrackData], zones: list[ZoneDef], cfg) -> list[EventDef]:
    events: list[EventDef] = []
    zmap = {z.id: z for z in zones}
    by_id = {t.id: t for t in tracks}

    for t in tracks:
        if len(t.points) < 2:
            continue
        p0, pN = t.points[0], t.points[-1]
        if t.last_seen - t.first_seen >= 1.0:  # informational scene entry / exit
            events.append(EventDef("OBJECT_ENTRY", t.id, p0.t, p0.t, p0.lat, p0.lon, details={"class": t.cls}))
            events.append(EventDef("OBJECT_EXIT", t.id, pN.t, pN.t, pN.lat, pN.lon, details={"class": t.cls}))

        if t.cls in FIRE_CLASSES:
            events.append(EventDef("FIRE_SMOKE", t.id, p0.t, pN.t, pN.lat, pN.lon,
                                   details={"class": t.cls, "n_detections": len(t.points)}))
            continue

        visits = zone_visits(t, zones)
        for zid, vs in visits.items():
            z = zmap[zid]
            for v in vs:
                if z.type in ALERT_ZONE_TYPES and _zone_applies(z, t.cls):
                    events.append(EventDef("ZONE_INTRUSION", t.id, v["enter"], v["exit"], v["lat"], v["lon"], zid,
                                           details={"zone": z.name, "zone_type": z.type, "class": t.cls,
                                                    "dwell_s": v["dwell"]}))
                if z.max_dwell_s and v["dwell"] > z.max_dwell_s:
                    events.append(EventDef("LOITERING", t.id, v["enter"], v["exit"], v["lat"], v["lon"], zid,
                                           details={"zone": z.name, "dwell_s": v["dwell"], "limit_s": z.max_dwell_s,
                                                    "reason": "dwell limit exceeded"}))

        # Obstacle: anything (vehicle, person, debris...) parked/lying on a road zone, or an explicit obstacle class
        for r in stationary_runs(t.points, cfg.stationary_radius_m, cfg.stationary_min_s):
            road = [z for z in zones if z.type == "road" and _inside(z, r["lat"], r["lon"])]
            if road or t.cls in OBSTACLE_CLASSES:
                events.append(EventDef("OBSTACLE", t.id, r["start"], r["end"], r["lat"], r["lon"],
                                       road[0].id if road else None,
                                       details={"class": t.cls, "duration_s": r["duration"],
                                                "zone": road[0].name if road else None}))
        if t.cls in VEHICLE_CLASSES:
            for r in stationary_runs(t.points, cfg.stationary_radius_m, cfg.stationary_min_s):
                in_zones = [z for z in zones if _inside(z, r["lat"], r["lon"])]
                events.append(EventDef("STATIONARY", t.id, r["start"], r["end"], r["lat"], r["lon"],
                                       in_zones[0].id if in_zones else None,
                                       details={"duration_s": r["duration"], "class": t.cls,
                                                "zones": [z.name for z in in_zones],
                                                "in_alert_zone": any(z.type in ALERT_ZONE_TYPES for z in in_zones)}))
        elif t.cls in PERSON_CLASSES:
            for r in stationary_runs(t.points, cfg.loiter_radius_m, cfg.loiter_min_s):
                events.append(EventDef("LOITERING", t.id, r["start"], r["end"], r["lat"], r["lon"],
                                       details={"duration_s": r["duration"], "reason": "lingering in small area"}))

        limit = SPEED_LIMIT_MPS.get(t.cls)
        turns, vmax = direction_changes(t), max_speed_mps(t)
        if turns >= 3 or (limit and vmax > limit):
            events.append(EventDef("UNUSUAL_ACTIVITY", t.id, p0.t, pN.t, pN.lat, pN.lon,
                                   details={"direction_changes": turns, "max_speed_mps": round(vmax, 2),
                                            "reason": "erratic movement" if turns >= 3 else "abnormal speed"}))

    events.extend(_proximity_events(tracks, by_id, zones, cfg))
    events.sort(key=lambda e: (e.start_t, e.event_type))
    return events


def _inside(z: ZoneDef, lat, lon) -> bool:
    from ..geospatial.geo import point_in_polygon
    return point_in_polygon(lat, lon, z.geometry)


def _proximity_events(tracks, by_id, zones, cfg) -> list[EventDef]:
    out = []
    persons = [t for t in tracks if t.cls in PERSON_CLASSES and len(t.points) >= 2]
    vehicles = [t for t in tracks if t.cls in VEHICLE_CLASSES and len(t.points) >= 2]
    for v in vehicles:
        v_runs = stationary_runs(v.points, cfg.stationary_radius_m, cfg.stationary_min_s)
        vp = {p.frame: p for p in v.points}
        for pr in persons:
            run = []
            for p in pr.points:
                q = vp.get(p.frame)
                if q and haversine_m(p.lat, p.lon, q.lat, q.lon) <= cfg.proximity_m:
                    run.append((p, q))
                else:
                    _flush(out, run, v, pr, v_runs, zones, cfg)
                    run = []
            _flush(out, run, v, pr, v_runs, zones, cfg)
    return out


def _flush(out, run, v, pr, v_runs, zones, cfg):
    if len(run) < 2:
        return
    t0, t1 = run[0][0].t, run[-1][0].t
    if t1 - t0 < cfg.proximity_min_s:
        return
    stationary = any(r["start"] <= t1 and r["end"] >= t0 for r in v_runs)
    mid_p, mid_q = run[len(run) // 2]
    zs = [z for z in zones if _inside(z, mid_q.lat, mid_q.lon)]
    min_d = min(haversine_m(p.lat, p.lon, q.lat, q.lon) for p, q in run)
    out.append(EventDef("PERSON_VEHICLE_PROXIMITY", v.id, t0, t1, mid_q.lat, mid_q.lon,
                        zs[0].id if zs else None, related_track_id=pr.id,
                        details={"min_distance_m": round(min_d, 2), "vehicle_stationary": stationary,
                                 "duration_s": round(t1 - t0, 2), "zone": zs[0].name if zs else None}))
