"""Temporal reasoning: turns per-frame positions into sequence-level facts."""
import math

from ..geospatial.geo import haversine_m, point_in_polygon
from .types import TrackData, ZoneDef


def zones_at(lat, lon, zones: list[ZoneDef]) -> list[int]:
    return [z.id for z in zones if point_in_polygon(lat, lon, z.geometry)]


def path_length_m(track: TrackData) -> float:
    p = track.points
    return sum(haversine_m(p[i].lat, p[i].lon, p[i + 1].lat, p[i + 1].lon) for i in range(len(p) - 1))


def stationary_runs(points, radius_m: float, min_s: float):
    """Maximal runs where the object stays within `radius_m` of the run's anchor for >= min_s.
    Returns [{start, end, duration, lat, lon, i0, i1}]."""
    runs, i, n = [], 0, len(points)
    while i < n:
        a, j = points[i], i + 1
        while j < n and haversine_m(a.lat, a.lon, points[j].lat, points[j].lon) <= radius_m:
            j += 1
        seg = points[i:j]
        dur = seg[-1].t - seg[0].t
        if dur >= min_s and len(seg) >= 2:
            runs.append({"start": seg[0].t, "end": seg[-1].t, "duration": round(dur, 2),
                         "lat": sum(p.lat for p in seg) / len(seg), "lon": sum(p.lon for p in seg) / len(seg),
                         "i0": i, "i1": j - 1})
            i = j
        else:
            i += 1
    return runs


def zone_visits(track: TrackData, zones: list[ZoneDef], gap_tolerance: int = 2):
    """Contiguous in-zone intervals per zone with small-gap debounce.
    Returns {zone_id: [{enter, exit, dwell, lat, lon}]}."""
    visits: dict[int, list] = {}
    open_: dict[int, dict] = {}
    miss: dict[int, int] = {}
    for p in track.points:
        inside = set(zones_at(p.lat, p.lon, zones))
        for zid in inside:
            if zid not in open_:
                open_[zid] = {"enter": p.t, "exit": p.t, "lat": p.lat, "lon": p.lon}
            open_[zid]["exit"] = p.t
            miss[zid] = 0
        for zid in list(open_):
            if zid not in inside:
                miss[zid] = miss.get(zid, 0) + 1
                if miss[zid] > gap_tolerance:
                    v = open_.pop(zid)
                    v["dwell"] = round(v["exit"] - v["enter"], 2)
                    visits.setdefault(zid, []).append(v)
    for zid, v in open_.items():
        v["dwell"] = round(v["exit"] - v["enter"], 2)
        visits.setdefault(zid, []).append(v)
    return visits


def direction_changes(track: TrackData, min_step_m=1.0, turn_deg=100.0) -> int:
    """Count sharp heading reversals using displacements of at least `min_step_m`."""
    pts, headings, last = track.points, [], None
    for p in pts:
        if last is None:
            last = p
            continue
        d = haversine_m(last.lat, last.lon, p.lat, p.lon)
        if d >= min_step_m:
            headings.append(math.atan2(p.lon - last.lon, p.lat - last.lat))
            last = p
    turns = 0
    for a, b in zip(headings, headings[1:]):
        diff = abs((math.degrees(b - a) + 180) % 360 - 180)
        if diff >= turn_deg:
            turns += 1
    return turns


def max_speed_mps(track: TrackData, window=3) -> float:
    p, best = track.points, 0.0
    for i in range(len(p) - window):
        dt = p[i + window].t - p[i].t
        if dt > 0:
            best = max(best, haversine_m(p[i].lat, p[i].lon, p[i + window].lat, p[i + window].lon) / dt)
    return best


def track_stats(track: TrackData, zones: list[ZoneDef], cfg) -> dict:
    visits = zone_visits(track, zones)
    runs = stationary_runs(track.points, cfg.stationary_radius_m, cfg.stationary_min_s)
    dur = track.last_seen - track.first_seen
    dist = path_length_m(track)
    return {
        "first_seen": track.first_seen, "last_seen": track.last_seen, "duration_s": round(dur, 2),
        "distance_m": round(dist, 2), "avg_speed_mps": round(dist / dur, 2) if dur > 0 else 0.0,
        "max_speed_mps": round(max_speed_mps(track), 2),
        "stationary_s": round(sum(r["duration"] for r in runs), 2),
        "direction_changes": direction_changes(track),
        "zones": {str(z): {"visits": len(v), "dwell_s": round(sum(x["dwell"] for x in v), 2)} for z, v in visits.items()},
    }
