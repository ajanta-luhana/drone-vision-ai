"""Explainable risk scoring (deterministic, unit-testable). Weights and bands are configurable."""
from ..core.config import settings
from ..services.types import EventDef

DEFAULT_WEIGHTS = {
    "ZONE_INTRUSION": 40,
    "STATIONARY": 15,
    "STATIONARY_DURATION_BONUS": 10,   # scaled by duration up to 3x the stationary threshold
    "STATIONARY_IN_ALERT_ZONE": 10,
    "PERSON_VEHICLE_PROXIMITY": 20,
    "LOITERING": 15,
    "UNUSUAL_ACTIVITY": 15,
    "FIRE_SMOKE": 70,
    "OBSTACLE": 25,
}
ZONE_MULT = {"road": 0.5, "restricted": 1.0, "emergency": 1.0, "construction": 0.7, "inspection": 0.4,
             "monitoring": 0.3, "custom": 0.5}
INFO_EVENTS = {"OBJECT_ENTRY", "OBJECT_EXIT"}
LABELS = {
    "ZONE_INTRUSION": "Restricted zone intrusion",
    "STATIONARY": "Extended stationary period",
    "PERSON_VEHICLE_PROXIMITY": "Person-vehicle proximity",
    "LOITERING": "Loitering",
    "UNUSUAL_ACTIVITY": "Unusual activity",
    "FIRE_SMOKE": "Fire/smoke detected",
    "OBSTACLE": "Obstruction on road",
}


def severity_for(score: int, bands=None) -> str:
    for name, upper in (bands or settings.risk_bands):
        if score <= upper:
            return name
    return "CRITICAL"


def assess(events: list[EventDef], stationary_min_s: float = 4.0, vlm_adjust: int = 0,
           weights: dict | None = None, bands=None) -> dict:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    score, reasons, factors = 0.0, [], []

    def add(label, pts, key):
        nonlocal score
        score += pts
        if label not in reasons:
            reasons.append(label)
        factors.append({"factor": key, "points": round(pts, 1)})

    for e in events:
        if e.event_type in INFO_EVENTS:
            continue
        d = e.details
        if e.event_type == "ZONE_INTRUSION":
            add(LABELS[e.event_type], w["ZONE_INTRUSION"] * ZONE_MULT.get(d.get("zone_type", "custom"), 0.5), "ZONE_INTRUSION")
        elif e.event_type == "STATIONARY":
            frac = min(1.0, d.get("duration_s", 0) / (3 * stationary_min_s))
            add(LABELS[e.event_type], w["STATIONARY"] + w["STATIONARY_DURATION_BONUS"] * frac, "STATIONARY")
            if d.get("in_alert_zone"):
                add(LABELS[e.event_type], w["STATIONARY_IN_ALERT_ZONE"], "STATIONARY_IN_ALERT_ZONE")
        elif e.event_type == "PERSON_VEHICLE_PROXIMITY":
            mult = 1.0 if d.get("vehicle_stationary") else 0.5
            add(LABELS[e.event_type], w["PERSON_VEHICLE_PROXIMITY"] * mult, "PERSON_VEHICLE_PROXIMITY")
        elif e.event_type in w:
            add(LABELS.get(e.event_type, e.event_type), w[e.event_type], e.event_type)

    score += max(-10, min(10, vlm_adjust))
    final = int(round(max(0, min(100, score))))
    return {"risk_score": final, "severity": severity_for(final, bands), "reasons": reasons, "factors": factors}
