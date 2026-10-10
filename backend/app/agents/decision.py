"""Decision Agent. Selects *software* actions only; never physical/autonomous ones.
All HIGH and CRITICAL incidents are routed to a human operator."""

ACTIONS = ("LOG", "ALERT", "REQUEST_REVIEW", "GENERATE_REPORT", "ESCALATE", "TRACK")


def decide(severity: str, reasons: list[str], object_class: str, event_types: set[str]) -> dict:
    if severity == "CRITICAL":
        actions = ["ALERT", "ESCALATE", "REQUEST_REVIEW", "GENERATE_REPORT", "TRACK"]
        rec = "Escalate to supervisor and start a human investigation immediately."
    elif severity == "HIGH":
        actions = ["ALERT", "REQUEST_REVIEW", "GENERATE_REPORT", "TRACK"]
        rec = "Human investigation."
    elif severity == "MEDIUM":
        actions = ["ALERT", "TRACK"]
        rec = "Keep tracking the object; review if the situation persists."
    else:
        actions = ["LOG"]
        rec = "No action required; logged for audit."
    if "FIRE_SMOKE" in event_types:
        rec = "Possible fire/smoke: verify visually and follow the fire-response SOP."
        if "ESCALATE" not in actions:
            actions.append("ESCALATE")
    return {
        "actions": actions,
        "recommendation": rec,
        "requires_review": severity in ("HIGH", "CRITICAL") or "FIRE_SMOKE" in event_types,
        "autonomous_physical_action": False,  # hard invariant of the platform
    }
