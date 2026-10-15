"""Outbound alerts for HIGH/CRITICAL incidents: Telegram bot and/or a generic webhook (Slack/Discord-compatible).
Best-effort by design: a failing alert channel must never break video processing."""
import logging

import httpx

from ..core.config import settings

log = logging.getLogger("dronevision.notify")
ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def enabled() -> dict:
    return {"telegram": bool(settings.telegram_bot_token and settings.telegram_chat_id), "webhook": bool(settings.alert_webhook_url),
            "min_severity": settings.alert_min_severity}


def format_message(m: dict) -> str:
    lines = [f"DroneVision AI - {m['severity']} incident {m['code']}", f"{m['object']} - risk {m['risk_score']}/100"]
    if m.get("reasons"):
        lines.append("Why: " + "; ".join(m["reasons"]))
    if m.get("recommendation"):
        lines.append("Recommendation: " + m["recommendation"])
    lines.append("Human review required." if m["severity"] in ("HIGH", "CRITICAL") else "")
    if settings.public_url:
        lines.append(f"{settings.public_url}")
    return "\n".join(l for l in lines if l)


def send_alert(m: dict) -> dict:
    """m: the incident message published on the live feed. Returns which channels accepted it."""
    sent = {"telegram": False, "webhook": False}
    if ORDER.get(m.get("severity"), 0) < ORDER.get(settings.alert_min_severity, 2):
        return sent
    text = format_message(m)
    if settings.telegram_bot_token and settings.telegram_chat_id:
        try:
            r = httpx.post(f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage", timeout=6,
                           json={"chat_id": settings.telegram_chat_id, "text": text, "disable_web_page_preview": True})
            sent["telegram"] = r.status_code == 200
        except Exception as e:                       # never log the URL: it contains the bot token
            log.warning("telegram alert failed: %s", type(e).__name__)
    if settings.alert_webhook_url:
        try:
            r = httpx.post(settings.alert_webhook_url, timeout=6, json={"text": text, "content": text})
            sent["webhook"] = 200 <= r.status_code < 300
        except Exception as e:
            log.warning("webhook alert failed: %s", type(e).__name__)
    return sent
