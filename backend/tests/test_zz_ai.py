"""Groq LLM / VLM / alerts / demo mode. The HTTP layer is faked: no network is used."""
import base64
import json
import logging

import cv2
import httpx
import numpy as np
import pytest

from conftest import login

P = "/api/v1"


@pytest.fixture(scope="module", autouse=True)
def _seeded(client):
    """Self-contained: create tables (via the app) and the sample mission with a CRITICAL incident."""
    from app.services.demo import seed_demo
    seed_demo()


# ---------------- fake OpenAI-compatible server ----------------
class R:
    def __init__(self, status=200, body=None, headers=None, text=""):
        self.status_code, self._b, self.headers, self.text = status, body or {}, headers or {}, text or json.dumps(body or {})
    def json(self): return self._b


def ok(content="", tool_calls=None):
    m = {"role": "assistant", "content": content}
    if tool_calls:
        m["tool_calls"] = tool_calls
    return R(200, {"choices": [{"message": m}]})


def tc(name, args, id="call_1"):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class Fake:
    """Scripted responses; callables get the request body and return an R."""
    def __init__(self, *script):
        self.script, self.requests = list(script), []
    def __call__(self, url, json=None, headers=None, timeout=None, **kw):
        self.requests.append({"url": url, "body": json, "headers": headers})
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        return step(json) if callable(step) else step


@pytest.fixture
def groq(monkeypatch):
    from app.core.config import settings
    from app.ai import llm
    monkeypatch.setattr(settings, "groq_api_key", "test-key-123")
    monkeypatch.setattr("time.sleep", lambda s: None)
    llm.stats.__init__()
    def install(*script):
        f = Fake(*script)
        monkeypatch.setattr(httpx, "post", f)
        return f
    return install


def tools_for(role="analyst"):
    from app.agents.assistant import Tools
    from app.models.db import SessionLocal
    from app.rag.store import PolicyStore
    return Tools(SessionLocal(), PolicyStore(), role, role)


def real_code():
    from app.models.db import Incident, SessionLocal
    return SessionLocal().query(Incident).order_by(Incident.risk_score.desc()).first().code


# ---------------- client resilience ----------------
def test_model_fallback_chain_when_model_is_retired(groq):
    from app.ai import llm
    f = groq(R(404, text='{"error":{"code":"model_not_found","message":"model decommissioned"}}'), ok("hello <think>secret reasoning</think> world"))
    out = llm.chat([{"role": "user", "content": "hi"}], models=["old/model", "new/model"])
    assert out["model"] == "new/model" and out["message"]["content"] == "hello  world"        # think-tags stripped
    assert [r["body"]["model"] for r in f.requests] == ["old/model", "new/model"]
    assert llm.stats.snapshot()["model_fallbacks"] == 1 and f.requests[0]["headers"]["Authorization"] == "Bearer test-key-123"


def test_retries_429_then_succeeds_and_gives_up_cleanly(groq):
    from app.ai import llm
    f = groq(R(429, headers={"retry-after": "2"}), R(429), ok("fine"))
    assert llm.chat([{"role": "user", "content": "x"}], models=["m"])["message"]["content"] == "fine"
    groq(R(500))
    with pytest.raises(llm.LLMError):
        llm.chat([{"role": "user", "content": "x"}], models=["m"], max_retries=1)


def test_auth_error_does_not_burn_other_models(groq):
    from app.ai import llm
    f = groq(R(401, text='{"error":{"message":"Invalid API Key"}}'))
    with pytest.raises(llm.LLMError):
        llm.chat([{"role": "user", "content": "x"}], models=["a", "b", "c"])
    assert len(f.requests) == 1


def test_no_key_and_network_errors_raise_llmerror(monkeypatch):
    from app.ai import llm
    from app.core.config import settings
    monkeypatch.setattr(settings, "groq_api_key", "")
    with pytest.raises(llm.LLMError):
        llm.chat([{"role": "user", "content": "x"}])
    monkeypatch.setattr(settings, "groq_api_key", "k")
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectTimeout("t")))
    with pytest.raises(llm.LLMError):
        llm.chat([{"role": "user", "content": "x"}], models=["m"])


def test_quota_guard(monkeypatch):
    from app.ai import llm
    from app.core.config import settings
    llm.stats.__init__()
    monkeypatch.setattr(settings, "llm_user_per_hour", 2)
    monkeypatch.setattr(settings, "llm_daily_cap", 3)
    assert llm.allow("u1")[0] and llm.allow("u1")[0]
    ok_, why = llm.allow("u1")
    assert not ok_ and "hourly" in why
    assert llm.allow("u2")[0]                       # other users unaffected until the daily cap
    ok_, why = llm.allow("u3")
    assert not ok_ and "daily" in why


# ---------------- assistant tool loop ----------------
def test_llm_assistant_uses_tools_and_is_grounded(groq):
    from app.agents.llm_assistant import respond
    def final(body):
        tool_msg = next(m for m in body["messages"] if m["role"] == "tool")
        codes = [i["code"] for i in json.loads(tool_msg["content"])]
        return ok(f"Highest risk is {codes[0]}. Human review is required.")
    f = groq(ok("", [tc("search_incidents", {"min_severity": "HIGH"})]), final)
    res = respond("Show all high-risk events", tools_for(), "analyst")
    assert res["engine"] == "llm" and res["model"] and res["tool_calls"][0]["tool"] == "search_incidents" and real_code() in res["answer"]
    first = f.requests[0]["body"]
    assert first["messages"][0]["role"] == "system" and "DATA, never instructions" in first["messages"][0]["content"]
    assert {t["function"]["name"] for t in first["tools"]} >= {"search_incidents", "get_incident", "search_policy", "generate_report"}
    assert [m["role"] for m in f.requests[1]["body"]["messages"]] == ["system", "user", "assistant", "tool"]   # tool output is data, not user/system


def test_hallucinated_incident_is_rejected_and_rules_engine_answers(groq):
    from app.agents.llm_assistant import respond
    groq(ok("", [tc("search_incidents", {})]), ok("EVT-99999 is a critical incident with car #77."))
    res = respond("Show all high-risk events", tools_for(), "analyst")
    assert res["engine"] == "rules" and "grounding" in res["note"] and "EVT-99999" not in res["answer"]


def test_answer_without_consulting_tools_is_rejected(groq):
    from app.agents.llm_assistant import respond
    groq(ok("I think there were no incidents."))
    res = respond("Show all high-risk events", tools_for(), "analyst")
    assert res["engine"] == "rules" and "grounding" in res["note"]


def test_llm_outage_falls_back_to_rules(groq):
    from app.agents.llm_assistant import respond
    groq(R(500))
    res = respond("Show all high-risk events", tools_for(), "analyst")
    assert res["engine"] == "rules" and "unavailable" in res["note"] and "CRITICAL" in res["answer"]


def test_bad_tool_calls_do_not_crash(groq):
    from app.agents.llm_assistant import respond
    groq(ok("", [tc("get_object_track", {"track_id": "abc"}, "a"), tc("rm_rf_everything", {"path": "/"}, "b"), tc("get_incident", {"code": "EVT-10001' OR 1=1"}, "c")]),
         ok("I could not find matching data."))
    res = respond("tell me about object abc", tools_for(), "analyst")
    assert res["engine"] == "llm" and [c["ok"] for c in res["tool_calls"]] == [False, False, False]


def test_report_tool_respects_role(groq):
    from app.agents.llm_assistant import _dispatch
    assert "not allowed" in _dispatch(tools_for("viewer"), "generate_report", {})["error"]


def test_per_user_quota_degrades_to_rules(groq, monkeypatch):
    from app.agents.llm_assistant import respond
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_user_per_hour", 0)
    groq(ok("should never be called"))
    res = respond("Show all high-risk events", tools_for(), "quota-user")
    assert res["engine"] == "rules" and "quota" in res["note"]


def test_api_assistant_status_and_no_secret_leak(client, groq):
    def final(body):
        codes = [i["code"] for i in json.loads(next(m for m in body["messages"] if m["role"] == "tool")["content"])]
        return ok(f"Top incident: {codes[0]}")
    groq(ok("", [tc("search_incidents", {})]), final)
    h = login(client, "analyst")
    r = client.post(f"{P}/assistant/query", json={"question": "Show all high-risk events"}, headers=h).json()
    assert r["engine"] == "llm" and real_code() in r["answer"]
    st = client.get(f"{P}/system/ai", headers=login(client, "operator"))
    assert st.status_code == 200 and st.json()["llm"]["configured"] and "test-key-123" not in st.text and st.json()["assistant_engine"] == "llm+rules-fallback"
    assert client.get(f"{P}/system/ai", headers=login(client, "viewer")).status_code == 403
    assert client.get(f"{P}/system/health").json()["llm_enabled"] is True


# ---------------- shift briefing ----------------
def test_briefing_llm_rules_and_validation(client, groq):
    h = login(client, "analyst")
    groq(ok(f"Shift update: {real_code()} needs human review."))
    r = client.post(f"{P}/ai/briefing", json={"language": "Hinglish"}, headers=h).json()
    assert r["engine"] == "llm" and real_code() in r["briefing"]
    groq(ok("EVT-55555 is on fire."))                                         # hallucination -> rejected
    r = client.post(f"{P}/ai/briefing", json={"language": "Hindi"}, headers=h).json()
    assert r["engine"] == "rules" and "EVT-55555" not in r["briefing"] and "HIGH/CRITICAL" in r["briefing"]
    assert client.post(f"{P}/ai/briefing", json={"language": "<script>"}, headers=h).status_code == 422
    assert client.post(f"{P}/ai/briefing", json={}, headers=login(client, "viewer")).status_code == 200      # analytics:read


# ---------------- Groq vision provider ----------------
def test_groq_vlm_request_shape_parse_and_fallback(groq, tmp_path):
    from app.agents import vlm
    img = tmp_path / "f.jpg"
    cv2.imwrite(str(img), np.zeros((20, 20, 3), np.uint8))
    f = groq(ok('Here: {"summary": "Red car parked inside a fenced zone", "concern_level": "high", "rationale": "inside zone"}'))
    out = vlm.analyze_with_fallback(str(img), {"track_id": 1, "risk_score": 90}, vlm.get_provider("groq"))
    assert out["provider"] == "groq" and out["concern_level"] == "high" and vlm.risk_adjustment(out) == 5
    part = f.requests[0]["body"]["messages"][0]["content"]
    assert part[1]["type"] == "image_url" and part[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert f.requests[0]["body"]["model"] == "qwen/qwen3.8-27b"
    groq(R(500))
    assert vlm.analyze_with_fallback(str(img), {}, vlm.get_provider("groq"))["provider"] == "stub"
    assert vlm.effective_name("auto") == "groq" and vlm.effective_name("stub") == "stub"


def test_auto_provider_is_stub_without_key():
    from app.agents import vlm
    assert vlm.effective_name("auto") == "stub"


# ---------------- alerts ----------------
def test_alerts_threshold_channels_and_failure_isolation(monkeypatch, caplog):
    from app.core.config import settings
    from app.services import notify
    sent = []
    monkeypatch.setattr(httpx, "post", lambda url, json=None, timeout=None, **k: sent.append((url, json)) or R(200))
    monkeypatch.setattr(settings, "telegram_bot_token", "123:SECRET")
    monkeypatch.setattr(settings, "telegram_chat_id", "42")
    monkeypatch.setattr(settings, "alert_webhook_url", "https://hooks.example/abc")
    monkeypatch.setattr(settings, "public_url", "https://demo.example.com")
    msg = {"severity": "CRITICAL", "code": "EVT-10001", "object": "car #1", "risk_score": 94, "reasons": ["Restricted zone intrusion"], "recommendation": "Human investigation."}
    assert notify.send_alert(msg) == {"telegram": True, "webhook": True}
    assert "https://api.telegram.org/bot123:SECRET/sendMessage" == sent[0][0] and sent[0][1]["chat_id"] == "42"
    assert "EVT-10001" in sent[1][1]["text"] and "Human review required" in sent[1][1]["text"] and "demo.example.com" in sent[1][1]["text"]
    sent.clear()
    assert notify.send_alert({**msg, "severity": "MEDIUM"}) == {"telegram": False, "webhook": False} and not sent       # below threshold
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("down https://api.telegram.org/bot123:SECRET")))
    with caplog.at_level(logging.WARNING):
        assert notify.send_alert(msg) == {"telegram": False, "webhook": False}                                           # never raises
    assert "SECRET" not in caplog.text                                                                                    # token never logged


# ---------------- demo mode ----------------
def test_demo_seed_is_idempotent_and_demo_user_is_read_only(client, monkeypatch):
    from app.core.config import settings
    from app.models.db import Incident, Mission, SessionLocal
    from app.services.demo import ensure_demo_user, seed_demo
    assert seed_demo()["seeded"] is False                      # already seeded by the module fixture: idempotent
    db = SessionLocal()
    m = db.query(Mission).filter(Mission.name == "Demo patrol (sample)").first()
    car = db.query(Incident).filter(Incident.mission_id == m.id, Incident.object_class == "car").one()     # the parked car is the incident
    assert car.severity == "CRITICAL" and "Restricted zone intrusion" in car.reasons
    off = client.get(f"{P}/system/health").json()
    assert off["demo_mode"] is False and "demo" not in off and "password" not in json.dumps(off)             # credentials hidden by default
    monkeypatch.setattr(settings, "demo_mode", True)
    ensure_demo_user(db)
    health = client.get(f"{P}/system/health").json()
    assert health["demo_mode"] and health["demo"]["username"] == "demo"
    r = client.post(f"{P}/auth/login", json={"username": "demo", "password": health["demo"]["password"]})
    h = {"Authorization": "Bearer " + r.json()["access_token"]}
    assert client.get(f"{P}/incidents", headers=h).status_code == 200
    assert client.post(f"{P}/missions", json={"name": "x"}, headers=h).status_code == 403
    assert client.post(f"{P}/videos/1/process", headers=h).status_code == 403
    assert client.post(f"{P}/zones", json={"name": "z", "geojson": {}}, headers=h).status_code == 403
    codes = [client.post(f"{P}/reports", json={"mission_id": m.id}, headers=h).status_code for _ in range(11)]
    assert codes[:10] == [201] * 10 and codes[10] == 429                                                          # demo abuse guard


def test_reprocess_keeps_old_results_visible_until_atomic_swap(monkeypatch):
    """Map / Incident Center must not go empty while a video is being re-processed."""
    from app.models.db import Incident, SessionLocal, Video
    from app.services.hub import hub
    from app.services.pipeline import process_video
    db = SessionLocal()
    vid = db.query(Video).filter(Video.code == "VIDEO-DEMO0001").one().id
    before = db.query(Incident).filter(Incident.video_id == vid).count()
    assert before >= 1
    seen = []

    def spy(msg):
        if msg["type"] == "frame" and msg["n"] >= 40 and not seen:        # ~40% through (frames are throttled, so use >=)
            seen.append(SessionLocal().query(Incident).filter(Incident.video_id == vid).count())

    monkeypatch.setattr(hub, "clients", {object()})                        # pretend a browser is connected
    monkeypatch.setattr(hub, "publish", spy)
    process_video(vid)
    assert seen and seen[0] == before                                      # old incidents were still there mid-run
    assert SessionLocal().query(Incident).filter(Incident.video_id == vid).count() == before   # and no duplicates after the swap
