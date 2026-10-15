"""Offline MOCK of Groq's OpenAI-compatible API (chat completions with tool calling + vision).

NOT an LLM: it follows keyword rules. It exists so the whole AI path (tool-calling loop, grounding check, vision provider,
briefing) can be exercised in development and CI with no API key and no network:

    python scripts/mock_groq.py                         # serves on :9000
    GROQ_BASE_URL=http://127.0.0.1:9000/openai/v1 GROQ_API_KEY=mock uvicorn app.main:app --app-dir backend

Real answers/quality come from real Groq models - set GROQ_API_KEY and leave GROQ_BASE_URL unset.
"""
import json
import re

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="mock-groq")
HINGLISH = re.compile(r"\b(kya|hai|hain|batao|bataiye|dikhao|dikha|kaun|kyun|kyu|mujhe|sabse|wala|kitna)\b", re.I)


def msg(content="", tool_calls=None):
    m = {"role": "assistant", "content": content}
    if tool_calls:
        m["tool_calls"] = tool_calls
    return {"choices": [{"message": m, "finish_reason": "stop"}], "model": "mock-groq"}


def call(name, args):
    return [{"id": "call_mock_1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]


def plan(q: str):
    ql = q.lower()
    code = re.search(r"evt-\d+", ql)
    track = re.search(r"#\s?(\d+)", ql)
    if code:
        return "get_incident", {"code": code.group(0).upper()}
    if re.search(r"procedure|policy|sop|protocol|rule", ql):
        return "search_policy", {"query": q}
    if "report" in ql and re.search(r"generate|create|make", ql):
        return "generate_report", {}
    if track and re.search(r"why|risk|explain|classif", ql):
        return "get_incident", {"track_id": int(track.group(1))}
    if track:
        return "get_object_track", {"track_id": int(track.group(1))}
    if re.search(r"zone|building|area", ql):
        return "search_zone", {"name_query": "Restricted"}
    if re.search(r"enter|intrusion|violation", ql):
        return "search_incidents", {"event_type": "ZONE_INTRUSION"}
    return "search_incidents", {"min_severity": "HIGH"}


def compose(question: str, tool_content: str) -> str:
    hi = bool(HINGLISH.search(question))
    try:
        data = json.loads(tool_content)
    except ValueError:
        return "I could not read the tool result."
    if isinstance(data, dict) and data.get("error"):
        return "I could not complete that: " + str(data["error"])
    if isinstance(data, list) and data and "code" in data[0]:
        t = data[0]
        ev = "; ".join(t.get("evidence") or [])
        if hi:
            return f"Sabse zyada risk wala incident {t['code']} hai: {t['object']}, {t['severity']} ({t['risk_score']}/100). {ev}. HIGH/CRITICAL incident par human review zaroori hai."
        extra = f" ({len(data)} incidents matched.)" if len(data) > 1 else ""
        return f"Top incident: {t['code']} - {t['object']}, {t['severity']} ({t['risk_score']}/100). {ev}.{extra} HIGH/CRITICAL incidents need human review."
    if isinstance(data, list) and data and "heading" in data[0]:
        return f"From {data[0]['source']} ({data[0]['heading']}): " + data[0]["text"][:420].replace("\n", " ")
    if isinstance(data, dict) and "incident_id" in data:
        why = "; ".join(data.get("reasons") or [])
        if hi:
            return f"{data['incident_id']} {data['severity']} ({data['risk_score']}/100) hai kyunki: {why}. Sujhav: {data['ai_recommendation']}"
        return f"{data['incident_id']} is {data['severity']} ({data['risk_score']}/100) because: {why}. Recommendation: {data['ai_recommendation']}"
    if isinstance(data, dict) and "stats" in data and "class" in data:
        s = data["stats"]
        return f"{data['class']} #{data['track_id']}: stationary {s.get('stationary_s', 0)} s, travelled {s.get('distance_m', 0)} m."
    if isinstance(data, dict) and "report_id" in data:
        return f"Report #{data['report_id']} generated for {data['mission']}."
    if not data:
        return "Database mein is naam se kuch nahi mila." if hi else "I could not find anything matching that in the database."
    return "Result: " + tool_content[:300]


@app.get("/openai/v1/models")
def models():
    return {"data": [{"id": "mock-groq"}]}


@app.post("/openai/v1/chat/completions")
async def completions(req: Request):
    body = await req.json()
    messages = body.get("messages", [])
    if not req.headers.get("authorization", "").startswith("Bearer "):
        return JSONResponse({"error": {"message": "Invalid API Key"}}, status_code=401)
    last = messages[-1] if messages else {}
    # vision
    if isinstance(last.get("content"), list) and any(p.get("type") == "image_url" for p in last["content"]):
        return msg(json.dumps({"summary": "Overhead view: a red vehicle is parked inside a fenced area with a person standing next to it.",
                               "concern_level": "high", "rationale": "Vehicle stopped inside the marked restricted zone with a person nearby (mock model)."}))
    # briefing (no tools)
    if not body.get("tools"):
        sys_txt = messages[0].get("content", "") if messages else ""
        if "briefing" in sys_txt:
            d = json.loads(last["content"])
            lang = re.search(r"Reply in ([A-Za-z ]+)\.", sys_txt)
            t = d["top_incidents"][0] if d["top_incidents"] else None
            if not t:
                return msg("No incidents recorded yet.")
            if lang and lang.group(1).strip().lower() in ("hinglish", "hindi"):
                return msg(f"Shift update: {d['incident_count']} incident, {d['high_or_critical']} HIGH/CRITICAL. Sabse bada risk {t['code']} ({t['object']}, {t['severity']} {t['risk_score']}/100). Human review baaki hai.")
            return msg(f"Shift update: {d['incident_count']} incident(s), {d['high_or_critical']} HIGH/CRITICAL. Highest risk {t['code']} ({t['object']}, {t['severity']} {t['risk_score']}/100). {d['pending_review']} awaiting human review.")
        return msg("OK")
    question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    if tool_msgs:
        return msg(compose(question, tool_msgs[-1]["content"]))
    name, args = plan(question)
    return msg("", call(name, args))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)
