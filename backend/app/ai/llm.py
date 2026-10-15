"""Minimal OpenAI-compatible chat client for Groq.

Design goals (Groq's free tier is generous but retires models and rate-limits aggressively):
  * model *fallback chain* - on "model not found / decommissioned" try the next model ID
  * retry 429 / 5xx with Retry-After (bounded), never hang the app (timeout)
  * never raise into callers that have a non-AI fallback: raises LLMError, callers catch
  * secrets never logged; usage counters feed GET /api/v1/system/ai
"""
import re
import threading
import time
from collections import deque

import httpx

from ..core.config import settings

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_BAD_MODEL = re.compile(r"model_not_found|decommission|does not exist|no longer supported|not_found|deprecat", re.I)


class LLMError(RuntimeError):
    pass


class LLMStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = self.ok = self.errors = self.retries = self.fallbacks = 0
        self.last_error = None
        self.last_model = None
        self.day_calls = deque()          # timestamps, for the daily cap
        self.user_calls = {}              # username -> deque[timestamps]

    def snapshot(self):
        with self.lock:
            return {"calls": self.calls, "ok": self.ok, "errors": self.errors, "retries": self.retries,
                    "model_fallbacks": self.fallbacks, "last_error": self.last_error, "last_model": self.last_model,
                    "calls_last_24h": len(self.day_calls)}


stats = LLMStats()


def configured() -> bool:
    return bool(settings.groq_api_key)


def allow(user: str | None) -> tuple[bool, str]:
    """Quota guard so a public demo cannot burn the whole free tier."""
    now = time.time()
    with stats.lock:
        while stats.day_calls and now - stats.day_calls[0] > 86400:
            stats.day_calls.popleft()
        if len(stats.day_calls) >= settings.llm_daily_cap:
            return False, "daily AI quota reached"
        if user:
            q = stats.user_calls.setdefault(user, deque())
            while q and now - q[0] > 3600:
                q.popleft()
            if len(q) >= settings.llm_user_per_hour:
                return False, "hourly AI quota reached for this user"
            q.append(now)
        stats.day_calls.append(now)
    return True, ""


def _clean(text):
    return _THINK.sub("", text or "").strip()


def chat(messages, tools=None, models=None, json_mode=False, temperature=0.1, max_tokens=900, max_retries=2) -> dict:
    """Returns {"message": {...}, "model": str}. message.content is cleaned text; message.tool_calls as in OpenAI."""
    if not configured():
        raise LLMError("GROQ_API_KEY is not set")
    chain = list(models or settings.groq_models)
    if not chain:
        raise LLMError("no Groq models configured")
    last = None
    for idx, model in enumerate(chain):
        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            body["tools"], body["tool_choice"] = tools, "auto"
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        for attempt in range(max_retries + 1):
            with stats.lock:
                stats.calls += 1
            try:
                r = httpx.post(f"{settings.groq_base_url}/chat/completions", json=body, timeout=settings.llm_timeout_s,
                               headers={"Authorization": f"Bearer {settings.groq_api_key}", "Content-Type": "application/json"})
            except httpx.HTTPError as e:                    # network / timeout
                last = f"network: {type(e).__name__}"
                with stats.lock:
                    stats.errors += 1
                break
            if r.status_code == 429 or r.status_code >= 500:
                with stats.lock:
                    stats.retries += 1
                    stats.errors += 1
                last = f"HTTP {r.status_code}"
                if attempt < max_retries:
                    try:
                        wait = min(float(r.headers.get("retry-after", 1 + attempt)), 8.0)
                    except (ValueError, TypeError):
                        wait = 1.0
                    time.sleep(wait)
                    continue
                break
            if r.status_code >= 400:
                text = r.text[:300] if hasattr(r, "text") else ""
                last = f"HTTP {r.status_code}: {text}"
                with stats.lock:
                    stats.errors += 1
                if _BAD_MODEL.search(text) or r.status_code in (404, 410):
                    with stats.lock:
                        stats.fallbacks += 1
                    break                                   # try next model in the chain
                raise_after = LLMError(last)
                with stats.lock:
                    stats.last_error = last
                raise raise_after                            # auth / bad request: no point retrying other models
            try:
                msg = r.json()["choices"][0]["message"]
            except (KeyError, IndexError, ValueError):
                last = "malformed response"
                break
            msg = dict(msg)
            msg["content"] = _clean(msg.get("content"))
            with stats.lock:
                stats.ok += 1
                stats.last_model, stats.last_error = model, None
            return {"message": msg, "model": model}
    with stats.lock:
        stats.last_error = last
    raise LLMError(last or "LLM unavailable")


def vision(image_b64: str, prompt: str, models=None, max_tokens=500) -> dict:
    content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}]
    return chat([{"role": "user", "content": content}], models=models or settings.groq_vision_models,
                json_mode=False, max_tokens=max_tokens, temperature=0.0)
