"""Vision-language model provider abstraction. The VLM is only called for selected evidence frames."""
import base64
import json
import re
from pathlib import Path

from ..core.config import settings

PROMPT = (
    "You are reviewing one evidence frame from aerial/drone footage for a human operator. "
    "Context (from automated detectors): {context}\n"
    "Describe only what is visible. Reply with JSON: "
    '{{"summary": str, "concern_level": "none"|"low"|"high", "rationale": str}}. '
    "Do not speculate about identity or intent."
)


class VLMProvider:
    name = "base"
    model = ""

    def analyze(self, image_path: str, context: dict) -> dict:  # pragma: no cover
        raise NotImplementedError


class StubVLM(VLMProvider):
    """Deterministic offline provider: summarises the detector context. Never calls a network."""
    name, model = "stub", "rule-summary-v1"

    def analyze(self, image_path, context):
        parts = [f"{context.get('object_class', 'object')} #{context.get('track_id')}"]
        if context.get("zone"):
            parts.append(f"inside {context['zone']}")
        if context.get("stationary_s"):
            parts.append(f"stationary ~{context['stationary_s']:.0f}s")
        if context.get("person_nearby"):
            parts.append("with a person nearby")
        concern = "high" if context.get("risk_score", 0) >= 51 else ("low" if context.get("risk_score", 0) > 25 else "none")
        return {"summary": "Frame shows " + ", ".join(parts) + ".", "concern_level": concern,
                "rationale": "Offline stub: derived from detector/tracker context, not from pixels."}


class AnthropicVLM(VLMProvider):
    """Calls the Anthropic Messages API with the key from ANTHROPIC_API_KEY. Falls back to the stub on any error."""
    name = "anthropic"

    def __init__(self):
        self.model = settings.vlm_model

    def analyze(self, image_path, context):
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        import httpx
        data = base64.b64encode(Path(image_path).read_bytes()).decode()
        body = {"model": self.model, "max_tokens": 400, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
            {"type": "text", "text": PROMPT.format(context=json.dumps(context))}]}]}
        r = httpx.post("https://api.anthropic.com/v1/messages", json=body, timeout=60, headers={
            "x-api-key": settings.anthropic_api_key, "anthropic-version": "2023-06-01",
            "content-type": "application/json"})
        r.raise_for_status()
        text = "".join(b.get("text", "") for b in r.json().get("content", []))
        m = re.search(r"\{.*\}", text, re.S)
        out = json.loads(m.group(0)) if m else {"summary": text[:300], "concern_level": "low", "rationale": ""}
        if out.get("concern_level") not in ("none", "low", "high"):
            out["concern_level"] = "low"
        return out


class GroqVLM(VLMProvider):
    """Evidence-frame analysis through Groq's OpenAI-compatible vision endpoint (model chain from GROQ_VISION_MODELS)."""
    name = "groq"

    def __init__(self):
        self.model = settings.groq_vision_models[0] if settings.groq_vision_models else ""

    def analyze(self, image_path, context):
        from ..ai import llm
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        out = llm.vision(b64, PROMPT.format(context=json.dumps(context)))
        self.model = out["model"]
        text = out["message"].get("content") or ""
        m = re.search(r"\{.*\}", text, re.S)
        res = json.loads(m.group(0)) if m else {"summary": text[:300], "concern_level": "low", "rationale": ""}
        if res.get("concern_level") not in ("none", "low", "high"):
            res["concern_level"] = "low"
        return {k: res.get(k, "") for k in ("summary", "concern_level", "rationale")}


def effective_name(name: str | None = None) -> str:
    name = name or settings.vlm_provider
    if name == "auto":
        from ..ai import llm
        return "groq" if llm.configured() else "stub"
    return name


def get_provider(name: str | None = None) -> VLMProvider:
    n = effective_name(name)
    return GroqVLM() if n == "groq" else (AnthropicVLM() if n == "anthropic" else StubVLM())


def analyze_with_fallback(image_path: str, context: dict, provider: VLMProvider | None = None) -> dict:
    provider = provider or get_provider()
    try:
        res = provider.analyze(image_path, context)
        used = provider
    except Exception as e:  # never let a VLM failure break the pipeline
        used = StubVLM()
        res = used.analyze(image_path, context)
        res["rationale"] += f" (fallback: {provider.name} failed: {type(e).__name__})"
    return {"provider": used.name, "model": used.model, **res}


def risk_adjustment(vlm: dict) -> int:
    if vlm.get("provider") == "stub":  # stub is derived from the risk itself; avoid double counting
        return 0
    return {"high": 5, "low": 0, "none": -5}.get(vlm.get("concern_level"), 0)
