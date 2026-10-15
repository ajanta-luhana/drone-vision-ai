"""Central configuration. All secrets come from environment variables (never hardcoded)."""
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _f(name, default):
    return float(os.getenv(name, default))


@dataclass
class Settings:
    app_name: str = "DroneVision AI"
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'storage' / 'dronevision.db'}"))
    storage_dir: Path = field(default_factory=lambda: Path(os.getenv("STORAGE_DIR", str(ROOT / "storage"))))
    policies_dir: Path = field(default_factory=lambda: Path(os.getenv("POLICIES_DIR", str(ROOT / "docs" / "policies"))))
    # Auth. If JWT_SECRET is unset a random per-process secret is used (tokens die on restart).
    jwt_secret: str = field(default_factory=lambda: os.getenv("JWT_SECRET") or secrets.token_urlsafe(48))
    jwt_ttl_minutes: int = field(default_factory=lambda: int(os.getenv("JWT_TTL_MINUTES", "480")))
    seed_password: str = field(default_factory=lambda: os.getenv("SEED_PASSWORD", ""))
    # Upload limits / validation
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "200")))
    allowed_ext: tuple = (".mp4", ".mov", ".avi")
    # Processing (3 GB RAM friendly defaults)
    detector: str = field(default_factory=lambda: os.getenv("DETECTOR", "blob"))  # blob | yolo
    yolo_weights: str = field(default_factory=lambda: os.getenv("YOLO_WEIGHTS", "yolov8n.pt"))
    yolo_imgsz: int = field(default_factory=lambda: int(os.getenv("YOLO_IMGSZ", "960")))
    blur_persons: bool = field(default_factory=lambda: os.getenv("BLUR_PERSONS", "0") == "1")
    process_fps: float = field(default_factory=lambda: _f("PROCESS_FPS", "5"))
    max_width: int = field(default_factory=lambda: int(os.getenv("MAX_WIDTH", "960")))
    conf_threshold: float = field(default_factory=lambda: _f("CONF_THRESHOLD", "0.35"))
    # Event rules (configurable)
    stationary_radius_m: float = field(default_factory=lambda: _f("STATIONARY_RADIUS_M", "2.0"))
    stationary_min_s: float = field(default_factory=lambda: _f("STATIONARY_MIN_S", "4"))
    proximity_m: float = field(default_factory=lambda: _f("PROXIMITY_M", "6"))
    proximity_min_s: float = field(default_factory=lambda: _f("PROXIMITY_MIN_S", "1"))
    loiter_radius_m: float = field(default_factory=lambda: _f("LOITER_RADIUS_M", "6"))
    loiter_min_s: float = field(default_factory=lambda: _f("LOITER_MIN_S", "8"))
    # VLM
    vlm_provider: str = field(default_factory=lambda: os.getenv("VLM_PROVIDER", "auto"))  # auto | stub | groq | anthropic
    # --- Groq (OpenAI-compatible). Model IDs are env-driven because Groq retires models often; comma list = fallback chain.
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_base_url: str = field(default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/"))
    groq_models: tuple = field(default_factory=lambda: tuple(m.strip() for m in os.getenv("GROQ_MODELS", "openai/gpt-oss-120b,openai/gpt-oss-20b").split(",") if m.strip()))
    groq_vision_models: tuple = field(default_factory=lambda: tuple(m.strip() for m in os.getenv("GROQ_VISION_MODELS", "qwen/qwen3.8-27b").split(",") if m.strip()))
    llm_timeout_s: float = field(default_factory=lambda: _f("LLM_TIMEOUT_S", "25"))
    llm_user_per_hour: int = field(default_factory=lambda: int(os.getenv("LLM_USER_PER_HOUR", "30")))   # protects the free quota
    llm_daily_cap: int = field(default_factory=lambda: int(os.getenv("LLM_DAILY_CAP", "400")))
    # --- alerts / demo
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    alert_webhook_url: str = field(default_factory=lambda: os.getenv("ALERT_WEBHOOK_URL", ""))
    alert_min_severity: str = field(default_factory=lambda: os.getenv("ALERT_MIN_SEVERITY", "HIGH"))
    public_url: str = field(default_factory=lambda: os.getenv("PUBLIC_URL", "").rstrip("/"))
    demo_mode: bool = field(default_factory=lambda: os.getenv("DEMO_MODE", "0") == "1")
    vlm_model: str = field(default_factory=lambda: os.getenv("VLM_MODEL", "claude-sonnet-4-6"))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    # Risk thresholds: upper bound of each band (configurable)
    risk_bands: tuple = (("LOW", 25), ("MEDIUM", 50), ("HIGH", 75), ("CRITICAL", 100))
    incident_min_score: int = field(default_factory=lambda: int(os.getenv("INCIDENT_MIN_SCORE", "26")))

    def ensure_dirs(self):
        for sub in ("videos", "annotated", "frames", "reports", "tmp"):
            (self.storage_dir / sub).mkdir(parents=True, exist_ok=True)


settings = Settings()
