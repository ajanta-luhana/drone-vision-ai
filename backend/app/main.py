import asyncio
import logging
import threading
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .api.routes import router
from .core.config import ROOT, settings
from .core.security import decode_token, has_permission, hash_password
from .services.demo import ensure_demo_user, seed_demo
from .services.hub import hub
from .models.db import ModelRecord, SessionLocal, User, Video, init_db

log = logging.getLogger("dronevision")
ROLES = ("admin", "operator", "analyst", "viewer")


def seed(db):
    """Create the four default roles on a fresh DB. Passwords come from SEED_PASSWORD or are generated and
    printed ONCE - nothing secret is hardcoded in the repository."""
    if db.query(User).count() == 0:
        shared = settings.seed_password
        for role in ROLES:
            pw = shared or secrets.token_urlsafe(12)
            db.add(User(username=role, password_hash=hash_password(pw), role=role))
            if not shared:
                log.warning("SEEDED USER  %s / %s   (shown once - set SEED_PASSWORD to choose your own)", role, pw)
        if shared:
            log.warning("Seeded users admin/operator/analyst/viewer with the password from SEED_PASSWORD")
    if db.query(ModelRecord).count() == 0:
        db.add_all([
            ModelRecord(name="blob-detector", version="v1", task="Vehicle/person detection (demo footage)",
                        dataset="synthetic_demo", metrics={"note": "evaluate with scripts/evaluate.py"}, status="demo"),
            ModelRecord(name="yolo", version=settings.yolo_weights, task="Object detection (real footage)",
                        dataset="COCO / VisDrone fine-tune recommended", metrics={}, status="optional"),
            ModelRecord(name="bytelike-tracker", version="v1", task="Multi-object tracking", dataset="-", metrics={}, status="production"),
            ModelRecord(name="vlm", version=f"{settings.vlm_provider}:{settings.vlm_model}", task="Evidence-frame analysis", dataset="-", metrics={}, status="production"),
            ModelRecord(name="risk-engine", version="rules-v1", task="Explainable risk scoring", dataset="-", metrics={}, status="production")])
    # a crash/restart must not leave videos stuck in "processing"
    for v in db.query(Video).filter(Video.status == "processing"):
        v.status, v.error = "failed", "Interrupted by server restart"
    db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    init_db()
    hub.bind(asyncio.get_running_loop())
    if len(settings.jwt_secret) < 32:
        log.warning("JWT_SECRET is shorter than 32 characters - use a long random value in production")
    db = SessionLocal()
    try:
        seed(db)
        if settings.demo_mode:
            ensure_demo_user(db)
            threading.Thread(target=seed_demo, daemon=True, name="demo-seed").start()   # no-op when already baked in
    finally:
        db.close()
    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.include_router(router)


@app.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    """Live progress / incident feed. Auth = first message {"token": "<jwt>"} (keeps tokens out of URLs/logs)."""
    await ws.accept()
    try:
        first = await asyncio.wait_for(ws.receive_json(), timeout=5)
        payload = decode_token(str(first.get("token", "")))
        if not has_permission(payload.get("role", ""), "incidents:read"):
            raise PermissionError
    except Exception:
        await ws.close(code=4401)
        return
    hub.clients.add(ws)
    await ws.send_json({"type": "hello", "user": payload.get("sub")})
    try:
        while True:
            await ws.receive_text()      # keep-alive; client messages are ignored
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(ws)

# demo assets (zone GeoJSON + sample clip) - contain no sensitive data
for url, folder in (("/sim", "simulations"), ("/demo", "datasets")):
    d = Path(ROOT) / folder
    if d.exists():
        app.mount(url, StaticFiles(directory=str(d)), name=folder)

frontend = Path(ROOT) / "frontend"
if frontend.exists():
    app.mount("/", StaticFiles(directory=str(frontend), html=True), name="frontend")
