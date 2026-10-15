"""Demo mode: a ready-made sample mission so a freshly deployed (ephemeral, free-tier) instance is never empty."""
import shutil

from ..core.config import ROOT, settings
from ..core.security import hash_password
from ..geospatial.geo import extract_geometry
from ..models.db import Mission, SessionLocal, User, Video, Zone, init_db
import json

DEMO_NAME = "Demo patrol (sample)"


def demo_credentials() -> dict:
    import os
    return {"username": "demo", "password": os.getenv("DEMO_PASSWORD", "demo-drone-2026"), "role": "analyst"}


def ensure_demo_user(db):
    c = demo_credentials()
    if not db.query(User).filter(User.username == c["username"]).first():
        db.add(User(username=c["username"], password_hash=hash_password(c["password"]), role=c["role"]))
        db.commit()


def seed_demo(session_factory=SessionLocal) -> dict:
    """Idempotent: creates zone + mission + video from the bundled synthetic clip and processes it."""
    from .pipeline import process_video
    init_db()
    db = session_factory()
    try:
        if db.query(Mission).filter(Mission.name == DEMO_NAME).first():
            return {"seeded": False}
        src = ROOT / "datasets" / "demo_aerial.mp4"
        zf = ROOT / "simulations" / "zone_restricted_a.geojson"
        if not src.exists() or not zf.exists():
            return {"seeded": False, "error": "demo assets missing"}
        if not db.query(Zone).filter(Zone.name == "Restricted Zone A", Zone.active.is_(True)).first():
            geom = extract_geometry(json.loads(zf.read_text()))
            db.add(Zone(name="Restricted Zone A", zone_type="restricted", geojson=geom, forbidden_classes=[], created_by="demo-seed"))
        m = Mission(code="TMP", name=DEMO_NAME, description="Synthetic sample: a vehicle parks in a restricted zone and a person approaches.",
                    created_by="demo-seed")
        db.add(m)
        db.flush()
        m.code = f"MISSION-{m.id:03d}"
        dest = settings.storage_dir / "videos" / "VIDEO-DEMO0001.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)
        v = Video(code="VIDEO-DEMO0001", mission_id=m.id, filename="demo_aerial.mp4", path=str(dest))
        db.add(v)
        db.commit()
        vid = v.id
    finally:
        db.close()
    stats = process_video(vid, session_factory)
    return {"seeded": True, "stats": stats}


if __name__ == "__main__":      # used at Docker build time: bake the sample data into the image
    print(seed_demo())
