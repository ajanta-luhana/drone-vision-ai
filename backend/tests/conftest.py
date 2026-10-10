import os
import sys
import tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp(prefix="dvai_test_")
os.environ["STORAGE_DIR"] = _tmp
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["SEED_PASSWORD"] = "test-password-123"
os.environ["JWT_SECRET"] = "test-secret-not-for-prod"
os.environ.setdefault("PROCESS_FPS", "8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="session")
def demo_assets():
    video = ROOT / "datasets" / "demo_aerial.mp4"
    zone = ROOT / "simulations" / "zone_restricted_a.geojson"
    if not video.exists() or not zone.exists():
        import make_demo_video
        make_demo_video.main()
    return video, zone


@pytest.fixture(scope="session")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


def login(client, role):
    r = client.post("/api/v1/auth/login", json={"username": role, "password": "test-password-123"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}
