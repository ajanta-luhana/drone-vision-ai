# Deployment
**Local:** `pip install -r requirements.txt`, `python scripts/make_demo_video.py`, `SEED_PASSWORD=... uvicorn app.main:app --app-dir backend`.
**Docker:** `cp .env.example .env` (edit), `docker compose up --build`.
**Production checklist:** set `JWT_SECRET` and `SEED_PASSWORD`; serve behind HTTPS (the token is a bearer credential);
move to PostgreSQL/PostGIS; replace the in-memory login throttle with Redis; add Prometheus metrics and a job queue (Celery/Redis);
set retention policy for `storage/` footage and evidence frames.
