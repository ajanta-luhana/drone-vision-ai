# Requirements traceability (spec v1.0 -> implementation)

**Done** = implemented and covered by an automated test · **Partial** = works with a stated limit · **Not verified** = code exists but could not be tested in the build environment · **V2** = deferred.
Test evidence: 69 backend tests (unit + full API pipeline + WebSocket) and 64 UI checks (`scripts/ui_test.mjs`: the real dashboard JS in jsdom against a live server, including the real-time live view). The new UI was also driven in headless Chromium and checked visually (`docs/screenshots/`).

| Spec | Requirement | Status | Where / notes |
|---|---|---|---|
| §7 | MP4/MOV/AVI upload, metadata | Done | content-validated, size-capped, server-chosen filenames. RTSP/WebRTC live streams: V2 |
| §8 | Detector, confidence, bboxes | Done / **Not verified on real footage** | `blob` (demo clip only) tested. `yolo` output parsing and class-flicker handling are unit-tested with a faked model; actual YOLO inference has **not** been run (no torch in the build sandbox). Build with `INSTALL_YOLO=1` |
| §8 | Fire / smoke classes | Partial | event + risk logic tested on synthetic tracks; needs a detector trained on those classes |
| §9 | Persistent IDs, trajectories | Done | class-group + majority-vote tracker. Demo clip: 5 objects -> 5 tracks, 1 ID switch (`scripts/evaluate.py`) |
| §10 | Dwell, entry/exit, stationary, direction changes, proximity | Done | `services/temporal.py` |
| §11 | Zone intrusion, stationary, loitering, proximity, unusual, fire/smoke, **obstacle** | Done | obstacle = anything stationary on a `road` zone, or class obstacle/debris |
| §12 | GeoJSON zones, PIP, zone types | Done | now incl. `road`; **zones can be drawn directly on the video frame** (`POST /videos/{id}/zone-from-pixels`) |
| §13 | Simulated GPS | Done | deterministic; pixel<->geo round-trip tested to <5 cm |
| §14 | VLM on selected frames | Done (logic) / **Not verified live** | Anthropic provider request/response parsing and fallback are tested with a faked HTTP layer; never called the real API |
| §15-17 | Risk, decision agent, human review | Done | enforced + tested |
| §18-19 | RAG + tool assistant | Done | BM25+stemming; rule-based router (not an LLM planner) |
| §20-21 | Dashboard | Done | 64 UI checks in jsdom (all tabs, mission create, zone draw, process, **live streaming**, review, assistant, reports, RBAC) plus screenshots from headless Chromium. Basemap tiles (need internet) and `<video>` H.264 playback were **not** verified |
| §22 | PDF/JSON/CSV reports | Done | |
| §23-24 | DB/infra | Partial | SQLite. PostgreSQL/PostGIS, Qdrant, Redis: V2 (schema is portable, not tested on Postgres) |
| §25 | WebSocket live events | Done | `/ws/events` (first-message token auth): throttled annotated frames + detections + GPS (live view), progress, incident (sent after DB commit), done, failed. Reconnects with back-off |
| §27-28 | Auth, RBAC, audit | Done | |
| §29-30 | Model registry, MLOps metrics | Partial | registry + latency/FPS/RSS + precision/recall/ID-switch on the demo clip. mAP / IDF1 on real data: needs labelled footage |
| §31 | Prometheus | Done | `/api/v1/system/prometheus` (bearer token). Grafana dashboards: V2 |
| §32 | Security | Done | |
| §33 | Privacy | Partial | `BLUR_PERSONS=1` blurs person regions in all stored/served video & evidence. Face / licence-plate-specific blurring: not applicable to the top-down demo, not implemented |
| §35-36 | 3 GB constraints | Done | ~150 MB peak on demo; Docker build no longer needs apt packages |

See `docs/limitations.md` for what is still genuinely open.
