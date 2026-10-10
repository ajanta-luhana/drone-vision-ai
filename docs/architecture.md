# Architecture

```
Aerial video -> ingestion (validated upload) -> frame sampling/resize -> detector (blob | YOLO)
  -> tracker (ByteTrack-style) -> pixel->geo projection (simulated GPS) -> temporal engine
  -> event engine -> risk engine -> [selected key frames -> VLM] -> decision agent
  -> incidents / events / reports -> REST API -> dashboard (map, incidents, assistant)
```

| Module | Path | Notes |
|---|---|---|
| Detector | `backend/app/tracking/detector.py` | provider interface; YOLO lazy-loaded |
| Tracker | `backend/app/tracking/tracker.py` | two-stage IoU, velocity prediction, persistent IDs |
| Geospatial | `backend/app/geospatial/geo.py` | point-in-polygon (holes), haversine, deterministic GPS simulator, pixel<->geo |
| Temporal / events | `backend/app/services/{temporal,events}.py` | stationary runs, dwell, proximity, loitering, unusual activity |
| Risk | `backend/app/risk/engine.py` | weighted, explainable, configurable bands |
| Decision agent | `backend/app/agents/decision.py` | software actions only; human review enforced for HIGH+ |
| VLM | `backend/app/agents/vlm.py` | provider abstraction (stub / Anthropic), max 3 calls per video, failure-safe |
| RAG | `backend/app/rag/store.py` | BM25 + light stemming over `docs/policies`; swap for Qdrant later |
| Assistant | `backend/app/agents/assistant.py` | tool router; answers composed only from tool output |
| Pipeline | `backend/app/services/pipeline.py` | single streaming pass, one job at a time |
| Reports | `backend/app/reports/generator.py` | PDF, JSON, CSV |

## Memory strategy (3 GB target; measured ~150 MB peak on the demo)
Frames are streamed (never buffered), time/resolution down-sampled, evidence frames re-read on demand, VLM only on
selected frames, one processing job at a time, YOLO loaded lazily.

## Geo model
Nadir camera, ground scale `2*alt*tan(hfov/2)/width` metres per pixel, rotated by heading. Zones are GeoJSON; objects are
tested in lat/lon, so zones are independent of any video's pixel space.
