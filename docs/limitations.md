# Known limitations (honest list)

1. **Real-footage accuracy is unmeasured.** The only measured numbers (precision ~0.99, recall ~0.98) are on the synthetic clip with the colour-blob detector.
   YOLO support is implemented and its parsing is tested, but no real aerial video has been run through it. COCO `yolov8n` is weak on small, top-down objects;
   expect to fine-tune on VisDrone/UAVDT for good results. Measure on your own labelled clip with `scripts/evaluate.py` as a template.
2. **VLM** defaults to an offline stub that restates detector context. The Anthropic provider is tested against a faked HTTP layer only.
3. **Assistant** is a rule-based tool router, not an LLM: unfamiliar phrasings get a "here is what I can do" answer instead of a guess (by design).
4. **GPS is simulated** (nadir camera, flat ground, no lens distortion). Zones are therefore only as accurate as the assumed altitude/FOV/heading.
5. **Risk weights** are hand-set and explainable, not learned or validated against incident outcomes.
6. **Fire/smoke** needs a model trained for those classes; the event logic is ready but no detector here emits them.
7. **Single worker, SQLite**: one processing job at a time; no Postgres/Redis/Qdrant, no RTSP, no Grafana dashboards.
8. **Browser coverage**: UI logic is tested in jsdom and the layout was checked in headless Chromium (software rendering, Chrome 153). Map tiles (internet), H.264 `<video>` playback and other browsers (Firefox/Safari) were not verified.
   Live frames are base64 JPEGs over WebSocket (~15 fps, 640 px): fine on a LAN/localhost, not designed for many simultaneous viewers or slow links.
9. **Bearer-token auth** stored in `sessionStorage`; put the app behind HTTPS in any real deployment. Login throttling is in-memory.
