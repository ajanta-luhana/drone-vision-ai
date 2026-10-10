# API (`/api/v1`, Bearer JWT; interactive docs at `/docs`)
Auth `POST /auth/login` `GET /auth/me` · Users `GET|POST /users` (admin) · `GET /audit` (admin) · `GET /models` (admin)
Missions `POST|GET /missions` `GET /missions/{id}` `GET /missions/{id}/geojson`
Videos `POST /videos/upload` `POST /videos/{id}/process` `GET /videos/{id}` `GET /videos/{id}/stream`
Data `GET /detections` `GET /tracks` · Zones `POST|GET /zones` `DELETE /zones/{id}` (admin)
Events `GET /events` `GET /events/{id}` · Incidents `GET /incidents` `GET /incidents/{id}` `GET /incidents/{id}/frame` `POST /incidents/{id}/review`
Reports `POST|GET /reports` `GET /reports/{id}` `GET /reports/{id}/download?format=pdf|json|csv`
Assistant `POST /assistant/query` · Analytics `GET /analytics/overview` · System `GET /system/health` `GET /system/metrics`

| Role | Can |
|---|---|
| admin | everything (users, zones, audit, models) |
| operator | missions, upload/process video, review incidents, reports, assistant |
| analyst | read everything, reports, assistant |
| viewer | read-only dashboard, map, incidents, reports |
