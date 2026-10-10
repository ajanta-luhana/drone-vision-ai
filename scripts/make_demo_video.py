"""Generate a deterministic synthetic aerial clip that reproduces the spec's §42 scenario:
3 vehicles + 2 people; Vehicle #2 drives into Restricted Zone A, stops, and a person approaches it.

Outputs:
  datasets/demo_aerial.mp4
  simulations/missions/demo_mission.json    (simulated GPS metadata, §7 / §13)
  simulations/zone_restricted_a.geojson     (Restricted Zone A polygon, GeoJSON)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.geospatial.geo import GeoMapper, GPSSimulator  # noqa: E402

W, H, FPS, DUR = 960, 540, 15, 20
MISSION = {"mission_id": "MISSION-001", "video_id": "VIDEO-001", "source": "synthetic_demo",
           "latitude": 24.8607, "longitude": 67.0011, "altitude": 60.0, "heading": 0.0,
           "hfov_deg": 84.0, "seed": 42, "path_mode": "hover"}
ZONE_PX = (560, 90, 860, 250)  # x1,y1,x2,y2 of Restricted Zone A in pixels


def lerp(a, b, u):
    return a + (b - a) * max(0.0, min(1.0, u))


def car2(t):  # drives along the road, turns north into the zone, then stays parked
    if t < 6:
        return (lerp(-30, 640, t / 6), 370)
    if t < 9:
        u = (t - 6) / 3
        return (lerp(640, 700, u), lerp(370, 170, u))
    return (700, 170)


def scene(t):
    objs = []
    objs.append(("car", (200, 80, 30), (lerp(-50, 1010, t / DUR), 372)))                 # car 1 (blue)
    objs.append(("car", (30, 30, 200), car2(t)))                                         # car 2 (red)
    objs.append(("car", (20, 110, 230), (lerp(1010, -50, t / DUR), 398)))                # car 3 (orange)
    if 10 <= t <= DUR:                                                                   # person approaching car 2
        u = (t - 10) / 4
        objs.append(("person", (180, 40, 200), (lerp(780, 735, u), lerp(250, 185, u))))
    if 2 <= t <= 18:                                                                     # bystander walking
        objs.append(("person", (180, 40, 200), (lerp(100, 500, (t - 2) / 16), 300)))
    return objs


def background():
    bg = np.zeros((H, W, 3), np.uint8)
    bg[:] = (60, 140, 60)                                       # grass
    cv2.rectangle(bg, (0, 340), (W, 420), (90, 90, 90), -1)      # road
    for x in range(0, W, 60):
        cv2.line(bg, (x, 380), (x + 30, 380), (200, 200, 200), 2)
    cv2.rectangle(bg, (ZONE_PX[0] - 6, ZONE_PX[1] - 6), (ZONE_PX[2] + 6, ZONE_PX[3] + 6), (150, 150, 150), 1)  # fence
    return bg


def main():
    out_dir = ROOT / "datasets"
    out_dir.mkdir(exist_ok=True)
    (ROOT / "simulations" / "missions").mkdir(parents=True, exist_ok=True)
    path = out_dir / "demo_aerial.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    bg, rng = background(), np.random.default_rng(7)
    for i in range(FPS * DUR):
        t = i / FPS
        f = bg.copy()
        for cls, col, (x, y) in scene(t):
            if cls == "car":
                cv2.rectangle(f, (int(x - 22), int(y - 12)), (int(x + 22), int(y + 12)), col, -1)
            else:
                cv2.circle(f, (int(x), int(y)), 7, col, -1)
        f = np.clip(f.astype(np.int16) + rng.normal(0, 2, f.shape).astype(np.int16), 0, 255).astype(np.uint8)
        w.write(f)
    w.release()

    sim = GPSSimulator(MISSION["latitude"], MISSION["longitude"], MISSION["altitude"], MISSION["heading"],
                       MISSION["path_mode"], MISSION["seed"])
    m, tel = GeoMapper(W, H, MISSION["hfov_deg"]), sim.at(0.0)
    x1, y1, x2, y2 = ZONE_PX
    ring = [m.pixel_to_geo(px, py, tel) for px, py in [(x1, y2), (x2, y2), (x2, y1), (x1, y1), (x1, y2)]]
    zone = {"type": "Feature", "properties": {"name": "Restricted Zone A", "zone_type": "restricted"},
            "geometry": {"type": "Polygon", "coordinates": [[[lon, lat] for lat, lon in ring]]}}
    (ROOT / "simulations" / "zone_restricted_a.geojson").write_text(json.dumps(zone, indent=2))
    (ROOT / "simulations" / "missions" / "demo_mission.json").write_text(json.dumps(MISSION, indent=2))
    print("wrote", path, "and simulation metadata")


if __name__ == "__main__":
    main()
