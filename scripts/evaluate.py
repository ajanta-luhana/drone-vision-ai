"""Measure detector/tracker quality on the synthetic demo clip, where ground truth is known (§30 MLOps metrics).
Reports detection precision / recall (IoU>=0.5), ID switches, track continuity and processing FPS.
For real footage, supply your own annotations and reuse `evaluate()`."""
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))
from app.tracking.detector import BlobDetector  # noqa: E402
from app.tracking.tracker import ByteLikeTracker, iou  # noqa: E402
import make_demo_video as g  # noqa: E402


def gt_boxes(t):
    out = []
    for k, (cls, _, (x, y)) in enumerate(g.scene(t)):
        half = (22, 12) if cls == "car" else (7, 7)
        out.append((f"{cls}{k}", cls, (x - half[0], y - half[1], x + half[0], y + half[1])))
    return out


def evaluate(process_fps=8):
    cap = cv2.VideoCapture(str(ROOT / "datasets" / "demo_aerial.mp4"))
    step = round(g.FPS / process_fps)
    det, tr = BlobDetector(), ByteLikeTracker()
    tp = fp = fn = switches = frames = 0
    last_assign, track_hits, t0, i = {}, {}, time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames += 1
            gts = [b for b in gt_boxes(i / g.FPS) if -10 < (b[2][0] + b[2][2]) / 2 < g.W + 10]  # visible only
            matched = tr.update(det.detect(frame), i)
            used = set()
            for gid, cls, box in gts:
                best = max(((iou(box, d.bbox), T, d) for T, d in matched if id(d) not in used), default=(0, None, None), key=lambda x: x[0])
                if best[0] >= 0.5:
                    tp += 1; used.add(id(best[2]))
                    if gid in last_assign and last_assign[gid] != best[1].id:
                        switches += 1
                    last_assign[gid] = best[1].id
                    track_hits[gid] = track_hits.get(gid, 0) + 1
                else:
                    fn += 1
            fp += len(matched) - len(used)
        i += 1
    dt = time.time() - t0
    return {"precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4),
            "id_switches": switches, "gt_objects": len(last_assign), "tracks_used": len(set(last_assign.values())),
            "frames": frames, "processing_fps": round(frames / dt, 1), "process_fps_setting": process_fps}


if __name__ == "__main__":
    for fps in (5, 8, 10):
        print(evaluate(fps))
