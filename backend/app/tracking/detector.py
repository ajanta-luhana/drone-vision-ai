"""Object detectors behind one interface.

* BlobDetector  - zero-model, CPU-only colour/contrast detector. Works on the synthetic demo footage and
                  is what makes the pipeline testable on a 3 GB machine with no weights. NOT for real footage.
* YoloDetector  - Ultralytics YOLO (lazy-loaded; install `ultralytics` + torch CPU separately). Use for real
                  aerial/UAV footage (ideally weights fine-tuned on VisDrone).
"""
from dataclasses import dataclass

import cv2
import numpy as np

COCO_MAP = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class Det:
    cls: str
    conf: float
    bbox: tuple  # x1, y1, x2, y2


class Detector:
    name = "base"
    version = "0"

    def detect(self, frame_bgr, conf_threshold: float = 0.35) -> list[Det]:  # pragma: no cover
        raise NotImplementedError


class BlobDetector(Detector):
    """Finds saturated, non-vegetation blobs. Large -> car, small -> person."""
    name, version = "blob-detector", "v1"

    def __init__(self, min_area=100, car_area=350):
        self.min_area, self.car_area = min_area, car_area
        self.kernel = np.ones((3, 3), np.uint8)

    def detect(self, frame_bgr, conf_threshold=0.35):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        sat = s > 110
        not_green = (h < 35) | (h > 85)
        mask = sat & not_green
        # Segment per hue bin so touching objects of different colours are not merged into one blob.
        bins = ((h.astype(np.int16) + 8) % 180) // 12
        cand = []
        for b in np.unique(bins[mask]):
            m = ((mask & (bins == b)).astype(np.uint8)) * 255
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.kernel, iterations=2)
            n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            for i in range(1, n):
                x, y, w, hh, area = stats[i]
                if area < self.min_area:
                    continue
                fill = area / float(w * hh)
                cls = "car" if area >= self.car_area else "person"
                conf = float(min(0.99, 0.6 + 0.4 * fill))
                if conf >= conf_threshold:
                    cand.append((int(area), Det(cls, round(conf, 3), (int(x), int(y), int(x + w), int(y + hh)))))
        # Drop edge-fringe fragments: small blobs touching a larger kept blob (codec colour bleed).
        cand.sort(key=lambda c: -c[0])
        kept, pad = [], 3
        for area, d in cand:
            x1, y1, x2, y2 = d.bbox
            if any(area < 0.25 * ka and x1 < k.bbox[2] + pad and x2 > k.bbox[0] - pad
                   and y1 < k.bbox[3] + pad and y2 > k.bbox[1] - pad for ka, k in kept):
                continue
            kept.append((area, d))
        out = [d for _, d in kept]
        return out


class YoloDetector(Detector):
    name = "yolo"

    def __init__(self, weights="yolov8n.pt", imgsz=960):
        self.weights, self.imgsz, self._model = weights, imgsz, None
        self.version = weights

    def _load(self):
        if self._model is None:  # lazy model loading (memory optimisation)
            try:
                from ultralytics import YOLO
            except ImportError as e:
                raise RuntimeError("DETECTOR=yolo needs `pip install ultralytics` (CPU torch is enough).") from e
            self._model = YOLO(self.weights)
        return self._model

    def detect(self, frame_bgr, conf_threshold=0.35):
        res = self._load().predict(frame_bgr, imgsz=self.imgsz, conf=conf_threshold, device="cpu", verbose=False)[0]
        out = []
        for b in res.boxes:
            c = COCO_MAP.get(int(b.cls))
            if c:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                out.append(Det(c, round(float(b.conf), 3), (x1, y1, x2, y2)))
        return out


def get_detector(name: str, weights: str = "yolov8n.pt") -> Detector:
    if name == "yolo":
        from ..core.config import settings
        return YoloDetector(weights, settings.yolo_imgsz)
    return BlobDetector()
