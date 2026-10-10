"""Lightweight multi-object tracker (ByteTrack-inspired, pure NumPy/Python).

Two-stage association: high-confidence detections first, low-confidence ones second to rescue
tracks through brief occlusions. Constant-velocity prediction + greedy IoU matching, class-consistent.
Persistent integer IDs are never reused within a run.
"""
from collections import Counter
from dataclasses import dataclass, field

# Detectors (esp. YOLO from above) flicker between similar classes; treat them as one group for association.
GROUPS = {"car": "vehicle", "truck": "vehicle", "bus": "vehicle", "van": "vehicle", "construction_equipment": "vehicle",
          "motorcycle": "twowheel", "bicycle": "twowheel", "person": "person"}


def group_of(cls: str) -> str:
    return GROUPS.get(cls, cls)


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


@dataclass
class TrackState:
    id: int
    cls: str
    bbox: tuple
    conf: float
    hits: int = 1
    misses: int = 0
    vel: tuple = (0.0, 0.0)
    confirmed: bool = False
    history: list = field(default_factory=list)
    votes: Counter = field(default_factory=Counter)

    def predicted(self):
        dx, dy = self.vel
        x1, y1, x2, y2 = self.bbox
        k = self.misses + 1
        return (x1 + dx * k, y1 + dy * k, x2 + dx * k, y2 + dy * k)


class ByteLikeTracker:
    def __init__(self, high_thresh=0.5, iou_thresh=0.15, max_age=10, min_hits=2):
        self.high, self.iou_thresh, self.max_age, self.min_hits = high_thresh, iou_thresh, max_age, min_hits
        self.tracks: list[TrackState] = []
        self._next_id = 1

    def _match(self, tracks, dets):
        pairs = []
        for ti, t in enumerate(tracks):
            pb = t.predicted()
            for di, d in enumerate(dets):
                if group_of(d.cls) != group_of(t.cls):
                    continue
                score = iou(pb, d.bbox)
                if score == 0:  # centroid fallback for fast/small objects
                    (cx1, cy1), (cx2, cy2) = _center(pb), _center(d.bbox)
                    dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                    size = max(pb[2] - pb[0], pb[3] - pb[1], 1)
                    if dist < size * 0.8:
                        score = 0.05 + 0.1 * (1 - dist / (size * 0.8))
                if score >= min(self.iou_thresh, 0.05):
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)
        mt, md, matches = set(), set(), []
        for score, ti, di in pairs:
            if ti in mt or di in md:
                continue
            mt.add(ti), md.add(di), matches.append((ti, di))
        return matches, [i for i in range(len(tracks)) if i not in mt], [i for i in range(len(dets)) if i not in md]

    def _apply(self, t: TrackState, d):
        c_old, c_new = _center(t.bbox), _center(d.bbox)
        steps = t.misses + 1
        t.vel = ((c_new[0] - c_old[0]) / steps, (c_new[1] - c_old[1]) / steps)
        t.bbox, t.conf = d.bbox, d.conf
        t.votes[d.cls] += 1
        t.cls = t.votes.most_common(1)[0][0]      # stable, majority-voted class
        t.hits += 1
        t.misses = 0
        if t.hits >= self.min_hits:
            t.confirmed = True

    def update(self, dets, frame_idx=0):
        """Returns list of (TrackState, Det) for confirmed tracks matched this frame."""
        high = [d for d in dets if d.conf >= self.high]
        low = [d for d in dets if d.conf < self.high]
        matched_pairs = []

        m1, ut, ud = self._match(self.tracks, high)
        for ti, di in m1:
            self._apply(self.tracks[ti], high[di]); matched_pairs.append((self.tracks[ti], high[di]))
        remaining = [self.tracks[i] for i in ut]
        m2, ut2, _ = self._match(remaining, low)
        for ti, di in m2:
            self._apply(remaining[ti], low[di]); matched_pairs.append((remaining[ti], low[di]))
        unmatched_tracks = [remaining[i] for i in ut2]
        for t in unmatched_tracks:
            t.misses += 1
        for di in ud:  # new tracks only from high-confidence leftovers
            d = high[di]
            self.tracks.append(TrackState(self._next_id, d.cls, d.bbox, d.conf, votes=Counter({d.cls: 1})))
            matched_pairs.append((self.tracks[-1], d))
            self._next_id += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
        return [(t, d) for t, d in matched_pairs if t.confirmed or t.hits >= self.min_hits]
