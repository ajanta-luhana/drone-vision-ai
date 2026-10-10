from dataclasses import dataclass, field

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "construction_equipment"}
PERSON_CLASSES = {"person"}


@dataclass
class Point:
    t: float          # seconds from video start
    frame: int
    lat: float
    lon: float
    px: float         # centre in processed-frame pixels
    py: float
    conf: float = 1.0
    bbox: tuple = (0, 0, 0, 0)


@dataclass
class TrackData:
    id: int
    cls: str
    points: list = field(default_factory=list)

    @property
    def first_seen(self):
        return self.points[0].t

    @property
    def last_seen(self):
        return self.points[-1].t


@dataclass
class ZoneDef:
    id: int
    name: str
    type: str
    geometry: dict
    forbidden_classes: list = field(default_factory=list)
    max_dwell_s: float | None = None


@dataclass
class EventDef:
    event_type: str
    track_id: int
    start_t: float
    end_t: float
    lat: float
    lon: float
    zone_id: int | None = None
    related_track_id: int | None = None
    details: dict = field(default_factory=dict)
