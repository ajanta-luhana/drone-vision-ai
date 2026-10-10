"""Geospatial primitives (pure Python, no heavy deps).

Coordinates are (lon, lat) inside GeoJSON and (lat, lon) in function args, as noted per function.
"""
import math
import random
from dataclasses import dataclass

EARTH_R = 6371008.8  # metres
M_PER_DEG_LAT = 111_320.0


# ---------- distance / polygon ----------
def haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def _ring_contains(ring, lon, lat) -> bool:
    """Ray-casting point-in-ring. ring = [[lon, lat], ...]."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > lat) != (yj > lat):
            x_cross = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_cross:
                inside = not inside
        j = i
    return inside


def point_in_polygon(lat: float, lon: float, geometry: dict) -> bool:
    """GeoJSON Polygon / MultiPolygon (outer ring minus holes)."""
    gtype = geometry.get("type")
    if gtype == "Polygon":
        polys = [geometry["coordinates"]]
    elif gtype == "MultiPolygon":
        polys = geometry["coordinates"]
    else:
        raise ValueError(f"Unsupported geometry type: {gtype}")
    for rings in polys:
        if _ring_contains(rings[0], lon, lat) and not any(_ring_contains(h, lon, lat) for h in rings[1:]):
            return True
    return False


def extract_geometry(geojson: dict) -> dict:
    """Accept Feature / FeatureCollection(first polygon) / bare geometry; return a validated geometry."""
    if not isinstance(geojson, dict):
        raise ValueError("GeoJSON must be an object")
    t = geojson.get("type")
    if t == "FeatureCollection":
        feats = geojson.get("features") or []
        if not feats:
            raise ValueError("FeatureCollection is empty")
        geojson = feats[0]
        t = geojson.get("type")
    if t == "Feature":
        geojson = geojson.get("geometry") or {}
        t = geojson.get("type")
    if t not in ("Polygon", "MultiPolygon"):
        raise ValueError("Zone geometry must be a Polygon or MultiPolygon")
    coords = geojson.get("coordinates")
    rings = coords[0] if t == "Polygon" else coords[0][0]
    if not coords or not rings or len(rings) < 4:
        raise ValueError("Polygon ring needs at least 4 coordinates (closed ring)")
    for pt in rings:
        lon, lat = pt[0], pt[1]
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("Coordinate out of range")
    return geojson


def polygon_centroid(geometry: dict):
    ring = geometry["coordinates"][0] if geometry["type"] == "Polygon" else geometry["coordinates"][0][0]
    pts = ring[:-1] if ring[0] == ring[-1] else ring
    return sum(p[1] for p in pts) / len(pts), sum(p[0] for p in pts) / len(pts)  # lat, lon


def make_box(center_lat, center_lon, width_m, height_m, east_m=0.0, north_m=0.0) -> dict:
    """Helper that builds a rectangular GeoJSON Polygon offset from a centre point."""
    clat = center_lat + north_m / M_PER_DEG_LAT
    clon = center_lon + east_m / (M_PER_DEG_LAT * math.cos(math.radians(center_lat)))
    dlat = (height_m / 2) / M_PER_DEG_LAT
    dlon = (width_m / 2) / (M_PER_DEG_LAT * math.cos(math.radians(center_lat)))
    ring = [[clon - dlon, clat - dlat], [clon + dlon, clat - dlat], [clon + dlon, clat + dlat],
            [clon - dlon, clat + dlat], [clon - dlon, clat - dlat]]
    return {"type": "Polygon", "coordinates": [ring]}


def offset_latlon(lat, lon, east_m, north_m):
    return (lat + north_m / M_PER_DEG_LAT,
            lon + east_m / (M_PER_DEG_LAT * math.cos(math.radians(lat))))


def local_xy_m(lat, lon, lat0, lon0):
    """Equirectangular local (east, north) metres relative to an origin."""
    return ((lon - lon0) * M_PER_DEG_LAT * math.cos(math.radians(lat0)), (lat - lat0) * M_PER_DEG_LAT)


# ---------- simulated GPS telemetry ----------
@dataclass
class Telemetry:
    lat: float
    lon: float
    altitude: float
    heading: float
    t: float  # seconds from video start


class GPSSimulator:
    """Deterministic telemetry: identical (seed, mode, t) -> identical output, independent of
    which frames the pipeline samples."""

    def __init__(self, lat, lon, altitude=60.0, heading=0.0, mode="hover", seed=42, speed_mps=3.0, noise_m=0.15):
        self.lat0, self.lon0, self.alt0, self.heading0 = lat, lon, altitude, heading
        self.mode, self.seed, self.speed, self.noise = mode, seed, speed_mps, noise_m

    def at(self, t: float) -> Telemetry:
        # noise is a function of the time bucket (1/10 s) so it is reproducible
        rng = random.Random(self.seed * 1_000_003 + int(round(t * 10)))
        east = rng.uniform(-self.noise, self.noise)
        north = rng.uniform(-self.noise, self.noise)
        heading = self.heading0
        if self.mode == "linear":
            rad = math.radians(self.heading0)
            east += math.sin(rad) * self.speed * t
            north += math.cos(rad) * self.speed * t
        lat, lon = offset_latlon(self.lat0, self.lon0, east, north)
        alt = self.alt0 + math.sin(t / 3.0) * 0.2
        return Telemetry(lat, lon, alt, heading, t)

    def stream(self, duration_s: float, hz: float = 1.0):
        n = int(duration_s * hz) + 1
        return [self.at(i / hz) for i in range(n)]


# ---------- pixel <-> geographic mapping (nadir camera) ----------
class GeoMapper:
    def __init__(self, frame_w: int, frame_h: int, hfov_deg: float = 84.0):
        self.w, self.h = frame_w, frame_h
        self.hfov = math.radians(hfov_deg)

    def metres_per_pixel(self, altitude: float) -> float:
        return 2 * altitude * math.tan(self.hfov / 2) / self.w

    def pixel_to_geo(self, px, py, tel: Telemetry):
        mpp = self.metres_per_pixel(tel.altitude)
        dx, dy = (px - self.w / 2) * mpp, (self.h / 2 - py) * mpp  # image-right, image-up
        th = math.radians(tel.heading)
        east = dx * math.cos(th) + dy * math.sin(th)
        north = -dx * math.sin(th) + dy * math.cos(th)
        return offset_latlon(tel.lat, tel.lon, east, north)

    def geo_to_pixel(self, lat, lon, tel: Telemetry):
        east, north = local_xy_m(lat, lon, tel.lat, tel.lon)
        th = math.radians(tel.heading)
        dx = east * math.cos(th) - north * math.sin(th)
        dy = east * math.sin(th) + north * math.cos(th)
        mpp = self.metres_per_pixel(tel.altitude)
        return self.w / 2 + dx / mpp, self.h / 2 - dy / mpp
