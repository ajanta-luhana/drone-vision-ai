import math
from app.geospatial.geo import (GeoMapper, GPSSimulator, extract_geometry, haversine_m, make_box, point_in_polygon)
import pytest

BOX = make_box(24.8607, 67.0011, 40, 30)


def test_point_in_polygon_inside_outside_edge():
    assert point_in_polygon(24.8607, 67.0011, BOX)
    assert not point_in_polygon(24.8700, 67.0011, BOX)
    assert not point_in_polygon(24.8607, 67.0100, BOX)


def test_polygon_with_hole():
    outer = BOX["coordinates"][0]
    hole = make_box(24.8607, 67.0011, 10, 10)["coordinates"][0]
    poly = {"type": "Polygon", "coordinates": [outer, hole]}
    assert not point_in_polygon(24.8607, 67.0011, poly)           # in the hole
    assert point_in_polygon(24.86082, 67.0011, poly)              # ~8 m north: inside outer ring, outside hole


def test_haversine_known_distance():
    # 0.001 deg latitude ~ 111.2 m
    assert haversine_m(24.0, 67.0, 24.001, 67.0) == pytest.approx(111.2, abs=0.5)
    assert haversine_m(1, 1, 1, 1) == 0


@pytest.mark.parametrize("bad", [{"type": "Point", "coordinates": [0, 0]}, {"type": "Polygon", "coordinates": [[[0, 0], [1, 1]]]},
                                 {"type": "Polygon", "coordinates": [[[200, 0], [1, 1], [2, 2], [200, 0]]]}, "x", {"type": "FeatureCollection", "features": []}])
def test_invalid_geojson_rejected(bad):
    with pytest.raises(ValueError):
        extract_geometry(bad)


def test_feature_wrapping_accepted():
    g = extract_geometry({"type": "Feature", "geometry": BOX})
    assert g["type"] == "Polygon"


def test_simulator_is_deterministic_and_sampling_independent():
    a = GPSSimulator(24.86, 67.0, 60, 10, "linear", seed=1)
    b = GPSSimulator(24.86, 67.0, 60, 10, "linear", seed=1)
    assert a.at(3.3) == b.at(3.3)
    assert a.at(3.3) != GPSSimulator(24.86, 67.0, 60, 10, "linear", seed=2).at(3.3)
    assert a.stream(5) == b.stream(5)


@pytest.mark.parametrize("heading", [0, 37, 90, 180, 271])
def test_pixel_geo_roundtrip(heading):
    sim = GPSSimulator(24.8607, 67.0011, 60, heading)
    tel, m = sim.at(1.0), GeoMapper(960, 540)
    for px, py in [(480, 270), (100, 50), (900, 500)]:
        lat, lon = m.pixel_to_geo(px, py, tel)
        x, y = m.geo_to_pixel(lat, lon, tel)
        assert x == pytest.approx(px, abs=0.05) and y == pytest.approx(py, abs=0.05)


def test_ground_scale_and_orientation():
    tel = GPSSimulator(24.8607, 67.0011, 60, 0, seed=1, noise_m=0).at(0)
    m = GeoMapper(960, 540, 84)
    assert m.metres_per_pixel(60) == pytest.approx(0.1125, abs=0.001)
    lat_up, _ = m.pixel_to_geo(480, 170, tel)       # 100 px up => north
    assert lat_up > tel.lat
    _, lon_r = m.pixel_to_geo(580, 270, tel)        # right => east
    assert lon_r > tel.lon
    lat_e, lon_e = m.pixel_to_geo(480, 170, GPSSimulator(24.8607, 67.0011, 60, 90, noise_m=0).at(0))
    assert lon_e > 67.0011 and abs(lat_e - 24.8607) < 1e-6      # heading east: image-up points east
