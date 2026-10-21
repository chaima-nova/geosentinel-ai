"""Boundary loading, validation, bbox containment and masking.

Every polygon here is a SYNTHETIC test shape. None is the London boundary, and no
test asserts London geometry. The official boundary has not been obtained.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from app.services.boundary_service import (
    BoundaryError,
    BoundaryUnavailableError,
    bbox_contains_boundary,
    load_boundary,
    mask_for_cells,
)

REPO = Path(__file__).resolve().parents[1]
STUDY_AREAS = json.loads(
    (REPO / "config" / "study_areas.json").read_text(encoding="utf-8")
)
LONDON = next(a for a in STUDY_AREAS["areas"] if a["id"] == "london")


def _write(tmp_path: Path, document: object, name: str = "synthetic.geojson") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _square(lon0: float, lat0: float, lon1: float, lat1: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [
            [[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]]
        ],
    }


def _feature_collection(*geometries: dict) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {}, "geometry": g} for g in geometries
        ],
    }


# --- Official boundary status is recorded honestly ----------------------------


def test_london_boundary_is_recorded_as_not_obtained_with_no_file() -> None:
    boundary = LONDON["boundary"]
    assert boundary["status"] == "not_obtained"
    assert (
        boundary["file"] is None
        and boundary["sha256"] is None
        and boundary["source"] is None
    )


def test_missing_official_boundary_raises_instead_of_using_the_envelope(
    tmp_path: Path,
) -> None:
    with pytest.raises(BoundaryUnavailableError):
        load_boundary(tmp_path / "greater_london.geojson")


@pytest.mark.parametrize("area_id", ["amsterdam", "rotterdam"])
def test_other_cities_have_no_boundary_either(area_id: str) -> None:
    area = next(a for a in STUDY_AREAS["areas"] if a["id"] == area_id)
    assert area["boundary"]["status"] == "not_obtained"


# --- Geometry and CRS validation ----------------------------------------------


def test_valid_polygon_loads_with_bounds_and_sha256(tmp_path: Path) -> None:
    path = _write(tmp_path, _feature_collection(_square(-0.5, 51.3, 0.3, 51.6)))
    loaded = load_boundary(path)
    assert loaded.bounds == pytest.approx((-0.5, 51.3, 0.3, 51.6))
    assert loaded.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert loaded.area_deg2 == pytest.approx(0.8 * 0.3)


def test_multipolygon_is_accepted(tmp_path: Path) -> None:
    island = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
            [[[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0], [2.0, 0.0]]],
        ],
    }
    loaded = load_boundary(_write(tmp_path, _feature_collection(island)))
    assert loaded.bounds == pytest.approx((0.0, 0.0, 3.0, 1.0))


def test_self_intersecting_polygon_is_rejected(tmp_path: Path) -> None:
    bowtie = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]],
    }
    with pytest.raises(BoundaryError, match="invalid"):
        load_boundary(_write(tmp_path, _feature_collection(bowtie)))


def test_non_wgs84_crs_is_refused_not_reprojected(tmp_path: Path) -> None:
    document = _feature_collection(_square(0, 0, 1, 1))
    document["crs"] = {
        "type": "name",
        "properties": {"name": "urn:ogc:def:crs:EPSG::27700"},
    }
    with pytest.raises(BoundaryError, match="only WGS84"):
        load_boundary(_write(tmp_path, document))


def test_out_of_range_coordinates_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(BoundaryError, match="outside WGS84"):
        load_boundary(_write(tmp_path, _feature_collection(_square(170, 0, 200, 1))))


def test_point_geometry_is_rejected(tmp_path: Path) -> None:
    point = {"type": "Point", "coordinates": [0.0, 0.0]}
    with pytest.raises(BoundaryError, match="must be a polygon"):
        load_boundary(_write(tmp_path, _feature_collection(point)))


def test_empty_document_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(BoundaryError, match="no geometry"):
        load_boundary(_write(tmp_path, {"type": "FeatureCollection", "features": []}))


def test_invalid_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.geojson"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(BoundaryError, match="not valid GeoJSON"):
        load_boundary(path)


def test_checksum_mismatch_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, _feature_collection(_square(0, 0, 1, 1)))
    with pytest.raises(BoundaryError, match="sha256 mismatch"):
        load_boundary(path, expected_sha256="0" * 64)


def test_matching_checksum_is_accepted(tmp_path: Path) -> None:
    path = _write(tmp_path, _feature_collection(_square(0, 0, 1, 1)))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_boundary(path, expected_sha256=digest).sha256 == digest


# --- Relationship between the catalogue bbox and the boundary -------------------


def test_bbox_containing_the_boundary_is_accepted(tmp_path: Path) -> None:
    loaded = load_boundary(
        _write(tmp_path, _feature_collection(_square(-0.4, 51.4, 0.2, 51.6)))
    )
    assert bbox_contains_boundary([-0.5, 51.3, 0.3, 51.7], loaded)


def test_bbox_that_clips_the_boundary_is_refused(tmp_path: Path) -> None:
    loaded = load_boundary(
        _write(tmp_path, _feature_collection(_square(-0.4, 51.4, 0.2, 51.6)))
    )
    # East edge of the bbox (0.1) is inside the boundary's east edge (0.2): the search would miss area.
    assert not bbox_contains_boundary([-0.5, 51.3, 0.1, 51.7], loaded)


# --- Masking --------------------------------------------------------------------


def test_mask_covers_the_expected_share_of_a_grid(tmp_path: Path) -> None:
    # A synthetic square covering the western half of a unit grid.
    loaded = load_boundary(
        _write(tmp_path, _feature_collection(_square(0.0, 0.0, 0.5, 1.0)))
    )
    xs, ys = np.meshgrid(np.linspace(0.005, 0.995, 100), np.linspace(0.005, 0.995, 100))
    mask = mask_for_cells(loaded, xs, ys)
    assert mask.shape == xs.shape
    assert mask.mean() == pytest.approx(0.5, abs=0.01)
    assert mask[:, :49].all() and not mask[:, 51:].any()


def test_mask_rejects_mismatched_grids(tmp_path: Path) -> None:
    loaded = load_boundary(_write(tmp_path, _feature_collection(_square(0, 0, 1, 1))))
    with pytest.raises(ValueError, match="same shape"):
        mask_for_cells(loaded, np.zeros((2, 2)), np.zeros((3, 2)))
