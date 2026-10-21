"""Study-area boundary loading, validation, containment and grid masking.

No boundary is bundled with the repository. ``load_boundary`` requires a real file,
and the study-area registry records the official source for it. When the file is
absent the loader raises ``BoundaryUnavailableError``. It never substitutes the
bounding-box envelope, which is only a catalogue-search extent.

GeoJSON (RFC 7946) coordinates are WGS84 longitude/latitude. A ``crs`` member that
names anything other than CRS84 or EPSG:4326 is refused rather than reprojected.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "BoundaryError",
    "BoundaryUnavailableError",
    "LoadedBoundary",
    "bbox_contains_boundary",
    "load_boundary",
    "mask_for_cells",
]


class BoundaryError(ValueError):
    """The boundary file exists but is not a usable polygon."""


class BoundaryUnavailableError(FileNotFoundError):
    """No official boundary file has been obtained for this area."""


@dataclass(frozen=True)
class LoadedBoundary:
    """A validated boundary polygon with its provenance.

    Attributes:
        geometry: Shapely (Multi)Polygon in WGS84 longitude/latitude.
        bounds: ``(lon_min, lat_min, lon_max, lat_max)`` of the geometry.
        sha256: SHA-256 of the file as read.
        area_deg2: Planar area in square degrees (for orientation only; not km²).
    """

    geometry: Any
    bounds: tuple[float, float, float, float]
    sha256: str
    area_deg2: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_polygons(obj: dict[str, Any]) -> list[dict[str, Any]]:
    kind = obj.get("type")
    if kind == "FeatureCollection":
        features = obj.get("features") or []
        return [
            f["geometry"] for f in features if isinstance(f, dict) and f.get("geometry")
        ]
    if kind == "Feature":
        return [obj["geometry"]] if obj.get("geometry") else []
    if kind in ("Polygon", "MultiPolygon"):
        return [obj]
    raise BoundaryError(f"unsupported GeoJSON type '{kind}'")


def load_boundary(path: Path, *, expected_sha256: str | None = None) -> LoadedBoundary:
    """Load and validate a boundary GeoJSON file.

    Raises ``BoundaryUnavailableError`` when ``path`` does not exist, and
    ``BoundaryError`` for every validation failure. Failures are never silent.
    """
    from shapely.geometry import shape
    from shapely.ops import unary_union

    if not path.is_file():
        raise BoundaryUnavailableError(f"no boundary file at {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BoundaryError(f"boundary is not valid GeoJSON: {exc}") from exc
    if not isinstance(document, dict):
        raise BoundaryError("boundary GeoJSON root must be an object")

    crs = document.get("crs")
    if crs is not None:
        name = (
            (crs.get("properties") or {}).get("name", "")
            if isinstance(crs, dict)
            else ""
        )
        if name not in (
            "urn:ogc:def:crs:OGC:1.3:CRS84",
            "EPSG:4326",
            "urn:ogc:def:crs:EPSG::4326",
        ):
            raise BoundaryError(
                f"boundary declares CRS '{name}'; only WGS84 is accepted (no reprojection)"
            )

    geometries = _collect_polygons(document)
    if not geometries:
        raise BoundaryError("boundary contains no geometry")
    parts = []
    for raw in geometries:
        if raw.get("type") not in ("Polygon", "MultiPolygon"):
            raise BoundaryError(
                f"boundary geometry must be a polygon, got {raw.get('type')}"
            )
        parts.append(shape(raw))
    geometry = unary_union(parts)

    if geometry.is_empty:
        raise BoundaryError("boundary geometry is empty")
    if not geometry.is_valid:
        from shapely.validation import explain_validity

        raise BoundaryError(
            f"boundary geometry is invalid: {explain_validity(geometry)}"
        )
    lon_min, lat_min, lon_max, lat_max = geometry.bounds
    if not (
        -180.0 <= lon_min and lon_max <= 180.0 and -90.0 <= lat_min and lat_max <= 90.0
    ):
        raise BoundaryError(
            "boundary coordinates lie outside WGS84 longitude/latitude range"
        )
    if geometry.area <= 0.0:
        raise BoundaryError("boundary has zero area")

    digest = _sha256(path)
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        raise BoundaryError(
            f"boundary sha256 mismatch: expected {expected_sha256}, got {digest}"
        )
    return LoadedBoundary(
        geometry=geometry,
        bounds=(lon_min, lat_min, lon_max, lat_max),
        sha256=digest,
        area_deg2=float(geometry.area),
    )


def bbox_contains_boundary(bbox: list[float], boundary: LoadedBoundary) -> bool:
    """True when the catalogue bbox fully contains the boundary.

    A bbox that does not contain the boundary would miss part of the study area in
    catalogue searches, so the caller must treat ``False`` as a configuration error.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    b_lon_min, b_lat_min, b_lon_max, b_lat_max = boundary.bounds
    return (
        lon_min <= b_lon_min
        and lat_min <= b_lat_min
        and lon_max >= b_lon_max
        and lat_max >= b_lat_max
    )


def mask_for_cells(
    boundary: LoadedBoundary, lons: np.ndarray, lats: np.ndarray
) -> np.ndarray:
    """Boolean mask of cell centres that fall inside the boundary (boundary-inclusive)."""
    import shapely
    from shapely import prepare

    if lons.shape != lats.shape:
        raise ValueError("lons and lats must have the same shape")
    prepare(boundary.geometry)
    flat = shapely.points(lons.ravel(), lats.ravel())
    inside = shapely.intersects(boundary.geometry, flat)
    return np.asarray(inside, dtype=bool).reshape(lons.shape)
