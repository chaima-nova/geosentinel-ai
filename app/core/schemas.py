"""Pydantic v2 data contracts shared by every GeoSentinel-AI agent.

These models are the wire format between agents, the HTTP API and the
``geo_context_embeddings`` table, so they are deliberately strict:

* ``extra="forbid"`` — a misspelled or unexpected field is a hard error rather
  than silently dropped data. This is what makes agent-to-agent handoffs safe
  to trust. Relax it to ``"ignore"`` if an upstream producer becomes chatty.
* Values are validated against the physical ranges the underlying satellite
  products actually use, which catches unit mistakes (a Kelvin land-surface
  temperature, a 0-1 cloud fraction instead of a percentage) at the boundary.
* Secrets never appear here; credentials live in :mod:`app.core.config`.

Example::

    from app.core.schemas import SpatialQueryRequest

    req = SpatialQueryRequest(
        query="drought stress in wheat belts",
        bbox=[2.5, 31.0, 3.5, 32.0],
        start_date="2024-06-01",
        end_date="2024-06-30",
    )
    print(req.bbox_wkt)  # POLYGON ((2.5 31, 3.5 31, 3.5 32, 2.5 32, 2.5 31))
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, ClassVar, Literal, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

__all__ = [
    "AuditStatus",
    "GeometryType",
    "GridStressLevel",
    "RiskAnalysisResult",
    "SatelliteLayerMetadata",
    "SpatialQueryRequest",
]


class _StrictModel(BaseModel):
    """Base for every GeoSentinel-AI contract: no undeclared fields allowed."""

    model_config: ClassVar[dict[str, Any]] = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------


#: GeoJSON object types accepted as a Feature ``geometry`` value (RFC 7946).
GeometryType = Literal[
    "Point",
    "MultiPoint",
    "LineString",
    "MultiLineString",
    "Polygon",
    "MultiPolygon",
    "GeometryCollection",
]

#: Every accepted geometry type, for error messages.
_GEOMETRY_TYPES: frozenset[str] = frozenset(get_args(GeometryType))


class GridStressLevel(StrEnum):
    """Canonical vocabulary for :attr:`RiskAnalysisResult.grid_stress_level`.

    Members are strings, so they serialise to plain JSON and compare equal to
    their literal value (``GridStressLevel.HIGH == "high"``).
    """

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class AuditStatus(StrEnum):
    """Canonical vocabulary for :attr:`RiskAnalysisResult.audit_status`.

    ``ESCALATED`` marks a result that failed an automatic check and needs a
    human decision, as opposed to ``FAILED``, which is a clean rejection.
    """

    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    ESCALATED = "escalated"


def _normalize_enum(value: Any, enum_cls: type[StrEnum], field_name: str) -> str:
    """Accept any casing/whitespace, return the canonical lowercase value.

    Raises:
        ValueError: If the value is not a member of ``enum_cls``. The message
            lists the accepted vocabulary so an agent can self-correct.
    """
    if isinstance(value, enum_cls):
        return value.value
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string, got {type(value).__name__}")

    candidate = value.strip().lower()
    try:
        return enum_cls(candidate).value
    except ValueError:
        allowed = ", ".join(member.value for member in enum_cls)
        raise ValueError(
            f"{field_name} must be one of [{allowed}], got {value!r}"
        ) from None


def _parse_iso8601(value: Any, field_name: str) -> datetime:
    """Parse an ISO 8601 date or datetime string into a ``datetime``.

    Accepts both date-only (``2024-06-01``, treated as midnight) and full
    datetime forms with or without a timezone offset.

    Raises:
        ValueError: If the value is empty or not ISO 8601.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO 8601 string, got {type(value).__name__}")

    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")

    # Python 3.11's fromisoformat already accepts a trailing "Z"; spelled out
    # here so the intent survives any future change of Python floor.
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(
            f"{field_name} must be an ISO 8601 date or datetime (e.g. '2024-06-01' "
            f"or '2024-06-01T12:30:00Z'), got {value!r}"
        ) from None


def _as_utc(moment: datetime) -> datetime:
    """Interpret a naive datetime as UTC so aware/naive values can compare."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# 1. Request contracts
# ---------------------------------------------------------------------------


class SpatialQueryRequest(_StrictModel):
    """A spatio-temporal semantic search request.

    Combines a natural-language ``query`` with a geographic bounding box and a
    time window, which maps directly onto the hybrid spatial + temporal +
    vector query indexed in ``infra/init.sql``.

    Attributes:
        query: Natural-language description of the climate risk to look for.
        bbox: ``[lon_min, lat_min, lon_max, lat_max]`` in decimal degrees,
            EPSG:4326 (WGS84), longitude first.
        start_date: Inclusive start of the time window, ISO 8601.
        end_date: Inclusive end of the time window, ISO 8601.
    """

    query: str = Field(
        min_length=1,
        max_length=2000,
        description="Natural-language description of the climate risk to search for.",
        examples=["drought stress in irrigated wheat belts"],
    )
    bbox: list[float] = Field(
        description=(
            "Bounding box as [lon_min, lat_min, lon_max, lat_max] in decimal "
            "degrees, EPSG:4326. Longitude first, western edge before eastern edge."
        ),
        examples=[[2.5, 31.0, 3.5, 32.0]],
    )
    start_date: str = Field(
        description="Inclusive start of the time window, ISO 8601 (date or datetime).",
        examples=["2024-06-01"],
    )
    end_date: str = Field(
        description="Inclusive end of the time window, ISO 8601 (date or datetime).",
        examples=["2024-06-30"],
    )

    @field_validator("bbox", mode="after")
    @classmethod
    def _validate_bbox(cls, value: list[float]) -> list[float]:
        """Enforce a well-formed, non-degenerate WGS84 bounding box."""
        if len(value) != 4:
            raise ValueError(
                "bbox must have exactly 4 values "
                f"[lon_min, lat_min, lon_max, lat_max], got {len(value)}"
            )

        lon_min, lat_min, lon_max, lat_max = value
        for label, coord, limit in (
            ("lon_min", lon_min, 180.0),
            ("lon_max", lon_max, 180.0),
        ):
            if not -limit <= coord <= limit:
                raise ValueError(f"bbox {label} must be within [-180, 180], got {coord}")
        for label, coord in (("lat_min", lat_min), ("lat_max", lat_max)):
            if not -90.0 <= coord <= 90.0:
                raise ValueError(f"bbox {label} must be within [-90, 90], got {coord}")

        if lon_min >= lon_max:
            raise ValueError(
                f"bbox lon_min ({lon_min}) must be strictly less than lon_max ({lon_max}); "
                "split antimeridian-crossing queries into two requests"
            )
        if lat_min >= lat_max:
            raise ValueError(
                f"bbox lat_min ({lat_min}) must be strictly less than lat_max ({lat_max})"
            )
        return value

    @field_validator("start_date", "end_date", mode="after")
    @classmethod
    def _validate_iso_dates(cls, value: str, info: ValidationInfo) -> str:
        """Reject date strings that cannot be parsed as ISO 8601."""
        _parse_iso8601(value, info.field_name or "date")
        return value

    @model_validator(mode="after")
    def _validate_time_window(self) -> SpatialQueryRequest:
        """Cross-field check: the window must run forwards in time.

        Declared as a ``model_validator`` rather than ``model_post_init`` so the
        failure surfaces as a ``ValidationError`` (a 422 in FastAPI) instead of
        an unhandled ``ValueError``.
        """
        if self.start_datetime > self.end_datetime:
            raise ValueError(
                f"start_date ({self.start_date!r}) must not be after "
                f"end_date ({self.end_date!r})"
            )
        return self

    @property
    def start_datetime(self) -> datetime:
        """``start_date`` parsed to a timezone-aware ``datetime`` (naive = UTC)."""
        return _as_utc(_parse_iso8601(self.start_date, "start_date"))

    @property
    def end_datetime(self) -> datetime:
        """``end_date`` parsed to a timezone-aware ``datetime`` (naive = UTC)."""
        return _as_utc(_parse_iso8601(self.end_date, "end_date"))

    @property
    def bbox_wkt(self) -> str:
        """The box as a closed WKT ``POLYGON``, ready for PostGIS ``ST_GeomFromText``."""
        lon_min, lat_min, lon_max, lat_max = self.bbox
        return (
            f"POLYGON (({lon_min} {lat_min}, {lon_max} {lat_min}, "
            f"{lon_max} {lat_max}, {lon_min} {lat_max}, {lon_min} {lat_min}))"
        )


# ---------------------------------------------------------------------------
# 2. Ingest contracts
# ---------------------------------------------------------------------------


class SatelliteLayerMetadata(_StrictModel):
    """Derived measurements for one satellite acquisition.

    Produced by the ingest agents from a STAC item (Sentinel-2, Landsat, MODIS)
    and stored alongside the embedding in ``geo_context_embeddings.metadata``.

    Attributes:
        stac_id: Identifier of the source STAC item, e.g. an ESA product name.
        cloud_cover: Scene cloud cover as a percentage of the footprint.
        ndvi_mean: Mean Normalised Difference Vegetation Index over the footprint.
        lst_celsius_max: Peak land-surface temperature over the footprint, °C.
        band_links: Band name to downloadable asset URL (STAC asset hrefs).
    """

    stac_id: str = Field(
        min_length=1,
        description="Identifier of the source STAC item or satellite product.",
        examples=["S2B_MSIL2A_20240615T103029_N0510_R108_T31SET"],
    )
    cloud_cover: float = Field(
        ge=0.0,
        le=100.0,
        description="Cloud cover over the footprint as a percentage, 0-100.",
        examples=[12.4],
    )
    ndvi_mean: float = Field(
        ge=-1.0,
        le=1.0,
        description="Mean NDVI over the footprint, -1 (bare/water) to 1 (dense vegetation).",
        examples=[0.62],
    )
    lst_celsius_max: float = Field(
        ge=-100.0,
        le=100.0,
        description=(
            "Maximum land-surface temperature over the footprint in degrees "
            "Celsius. Bounded to [-100, 100] so a value accidentally left in "
            "Kelvin (~300) is rejected instead of silently stored."
        ),
        examples=[41.8],
    )
    band_links: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Band or asset name mapped to its downloadable URL, e.g. "
            "{'B04': 'https://...', 'B08': 'https://...', 'ndvi': 'https://...'}."
        ),
        examples=[{"B04": "https://example.org/B04.jp2", "B08": "https://example.org/B08.jp2"}],
    )

    @field_validator("band_links", mode="after")
    @classmethod
    def _validate_band_links(cls, value: dict[str, str]) -> dict[str, str]:
        """Require non-empty band names and href-looking values."""
        for band, href in value.items():
            if not band.strip():
                raise ValueError("band_links keys must be non-empty band names")
            if not href.strip():
                raise ValueError(f"band_links[{band!r}] must be a non-empty URL")
        return value


# ---------------------------------------------------------------------------
# 3. Analysis contracts
# ---------------------------------------------------------------------------


class RiskAnalysisResult(_StrictModel):
    """Verdict produced by the risk-analysis agent for one spatial query.

    Attributes:
        heps_score: Composite HEPS score. The scale is owned by the producing
            agent and is intentionally left unconstrained here so the model
            does not silently clamp a rescaled index.
        grid_stress_level: Human-readable stress class for the target grid cell.
        geojson_features: GeoJSON ``FeatureCollection`` (or single ``Feature``)
            of the geometries the verdict applies to.
        audit_status: Outcome of the automatic audit of this result.
    """

    heps_score: float = Field(
        description=(
            "Composite HEPS score. Unconstrained by design: the producing "
            "agent owns the scale, and clamping a rescaled index here would "
            "hide the change rather than surface it."
        ),
        examples=[0.78],
    )
    grid_stress_level: str = Field(
        description=(
            f"Stress class for the grid cell; one of "
            f"{[level.value for level in GridStressLevel]}. Case-insensitive on "
            "input, normalised to lowercase on output."
        ),
        examples=["high"],
    )
    geojson_features: dict[str, Any] = Field(
        description=(
            "GeoJSON (RFC 7946) FeatureCollection or single Feature describing "
            "the geometries this verdict covers."
        ),
    )
    audit_status: str = Field(
        description=(
            f"Audit outcome; one of {[status.value for status in AuditStatus]}. "
            "Case-insensitive on input, normalised to lowercase on output."
        ),
        examples=["passed"],
    )

    @field_validator("grid_stress_level", mode="before")
    @classmethod
    def _normalize_grid_stress_level(cls, value: Any) -> str:
        """Map any casing to the canonical :class:`GridStressLevel` value."""
        return _normalize_enum(value, GridStressLevel, "grid_stress_level")

    @field_validator("audit_status", mode="before")
    @classmethod
    def _normalize_audit_status(cls, value: Any) -> str:
        """Map any casing to the canonical :class:`AuditStatus` value."""
        return _normalize_enum(value, AuditStatus, "audit_status")

    @field_validator("geojson_features", mode="after")
    @classmethod
    def _validate_geojson(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Enforce the minimum shape RFC 7946 requires for features."""
        kind = value.get("type")
        if kind is None:
            raise ValueError("geojson_features must include a 'type' member")
        if not isinstance(kind, str):
            raise ValueError("geojson_features 'type' must be a string")

        if kind == "FeatureCollection":
            features = value.get("features")
            if not isinstance(features, list):
                raise ValueError("a FeatureCollection must have a 'features' list")
            for index, feature in enumerate(features):
                _validate_feature(feature, f"geojson_features.features[{index}]")
        elif kind == "Feature":
            _validate_feature(value, "geojson_features")
        else:
            raise ValueError(
                "geojson_features must be a GeoJSON 'FeatureCollection' or "
                f"'Feature', got {kind!r}"
            )
        return value

    @property
    def is_audited(self) -> bool:
        """True when the automatic audit has reached a terminal verdict."""
        return self.audit_status in (AuditStatus.PASSED, AuditStatus.FAILED)

    @property
    def feature_count(self) -> int:
        """Number of features covered, or 1 for a bare ``Feature``."""
        if self.geojson_features.get("type") == "FeatureCollection":
            return len(self.geojson_features.get("features", []))
        return 1


def _validate_feature(feature: Any, path: str) -> None:
    """Check one GeoJSON ``Feature`` has the members RFC 7946 requires.

    Args:
        feature: Candidate feature object.
        path: Dotted location used in error messages.

    Raises:
        ValueError: If the object is not a well-formed GeoJSON Feature.
    """
    if not isinstance(feature, dict):
        raise ValueError(f"{path} must be a GeoJSON Feature object")
    if feature.get("type") != "Feature":
        raise ValueError(f"{path} must have type 'Feature', got {feature.get('type')!r}")
    if "geometry" not in feature:
        raise ValueError(f"{path} must include a 'geometry' member (null is allowed)")
    if "properties" not in feature:
        raise ValueError(f"{path} must include a 'properties' member ({{}} is allowed)")

    geometry = feature["geometry"]
    if geometry is None:  # RFC 7946 permits a null geometry
        return
    if not isinstance(geometry, dict):
        raise ValueError(f"{path}.geometry must be a GeoJSON geometry object or null")
    geom_type = geometry.get("type")
    if geom_type not in _GEOMETRY_TYPES:
        raise ValueError(
            f"{path}.geometry.type must be one of {sorted(_GEOMETRY_TYPES)}, got {geom_type!r}"
        )
