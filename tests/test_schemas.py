"""Tests for :mod:`app.core.schemas`.

Ported from the throwaway verification script used to build the module, so the
contracts are pinned permanently rather than only at authoring time.
"""

from __future__ import annotations

import typing
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError
from shapely import wkt

from app.core.schemas import (
    AuditStatus,
    GeometryType,
    GridStressLevel,
    RiskAnalysisResult,
    SatelliteLayerMetadata,
    SpatialQueryRequest,
)


#: Sentinel so ``make_feature(geometry=None)`` can express a real GeoJSON null
#: geometry instead of falling through to the default.
_UNSET = object()


def make_feature(geometry: dict | None = _UNSET) -> dict:  # type: ignore[assignment]
    """A minimal valid GeoJSON Feature.

    Args:
        geometry: Geometry to use. Omit for a default Point; pass ``None`` for
            an explicit GeoJSON null geometry.
    """
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [0, 0]} if geometry is _UNSET else geometry,
        "properties": {},
    }


def make_collection(features: list[dict] | None = None) -> dict:
    """A minimal valid GeoJSON FeatureCollection."""
    return {"type": "FeatureCollection", "features": features if features is not None else [make_feature()]}


def make_request(**overrides) -> SpatialQueryRequest:
    """A valid SpatialQueryRequest with individual fields overridable."""
    payload = {
        "query": "drought stress in wheat belts",
        "bbox": [2.5, 31.0, 3.5, 32.0],
        "start_date": "2024-06-01",
        "end_date": "2024-06-30",
    }
    payload.update(overrides)
    return SpatialQueryRequest(**payload)


def make_layer(**overrides) -> SatelliteLayerMetadata:
    """A valid SatelliteLayerMetadata with individual fields overridable."""
    payload = {
        "stac_id": "S2B_MSIL2A_20240615T103029_N0510_R108_T31SET",
        "cloud_cover": 12.4,
        "ndvi_mean": 0.62,
        "lst_celsius_max": 41.8,
        "band_links": {"B04": "https://example.org/B04.jp2"},
    }
    payload.update(overrides)
    return SatelliteLayerMetadata(**payload)


def make_result(**overrides) -> RiskAnalysisResult:
    """A valid RiskAnalysisResult with individual fields overridable."""
    payload = {
        "heps_score": 0.78,
        "grid_stress_level": "high",
        "geojson_features": make_collection(),
        "audit_status": "passed",
    }
    payload.update(overrides)
    return RiskAnalysisResult(**payload)


# ---------------------------------------------------------------------------
# SpatialQueryRequest
# ---------------------------------------------------------------------------


class TestSpatialQueryRequestTypes:
    """The field types stay exactly as specified."""

    def test_bbox_is_a_list_of_floats(self) -> None:
        request = make_request()
        assert isinstance(request.bbox, list)
        assert all(isinstance(value, float) for value in request.bbox)

    def test_dates_stay_strings(self) -> None:
        request = make_request()
        assert isinstance(request.start_date, str)
        assert isinstance(request.end_date, str)

    def test_integers_are_coerced_to_float(self) -> None:
        assert all(isinstance(v, float) for v in make_request(bbox=[0, 0, 1, 1]).bbox)


class TestSpatialQueryRequestHelpers:
    """The derived properties."""

    def test_start_datetime_is_utc_midnight_for_date_only(self) -> None:
        assert make_request().start_datetime == datetime(2024, 6, 1, tzinfo=timezone.utc)

    def test_end_datetime(self) -> None:
        assert make_request().end_datetime == datetime(2024, 6, 30, tzinfo=timezone.utc)

    def test_bbox_wkt_is_a_valid_closed_polygon(self) -> None:
        geometry = wkt.loads(make_request().bbox_wkt)
        assert geometry.geom_type == "Polygon"
        assert geometry.is_valid
        assert geometry.exterior.is_closed

    def test_bbox_wkt_bounds_match_input(self) -> None:
        assert wkt.loads(make_request().bbox_wkt).bounds == (2.5, 31.0, 3.5, 32.0)

    def test_bbox_wkt_area(self) -> None:
        assert wkt.loads(make_request().bbox_wkt).area == pytest.approx(1.0)


class TestBboxValidation:
    """Bounding-box rules."""

    @pytest.mark.parametrize(
        "bbox, fragment",
        [
            ([1.0, 2.0, 3.0], "exactly 4 values"),
            ([1.0, 2.0, 3.0, 4.0, 5.0], "exactly 4 values"),
            ([], "exactly 4 values"),
            ([-200.0, 0.0, 10.0, 10.0], "lon_min"),
            ([0.0, 0.0, 200.0, 10.0], "lon_max"),
            ([0.0, -100.0, 10.0, 10.0], "lat_min"),
            ([0.0, 0.0, 10.0, 95.0], "lat_max"),
            ([10.0, 0.0, 5.0, 10.0], "strictly less than lon_max"),
            ([5.0, 5.0, 5.0, 10.0], "strictly less than lon_max"),
            ([0.0, 10.0, 10.0, 5.0], "strictly less than lat_max"),
        ],
    )
    def test_rejected(self, bbox: list[float], fragment: str) -> None:
        with pytest.raises(ValidationError, match=fragment):
            make_request(bbox=bbox)

    @pytest.mark.parametrize(
        "bbox",
        [
            [0.0, 0.0, 1.0, 1.0],
            [-10.0, -10.0, 10.0, 0.0],
            [-180.0, -90.0, 180.0, 90.0],
            [179.0, 89.0, 180.0, 90.0],
        ],
    )
    def test_accepted(self, bbox: list[float]) -> None:
        assert make_request(bbox=bbox).bbox == bbox

    def test_antimeridian_crossing_is_rejected_with_guidance(self) -> None:
        with pytest.raises(ValidationError, match="antimeridian"):
            make_request(bbox=[170.0, 0.0, -170.0, 10.0])


class TestDateValidation:
    """ISO 8601 handling and window ordering."""

    @pytest.mark.parametrize(
        "start, end, fragment",
        [
            ("2024-13-01", "2024-06-30", "ISO 8601"),
            ("01/06/2024", "2024-06-30", "ISO 8601"),
            ("not a date", "2024-06-30", "ISO 8601"),
            ("2024-06-01", "31/12/2024", "ISO 8601"),
            # Whitespace is stripped first, so both report "empty" not "ISO 8601".
            ("", "2024-06-30", "must not be empty"),
            ("   ", "2024-06-30", "must not be empty"),
            ("2024-06-30", "2024-06-01", "must not be after"),
        ],
    )
    def test_rejected(self, start: str, end: str, fragment: str) -> None:
        with pytest.raises(ValidationError, match=fragment):
            make_request(start_date=start, end_date=end)

    def test_error_names_the_offending_field(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            make_request(end_date="31/12/2024")
        assert excinfo.value.errors()[0]["loc"] == ("end_date",)

    def test_backwards_window_is_a_validation_error_not_a_value_error(self) -> None:
        """FastAPI turns ValidationError into 422; a bare ValueError would 500."""
        with pytest.raises(ValidationError):
            make_request(start_date="2024-06-30", end_date="2024-06-01")

    def test_z_suffix_accepted(self) -> None:
        request = make_request(start_date="2024-06-01T00:00:00Z", end_date="2024-06-30T23:59:59Z")
        assert request.start_datetime.tzinfo is not None

    def test_explicit_offset_preserves_instant(self) -> None:
        request = make_request(start_date="2024-06-01T12:30:00+01:00", end_date="2024-06-02")
        assert request.start_datetime.hour == 12

    def test_equal_start_and_end_allowed(self) -> None:
        assert make_request(start_date="2024-06-01", end_date="2024-06-01").start_date == "2024-06-01"

    def test_naive_end_is_normalised_to_utc(self) -> None:
        request = make_request(start_date="2024-06-02T00:00:00Z", end_date="2024-06-02")
        assert request.end_datetime.tzinfo is not None
        assert request.start_datetime == request.end_datetime

    def test_backwards_mixed_naive_aware_is_rejected(self) -> None:
        """The naive/aware comparison must not raise TypeError."""
        with pytest.raises(ValidationError, match="must not be after"):
            make_request(start_date="2024-06-03T00:00:00Z", end_date="2024-06-02")

    def test_offset_is_compared_as_an_instant_not_a_clock_time(self) -> None:
        # 23:00+02:00 is 21:00Z, which is before 22:00Z.
        request = make_request(start_date="2024-06-01T23:00:00+02:00", end_date="2024-06-01T22:00:00Z")
        assert request.start_datetime < request.end_datetime


class TestQueryField:
    """The natural-language query."""

    def test_whitespace_is_stripped(self) -> None:
        assert make_request(query="  drought  ").query == "drought"

    def test_blank_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_request(query="   ")

    def test_overlong_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_request(query="x" * 2001)


# ---------------------------------------------------------------------------
# SatelliteLayerMetadata
# ---------------------------------------------------------------------------


class TestSatelliteLayerMetadata:
    """Physical ranges and unit guards."""

    def test_band_links_omitted_defaults_to_empty(self) -> None:
        layer = SatelliteLayerMetadata(
            stac_id="x", cloud_cover=0.0, ndvi_mean=0.0, lst_celsius_max=0.0
        )
        assert layer.band_links == {}

    def test_band_links_typed_as_str_to_str(self) -> None:
        links = make_layer().band_links
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in links.items())

    @pytest.mark.parametrize("cloud", [-0.1, 100.1])
    def test_cloud_cover_range(self, cloud: float) -> None:
        with pytest.raises(ValidationError):
            make_layer(cloud_cover=cloud)

    def test_cloud_cover_boundaries_accepted(self) -> None:
        assert make_layer(cloud_cover=0.0).cloud_cover == 0.0
        assert make_layer(cloud_cover=100.0).cloud_cover == 100.0

    @pytest.mark.parametrize("ndvi", [-1.01, 1.01])
    def test_ndvi_range(self, ndvi: float) -> None:
        with pytest.raises(ValidationError):
            make_layer(ndvi_mean=ndvi)

    @pytest.mark.parametrize("lst", [-100.1, 100.1])
    def test_lst_range(self, lst: float) -> None:
        with pytest.raises(ValidationError):
            make_layer(lst_celsius_max=lst)

    def test_kelvin_mistake_is_rejected(self) -> None:
        """A land-surface temperature left in Kelvin (~300) must not be stored."""
        with pytest.raises(ValidationError):
            make_layer(lst_celsius_max=314.2)

    def test_realistic_extreme_lst_accepted(self) -> None:
        assert make_layer(lst_celsius_max=-89.2).lst_celsius_max == -89.2

    def test_blank_stac_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_layer(stac_id="   ")

    def test_non_string_band_value_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_layer(band_links={"B04": 12345})

    def test_empty_band_url_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_layer(band_links={"B04": "   "})

    def test_empty_band_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_layer(band_links={"   ": "https://example.org/B04.jp2"})


# ---------------------------------------------------------------------------
# RiskAnalysisResult
# ---------------------------------------------------------------------------


class TestControlledVocabularies:
    """Stress level and audit status normalisation."""

    @pytest.mark.parametrize("raw, canonical", [("HIGH", "high"), (" High ", "high"), ("extreme", "extreme")])
    def test_grid_stress_level_normalises(self, raw: str, canonical: str) -> None:
        assert make_result(grid_stress_level=raw).grid_stress_level == canonical

    def test_enum_member_accepted(self) -> None:
        assert make_result(grid_stress_level=GridStressLevel.EXTREME).grid_stress_level == "extreme"

    def test_field_stays_a_plain_string(self) -> None:
        assert type(make_result().grid_stress_level) is str

    def test_unknown_stress_level_rejected_listing_vocabulary(self) -> None:
        with pytest.raises(ValidationError, match="low, moderate, high, extreme"):
            make_result(grid_stress_level="apocalyptic")

    @pytest.mark.parametrize("raw, canonical", [("PASSED", "passed"), ("Escalated", "escalated"), ("pending", "pending")])
    def test_audit_status_normalises(self, raw: str, canonical: str) -> None:
        assert make_result(audit_status=raw).audit_status == canonical

    def test_unknown_audit_status_rejected(self) -> None:
        with pytest.raises(ValidationError, match="pending, passed, failed, escalated"):
            make_result(audit_status="maybe")

    @pytest.mark.parametrize("status, audited", [("passed", True), ("failed", True), ("pending", False), ("escalated", False)])
    def test_is_audited(self, status: str, audited: bool) -> None:
        assert make_result(audit_status=status).is_audited is audited

    def test_heps_score_is_unbounded_by_design(self) -> None:
        """No scale is imposed: the producing agent owns it."""
        assert make_result(heps_score=0.0).heps_score == 0.0
        assert make_result(heps_score=-5.0).heps_score == -5.0
        assert make_result(heps_score=999.0).heps_score == 999.0


class TestGeoJsonValidation:
    """RFC 7946 minimum shape."""

    def test_feature_collection_accepted(self) -> None:
        assert make_result(geojson_features=make_collection()).feature_count == 1

    def test_bare_feature_accepted(self) -> None:
        assert make_result(geojson_features=make_feature()).feature_count == 1

    def test_null_geometry_allowed(self) -> None:
        assert make_result(geojson_features=make_feature(geometry=None)).feature_count == 1

    def test_missing_type_rejected(self) -> None:
        with pytest.raises(ValidationError, match="'type' member"):
            make_result(geojson_features={"features": []})

    def test_bare_geometry_rejected(self) -> None:
        with pytest.raises(ValidationError, match="FeatureCollection"):
            make_result(geojson_features={"type": "Point", "coordinates": [0, 0]})

    def test_non_list_features_rejected(self) -> None:
        with pytest.raises(ValidationError, match="'features' list"):
            make_result(geojson_features={"type": "FeatureCollection", "features": {}})

    def test_feature_without_properties_rejected(self) -> None:
        bad = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": None}]}
        with pytest.raises(ValidationError, match="'properties'"):
            make_result(geojson_features=bad)

    def test_feature_without_geometry_rejected(self) -> None:
        bad = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}}]}
        with pytest.raises(ValidationError, match="'geometry'"):
            make_result(geojson_features=bad)

    def test_unknown_geometry_type_rejected(self) -> None:
        with pytest.raises(ValidationError, match="geometry.type"):
            make_result(geojson_features=make_feature(geometry={"type": "Blob"}))

    def test_non_feature_inside_collection_rejected(self) -> None:
        bad = {"type": "FeatureCollection", "features": ["not a feature"]}
        with pytest.raises(ValidationError, match="Feature object"):
            make_result(geojson_features=bad)

    def test_feature_count_for_a_collection(self) -> None:
        collection = make_collection([make_feature(), make_feature(), make_feature()])
        assert make_result(geojson_features=collection).feature_count == 3

    @pytest.mark.parametrize("geometry_type", typing.get_args(GeometryType))
    def test_every_declared_geometry_type_is_accepted(self, geometry_type: str) -> None:
        coordinates = {
            "Point": [0, 0],
            "MultiPoint": [[0, 0], [1, 1]],
            "LineString": [[0, 0], [1, 1]],
            "MultiLineString": [[[0, 0], [1, 1]]],
            "Polygon": [[[0, 0], [1, 1]]],
            "MultiPolygon": [[[[0, 0], [1, 1]]]],
        }
        geometry = (
            {"type": geometry_type, "geometries": []}
            if geometry_type == "GeometryCollection"
            else {"type": geometry_type, "coordinates": coordinates[geometry_type]}
        )
        result = make_result(geojson_features=make_feature(geometry=geometry))
        assert result.geojson_features["geometry"]["type"] == geometry_type


# ---------------------------------------------------------------------------
# Cross-cutting
# ---------------------------------------------------------------------------


class TestStrictnessAndDocs:
    """extra='forbid', descriptions, and round-tripping."""

    @pytest.mark.parametrize(
        "factory, extra_key",
        [
            (make_request, "region"),
            (make_layer, "sensor"),
            (make_result, "note"),
        ],
    )
    def test_unknown_field_is_forbidden(self, factory, extra_key: str) -> None:
        with pytest.raises(ValidationError) as excinfo:
            factory(**{extra_key: "surprise"})
        assert excinfo.value.errors()[0]["type"] == "extra_forbidden"

    @pytest.mark.parametrize(
        "model", [SpatialQueryRequest, SatelliteLayerMetadata, RiskAnalysisResult]
    )
    def test_has_a_docstring(self, model) -> None:
        assert model.__doc__ and model.__doc__.strip()

    @pytest.mark.parametrize(
        "model", [SpatialQueryRequest, SatelliteLayerMetadata, RiskAnalysisResult]
    )
    def test_emits_json_schema_with_every_field_described(self, model) -> None:
        schema = model.model_json_schema()
        assert "properties" in schema
        undescribed = [k for k, v in schema["properties"].items() if not v.get("description")]
        assert undescribed == []

    @pytest.mark.parametrize(
        "model, instance",
        [
            (SpatialQueryRequest, make_request()),
            (SatelliteLayerMetadata, make_layer()),
            (RiskAnalysisResult, make_result()),
        ],
    )
    def test_json_round_trips_equal(self, model, instance) -> None:
        assert model.model_validate_json(instance.model_dump_json()) == instance

    def test_assignment_is_validated(self) -> None:
        """validate_assignment catches mutation after construction."""
        layer = make_layer()
        with pytest.raises(ValidationError):
            layer.cloud_cover = 500.0

class TestDefensiveBranches:
    """Type-guard branches that a well-behaved caller never reaches."""

    def test_null_geometry_really_is_null(self) -> None:
        """Guards the test helper itself: None must survive, not become a Point."""
        assert make_feature(geometry=None)["geometry"] is None
        assert make_feature()["geometry"]["type"] == "Point"

    def test_non_string_enum_value_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be a string"):
            make_result(grid_stress_level=42)

    def test_non_string_geojson_type_rejected(self) -> None:
        with pytest.raises(ValidationError, match="'type' must be a string"):
            make_result(geojson_features={"type": 123, "features": []})

    def test_mistyped_feature_rejected(self) -> None:
        bad = {"type": "FeatureCollection", "features": [{"type": "Featuree", "geometry": None, "properties": {}}]}
        with pytest.raises(ValidationError, match="must have type 'Feature'"):
            make_result(geojson_features=bad)

    def test_non_object_geometry_rejected(self) -> None:
        with pytest.raises(ValidationError, match="geometry object or null"):
            make_result(geojson_features=make_feature(geometry="not-an-object"))

    def test_iso_helper_rejects_non_string(self) -> None:
        from app.core.schemas import _parse_iso8601

        with pytest.raises(ValueError, match="must be an ISO 8601 string"):
            _parse_iso8601(20240601, "start_date")
