"""Regression tests for Sentinel-2 reflectance scaling, SCL masking and coverage.

Covers the corrections to the satellite NDVI path:

* L2A digital numbers are offset (-1000 from processing baseline 04.00) and
  divided by 10000 before the NDVI ratio.
* Pixels flagged by the scene classification layer are excluded, not counted as
  vegetation.
* Usable-pixel coverage is reported, and NDVI is withheld when coverage is below
  the configured minimum.
* Unavailable satellite evidence propagates into the coordinator's evidence
  workflow.

Everything here is offline: STAC is stubbed with httpx.MockTransport and rasters
are in-memory samples.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from app.agents.coordinator import CoordinatorAgent
from app.core.schemas import AuditStatus, SatelliteLayerMetadata, SpatialQueryRequest
from app.services.evidence_auditor import audit_conclusion
from app.services.risk_service import EvidenceProvenance, VulnerabilityContext
from app.services.satellite_service import (
    NDVI_NO_DATA,
    SCL_INVALID_CLASSES,
    UNAVAILABLE_STAC_ID,
    BandSample,
    NdviResult,
    SatelliteDataService,
    boa_add_offset,
    compute_ndvi,
    processing_baseline,
    scl_invalid_mask,
)
from tests.conftest import make_item, stac_response

# Reflectance used throughout (not DN). Hand values are noted where asserted.
RHO_RED = np.array([[0.1, 0.2], [0.3, 0.4]])
RHO_NIR = np.array([[0.5, 0.6], [0.7, 0.8]])
#: Mean NDVI over the four pixels, by hand:
#:   ((0.5-0.1)/0.6 + (0.6-0.2)/0.8 + (0.7-0.3)/1.0 + (0.8-0.4)/1.2) / 4 = 0.475
NDVI_OF_RHO = 0.475
#: Mean NDVI over the three pixels with (1, 0) excluded: (0.6667+0.5+0.3333)/3.
NDVI_WITHOUT_ONE_PIXEL = 0.5


def to_dn(rho: np.ndarray, offset: float = -1000.0, quantification: float = 10000.0) -> np.ndarray:
    """Encode reflectance as DN for a given offset: DN = rho * QV - offset."""
    return rho * quantification - offset


def vegetation(shape: tuple[int, int] = (2, 2)) -> np.ndarray:
    return np.full(shape, 4, dtype="uint8")


def scene(tag: str, cloud: float, item_id: str | None = None, baseline: str | None = "05.10") -> dict[str, Any]:
    """A STAC item whose band and SCL hrefs are unique to ``tag``."""
    item = make_item(
        item_id or f"S2B_MSIL2A_20240615T103029_N{(baseline or '0000').replace('.', '')}_R108_T31SET_{tag}",
        cloud_cover=cloud,
        red_href=f"https://eodata.example/{tag}_B04.jp2",
        nir_href=f"https://eodata.example/{tag}_B08.jp2",
        extra_assets={"SCL": {"href": f"https://eodata.example/{tag}_SCL.jp2"}},
    )
    if baseline is None:
        item["id"] = f"S2B_MSIL2A_20240615T103029_R108_T31SET_{tag}"
    return item


def load(reader: Any, tag: str, rho_red: np.ndarray, rho_nir: np.ndarray, scl: np.ndarray, offset: float = -1000.0) -> None:
    reader.samples[f"https://eodata.example/{tag}_B04.jp2"] = BandSample(to_dn(rho_red, offset))
    reader.samples[f"https://eodata.example/{tag}_B08.jp2"] = BandSample(to_dn(rho_nir, offset))
    reader.samples[f"https://eodata.example/{tag}_SCL.jp2"] = BandSample(scl)


# ---------------------------------------------------------------------------
# Processing baseline and BOA offset
# ---------------------------------------------------------------------------


class TestBoaOffset:
    @pytest.mark.parametrize("baseline", [4.0, 5.1, 6.0])
    def test_offset_applies_from_baseline_04_00(self, baseline: float) -> None:
        assert boa_add_offset(baseline) == -1000.0

    @pytest.mark.parametrize("baseline", [2.12, 3.99])
    def test_earlier_baselines_have_no_offset(self, baseline: float) -> None:
        assert boa_add_offset(baseline) == 0.0

    def test_unknown_baseline_is_none_not_zero(self) -> None:
        assert boa_add_offset(None) is None


class TestProcessingBaseline:
    def test_read_from_property_string(self) -> None:
        assert processing_baseline({"properties": {"s2:processing_baseline": "05.10"}}) == pytest.approx(5.1)

    def test_read_from_property_number(self) -> None:
        assert processing_baseline({"properties": {"s2:processing_baseline": 4.0}}) == 4.0

    def test_read_from_product_name(self) -> None:
        item = {"id": "S2A_MSIL2A_20210101T000000_N0209_R001_T31SET_20210101T000001"}
        assert processing_baseline(item) == pytest.approx(2.09)

    def test_property_takes_precedence_over_name(self) -> None:
        item = {"id": "X_N0209_Y", "properties": {"s2:processing_baseline": "05.10"}}
        assert processing_baseline(item) == pytest.approx(5.1)

    @pytest.mark.parametrize("value", ["", "abc", "-1", "nan", True, None, [], "0"])
    def test_malformed_property_falls_through_to_unknown(self, value: Any) -> None:
        assert processing_baseline({"id": "no-token", "properties": {"s2:processing_baseline": value}}) is None

    def test_unknown_when_neither_present(self) -> None:
        assert processing_baseline({"id": "REAL_SCENE"}) is None


# ---------------------------------------------------------------------------
# Reflectance scaling
# ---------------------------------------------------------------------------


class TestReflectanceScaling:
    def test_offset_recovers_the_true_ndvi(self) -> None:
        red = BandSample(to_dn(RHO_RED))
        nir = BandSample(to_dn(RHO_NIR))
        result = compute_ndvi(red, nir, reflectance_offset=-1000.0, quantification_value=10000.0, scl_invalid=None)
        assert result.mean == pytest.approx(NDVI_OF_RHO)
        assert result.valid_coverage == pytest.approx(1.0)

    def test_skipping_the_offset_gives_a_different_ndvi(self) -> None:
        """Documents the defect: the raw DN ratio is not the reflectance NDVI."""
        red = BandSample(to_dn(RHO_RED))
        nir = BandSample(to_dn(RHO_NIR))
        raw = compute_ndvi(red, nir, reflectance_offset=0.0, quantification_value=10000.0, scl_invalid=None)
        assert raw.mean is not None
        assert abs(raw.mean - NDVI_OF_RHO) > 0.05

    async def test_same_surface_gives_same_ndvi_on_both_baselines(self, service_factory, band_reader, valid_window) -> None:
        """An old pre-04.00 product and a new one describing the same surface agree."""
        legacy_scl = vegetation()
        band_reader.samples["https://eodata.example/old_B04.jp2"] = BandSample(RHO_RED * 10000.0)
        band_reader.samples["https://eodata.example/old_B08.jp2"] = BandSample(RHO_NIR * 10000.0)
        band_reader.samples["https://eodata.example/old_SCL.jp2"] = BandSample(legacy_scl)
        load(band_reader, "new", RHO_RED, RHO_NIR, legacy_scl)
        old = scene("old", 5.0, baseline="03.00")
        new = scene("new", 5.0, baseline="05.10")

        async with service_factory(stac_response([old]), reader=band_reader) as service:
            old_result = await service.fetch_sentinel_indices(**valid_window)
        async with service_factory(stac_response([new]), reader=band_reader) as service:
            new_result = await service.fetch_sentinel_indices(**valid_window)

        assert old_result["ndvi_mean"] == pytest.approx(NDVI_OF_RHO)
        assert new_result["ndvi_mean"] == pytest.approx(NDVI_OF_RHO)
        assert old_result["reflectance_offset"] == 0.0
        assert new_result["reflectance_offset"] == -1000.0


# ---------------------------------------------------------------------------
# Scene classification mask
# ---------------------------------------------------------------------------


class TestSclMask:
    @pytest.mark.parametrize("scl_class", sorted(SCL_INVALID_CLASSES))
    def test_each_invalid_class_is_masked(self, scl_class: int) -> None:
        mask = scl_invalid_mask(np.array([[scl_class]], dtype="uint8"), (1, 1))
        assert mask[0, 0]

    @pytest.mark.parametrize("scl_class", [4, 5, 6, 7, 11])
    def test_usable_classes_are_kept(self, scl_class: int) -> None:
        mask = scl_invalid_mask(np.array([[scl_class]], dtype="uint8"), (1, 1))
        assert not mask[0, 0]

    def test_single_cloud_pixel_survives_downsampling(self) -> None:
        """Nearest-neighbour decimation can drop one pixel; any-pixel reduction cannot."""
        scl = np.full((4, 4), 4, dtype="uint8")
        scl[0, 0] = 9  # one high-probability cloud pixel in the top-left 2x2 block
        mask = scl_invalid_mask(scl, (2, 2))
        assert mask.tolist() == [[True, False], [False, False]]

    def test_upsampling_maps_each_target_cell_to_its_source(self) -> None:
        scl = np.array([[9, 4], [4, 4]], dtype="uint8")
        mask = scl_invalid_mask(scl, (4, 4))
        assert mask[:2, :2].all()
        assert not mask[2:, 2:].any()

    def test_non_2d_input_is_rejected(self) -> None:
        from app.services.satellite_service import SatelliteServiceError

        with pytest.raises(SatelliteServiceError):
            scl_invalid_mask(np.zeros((2, 2, 2)), (2, 2))


# ---------------------------------------------------------------------------
# Coverage arithmetic
# ---------------------------------------------------------------------------


class TestCoverage:
    def test_nodata_pixel_reduces_coverage_and_is_not_counted_as_zero(self) -> None:
        red = BandSample(to_dn(RHO_RED).copy(), nodata=0.0)
        red.values[1, 0] = 0.0  # DN 0 is no-data
        nir = BandSample(to_dn(RHO_NIR))
        result = compute_ndvi(red, nir, reflectance_offset=-1000.0, quantification_value=10000.0, scl_invalid=None)
        assert result.valid_coverage == pytest.approx(0.75)
        assert result.mean == pytest.approx(NDVI_WITHOUT_ONE_PIXEL)

    def test_masked_cloud_is_excluded_from_the_mean(self) -> None:
        """A cloud whose reflectance would drag NDVI down must not contribute."""
        red = BandSample(to_dn(RHO_RED))
        nir = BandSample(to_dn(RHO_NIR))
        mask = np.array([[False, False], [True, False]])
        result = compute_ndvi(red, nir, reflectance_offset=-1000.0, quantification_value=10000.0, scl_invalid=mask)
        assert result.valid_coverage == pytest.approx(0.75)
        assert result.mean == pytest.approx(NDVI_WITHOUT_ONE_PIXEL)

    def test_fully_masked_footprint_has_no_ndvi_and_zero_coverage(self) -> None:
        red = BandSample(to_dn(RHO_RED))
        nir = BandSample(to_dn(RHO_NIR))
        result = compute_ndvi(
            red, nir, reflectance_offset=-1000.0, quantification_value=10000.0,
            scl_invalid=np.ones((2, 2), dtype=bool),
        )
        assert result == NdviResult(None, 0.0)

    def test_mask_shape_mismatch_refuses_ndvi(self) -> None:
        red = BandSample(to_dn(RHO_RED))
        nir = BandSample(to_dn(RHO_NIR))
        result = compute_ndvi(
            red, nir, reflectance_offset=-1000.0, quantification_value=10000.0,
            scl_invalid=np.zeros((3, 3), dtype=bool),
        )
        assert result == NdviResult(None, None)


# ---------------------------------------------------------------------------
# Service gating: coverage threshold, missing inputs, candidate fallback
# ---------------------------------------------------------------------------


class TestCoverageGate:
    async def test_low_coverage_withholds_ndvi_and_reports_it(self, service_factory, band_reader, valid_window) -> None:
        scl = vegetation()
        scl[0, :] = 9  # half the footprint is cloud
        load(band_reader, "a", RHO_RED, RHO_NIR, scl)
        async with service_factory(stac_response([scene("a", 5.0)]), reader=band_reader) as service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["ndvi_mean"] == NDVI_NO_DATA
        assert result["ndvi_valid_coverage"] == pytest.approx(0.5)
        assert result["stac_id"] != UNAVAILABLE_STAC_ID

    async def test_threshold_is_inclusive(self, service_factory, band_reader, valid_window) -> None:
        scl = vegetation()
        scl[1, 1] = 3  # 75% usable
        load(band_reader, "a", RHO_RED, RHO_NIR, scl)
        async with service_factory(
            stac_response([scene("a", 5.0)]), reader=band_reader, min_valid_coverage=0.75
        ) as service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["ndvi_mean"] != NDVI_NO_DATA
        # Pixel (1,1) (NDVI 0.4/1.2 = 0.3333) is masked; the rest are
        # 0.4/0.6, 0.4/0.8, 0.4/1.0: (0.6667 + 0.5 + 0.4) / 3 = 0.5222.
        assert result["ndvi_mean"] == pytest.approx(0.5222222, abs=1e-6)

    async def test_missing_scl_withholds_ndvi(self, service_factory, band_reader, valid_window) -> None:
        load(band_reader, "a", RHO_RED, RHO_NIR, vegetation())
        item = scene("a", 5.0)
        del item["assets"]["SCL"]
        async with service_factory(stac_response([item]), reader=band_reader) as service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["ndvi_mean"] == NDVI_NO_DATA
        assert result["ndvi_valid_coverage"] is None
        assert result["reflectance_offset"] == -1000.0  # the baseline was still known

    async def test_unknown_baseline_withholds_ndvi(self, service_factory, band_reader, valid_window) -> None:
        load(band_reader, "a", RHO_RED, RHO_NIR, vegetation())
        item = scene("a", 5.0, item_id="REAL_SCENE_NO_BASELINE")
        async with service_factory(stac_response([item]), reader=band_reader) as service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["ndvi_mean"] == NDVI_NO_DATA
        assert result["reflectance_offset"] is None
        assert result["processing_baseline"] is None

    async def test_next_clear_scene_is_used_when_the_clearest_is_too_cloudy(
        self, service_factory, band_reader, valid_window
    ) -> None:
        cloudy = vegetation()
        cloudy[0, :] = 9
        load(band_reader, "first", RHO_RED, RHO_NIR, cloudy)
        load(band_reader, "second", RHO_RED, RHO_NIR, vegetation())
        payload = stac_response([scene("first", 2.0), scene("second", 8.0)])
        async with service_factory(payload, reader=band_reader) as service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["stac_id"].endswith("_second")
        assert result["ndvi_valid_coverage"] == pytest.approx(1.0)
        assert result["ndvi_mean"] == pytest.approx(NDVI_OF_RHO)

    async def test_max_candidates_one_disables_fallback(self, service_factory, band_reader, valid_window) -> None:
        cloudy = vegetation()
        cloudy[0, :] = 9
        load(band_reader, "first", RHO_RED, RHO_NIR, cloudy)
        load(band_reader, "second", RHO_RED, RHO_NIR, vegetation())
        payload = stac_response([scene("first", 2.0), scene("second", 8.0)])
        async with service_factory(payload, reader=band_reader, max_candidates=1) as service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["ndvi_mean"] == NDVI_NO_DATA
        assert result["stac_id"].endswith("_first")

    async def test_best_attempt_is_reported_when_every_candidate_fails(
        self, service_factory, band_reader, valid_window
    ) -> None:
        worse = vegetation()
        worse[:, :] = 9
        better = vegetation()
        better[0, :] = 9
        load(band_reader, "worse", RHO_RED, RHO_NIR, worse)
        load(band_reader, "better", RHO_RED, RHO_NIR, better)
        payload = stac_response([scene("worse", 2.0), scene("better", 8.0)])
        async with service_factory(payload, reader=band_reader) as service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["ndvi_mean"] == NDVI_NO_DATA
        assert result["stac_id"].endswith("_better")
        assert result["ndvi_valid_coverage"] == pytest.approx(0.5)

    def test_invalid_coverage_threshold_rejected(self) -> None:
        from app.services.satellite_service import SatelliteServiceError

        with pytest.raises(SatelliteServiceError):
            SatelliteDataService(min_valid_coverage=1.5)
        with pytest.raises(SatelliteServiceError):
            SatelliteDataService(max_candidates=0)


# ---------------------------------------------------------------------------
# Schema additivity
# ---------------------------------------------------------------------------


class TestSchemaAdditivity:
    def test_legacy_payload_without_new_fields_still_validates(self) -> None:
        layer = SatelliteLayerMetadata.model_validate(
            {"stac_id": "x", "cloud_cover": 1.0, "ndvi_mean": 0.2, "lst_celsius_max": 0.0}
        )
        assert layer.ndvi_valid_coverage is None
        assert layer.reflectance_offset is None
        assert layer.processing_baseline is None

    @pytest.mark.parametrize("coverage", [-0.1, 1.2])
    def test_coverage_is_bounded(self, coverage: float) -> None:
        with pytest.raises(ValidationError):
            SatelliteLayerMetadata(
                stac_id="x", cloud_cover=1.0, ndvi_mean=0.2, lst_celsius_max=0.0,
                ndvi_valid_coverage=coverage,
            )

    async def test_degraded_document_validates_with_null_coverage(self, service_factory, valid_window) -> None:
        async with service_factory(exc=None, payload=stac_response([])) as service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["stac_id"] == UNAVAILABLE_STAC_ID
        assert result["ndvi_valid_coverage"] is None


# ---------------------------------------------------------------------------
# Propagation into the evidence workflow
# ---------------------------------------------------------------------------


class _VerifiedDemographics:
    async def fetch(self, bbox: Any) -> VulnerabilityContext:
        return VulnerabilityContext(
            provenance=EvidenceProvenance.VERIFIED, source="test fixture",
            population_density=8_000.0, social_vulnerability=0.8,
        )


class _WithheldSatellite:
    """Returns a layer whose NDVI was withheld for insufficient coverage."""

    async def fetch_sentinel_indices(self, bbox: Any, start_date: str, end_date: str) -> dict:
        return {
            "stac_id": "S2B_TEST_CLOUDY", "cloud_cover": 5.0, "ndvi_mean": NDVI_NO_DATA,
            "lst_celsius_max": 0.0, "band_links": {}, "ndvi_valid_coverage": 0.4,
            "reflectance_offset": -1000.0, "processing_baseline": "05.10",
        }


async def test_withheld_ndvi_is_invalid_evidence_and_never_passes() -> None:
    agent = CoordinatorAgent(
        satellite_service=_WithheldSatellite(), vulnerability_provider=_VerifiedDemographics()
    )
    request = SpatialQueryRequest(
        query="heat", bbox=[5.2, 31.8, 5.5, 32.1], start_date="2024-06-01", end_date="2024-06-30"
    )
    result = await agent.execute_pipeline(request)
    satellite = next(e for e in result.evidence if e.source == "satellite")
    assert satellite.provenance == "invalid"
    assert "40% of the footprint usable" in (satellite.detail or "")
    assert result.audit_status != AuditStatus.PASSED.value
    assert result.heps_score is not None  # equity alone: labelled partial, not passed


def test_auditor_blocks_a_pass_on_invalid_satellite_evidence() -> None:
    violations = audit_conclusion(
        heps_score=0.5, stress_level="moderate", audit_status="passed", data_complete=True,
        vulnerability_provenance=EvidenceProvenance.VERIFIED,
        satellite_provenance=EvidenceProvenance.INVALID,
    )
    assert any("satellite evidence" in v for v in violations)
