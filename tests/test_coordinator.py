"""Tests for :mod:`app.agents.coordinator`."""

from __future__ import annotations

import logging
from typing import Any

import httpx
import numpy as np
import pytest

from app.agents.coordinator import CoordinatorAgent
from app.core.schemas import (
    AuditStatus,
    GridStressLevel,
    RiskAnalysisResult,
    SpatialQueryRequest,
)
from app.services.risk_service import (
    RiskWeights,
    StressThresholds,
    VulnerabilityContext,
    VulnerabilityProvider,
)
from app.services.satellite_service import (
    NDVI_NO_DATA,
    UNAVAILABLE_STAC_ID,
    BandSample,
    SatelliteDataService,
)

#: Pixel samples matching the stub hrefs in tests/conftest.py.
_RED = np.array([[0.1, 0.2], [0.3, 0.4]])
_NIR = np.array([[0.5, 0.6], [0.7, 0.8]])
BAND_SAMPLES = {
    "https://eodata.dataspace.copernicus.eu/B04.jp2": BandSample(_RED),
    "https://eodata.dataspace.copernicus.eu/B08.jp2": BandSample(_NIR),
}
#: Mean NDVI the real service derives from those pixels, by hand:
#:   ((0.4/0.6) + (0.4/0.8) + (0.4/1.0) + (0.4/1.2)) / 4
PIPELINE_NDVI = 0.475
#: Cooling deficit for that NDVI: 1 - (0.475 - 0.1) / (0.8 - 0.1)
PIPELINE_COOLING_DEFICIT = 1 - (PIPELINE_NDVI - 0.1) / 0.7

COORDINATOR_LOGGER = "app.agents.coordinator"


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def make_layer(
    *,
    stac_id: str = "S2B_TEST_SCENE",
    cloud_cover: float = 5.0,
    ndvi_mean: float = 0.15,
    lst_celsius_max: float = 0.0,
) -> dict[str, Any]:
    """A SatelliteLayerMetadata-shaped document."""
    return {
        "stac_id": stac_id,
        "cloud_cover": cloud_cover,
        "ndvi_mean": ndvi_mean,
        "lst_celsius_max": lst_celsius_max,
        "band_links": {"B04": "https://example.org/B04.jp2"},
    }


def degraded_layer() -> dict[str, Any]:
    """The document SatelliteDataService returns when retrieval fails."""
    return {
        "stac_id": UNAVAILABLE_STAC_ID,
        "cloud_cover": 100.0,
        "ndvi_mean": NDVI_NO_DATA,
        "lst_celsius_max": 0.0,
        "band_links": {},
    }


class StubSatellite:
    """SatelliteDataService stand-in returning a canned layer."""

    def __init__(self, layer: dict[str, Any] | None = None, exc: Exception | None = None) -> None:
        self.layer = layer if layer is not None else make_layer()
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    async def fetch_sentinel_indices(self, bbox: list, start_date: str, end_date: str) -> dict:
        self.calls.append({"bbox": bbox, "start_date": start_date, "end_date": end_date})
        if self.exc is not None:
            raise self.exc
        return self.layer


class StubVulnerability:
    """VulnerabilityProvider stand-in recording the bbox it was asked about."""

    def __init__(
        self, density: float = 8_000.0, svi: float = 0.8, exc: Exception | None = None
    ) -> None:
        self.context = VulnerabilityContext(density, svi)
        self.exc = exc
        self.calls: list[Any] = []

    async def fetch(self, bbox: Any) -> VulnerabilityContext:
        self.calls.append(bbox)
        if self.exc is not None:
            raise self.exc
        return self.context


def make_request(**overrides: Any) -> SpatialQueryRequest:
    """A valid request over Ouargla."""
    payload: dict[str, Any] = {
        "query": "heat stress in dense housing",
        "bbox": [5.2, 31.8, 5.5, 32.1],
        "start_date": "2024-06-01",
        "end_date": "2024-06-30",
    }
    payload.update(overrides)
    return SpatialQueryRequest(**payload)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestExecutePipeline:
    """The normal flow."""

    async def test_returns_a_valid_result(self) -> None:
        agent = CoordinatorAgent(satellite_service=StubSatellite())
        result = await agent.execute_pipeline(make_request())
        assert isinstance(result, RiskAnalysisResult)

    async def test_request_is_forwarded_to_the_satellite_tool(self) -> None:
        satellite = StubSatellite()
        agent = CoordinatorAgent(satellite_service=satellite)
        request = make_request()
        await agent.execute_pipeline(request)
        assert satellite.calls == [
            {"bbox": request.bbox, "start_date": "2024-06-01", "end_date": "2024-06-30"}
        ]

    async def test_request_bbox_is_forwarded_to_the_vulnerability_tool(self) -> None:
        vuln = StubVulnerability()
        agent = CoordinatorAgent(satellite_service=StubSatellite(), vulnerability_provider=vuln)
        await agent.execute_pipeline(make_request())
        assert vuln.calls == [[5.2, 31.8, 5.5, 32.1]]

    async def test_stub_provider_satisfies_the_protocol(self) -> None:
        assert isinstance(StubVulnerability(), VulnerabilityProvider)

    async def test_score_matches_the_engine_given_the_same_inputs(self) -> None:
        """The coordinator must not silently re-derive the formula."""
        from app.services.risk_service import compute_heps

        satellite = StubSatellite(make_layer(ndvi_mean=0.15))
        vuln = StubVulnerability(8_000.0, 0.8)
        agent = CoordinatorAgent(satellite_service=satellite, vulnerability_provider=vuln)
        result = await agent.execute_pipeline(make_request())

        expected = compute_heps(
            lst_celsius=None,  # Sentinel-2 supplies no thermal band
            ndvi=0.15,
            vulnerability=VulnerabilityContext(8_000.0, 0.8),
        )
        assert result.heps_score == pytest.approx(expected.heps_score)
        assert result.grid_stress_level == expected.stress_level.value

    async def test_geojson_covers_the_requested_bbox(self) -> None:
        agent = CoordinatorAgent(satellite_service=StubSatellite())
        result = await agent.execute_pipeline(make_request())
        geometry = result.geojson_features["features"][0]["geometry"]
        assert geometry["type"] == "Polygon"
        ring = geometry["coordinates"][0]
        assert ring[0] == ring[-1]  # closed
        assert [round(v, 6) for v in ring[0]] == [5.2, 31.8]
        assert [round(v, 6) for v in ring[2]] == [5.5, 32.1]

    async def test_geojson_carries_the_audit_trail(self) -> None:
        agent = CoordinatorAgent(satellite_service=StubSatellite())
        result = await agent.execute_pipeline(make_request())
        properties = result.geojson_features["features"][0]["properties"]
        assert properties["query"] == "heat stress in dense housing"
        assert properties["stac_id"] == "S2B_TEST_SCENE"
        assert properties["ndvi_mean"] == 0.15
        assert "weights_used" in properties
        assert properties["factors"]["cooling_deficit"] == pytest.approx(1 - 0.05 / 0.7)

    async def test_feature_count(self) -> None:
        agent = CoordinatorAgent(satellite_service=StubSatellite())
        result = await agent.execute_pipeline(make_request())
        assert result.feature_count == 1


# ---------------------------------------------------------------------------
# Audit status
# ---------------------------------------------------------------------------


class TestAuditStatus:
    """Which verdict the run earns."""

    async def test_escalated_when_lst_is_absent(self) -> None:
        """The default: Sentinel-2 has no thermal band, so the score is partial."""
        agent = CoordinatorAgent(satellite_service=StubSatellite())
        result = await agent.execute_pipeline(make_request())
        assert result.audit_status == AuditStatus.ESCALATED.value

    async def test_passed_when_every_component_is_observed(self) -> None:
        """A thermal source flips the run to a clean pass."""
        satellite = StubSatellite(make_layer(lst_celsius_max=44.5))
        agent = CoordinatorAgent(satellite_service=satellite, provides_lst=True)
        result = await agent.execute_pipeline(make_request())
        assert result.audit_status == AuditStatus.PASSED.value
        properties = result.geojson_features["features"][0]["properties"]
        assert properties["data_complete"] is True
        assert properties["factors"]["heat_exposure"] == pytest.approx((44.5 - 15) / 35)

    async def test_provides_lst_false_ignores_the_placeholder(self) -> None:
        """A 0.0 placeholder must never be scored as 'no heat'."""
        satellite = StubSatellite(make_layer(lst_celsius_max=0.0))
        agent = CoordinatorAgent(satellite_service=satellite, provides_lst=False)
        result = await agent.execute_pipeline(make_request())
        properties = result.geojson_features["features"][0]["properties"]
        assert properties["factors"]["heat_exposure"] is None
        assert result.audit_status == AuditStatus.ESCALATED.value

    async def test_escalated_when_satellite_degrades(self) -> None:
        """A degraded tool is not a failure: the equity score still stands."""
        agent = CoordinatorAgent(
            satellite_service=StubSatellite(degraded_layer()),
            vulnerability_provider=StubVulnerability(8_000.0, 0.8),
        )
        result = await agent.execute_pipeline(make_request())
        assert result.audit_status == AuditStatus.ESCALATED.value
        properties = result.geojson_features["features"][0]["properties"]
        assert properties["factors"]["cooling_deficit"] is None
        # Equity alone, renormalised to full weight:
        #   0.5 * (8000/10000) + 0.5 * 0.8 = 0.4 + 0.4 = 0.8
        assert result.heps_score == pytest.approx(0.8)

    async def test_failed_when_satellite_raises(self) -> None:
        agent = CoordinatorAgent(
            satellite_service=StubSatellite(exc=RuntimeError("tool exploded"))
        )
        result = await agent.execute_pipeline(make_request())
        assert result.audit_status == AuditStatus.FAILED.value
        assert result.heps_score == 0.0
        assert result.grid_stress_level == GridStressLevel.LOW.value

    async def test_failed_result_still_describes_the_area(self) -> None:
        agent = CoordinatorAgent(
            satellite_service=StubSatellite(exc=RuntimeError("tool exploded"))
        )
        result = await agent.execute_pipeline(make_request())
        properties = result.geojson_features["features"][0]["properties"]
        assert "tool exploded" in properties["error"]
        assert result.geojson_features["features"][0]["geometry"]["type"] == "Polygon"

    async def test_failed_when_vulnerability_raises(self) -> None:
        """No equity component means it is not a Heat *Equity* score at all."""
        agent = CoordinatorAgent(
            satellite_service=StubSatellite(),
            vulnerability_provider=StubVulnerability(exc=ValueError("census unavailable")),
        )
        result = await agent.execute_pipeline(make_request())
        assert result.audit_status == AuditStatus.FAILED.value

    async def test_failed_when_satellite_returns_a_malformed_layer(self) -> None:
        agent = CoordinatorAgent(satellite_service=StubSatellite({"stac_id": "x"}))
        result = await agent.execute_pipeline(make_request())
        assert result.audit_status == AuditStatus.FAILED.value

    async def test_failed_when_the_engine_cannot_score(self) -> None:
        """Zero equity weight plus no satellite evidence leaves nothing to blend."""
        agent = CoordinatorAgent(
            satellite_service=StubSatellite(degraded_layer()),
            weights=RiskWeights(heat=0.5, cooling=0.5, vulnerability=0.0),
        )
        result = await agent.execute_pipeline(make_request())
        assert result.audit_status == AuditStatus.FAILED.value


class TestGraphRouting:
    """Conditional edges short-circuit correctly."""

    async def test_assess_is_skipped_after_satellite_failure(self, caplog) -> None:
        agent = CoordinatorAgent(
            satellite_service=StubSatellite(exc=RuntimeError("boom"))
        )
        with caplog.at_level(logging.INFO, logger=COORDINATOR_LOGGER):
            await agent.execute_pipeline(make_request())
        phases = [r.message for r in caplog.records if "phase=assess" in r.message]
        assert phases == []

    async def test_vulnerability_is_skipped_after_satellite_failure(self) -> None:
        vuln = StubVulnerability()
        agent = CoordinatorAgent(
            satellite_service=StubSatellite(exc=RuntimeError("boom")),
            vulnerability_provider=vuln,
        )
        await agent.execute_pipeline(make_request())
        assert vuln.calls == []

    async def test_assess_is_skipped_after_vulnerability_failure(self, caplog) -> None:
        agent = CoordinatorAgent(
            satellite_service=StubSatellite(),
            vulnerability_provider=StubVulnerability(exc=RuntimeError("boom")),
        )
        with caplog.at_level(logging.INFO, logger=COORDINATOR_LOGGER):
            await agent.execute_pipeline(make_request())
        assert [r for r in caplog.records if "phase=assess" in r.message] == []

    async def test_consolidate_always_runs(self, caplog) -> None:
        agent = CoordinatorAgent(
            satellite_service=StubSatellite(exc=RuntimeError("boom"))
        )
        with caplog.at_level(logging.INFO, logger=COORDINATOR_LOGGER):
            await agent.execute_pipeline(make_request())
        assert any("phase=consolidate" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Sentinel translation
# ---------------------------------------------------------------------------


class TestSentinelTranslation:
    """Tool sentinels become explicit 'unavailable', never real zeros."""

    async def test_ndvi_no_data_becomes_missing(self) -> None:
        satellite = StubSatellite(make_layer(ndvi_mean=NDVI_NO_DATA))
        agent = CoordinatorAgent(satellite_service=satellite)
        result = await agent.execute_pipeline(make_request())
        properties = result.geojson_features["features"][0]["properties"]
        assert properties["factors"]["cooling_deficit"] is None

    async def test_ndvi_no_data_does_not_score_as_water(self) -> None:
        """NDVI -1.0 is the sentinel; scoring it would give a maximal deficit."""
        agent = CoordinatorAgent(
            satellite_service=StubSatellite(make_layer(ndvi_mean=NDVI_NO_DATA)),
            vulnerability_provider=StubVulnerability(8_000.0, 0.8),
        )
        result = await agent.execute_pipeline(make_request())
        # Equity alone: 0.5 * 0.8 + 0.5 * 0.8 = 0.8.
        assert result.heps_score == pytest.approx(0.8)

        # Had -1.0 been read as a real measurement, the cooling deficit would be
        # maximal (1.0) and the score inflated to 0.5833*1.0 + 0.4167*0.8.
        misread = 0.35 / 0.60 * 1.0 + 0.25 / 0.60 * 0.8
        assert result.heps_score < misread

    async def test_unavailable_scene_yields_no_satellite_factors(self) -> None:
        agent = CoordinatorAgent(satellite_service=StubSatellite(degraded_layer()))
        result = await agent.execute_pipeline(make_request())
        properties = result.geojson_features["features"][0]["properties"]
        assert properties["factors"]["heat_exposure"] is None
        assert properties["factors"]["cooling_deficit"] is None
        assert properties["stac_id"] == UNAVAILABLE_STAC_ID


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    """Weights, thresholds and defaults."""

    async def test_custom_thresholds_change_the_label_not_the_score(self) -> None:
        satellite_a = StubSatellite(make_layer())
        satellite_b = StubSatellite(make_layer())
        lenient = CoordinatorAgent(
            satellite_service=satellite_a,
            thresholds=StressThresholds(moderate=0.9, high=0.95, extreme=0.99),
        )
        strict = CoordinatorAgent(
            satellite_service=satellite_b,
            thresholds=StressThresholds(moderate=0.1, high=0.2, extreme=0.3),
        )
        a = await lenient.execute_pipeline(make_request())
        b = await strict.execute_pipeline(make_request())
        assert a.heps_score == pytest.approx(b.heps_score)
        assert a.grid_stress_level != b.grid_stress_level
        assert b.grid_stress_level == GridStressLevel.EXTREME.value

    async def test_custom_weights_change_the_score(self) -> None:
        cooling_heavy = CoordinatorAgent(
            satellite_service=StubSatellite(make_layer()),
            vulnerability_provider=StubVulnerability(),
            weights=RiskWeights(heat=0.0, cooling=0.5, vulnerability=0.5),
        )
        result = await cooling_heavy.execute_pipeline(make_request())
        properties = result.geojson_features["features"][0]["properties"]
        assert properties["weights_used"].keys() == {"cooling", "vulnerability"}

    async def test_defaults_are_constructed_when_omitted(self) -> None:
        """No injected tools: real service and placeholder provider."""
        agent = CoordinatorAgent()
        assert agent._satellite is not None
        assert agent._provides_lst is False

    async def test_pipeline_is_reusable(self) -> None:
        satellite = StubSatellite()
        agent = CoordinatorAgent(satellite_service=satellite)
        first = await agent.execute_pipeline(make_request(query="first"))
        second = await agent.execute_pipeline(make_request(query="second"))
        assert first.heps_score == pytest.approx(second.heps_score)
        assert len(satellite.calls) == 2
        assert first.geojson_features["features"][0]["properties"]["query"] == "first"
        assert second.geojson_features["features"][0]["properties"]["query"] == "second"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestLogging:
    """Every phase is logged, at the level its outcome deserves."""

    async def test_all_phases_are_logged_on_success(self, caplog) -> None:
        agent = CoordinatorAgent(satellite_service=StubSatellite())
        with caplog.at_level(logging.INFO, logger=COORDINATOR_LOGGER):
            await agent.execute_pipeline(make_request())
        text = "\n".join(r.message for r in caplog.records)
        for phase in ("gather_satellite", "gather_vulnerability", "assess", "consolidate"):
            assert f"phase={phase}" in text, f"no log for {phase}"
        assert "pipeline start" in text
        assert "pipeline end" in text

    async def test_pipeline_end_reports_the_verdict(self, caplog) -> None:
        agent = CoordinatorAgent(satellite_service=StubSatellite())
        with caplog.at_level(logging.INFO, logger=COORDINATOR_LOGGER):
            await agent.execute_pipeline(make_request())
        end = [r.message for r in caplog.records if "pipeline end" in r.message]
        assert len(end) == 1
        assert "audit=escalated" in end[0]
        assert "phases=['gather_satellite', 'gather_vulnerability', 'assess', 'consolidate']" in end[0]

    async def test_missing_thermal_source_is_warned(self, caplog) -> None:
        agent = CoordinatorAgent(satellite_service=StubSatellite())
        with caplog.at_level(logging.WARNING, logger=COORDINATOR_LOGGER):
            await agent.execute_pipeline(make_request())
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("no thermal band" in w for w in warnings)

    async def test_degraded_scene_is_warned(self, caplog) -> None:
        agent = CoordinatorAgent(satellite_service=StubSatellite(degraded_layer()))
        with caplog.at_level(logging.WARNING, logger=COORDINATOR_LOGGER):
            await agent.execute_pipeline(make_request())
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("no usable scene" in w for w in warnings)

    async def test_tool_failures_are_logged_as_errors(self, caplog) -> None:
        agent = CoordinatorAgent(
            satellite_service=StubSatellite(exc=RuntimeError("tool exploded"))
        )
        with caplog.at_level(logging.ERROR, logger=COORDINATOR_LOGGER):
            await agent.execute_pipeline(make_request())
        errors = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("phase=gather_satellite failed" in e for e in errors)
        assert any("RuntimeError" in e for e in errors)

    async def test_success_is_not_noisy_at_error_level(self, caplog) -> None:
        agent = CoordinatorAgent(
            satellite_service=StubSatellite(make_layer(lst_celsius_max=40.0)),
            provides_lst=True,
        )
        with caplog.at_level(logging.DEBUG, logger=COORDINATOR_LOGGER):
            await agent.execute_pipeline(make_request())
        assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


class TestRobustness:
    """The pipeline never lets a tool exception escape."""

    @pytest.mark.parametrize(
        "exc",
        [RuntimeError("boom"), ValueError("bad value"), TimeoutError("timed out"), KeyError("missing")],
    )
    async def test_any_satellite_exception_is_contained(self, exc: Exception) -> None:
        agent = CoordinatorAgent(satellite_service=StubSatellite(exc=exc))
        result = await agent.execute_pipeline(make_request())
        assert isinstance(result, RiskAnalysisResult)
        assert result.audit_status == AuditStatus.FAILED.value

    @pytest.mark.parametrize("exc", [RuntimeError("boom"), ConnectionError("offline")])
    async def test_any_vulnerability_exception_is_contained(self, exc: Exception) -> None:
        agent = CoordinatorAgent(
            satellite_service=StubSatellite(), vulnerability_provider=StubVulnerability(exc=exc)
        )
        result = await agent.execute_pipeline(make_request())
        assert result.audit_status == AuditStatus.FAILED.value

    async def test_result_survives_a_json_round_trip(self) -> None:
        agent = CoordinatorAgent(satellite_service=StubSatellite())
        result = await agent.execute_pipeline(make_request())
        restored = RiskAnalysisResult.model_validate_json(result.model_dump_json())
        assert restored == result

    async def test_extreme_bbox_is_accepted(self) -> None:
        agent = CoordinatorAgent(satellite_service=StubSatellite())
        result = await agent.execute_pipeline(
            make_request(bbox=[-180.0, -90.0, 180.0, 90.0])
        )
        assert isinstance(result, RiskAnalysisResult)


# ---------------------------------------------------------------------------
# Real component wiring
# ---------------------------------------------------------------------------


class TestRealComponentWiring:
    """The coordinator against the real SatelliteDataService, not a stub.

    Every other test here injects ``StubSatellite``, which proves the
    coordinator's logic but not that the two real classes agree on the shape of
    the document passing between them. HTTP is still mocked, so this runs
    offline, but ``SatelliteDataService`` executes for real.
    """

    def _service(self, payload: Any) -> SatelliteDataService:
        return SatelliteDataService(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
            band_reader=lambda href: BAND_SAMPLES[href],
        )

    async def test_full_pipeline_over_the_real_service(self) -> None:
        from tests.conftest import make_item, stac_response

        payload = stac_response([make_item("REAL_SCENE", cloud_cover=3.0)])
        agent = CoordinatorAgent(
            satellite_service=self._service(payload),
            vulnerability_provider=StubVulnerability(8_000.0, 0.8),
        )
        result = await agent.execute_pipeline(make_request())

        assert result.audit_status == AuditStatus.ESCALATED.value  # no thermal source
        properties = result.geojson_features["features"][0]["properties"]
        assert properties["stac_id"] == "REAL_SCENE"
        assert properties["cloud_cover"] == 3.0
        # NDVI comes from the pixels, not from a canned field.
        assert properties["ndvi_mean"] == pytest.approx(PIPELINE_NDVI)
        assert properties["factors"]["cooling_deficit"] == pytest.approx(PIPELINE_COOLING_DEFICIT)
        assert result.heps_score == pytest.approx(
            0.35 / 0.60 * PIPELINE_COOLING_DEFICIT + 0.25 / 0.60 * 0.8
        )

    async def test_full_pipeline_when_the_real_service_degrades(self) -> None:
        from tests.conftest import stac_response

        agent = CoordinatorAgent(
            satellite_service=self._service(stac_response([])),
            vulnerability_provider=StubVulnerability(8_000.0, 0.8),
        )
        result = await agent.execute_pipeline(make_request())
        assert result.audit_status == AuditStatus.ESCALATED.value
        properties = result.geojson_features["features"][0]["properties"]
        assert properties["stac_id"] == UNAVAILABLE_STAC_ID
        assert result.heps_score == pytest.approx(0.8)

    async def test_full_pipeline_when_the_api_is_unreachable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        service = SatelliteDataService(
            transport=httpx.MockTransport(handler), band_reader=lambda href: BAND_SAMPLES[href]
        )
        agent = CoordinatorAgent(
            satellite_service=service, vulnerability_provider=StubVulnerability(8_000.0, 0.8)
        )
        result = await agent.execute_pipeline(make_request())
        # The service degrades rather than raising, so the pipeline degrades too.
        assert result.audit_status == AuditStatus.ESCALATED.value
        assert result.heps_score == pytest.approx(0.8)
