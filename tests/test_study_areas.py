"""Study-area configuration, city switching and unavailable-data behaviour.

Everything here is offline. Rasters are synthetic and written to ``tmp_path``.
Nothing in this file proves that London is validated: it proves that the
configuration is consistent and that the pipeline withholds results when the
real inputs are missing or do not cover the requested area.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.agents.coordinator import CoordinatorAgent
from app.core.schemas import SpatialQueryRequest
from app.services.risk_service import EvidenceProvenance
from app.services.worldpop_provider import WorldPopExposureProvider
from tests.test_coordinator import StubSatellite

REPO = Path(__file__).resolve().parents[1]
STUDY_AREAS = json.loads((REPO / "config" / "study_areas.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((REPO / "config" / "vulnerability_sources.json").read_text(encoding="utf-8"))
AREAS = {area["id"]: area for area in STUDY_AREAS["areas"]}

RES = 1 / 1200  # 3 arc-seconds, the WorldPop 100 m grid
LONDON_BBOX = AREAS["london"]["bbox"]
LONDON_INNER_BOX = [-0.2, 51.45, -0.1, 51.55]  # well inside the London envelope
OUARGLA_BBOX = AREAS["ouargla"]["bbox"]
VALID_VALIDATION_STATES = {"not_validated", "not_started", "validated"}
VALID_FILE_STATES = {"not_downloaded", "downloaded_unvalidated", "validated"}


def _raster(path: Path, bounds: tuple[float, float, float, float], value: float = 100.0) -> Path:
    """Synthetic EPSG:4326 raster at the 100 m grid covering ``bounds``.

    ``bounds`` is ``(west, south, east, north)``.
    """
    west, south, east, north = bounds
    width = round((east - west) / RES)
    height = round((north - south) / RES)
    data = np.full((1, height, width), value, dtype="float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(west, north, RES, RES),
    ) as destination:
        destination.write(data)
    return path


def _london_raster(tmp_path: Path) -> Path:
    west, south, east, north = LONDON_BBOX
    return _raster(tmp_path / "london.tif", (west, south, east, north))


def _ouargla_raster(tmp_path: Path) -> Path:
    return _raster(tmp_path / "ouargla.tif", (4.9, 31.5, 5.8, 32.4))


# --- Registry -----------------------------------------------------------------


def test_default_area_is_london_and_is_the_primary_pilot() -> None:
    assert STUDY_AREAS["default_area"] == "london"
    assert AREAS["london"]["role"] == "primary_pilot"
    assert AREAS["london"]["bbox"] is not None


@pytest.mark.parametrize("area_id", [a["id"] for a in STUDY_AREAS["areas"] if a["bbox"] is not None])
def test_every_configured_bbox_passes_the_api_request_validator(area_id: str) -> None:
    # The same validator the API applies to every request, so a preset can never
    # be one the backend rejects.
    west, south, east, north = AREAS[area_id]["bbox"]
    request = SpatialQueryRequest(
        query="configuration check",
        bbox=[west, south, east, north],
        start_date="2024-06-01",
        end_date="2024-06-30",
    )
    assert request.bbox == [west, south, east, north]


@pytest.mark.parametrize("area_id", ["amsterdam", "rotterdam"])
def test_second_and_third_cities_are_unavailable_until_their_extent_is_verified(area_id: str) -> None:
    area = AREAS[area_id]
    assert area["bbox"] is None
    assert area["unavailable_reason"]
    assert area["validation"] != "validated"


def test_ouargla_is_kept_as_an_explicit_demo_and_is_not_the_pilot() -> None:
    area = AREAS["ouargla"]
    assert area["role"] == "demo_legacy"
    assert area["bbox"] == [5.1, 31.7, 5.6, 32.2]
    assert "demo" in area["label"].lower()
    assert area["validation"] != "validated"


def test_no_area_is_marked_validated_by_configuration_alone() -> None:
    # Validation must come from evidence (a real raster, checksum and run), not from a
    # config edit. Until a validation record exists, every area must say so.
    for area in STUDY_AREAS["areas"]:
        assert area["validation"] in VALID_VALIDATION_STATES, area["id"]
        if area["validation"] == "validated":
            pytest.fail(f"{area['id']} is marked validated without a validation record")


def test_london_envelope_is_not_described_as_the_city_boundary() -> None:
    basis = AREAS["london"]["bbox_basis"].lower()
    assert "not verified" in basis
    assert "boundary" in basis


def test_manifest_entries_reference_known_areas_and_use_no_placeholder_hashes() -> None:
    manifest_areas: set[str] = set()
    for dataset in MANIFEST["datasets"].values():
        for area_id, entry in dataset.get("areas", {}).items():
            manifest_areas.add(area_id)
            assert area_id in AREAS
            assert entry["validation_status"] in VALID_FILE_STATES
            checksum = entry["sha256"]
            assert checksum is None or re.fullmatch(r"[0-9a-f]{64}", checksum)
            assert entry["url"].startswith("https://")
    assert {"london", "amsterdam", "rotterdam"} <= manifest_areas
    assert "ouargla" not in manifest_areas  # no raster is mapped for the demo area


def test_worldpop_global2_is_recorded_as_cc_by_with_the_reference_year() -> None:
    dataset = MANIFEST["datasets"]["worldpop_global2_r2025a_100m_2020_constrained"]
    assert dataset["licence"] == "CC BY 4.0"
    assert dataset["reference_year"] == 2020
    assert dataset["resolution_m"] == 100
    assert dataset["common_across_pilot_cities"] is True


# --- Manifest-to-area resolution (fail closed) ---------------------------------


@pytest.fixture(scope="module")
def fetch_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fetch_worldpop_for_areas", REPO / "tools" / "vulnerability" / "fetch_worldpop.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_london_resolves_to_the_gbr_file(fetch_script) -> None:
    key, _dataset, entry, study = fetch_script.resolve_area("london")
    assert key == "worldpop_global2_r2025a_100m_2020_constrained"
    assert entry["iso3"] == "GBR"
    assert entry["filename"] == "gbr_pop_2020_CN_100m_R2025A_v1.tif"
    assert study["id"] == "london"


@pytest.mark.parametrize(
    ("area_id", "fragment"),
    [
        ("amsterdam", "no verified bounding box"),
        ("ouargla", "no population raster is listed"),
        ("atlantis", "unknown study area"),
    ],
)
def test_unresolvable_areas_fail_closed(fetch_script, area_id: str, fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        fetch_script.resolve_area(area_id)


def test_a_bbox_with_no_manifest_entry_fails_closed(fetch_script, monkeypatch) -> None:
    # A configured bbox but no raster mapping must not borrow another city's file.
    areas = json.loads(json.dumps(STUDY_AREAS))
    areas["areas"].append(
        {"id": "testcity", "label": "Test", "bbox": [0.0, 0.0, 0.1, 0.1], "validation": "not_started"}
    )
    with pytest.raises(ValueError, match="no population raster is listed"):
        fetch_script.resolve_area("testcity", manifest=MANIFEST, areas=areas)


# --- City switching: real coverage decides, not the city name -------------------


@pytest.mark.asyncio
async def test_london_bbox_with_a_london_raster_gives_verified_exposure(tmp_path: Path) -> None:
    context = await WorldPopExposureProvider(_london_raster(tmp_path)).fetch(LONDON_INNER_BOX)
    assert context.provenance == EvidenceProvenance.VERIFIED
    assert context.population_density is not None and context.population_density > 0
    assert "population exposure only" in context.source


@pytest.mark.asyncio
async def test_ouargla_bbox_is_refused_by_a_london_raster(tmp_path: Path) -> None:
    context = await WorldPopExposureProvider(_london_raster(tmp_path)).fetch(OUARGLA_BBOX)
    assert context.provenance == EvidenceProvenance.INVALID
    assert context.population_density is None


@pytest.mark.asyncio
async def test_london_bbox_is_refused_by_an_algeria_raster(tmp_path: Path) -> None:
    # Guards against reusing the Algeria raster for London.
    context = await WorldPopExposureProvider(_ouargla_raster(tmp_path)).fetch(LONDON_BBOX)
    assert context.provenance == EvidenceProvenance.INVALID
    assert context.population_density is None


@pytest.mark.asyncio
async def test_a_london_raster_that_covers_only_half_the_area_is_refused(tmp_path: Path) -> None:
    west, south, east, north = LONDON_BBOX
    half = _raster(tmp_path / "half.tif", (west, south, (west + east) / 2, north))
    context = await WorldPopExposureProvider(half).fetch(LONDON_BBOX)
    assert context.provenance == EvidenceProvenance.INVALID
    assert context.population_density is None


@pytest.mark.asyncio
async def test_missing_london_raster_is_reported_as_missing_not_zero(tmp_path: Path) -> None:
    context = await WorldPopExposureProvider(tmp_path / "absent.tif").fetch(LONDON_INNER_BOX)
    assert context.provenance == EvidenceProvenance.MISSING
    assert context.population_density is None


# --- Audit propagation: unavailable London evidence withholds the score ---------


@pytest.mark.asyncio
async def test_london_query_without_population_raster_withholds_the_score() -> None:
    agent = CoordinatorAgent(satellite_service=StubSatellite())
    request = SpatialQueryRequest(
        query="London heat exposure pilot",
        bbox=LONDON_BBOX,
        start_date="2024-06-01",
        end_date="2024-06-30",
    )
    result = await agent.execute_pipeline(request)
    assert result.heps_score is None
    assert result.grid_stress_level is None
    assert result.evidence_status != "complete"
    demographics = next(e for e in result.evidence if e.source == "demographics")
    assert demographics.provenance == EvidenceProvenance.MISSING.value


@pytest.mark.asyncio
async def test_london_exposure_alone_never_yields_a_complete_score(tmp_path: Path) -> None:
    from app.services.worldpop_provider import WorldPopExposureProvider as Provider

    agent = CoordinatorAgent(
        satellite_service=StubSatellite(),
        vulnerability_provider=Provider(_london_raster(tmp_path)),
    )
    request = SpatialQueryRequest(
        query="London exposure only",
        bbox=LONDON_INNER_BOX,
        start_date="2024-06-01",
        end_date="2024-06-30",
    )
    result = await agent.execute_pipeline(request)
    assert result.heps_score is None
    assert result.evidence_status != "complete"
    demographics = next(e for e in result.evidence if e.source == "demographics")
    assert demographics.provenance == EvidenceProvenance.VERIFIED.value
    assert "social vulnerability not provided" in (demographics.detail or "")
