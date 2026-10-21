"""Tests for the WorldPop population-exposure provider.

Synthetic rasters (written to ``tmp_path``) cover the arithmetic and every
fail-closed branch. The only test that touches real data is the smoke test at the
bottom, which runs only when ``GEOSENTINEL_WORLDPOP_RASTER`` points at a
downloaded WorldPop GeoTIFF and skips otherwise. Nothing is downloaded here.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.agents.coordinator import CoordinatorAgent
from app.core.schemas import SpatialQueryRequest
from app.services.risk_service import (
    EvidenceProvenance,
    VulnerabilityContext,
    compute_heps,
)
from app.services.worldpop_provider import (
    WORLDPOP_RESOLUTION_DEG,
    WorldPopExposureProvider,
    spherical_bbox_area_km2,
)

RES = WORLDPOP_RESOLUTION_DEG  # 1/1200 deg, the 100 m product
LON0, LAT_TOP = 5.0, 31.92  # synthetic origin near Ouargla
SIZE = 24  # 24 cells = 0.02 deg
BOX = [LON0, LAT_TOP - SIZE * RES, LON0 + SIZE * RES, LAT_TOP]  # the whole raster
UNIFORM = 2.0  # people per cell


def _write(
    path: Path,
    data: np.ndarray,
    *,
    crs: str = "EPSG:4326",
    res: float = RES,
    nodata: float | None = None,
) -> Path:
    height, width = data.shape[-2:]
    count = 1 if data.ndim == 2 else data.shape[0]
    arr = data if data.ndim == 3 else data[np.newaxis]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=arr.dtype,
        crs=crs,
        transform=from_origin(LON0, LAT_TOP, res, res),
        nodata=nodata,
    ) as destination:
        destination.write(arr)
    return path


@pytest.fixture
def uniform_raster(tmp_path: Path) -> Path:
    """Synthetic 24x24 raster, 2 people in every cell."""
    return _write(tmp_path / "synthetic_uniform.tif", np.full((SIZE, SIZE), UNIFORM, "float32"))


def _reason(context: VulnerabilityContext) -> str:
    return context.source


class TestArithmetic:
    def test_sphere_area_of_one_degree_square_at_equator(self) -> None:
        # Known value: about 12,364 km² for a 1°x1° cell at the equator.
        assert spherical_bbox_area_km2([0.0, 0.0, 1.0, 1.0]) == pytest.approx(12364, rel=1e-3)

    @pytest.mark.asyncio
    async def test_uniform_raster_density_is_total_over_area(self, uniform_raster: Path) -> None:
        context = await WorldPopExposureProvider(uniform_raster).fetch(BOX)
        expected_total = UNIFORM * SIZE * SIZE
        expected = expected_total / spherical_bbox_area_km2(BOX)
        assert context.provenance is EvidenceProvenance.VERIFIED
        assert context.population_density == pytest.approx(expected, rel=1e-9)

    @pytest.mark.asyncio
    async def test_half_cell_offset_uses_fractional_overlap(self, uniform_raster: Path) -> None:
        # Shift the AOI by half a cell in each direction, so every edge cell is
        # partly covered. Uniform counts make the expected total exact: the AOI
        # covers (dlon/res) * (dlat/res) cells' worth of population.
        half = RES / 2
        box = [LON0 + half, LAT_TOP - SIZE * RES + half, LON0 + SIZE * RES - half, LAT_TOP - half]
        context = await WorldPopExposureProvider(uniform_raster).fetch(box)
        cells = ((box[2] - box[0]) / RES) * ((box[3] - box[1]) / RES)
        expected = UNIFORM * cells / spherical_bbox_area_km2(box)
        assert context.population_density == pytest.approx(expected, rel=1e-9)

    @pytest.mark.asyncio
    async def test_context_is_population_exposure_only(self, uniform_raster: Path) -> None:
        context = await WorldPopExposureProvider(uniform_raster).fetch(BOX)
        assert context.social_vulnerability is None
        assert not context.is_scoreable
        assert "CC BY 4.0" in context.source
        assert "population exposure only" in context.source
        assert "social vulnerability not provided" in context.source


class TestFailClosed:
    @pytest.mark.asyncio
    async def test_missing_file_is_missing(self, tmp_path: Path) -> None:
        context = await WorldPopExposureProvider(tmp_path / "absent.tif").fetch(BOX)
        assert context.provenance is EvidenceProvenance.MISSING
        assert context.population_density is None
        assert "not found" in _reason(context)

    @pytest.mark.asyncio
    async def test_unreadable_file_is_invalid_not_placeholder(self, tmp_path: Path) -> None:
        bad = tmp_path / "not_a_raster.tif"
        bad.write_bytes(b"this is not a tiff")
        context = await WorldPopExposureProvider(bad).fetch(BOX)
        assert context.provenance is EvidenceProvenance.INVALID
        assert context.population_density is None

    @pytest.mark.asyncio
    async def test_wrong_crs_is_rejected_not_reprojected(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "utm.tif",
            np.full((SIZE, SIZE), UNIFORM, "float32"),
            crs="EPSG:32631",
            res=RES,
        )
        context = await WorldPopExposureProvider(path).fetch(BOX)
        assert context.provenance is EvidenceProvenance.INVALID
        assert "EPSG:4326" in _reason(context)

    @pytest.mark.asyncio
    async def test_wrong_resolution_is_rejected(self, tmp_path: Path) -> None:
        # A 1 km product must not be read as if it were the 100 m product.
        path = _write(tmp_path / "km.tif", np.full((SIZE, SIZE), UNIFORM, "float32"), res=1 / 120)
        context = await WorldPopExposureProvider(path).fetch(BOX)
        assert context.provenance is EvidenceProvenance.INVALID
        assert "pixel size" in _reason(context)

    @pytest.mark.asyncio
    async def test_multi_band_is_rejected(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "two_band.tif",
            np.full((2, SIZE, SIZE), UNIFORM, "float32"),
        )
        context = await WorldPopExposureProvider(path).fetch(BOX)
        assert context.provenance is EvidenceProvenance.INVALID
        assert "one band" in _reason(context)

    @pytest.mark.asyncio
    async def test_partial_aoi_is_not_silently_understated(self, uniform_raster: Path) -> None:
        # Extend the AOI 0.01 deg west of the raster: half of it has no data.
        box = [LON0 - 0.01, BOX[1], LON0 + 0.01, BOX[3]]
        context = await WorldPopExposureProvider(uniform_raster).fetch(box)
        assert context.provenance is EvidenceProvenance.INVALID
        assert context.population_density is None
        assert "not fully covered" in _reason(context)

    @pytest.mark.asyncio
    async def test_aoi_outside_raster_is_invalid(self, uniform_raster: Path) -> None:
        context = await WorldPopExposureProvider(uniform_raster).fetch([10.0, 31.0, 10.1, 31.1])
        assert context.provenance is EvidenceProvenance.INVALID
        assert context.population_density is None

    @pytest.mark.asyncio
    async def test_nodata_inside_aoi_is_invalid(self, tmp_path: Path) -> None:
        data = np.full((SIZE, SIZE), UNIFORM, "float32")
        data[10, 10] = -9999.0
        path = _write(tmp_path / "holes.tif", data, nodata=-9999.0)
        context = await WorldPopExposureProvider(path).fetch(BOX)
        assert context.provenance is EvidenceProvenance.INVALID
        assert "nodata" in _reason(context)

    @pytest.mark.asyncio
    async def test_nodata_outside_aoi_does_not_block(self, tmp_path: Path) -> None:
        data = np.full((SIZE, SIZE), UNIFORM, "float32")
        data[0, 0] = -9999.0  # top-left cell, which the AOI below excludes
        path = _write(tmp_path / "far_hole.tif", data, nodata=-9999.0)
        box = [LON0 + 5 * RES, LAT_TOP - SIZE * RES, LON0 + SIZE * RES, LAT_TOP - 5 * RES]
        context = await WorldPopExposureProvider(path).fetch(box)
        assert context.provenance is EvidenceProvenance.VERIFIED

    @pytest.mark.asyncio
    async def test_negative_counts_are_rejected(self, tmp_path: Path) -> None:
        data = np.full((SIZE, SIZE), UNIFORM, "float32")
        data[3, 3] = -1.0
        path = _write(tmp_path / "negative.tif", data)
        context = await WorldPopExposureProvider(path).fetch(BOX)
        assert context.provenance is EvidenceProvenance.INVALID
        assert "negative" in _reason(context)

    @pytest.mark.asyncio
    async def test_malformed_bbox_is_invalid(self, uniform_raster: Path) -> None:
        context = await WorldPopExposureProvider(uniform_raster).fetch([5.0, 32.0, 5.1, 31.0])
        assert context.provenance is EvidenceProvenance.INVALID
        assert "malformed" in _reason(context)


class TestCompositeStaysWithheld:
    @pytest.mark.asyncio
    async def test_population_only_context_cannot_produce_heps(self, uniform_raster: Path) -> None:
        context = await WorldPopExposureProvider(uniform_raster).fetch(BOX)
        assessment = compute_heps(lst_celsius=40.0, ndvi=0.2, vulnerability=context)
        assert assessment.heps_score is None
        assert assessment.stress_level is None
        assert assessment.data_complete is False

    @pytest.mark.asyncio
    async def test_coordinator_withholds_score_with_population_only(
        self, uniform_raster: Path
    ) -> None:
        from tests.test_coordinator import StubSatellite

        provider = WorldPopExposureProvider(uniform_raster)
        agent = CoordinatorAgent(satellite_service=StubSatellite(), vulnerability_provider=provider)
        request = SpatialQueryRequest(
            query="population exposure check",
            bbox=BOX,
            start_date="2024-06-01",
            end_date="2024-06-30",
        )
        result = await agent.execute_pipeline(request)
        assert result.heps_score is None
        assert result.grid_stress_level is None
        assert result.evidence_status != "complete"
        demographics = next(e for e in result.evidence if e.source == "demographics")
        assert demographics.provenance == EvidenceProvenance.VERIFIED.value
        assert "population exposure only" in demographics.detail


SMOKE_PATH = os.environ.get("GEOSENTINEL_WORLDPOP_RASTER", "")
OUARGLA_BBOX = [5.1, 31.7, 5.6, 32.2]  # the frontend's Ouargla preset


@pytest.mark.skipif(
    not SMOKE_PATH or not Path(SMOKE_PATH).is_file(),
    reason="set GEOSENTINEL_WORLDPOP_RASTER to a downloaded WorldPop GeoTIFF to run",
)
@pytest.mark.asyncio
async def test_real_worldpop_smoke_over_ouargla() -> None:
    """Real-data smoke test. Runs only where a downloaded WorldPop raster exists."""
    context = await WorldPopExposureProvider(SMOKE_PATH).fetch(OUARGLA_BBOX)
    assert context.provenance is EvidenceProvenance.VERIFIED, context.source
    assert context.population_density is not None
    assert math.isfinite(context.population_density)
    assert context.population_density >= 0
    assert context.social_vulnerability is None
