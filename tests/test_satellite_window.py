"""AOI-windowed reads, footprint coverage and water/snow policy on real GeoTIFFs.

These tests write small **synthetic** Sentinel-2-like rasters to ``tmp_path`` in
UTM (EPSG:32631, the zone of Ouargla) and read them through the production
rasterio code path. They need no network and no real product.

Synthetic layout, 10 m grid, 1000 x 1000 px (10 km square):

* left half (columns 0-499): vegetation, reflectance red 0.1, NIR 0.3 (NDVI 0.5),
  SCL class 4 (vegetation).
* right half: cloud (SCL 9), reflectance red 0.5, NIR 0.2 (NDVI about -0.43).
  Its values are deliberately different so a read that ignores the AOI or the mask
  shows up as a wrong mean.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from app.services.satellite_service import (
    NDVI_NO_DATA,
    BandSample,
    OutsideSceneError,
    SatelliteDataService,
    compute_ndvi,
    read_band_sample,
    read_scl_sample,
    scl_invalid_mask,
)

UTM = "EPSG:32631"
SIZE = 1000  # 10 m pixels -> 10 km
ORIGIN_X = 500_000.0  # UTM easting of the west edge (synthetic)
ORIGIN_Y = 3_550_000.0  # UTM northing of the north edge (synthetic)
HALF = SIZE // 2
DN_OFFSET = 1000  # BOA_ADD_OFFSET = -1000 means DN = reflectance * 10000 + 1000


def _dn(rho: float) -> int:
    return round(rho * 10000) + DN_OFFSET


def _bands() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    red = np.empty((SIZE, SIZE), dtype="uint16")
    nir = np.empty((SIZE, SIZE), dtype="uint16")
    scl = np.empty((SIZE // 2, SIZE // 2), dtype="uint8")
    red[:, :HALF], nir[:, :HALF] = _dn(0.1), _dn(0.3)
    red[:, HALF:], nir[:, HALF:] = _dn(0.5), _dn(0.2)
    scl[:, : HALF // 2], scl[:, HALF // 2 :] = 4, 9
    return red, nir, scl


def _write(path: Path, array: np.ndarray, pixel: float) -> Path:
    transform = from_origin(ORIGIN_X, ORIGIN_Y, pixel, pixel)
    height, width = array.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=array.dtype,
        crs=UTM,
        transform=transform,
        nodata=0 if array.dtype != "uint8" else None,
    ) as destination:
        destination.write(array, 1)
    return path


@pytest.fixture
def scene(tmp_path: Path) -> dict[str, Path]:
    red, nir, scl = _bands()
    return {
        "B04": _write(tmp_path / "B04_10m.tif", red, 10.0),
        "B08": _write(tmp_path / "B08_10m.tif", nir, 10.0),
        "SCL": _write(tmp_path / "SCL_20m.tif", scl, 20.0),
    }


def _lonlat(west: float, south: float, east: float, north: float) -> list[float]:
    """Convert a UTM rectangle to an EPSG:4326 bbox, as the service receives it."""
    left, bottom, right, top = transform_bounds(UTM, "EPSG:4326", west, south, east, north)
    return [left, bottom, right, top]


LEFT_BBOX = _lonlat(ORIGIN_X, ORIGIN_Y - SIZE * 10.0, ORIGIN_X + HALF * 10.0, ORIGIN_Y)
RIGHT_BBOX = _lonlat(ORIGIN_X + HALF * 10.0, ORIGIN_Y - SIZE * 10.0, ORIGIN_X + SIZE * 10.0, ORIGIN_Y)
WHOLE_BBOX = _lonlat(ORIGIN_X, ORIGIN_Y - SIZE * 10.0, ORIGIN_X + SIZE * 10.0, ORIGIN_Y)
FAR_BBOX = [10.0, 31.0, 10.5, 31.5]  # well outside the synthetic scene


def _ndvi(scene: dict[str, Path], bbox: list[float] | None) -> tuple[float | None, float]:
    red = read_band_sample(str(scene["B04"]), max_sample_pixels=10_000_000, bbox=bbox)
    nir = read_band_sample(str(scene["B08"]), max_sample_pixels=10_000_000, bbox=bbox)
    scl = read_scl_sample(str(scene["SCL"]), bbox=bbox)
    invalid = scl_invalid_mask(scl.values, np.shape(red.values))
    result = compute_ndvi(
        red,
        nir,
        reflectance_offset=-1000.0,
        quantification_value=10000.0,
        scl_invalid=invalid,
    )
    assert result.valid_coverage is not None
    return result.mean, result.valid_coverage


class TestAoiWindow:
    def test_aoi_read_contains_only_requested_window(self, scene: dict[str, Path]) -> None:
        sample = read_band_sample(str(scene["B04"]), max_sample_pixels=10_000_000, bbox=LEFT_BBOX)
        values = np.asarray(sample.values, dtype="float64")
        # Only the vegetation half is inside the AOI, so every sample is its DN.
        assert np.allclose(values, _dn(0.1), atol=1)
        assert sample.footprint_fraction == pytest.approx(1.0, abs=0.01)

    def test_whole_scene_read_is_unchanged_when_no_bbox(self, scene: dict[str, Path]) -> None:
        sample = read_band_sample(str(scene["B04"]), max_sample_pixels=10_000_000)
        assert sample.footprint_fraction == 1.0
        assert np.asarray(sample.values).shape == (SIZE, SIZE)

    def test_decimation_budget_applies_to_the_window(self, scene: dict[str, Path]) -> None:
        sample = read_band_sample(str(scene["B04"]), max_sample_pixels=16, bbox=WHOLE_BBOX)
        assert np.asarray(sample.values).size <= 16 * 4  # within a small factor of budget

    def test_aoi_ndvi_on_clear_half_is_not_diluted_by_the_scene(
        self, scene: dict[str, Path]
    ) -> None:
        mean, coverage = _ndvi(scene, LEFT_BBOX)
        assert mean == pytest.approx(0.5, abs=1e-3)
        assert coverage == pytest.approx(1.0, abs=0.01)

    def test_whole_scene_request_reports_cloud_as_lost_coverage(
        self, scene: dict[str, Path]
    ) -> None:
        mean, coverage = _ndvi(scene, WHOLE_BBOX)
        # Cloud covers half the scene, so usable coverage is about 0.5 and the
        # mean is taken over the clear half only.
        assert mean == pytest.approx(0.5, abs=1e-3)
        assert coverage == pytest.approx(0.5, abs=0.02)

    def test_mixed_request_has_coverage_below_default_threshold(
        self, scene: dict[str, Path]
    ) -> None:
        _, coverage = _ndvi(scene, WHOLE_BBOX)
        assert coverage < 0.9

    def test_aoi_partly_outside_scene_scales_coverage(self, scene: dict[str, Path]) -> None:
        # Extend the clear half 5 km west, beyond the scene edge: half the AOI is
        # outside the raster and must not count as usable.
        extended = _lonlat(
            ORIGIN_X - HALF * 10.0,
            ORIGIN_Y - SIZE * 10.0,
            ORIGIN_X + HALF * 10.0,
            ORIGIN_Y,
        )
        sample = read_band_sample(str(scene["B04"]), max_sample_pixels=10_000_000, bbox=extended)
        assert sample.footprint_fraction == pytest.approx(0.5, abs=0.02)
        _, coverage = _ndvi(scene, extended)
        assert coverage == pytest.approx(0.5, abs=0.02)

    def test_aoi_outside_scene_raises_outside_error(self, scene: dict[str, Path]) -> None:
        with pytest.raises(OutsideSceneError):
            read_band_sample(str(scene["B04"]), bbox=FAR_BBOX)
        with pytest.raises(OutsideSceneError):
            read_scl_sample(str(scene["SCL"]), bbox=FAR_BBOX)


class TestWaterAndSnowPolicy:
    def test_water_pixels_are_excluded_from_mean_and_coverage(self, tmp_path: Path) -> None:
        red, nir, scl = _bands()
        # Make a 100-column strip of the clear half open water (SCL 6). Water
        # reflects so that its NDVI is negative; it must not enter the mean.
        strip_cols = slice(0, HALF // 5)  # 100 px at 10 m, 50 px at 20 m SCL
        red[:, strip_cols] = _dn(0.05)
        nir[:, strip_cols] = _dn(0.02)
        scl[:, : HALF // 10] = 6
        paths = {
            "B04": _write(tmp_path / "B04_10m.tif", red, 10.0),
            "B08": _write(tmp_path / "B08_10m.tif", nir, 10.0),
            "SCL": _write(tmp_path / "SCL_20m.tif", scl, 20.0),
        }
        mean, coverage = _ndvi(paths, LEFT_BBOX)
        assert mean == pytest.approx(0.5, abs=1e-3)  # water excluded, not averaged in
        assert coverage == pytest.approx(0.8, abs=0.02)  # 100 of 500 clear columns are water


def _service(item: dict) -> SatelliteDataService:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"type": "FeatureCollection", "features": [item]})

    return SatelliteDataService(
        transport=httpx.MockTransport(handler),
        min_valid_coverage=0.9,
        max_sample_pixels=100_000,
    )


def _item(scene: dict[str, Path]) -> dict:
    return {
        "id": "S2A_MSIL2A_20240601T100000_N0511_R022_T31SGA_20240601T120000",
        "properties": {"eo:cloud_cover": 5.0, "s2:processing_baseline": "05.11"},
        "assets": {
            "B04_10m": {"href": str(scene["B04"])},
            "B08_10m": {"href": str(scene["B08"])},
            "SCL_20m": {"href": str(scene["SCL"])},
        },
    }


class TestServiceEndToEndOnRasters:
    """Default production readers, real local rasters, stubbed STAC search."""

    async def test_clear_aoi_returns_ndvi_with_coverage(self, scene: dict[str, Path]) -> None:
        service = _service(_item(scene))
        async with service:
            layer = await service.fetch_sentinel_indices(
                bbox=LEFT_BBOX, start_date="2024-06-01", end_date="2024-06-30"
            )
        assert layer["ndvi_mean"] == pytest.approx(0.5, abs=1e-3)
        assert layer["ndvi_valid_coverage"] == pytest.approx(1.0, abs=0.01)
        assert layer["reflectance_offset"] == -1000.0

    async def test_partly_cloudy_aoi_withholds_ndvi(self, scene: dict[str, Path]) -> None:
        service = _service(_item(scene))
        async with service:
            layer = await service.fetch_sentinel_indices(
                bbox=WHOLE_BBOX, start_date="2024-06-01", end_date="2024-06-30"
            )
        assert layer["ndvi_mean"] == NDVI_NO_DATA
        assert layer["ndvi_valid_coverage"] == pytest.approx(0.5, abs=0.02)

    async def test_aoi_outside_scene_reports_zero_coverage(self, scene: dict[str, Path]) -> None:
        service = _service(_item(scene))
        async with service:
            layer = await service.fetch_sentinel_indices(
                bbox=FAR_BBOX, start_date="2024-06-01", end_date="2024-06-30"
            )
        assert layer["ndvi_mean"] == NDVI_NO_DATA
        assert layer["ndvi_valid_coverage"] == 0.0


def test_band_sample_defaults_to_full_footprint() -> None:
    sample = BandSample(values=np.zeros((2, 2)))
    assert sample.footprint_fraction == 1.0
