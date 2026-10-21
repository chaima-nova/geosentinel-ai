"""Landsat C2 L2 ST_B10 decoder tests. Synthetic data only; no real scene was read.

The USGS worked example (DN 44947 -> 302.6 K) is used as the conversion reference.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.services.landsat_lst_service import (
    KELVIN_OFFSET,
    ST_FILL_VALUE,
    ST_VALID_MIN_DN,
    decode_st_b10,
    read_lst_sample,
    summarise_lst,
)

ACQUIRED = date(2024, 6, 15)


def test_usgs_worked_example_converts_to_kelvin_and_celsius() -> None:
    celsius = decode_st_b10(np.array([44947], dtype=np.uint16))
    assert celsius[0] == pytest.approx(302.6 - KELVIN_OFFSET, abs=0.05)


def test_fill_value_and_dn_below_valid_minimum_are_nan_not_zero() -> None:
    dn = np.array(
        [ST_FILL_VALUE, ST_VALID_MIN_DN - 1, ST_VALID_MIN_DN], dtype=np.uint16
    )
    celsius = decode_st_b10(dn)
    assert np.isnan(celsius[0])
    assert np.isnan(celsius[1])
    assert np.isfinite(celsius[2])


def test_collection_one_fill_value_is_rejected_not_converted() -> None:
    # Collection 1 used -9999 as fill. Under Collection 2 rules it is invalid.
    celsius = decode_st_b10(np.array([-9999], dtype=np.int16))
    assert np.isnan(celsius[0])


def test_float_input_is_refused() -> None:
    with pytest.raises(ValueError, match="integer array"):
        decode_st_b10(np.array([44947.0]))


def test_input_is_not_modified() -> None:
    dn = np.array([0, 44947], dtype=np.uint16)
    before = dn.copy()
    decode_st_b10(dn)
    assert np.array_equal(dn, before)


def test_complete_coverage_gives_a_mean_and_states_clouds_are_unscreened() -> None:
    sample = summarise_lst(np.full((10, 10), 44947, dtype=np.uint16), acquired=ACQUIRED)
    assert sample.mean_c is not None
    assert sample.valid_fraction == 1.0
    assert sample.qa_mask_applied is False
    assert "clouds are not screened" in sample.reason


def test_low_coverage_withholds_the_mean_and_keeps_the_extremes() -> None:
    dn = np.full((10, 10), 44947, dtype=np.uint16)
    dn[:5, :] = 0  # half fill
    sample = summarise_lst(dn, acquired=ACQUIRED, min_valid_fraction=0.9)
    assert sample.mean_c is None
    assert sample.valid_fraction == pytest.approx(0.5)
    assert sample.min_c is not None and sample.max_c is not None
    assert "mean withheld" in sample.reason


def test_all_fill_area_has_no_mean_rather_than_zero() -> None:
    sample = summarise_lst(np.zeros((4, 4), dtype=np.uint16), acquired=ACQUIRED)
    assert sample.mean_c is None
    assert sample.valid_fraction == 0.0
    assert sample.reason == "no valid pixels"


def test_empty_area_is_reported_as_empty() -> None:
    sample = summarise_lst(np.zeros((0, 0), dtype=np.uint16), acquired=ACQUIRED)
    assert sample.mean_c is None and sample.reason == "empty area"


def test_threshold_must_be_a_fraction() -> None:
    with pytest.raises(ValueError):
        summarise_lst(
            np.ones((2, 2), dtype=np.uint16), acquired=ACQUIRED, min_valid_fraction=1.5
        )


def _st_raster(
    path: Path, bounds: tuple[float, float, float, float], crs: str = "EPSG:4326"
) -> Path:
    west, south, east, north = bounds
    res = 0.001
    width = round((east - west) / res)
    height = round((north - south) / res)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint16",
        crs=crs,
        transform=from_origin(west, north, res, res),
    ) as destination:
        destination.write(np.full((height, width), 44947, dtype=np.uint16), 1)
    return path


def test_reads_a_synthetic_raster_over_a_contained_bbox(tmp_path: Path) -> None:
    raster = _st_raster(tmp_path / "st.tif", (-1.0, 51.0, 1.0, 52.0))
    sample = read_lst_sample(raster, acquired=ACQUIRED, bbox=[-0.2, 51.4, -0.1, 51.5])
    assert sample.mean_c == pytest.approx(302.6 - KELVIN_OFFSET, abs=0.05)


def test_refuses_a_raster_that_does_not_contain_the_bbox(tmp_path: Path) -> None:
    raster = _st_raster(tmp_path / "st.tif", (-0.2, 51.0, 1.0, 52.0))
    with pytest.raises(ValueError, match="does not fully contain"):
        read_lst_sample(raster, acquired=ACQUIRED, bbox=[-0.5, 51.4, -0.1, 51.5])


def test_refuses_a_raster_that_is_not_wgs84(tmp_path: Path) -> None:
    raster = _st_raster(tmp_path / "st.tif", (-1.0, 51.0, 1.0, 52.0), crs="EPSG:27700")
    with pytest.raises(ValueError, match="expected EPSG:4326"):
        read_lst_sample(raster, acquired=ACQUIRED, bbox=[-0.2, 51.4, -0.1, 51.5])
