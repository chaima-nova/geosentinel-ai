"""Landsat Collection 2 Level-2 land-surface temperature (ST_B10) decoder.

Implemented but NOT validated against real London data in this environment: the
USGS/Earth Search endpoints were unreachable, so every test here uses synthetic
arrays or synthetic rasters. Nothing in this module is wired into the risk score.

Verified conventions (USGS "How do I use a scale factor with Landsat Level-2
science products?" and the Collection 2 Level-2 Science Product Guide):

* ST = DN * 0.00341802 + 149.0, in kelvin, for ST_B10 (unsigned 16-bit).
* Fill value is 0. The USGS FAQ gives a valid range of 293-65535 DN.
* Collection 1 used -9999 and a different scale. Collection 1 data is refused.

Not implemented: QA_PIXEL cloud, cloud-shadow and snow masking. The bit positions
were not confirmed from an official table in this session, so
``require_qa_mask`` is always ``True`` and every call reports that cloud
contamination is unscreened, and no result from this module is "validated" on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "KELVIN_OFFSET",
    "ST_FILL_VALUE",
    "ST_OFFSET_K",
    "ST_SCALE",
    "ST_VALID_MIN_DN",
    "LstSample",
    "decode_st_b10",
    "read_lst_sample",
    "summarise_lst",
]

ST_SCALE = 0.00341802
ST_OFFSET_K = 149.0
ST_FILL_VALUE = 0
ST_VALID_MIN_DN = 293
KELVIN_OFFSET = 273.15


@dataclass(frozen=True)
class LstSample:
    """Summary of valid land-surface temperature over one area.

    Attributes:
        acquired: Scene acquisition date from the scene metadata (not inferred).
        mean_c: Mean of valid pixels in degrees Celsius, or ``None`` when coverage
            is below the threshold. Never 0 for missing data.
        valid_fraction: Share of the area's pixels that decoded to a valid temperature.
        min_c: Minimum valid temperature in Celsius, or ``None``.
        max_c: Maximum valid temperature in Celsius, or ``None``.
        qa_mask_applied: Always ``False`` in this version (see module docstring).
        reason: Why ``mean_c`` is withheld, or a note on the result.
    """

    acquired: date
    mean_c: float | None
    valid_fraction: float
    min_c: float | None
    max_c: float | None
    qa_mask_applied: bool
    reason: str


def decode_st_b10(dn: np.ndarray) -> np.ndarray:
    """Convert ST_B10 digital numbers to Celsius; invalid pixels become NaN.

    Fill (0) and DN below the USGS valid minimum are invalid. The input is never
    modified and invalid pixels are never converted to zero.
    """
    values = np.asarray(dn)
    if values.dtype.kind not in "ui":
        raise ValueError(f"ST_B10 must be an integer array, got dtype {values.dtype}")
    valid = (values != ST_FILL_VALUE) & (values >= ST_VALID_MIN_DN)
    kelvin = values.astype(np.float64) * ST_SCALE + ST_OFFSET_K
    celsius = kelvin - KELVIN_OFFSET
    return np.where(valid, celsius, np.nan)


def summarise_lst(
    dn: np.ndarray,
    *,
    acquired: date,
    min_valid_fraction: float = 0.9,
) -> LstSample:
    """Summarise valid temperature for an area, withholding the mean when coverage is low.

    The whole array is treated as the area. Callers pass a window already cropped to it.
    """
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be within [0, 1]")
    celsius = decode_st_b10(dn)
    total = celsius.size
    if total == 0:
        return LstSample(acquired, None, 0.0, None, None, False, "empty area")
    finite = np.isfinite(celsius)
    valid_fraction = float(finite.sum()) / float(total)
    if valid_fraction == 0.0:
        return LstSample(acquired, None, 0.0, None, None, False, "no valid pixels")
    minimum = float(celsius[finite].min())
    maximum = float(celsius[finite].max())
    if valid_fraction < min_valid_fraction:
        return LstSample(
            acquired,
            None,
            valid_fraction,
            minimum,
            maximum,
            False,
            f"valid coverage {valid_fraction:.3f} is below {min_valid_fraction:.3f}; mean withheld",
        )
    return LstSample(
        acquired,
        float(celsius[finite].mean()),
        valid_fraction,
        minimum,
        maximum,
        False,
        "QA_PIXEL cloud/shadow masking not applied; clouds are not screened",
    )


def read_lst_sample(
    path: Path,
    *,
    acquired: date,
    bbox: list[float],
    min_valid_fraction: float = 0.9,
) -> LstSample:
    """Read the ST_B10 GeoTIFF over ``bbox`` (EPSG:4326) and summarise it.

    The raster must be EPSG:4326 and must fully contain ``bbox``; otherwise the
    result is an error rather than a partial mean.
    """
    import rasterio
    from rasterio.windows import from_bounds

    lon_min, lat_min, lon_max, lat_max = bbox
    with rasterio.open(path) as source:
        if source.crs is None or source.crs.to_epsg() != 4326:
            raise ValueError(f"ST_B10 raster CRS is {source.crs}, expected EPSG:4326")
        bounds = source.bounds
        if (
            bounds.left > lon_min
            or bounds.right < lon_max
            or bounds.bottom > lat_min
            or bounds.top < lat_max
        ):
            raise ValueError("ST_B10 raster does not fully contain the requested bbox")
        window = from_bounds(lon_min, lat_min, lon_max, lat_max, source.transform)
        data: Any = source.read(1, window=window)
    return summarise_lst(data, acquired=acquired, min_valid_fraction=min_valid_fraction)
