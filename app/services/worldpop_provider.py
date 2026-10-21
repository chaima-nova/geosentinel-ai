"""WorldPop population-exposure provider.

Reads a locally downloaded WorldPop GeoTIFF of population counts per grid cell
and returns the mean population density over a bounding box, with provenance.

Scope (read before use)
-----------------------
This provider supplies **population exposure only**. A population count raster
does not measure social vulnerability, so the returned context always has
``social_vulnerability = None``. :meth:`~app.services.risk_service.VulnerabilityContext.is_scoreable`
is therefore ``False`` and no HEPS composite score is produced from it. This is
deliberate: population exposure alone is not a complete vulnerability index (see
``docs/data/vulnerability-dataset-selection.md``).

Fail-closed validation
----------------------
A context is returned as ``VERIFIED`` only when all of these hold; otherwise the
provider reports ``MISSING`` or ``INVALID`` with the reason in the source text:

* the file exists and is a single-band raster;
* its CRS is EPSG:4326 (no silent reprojection);
* its pixel size matches the WorldPop 3 arc-second product (about 100 m);
* the whole requested area lies inside the raster (a partial AOI would
  understate density);
* no nodata cell and no negative count lies inside the requested area.

No value is ever substituted for a missing cell, and nothing falls back to a
placeholder provider.

Units: persons per square kilometre, from people per cell and the spherical area
of the requested bounding box (mean Earth radius 6371.0088 km).
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from app.services.risk_service import EvidenceProvenance, VulnerabilityContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rasterio.io import DatasetReader
    from rasterio.windows import Window

logger = logging.getLogger(__name__)

__all__ = [
    "WORLDPOP_RESOLUTION_DEG",
    "WORLDPOP_SOURCE_LABEL",
    "WorldPopExposureProvider",
    "spherical_bbox_area_km2",
]

#: WorldPop 3 arc-second grid (about 100 m at the equator), in degrees.
WORLDPOP_RESOLUTION_DEG = 1.0 / 1200.0

#: Attribution text carried into the audit trail. The product and reference year
#: are those of the dataset the operator downloads; see the selection document.
WORLDPOP_SOURCE_LABEL = (
    "WorldPop 2020 constrained, UN-adjusted, 100 m population counts "
    "(CC BY 4.0; doi:10.5258/SOTON/WP00685)"
)

#: Mean Earth radius (IUGG), in km.
EARTH_RADIUS_KM = 6371.0088

_RESOLUTION_TOLERANCE_DEG = 1e-6
_EXPOSURE_ONLY_NOTE = "population exposure only; social vulnerability not provided"


def spherical_bbox_area_km2(bbox: Sequence[float]) -> float:
    """Area of an EPSG:4326 bounding box on a sphere, in km².

    Args:
        bbox: ``[lon_min, lat_min, lon_max, lat_max]`` in decimal degrees.

    Returns:
        The area in square kilometres.
    """
    lon_min, lat_min, lon_max, lat_max = (float(v) for v in bbox)
    delta_lon = math.radians(lon_max - lon_min)
    return (
        EARTH_RADIUS_KM**2
        * delta_lon
        * (math.sin(math.radians(lat_max)) - math.sin(math.radians(lat_min)))
    )


class WorldPopExposureProvider:
    """Mean population density over a bbox, from a local WorldPop GeoTIFF.

    Implements :class:`~app.services.risk_service.VulnerabilityProvider`.

    Args:
        raster_path: Path to the downloaded GeoTIFF. Not read until :meth:`fetch`.
        source_label: Attribution text placed in the audit trail.
        expected_resolution_deg: Required pixel size in degrees. Defaults to the
            100 m product. Other rasters are rejected rather than mis-scaled.
    """

    def __init__(
        self,
        raster_path: str | Path,
        *,
        source_label: str = WORLDPOP_SOURCE_LABEL,
        expected_resolution_deg: float = WORLDPOP_RESOLUTION_DEG,
    ) -> None:
        self._path = Path(raster_path)
        self._label = source_label
        self._resolution = expected_resolution_deg

    async def fetch(self, bbox: Sequence[float]) -> VulnerabilityContext:
        """Return the population-exposure context for ``bbox``.

        Never raises for data problems; they are reported as ``MISSING`` or
        ``INVALID`` contexts so the pipeline can withhold the score.
        """
        return await asyncio.to_thread(self._read_blocking, list(bbox))

    # --- internals ---------------------------------------------------------

    def _context(self, provenance: EvidenceProvenance, reason: str) -> VulnerabilityContext:
        """A context without values, carrying the reason in its source text."""
        return VulnerabilityContext(
            provenance=provenance,
            source=f"{self._label}: {reason}",
        )

    def _read_blocking(self, bbox: list[float]) -> VulnerabilityContext:
        if not self._path.is_file():
            return self._context(
                EvidenceProvenance.MISSING,
                f"raster not found at configured path ({self._path.name})",
            )
        if not _valid_bbox(bbox):
            return self._context(EvidenceProvenance.INVALID, "requested bbox is malformed")

        import rasterio
        from rasterio.errors import RasterioError

        try:
            with rasterio.open(self._path) as source:
                return self._from_dataset(source, bbox)
        except (RasterioError, OSError) as exc:
            logger.warning("WorldPop raster unreadable: %s: %s", type(exc).__name__, exc)
            return self._context(
                EvidenceProvenance.INVALID, f"raster could not be read ({type(exc).__name__})"
            )

    def _from_dataset(self, source: DatasetReader, bbox: list[float]) -> VulnerabilityContext:
        from rasterio.errors import WindowError
        from rasterio.windows import Window, from_bounds

        if source.crs is None or source.crs.to_epsg() != 4326:
            return self._context(EvidenceProvenance.INVALID, "CRS is not EPSG:4326")
        if source.count != 1:
            return self._context(EvidenceProvenance.INVALID, "raster must have exactly one band")
        res_x, res_y = source.res
        if (
            abs(abs(res_x) - self._resolution) > _RESOLUTION_TOLERANCE_DEG
            or abs(abs(res_y) - self._resolution) > _RESOLUTION_TOLERANCE_DEG
        ):
            return self._context(
                EvidenceProvenance.INVALID,
                f"pixel size {res_x:g}x{res_y:g} deg does not match the expected "
                f"{self._resolution:g} deg product",
            )

        # Requested area in raster pixel space. Positive weights only where the
        # AOI overlaps a cell, so cell-edge alignment does not bias the mean.
        requested = from_bounds(*bbox, transform=source.transform)
        full = Window(0, 0, source.width, source.height)
        try:
            inside = requested.intersection(full)
        except WindowError:
            return self._context(EvidenceProvenance.INVALID, "requested area lies outside the raster")
        covered = (inside.width * inside.height) / (requested.width * requested.height)
        if covered < 1.0 - 1e-6:
            return self._context(
                EvidenceProvenance.INVALID,
                f"requested area is only {covered:.1%} inside the raster; not fully covered",
            )

        col0 = max(0, math.floor(inside.col_off))
        row0 = max(0, math.floor(inside.row_off))
        col1 = min(source.width, math.ceil(inside.col_off + inside.width))
        row1 = min(source.height, math.ceil(inside.row_off + inside.height))
        window = Window(col0, row0, col1 - col0, row1 - row0)
        values = source.read(1, window=window, masked=False).astype("float64")
        nodata = source.nodata

        weights = _overlap_weights(inside, window)
        nodata_mask = ~np.isfinite(values)
        if nodata is not None:
            nodata_mask |= values == float(nodata)
        if np.any(nodata_mask & (weights > 0)):
            return self._context(
                EvidenceProvenance.INVALID, "nodata cells lie inside the requested area"
            )
        values = np.where(nodata_mask, 0.0, values)
        if np.any(values < 0):
            return self._context(
                EvidenceProvenance.INVALID, "negative population counts inside the requested area"
            )

        total = float(np.sum(values * weights))
        area_km2 = spherical_bbox_area_km2(bbox)
        density = total / area_km2 if area_km2 > 0 else math.nan
        if not math.isfinite(density):
            return self._context(EvidenceProvenance.INVALID, "density could not be computed")

        logger.info(
            "WorldPop exposure: %.1f persons over %.3f km2 -> %.1f persons/km2",
            total,
            area_km2,
            density,
        )
        return VulnerabilityContext(
            provenance=EvidenceProvenance.VERIFIED,
            source=f"{self._label}; {_EXPOSURE_ONLY_NOTE}",
            population_density=density,
            social_vulnerability=None,
        )


def _valid_bbox(bbox: list[float]) -> bool:
    if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
        return False
    lon_min, lat_min, lon_max, lat_max = bbox
    return -180 <= lon_min < lon_max <= 180 and -90 <= lat_min < lat_max <= 90


def _overlap_weights(requested: Window, window: Window) -> np.ndarray:
    """Fraction of each cell in ``window`` covered by ``requested`` (outer product of 1-D weights)."""
    rows = np.arange(window.row_off, window.row_off + window.height)
    cols = np.arange(window.col_off, window.col_off + window.width)
    row_w = np.clip(
        np.minimum(rows + 1, requested.row_off + requested.height)
        - np.maximum(rows, requested.row_off),
        0.0,
        1.0,
    )
    col_w = np.clip(
        np.minimum(cols + 1, requested.col_off + requested.width)
        - np.maximum(cols, requested.col_off),
        0.0,
        1.0,
    )
    return np.outer(row_w, col_w)
