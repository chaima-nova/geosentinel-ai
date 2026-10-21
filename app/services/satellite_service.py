"""Satellite data retrieval for GeoSentinel-AI.

Queries the `Copernicus Data Space Ecosystem`_ (CDSE) STAC API for Sentinel-2
acquisitions over a bounding box and time window, keeps only scenes below a
cloud-cover threshold, samples the red (B04) and near-infrared (B08) bands, and
returns vegetation indices shaped as a
:class:`~app.core.schemas.SatelliteLayerMetadata`.

.. _Copernicus Data Space Ecosystem: https://dataspace.copernicus.eu/stac

Design notes
------------
Everything that touches the outside world is injectable, so the service is
testable without network or GDAL access:

* ``transport`` — an :class:`httpx.AsyncBaseTransport`. Tests pass an
  :class:`httpx.MockTransport`.
* ``band_reader`` — a callable ``(href) -> BandSample``. Tests pass an in-memory
  reader instead of downloading JPEG2000 tiles.

Failure handling
----------------
``fetch_sentinel_indices`` is a *best-effort* endpoint: it never raises for
ordinary operational failures (unreachable API, no scene found, missing band,
unreadable raster). Instead it returns a valid
:class:`~app.core.schemas.SatelliteLayerMetadata` dictionary using the
documented sentinel values below, and logs the reason at ``WARNING``.

Because :class:`~app.core.schemas.SatelliteLayerMetadata` declares
``extra="forbid"`` and requires every field, the sentinel is carried in
``stac_id``:

* ``stac_id == UNAVAILABLE_STAC_ID`` — the whole retrieval failed.
* ``ndvi_mean == NDVI_NO_DATA`` — NDVI could not be computed.

Callers that need a machine-readable status rather than a sentinel should wrap
this service; see the module README note.

Reflectance and masking
-----------------------
Sentinel-2 L2A digital numbers (DN) are *not* reflectance. From processing
baseline 04.00 (25 Jan 2022) onward, ``reflectance = (DN + BOA_ADD_OFFSET) /
QUANTIFICATION_VALUE`` with ``BOA_ADD_OFFSET = -1000`` and
``QUANTIFICATION_VALUE = 10000``. Earlier baselines have no offset. The baseline
is read from the STAC item (``s2:processing_baseline``) or from the product name
(``_N0510_``). If it cannot be determined, NDVI is not computed.

Metadata checks (``resolve_reflectance_offset``) run before any pixel is read.
The baseline in ``s2:processing_baseline`` must agree with the product name, and
any ``raster:bands`` scale/offset an asset declares must equal the L2A convention
``(DN + offset) / 10000``. Otherwise NDVI is withheld with a stated reason. The
convention was cross-checked against the product-level ``BOA_ADD_OFFSET_VALUES_LIST``
as read by GDAL and satpy. The full policy is in
``docs/satellite/sentinel2-scientific-validation.md``.

Area of interest
----------------
When a bounding box is requested, only its window of each raster is read. The
SCL is read at native resolution over the same window. ``footprint_fraction``
records how much of the requested area lies inside the scene. Coverage is
multiplied by it, so an area that extends past the scene edge loses coverage
rather than being scored as if it were usable.

Pixels are excluded from NDVI when the Scene Classification Layer (SCL) marks
them as no-data, defective, dark, cloud shadow, cloud (medium or high
probability), cirrus, water or snow/ice. Water and snow are excluded by policy
because they would bias a vegetation indicator; an area dominated by them is
therefore withheld. ``ndvi_valid_coverage`` reports the fraction of the requested
area that was usable, and an NDVI whose coverage is below ``min_valid_coverage``
is reported as ``NDVI_NO_DATA`` rather than as a value.

Known limitations
-----------------
* Band and SCL asset keys are matched for both the canonical names (``B04``,
  ``SCL``) and the CDSE names (``B04_10m``, ``SCL_20m``). The CDSE names come
  from the MIT-licensed mapchete-eo mapping; live CDSE items were not reachable
  from the build environment, so they are unverified against a live response.
* Sentinel-2 carries **no thermal band**, so ``lst_celsius_max`` cannot come
  from this collection and is always ``LST_NOT_AVAILABLE`` here. Land-surface
  temperature needs Sentinel-3 SLSTR or Landsat TIRS.
* Mean NDVI requires reading actual pixels. The AOI window is decimated to at most
  ``max_sample_pixels`` pixels with average resampling, so for a city-sized area
  the result is an estimate over a coarse grid (about 16 x 16 cells at the
  default budget), not a full-resolution statistic.
* Window edges are rounded to the nearest pixel boundary, so a bounding box is
  represented to within half a 10 m pixel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import Any

import httpx
import numpy as np

from app.core.schemas import SatelliteLayerMetadata, SpatialQueryRequest

logger = logging.getLogger(__name__)

__all__ = [
    "BOA_ADD_OFFSET_PB04",
    "DEFAULT_MIN_VALID_COVERAGE",
    "NDVI_NO_DATA",
    "NIR_BAND",
    "RED_BAND",
    "SCL_INVALID_CLASSES",
    "SCL_NON_VEGETATION_CLASSES",
    "UNAVAILABLE_STAC_ID",
    "BandSample",
    "NdviResult",
    "OutsideSceneError",
    "SatelliteDataService",
    "SatelliteServiceError",
    "boa_add_offset",
    "compute_ndvi",
    "mean_ndvi",
    "processing_baseline",
    "read_band_sample",
    "read_scl_sample",
    "resolve_reflectance_offset",
    "scl_invalid_mask",
]

#: Root of the CDSE STAC API.
DEFAULT_STAC_URL = "https://dataspace.copernicus.eu/stac"

#: Sentinel-2 optical collection on CDSE.
DEFAULT_COLLECTION = "SENTINEL-2"

#: Sentinel-2 red band (band 4, ~665 nm), 10 m native resolution.
RED_BAND = "B04"
#: Sentinel-2 near-infrared band (band 8, ~842 nm), 10 m native resolution.
NIR_BAND = "B08"

#: Sentinel-2 keys used in STAC ``assets``, in preferred order.
RED_BAND_KEYS = ("B04_10m", "B04", "b04", "red", "B4")
NIR_BAND_KEYS = ("B08_10m", "B08", "b08", "nir", "B8")

#: Property keys that may carry scene cloud cover, in preferred order.
CLOUD_COVER_KEYS = ("eo:cloud_cover", "cloudCover", "cloud_cover", "s2:cloud_cover")

#: ``stac_id`` returned when the retrieval failed entirely.
UNAVAILABLE_STAC_ID = "unavailable"

#: ``ndvi_mean`` returned when NDVI could not be computed. It is a legal NDVI
#: value (bare/water), so pair it with the log record rather than trusting it.
NDVI_NO_DATA = -1.0

#: Sentinel-2 scene classification layer (SCL), 20 m, one class per pixel.
SCL_BAND = "SCL"
SCL_KEYS = ("SCL_20m", "SCL", "scl")
#: SCL classes treated as unusable for NDVI: 0 no data, 1 saturated or defective,
#: 2 dark area, 3 cloud shadow, 8 cloud medium probability, 9 cloud high
#: probability, 10 thin cirrus. Vegetation (4), bare soil (5), water (6),
#: unclassified (7) and snow (11) are kept.
SCL_INVALID_CLASSES: frozenset[int] = frozenset({0, 1, 2, 3, 8, 9, 10})
#: SCL classes excluded from the *vegetation* NDVI mean by policy: 6 water and
#: 11 snow/ice. Water has negative NDVI and snow is not vegetation, so keeping
#: them would bias an urban vegetation indicator. Excluding them lowers
#: ``ndvi_valid_coverage``, so an AOI dominated by water or snow is withheld
#: rather than reported. This is a documented policy choice, not a
#: scientifically established rule (see docs/satellite/sentinel2-scientific-validation.md).
SCL_NON_VEGETATION_CLASSES: frozenset[int] = frozenset({6, 11})

#: Sentinel-2 L2A quantification value (reflectance = DN / 10000 before offset).
QUANTIFICATION_VALUE = 10000.0
#: First processing baseline whose L2A DN carry the BOA additive offset.
BOA_OFFSET_FIRST_BASELINE = 4.00
#: BOA_ADD_OFFSET for baseline >= 04.00, per the Copernicus L2A product guide.
BOA_ADD_OFFSET_PB04 = -1000.0

#: Default minimum fraction of the requested area that must be usable (not masked
#: by SCL, and inside the scene) for NDVI to be reported.
#: This is a POLICY CHOICE, not a scientifically established threshold. It is
#: configurable through ``SatelliteDataService(min_valid_coverage=...)``. Below it,
#: NDVI is withheld as ``NDVI_NO_DATA`` and the evidence auditor treats the
#: satellite evidence as unusable. See docs/satellite/sentinel2-scientific-validation.md §4.
DEFAULT_MIN_VALID_COVERAGE = 0.9
#: How many clear-enough scenes to try before giving up on NDVI.
DEFAULT_MAX_CANDIDATES = 3

#: ``cloud_cover`` returned when the retrieval failed entirely: "fully opaque",
#: i.e. nothing usable was observed.
CLOUD_COVER_UNKNOWN = 100.0

#: ``lst_celsius_max`` placeholder — Sentinel-2 has no thermal band.
LST_NOT_AVAILABLE = 0.0

#: Query text used only to reuse :class:`SpatialQueryRequest` validation.
_VALIDATION_QUERY = "sentinel-2 index retrieval"

#: Type for the pluggable band reader.
BandReader = Callable[[str], "BandSample"]


@dataclass(frozen=True)
class NdviResult:
    """Outcome of an NDVI computation over one footprint.

    Attributes:
        mean: Mean NDVI over usable pixels, or ``None`` if none were usable or
            the inputs could not be read.
        valid_coverage: Fraction of the footprint that was usable, in [0, 1],
            or ``None`` if the inputs could not be read at all.
    """

    mean: float | None
    valid_coverage: float | None


class SatelliteServiceError(RuntimeError):
    """Raised only for programming errors, never for operational failures."""


class OutsideSceneError(SatelliteServiceError):
    """The requested area does not intersect the raster at all."""


@dataclass(frozen=True)
class BandSample:
    """Decimated pixel samples of a single band.

    Attributes:
        values: 2-D array of reflectance samples, already flattened of its band
            axis.
        nodata: The source's nodata sentinel, or ``None`` if it declares none.
        footprint_fraction: Share of the requested area that lies inside the
            raster, in (0, 1]. ``1.0`` when no area was requested.
    """

    values: Any  # numpy.ndarray; typed loosely to avoid a hard numpy import here
    nodata: float | None = None
    footprint_fraction: float = 1.0


class SatelliteDataService:
    """Asynchronous client for Sentinel-2 STAC retrieval and index computation.

    Args:
        base_url: STAC API root. Override in tests to point at a stub.
        collection: STAC collection identifier to search.
        max_cloud_cover: Scenes with cloud cover at or above this percentage are
            discarded. Applied client-side as well as in the query.
        limit: Maximum number of scenes to ask the API for.
        timeout: Per-request HTTP timeout in seconds.
        max_sample_pixels: Upper bound on pixels read per band when computing
            NDVI. Keeps memory and bandwidth bounded on 10980x10980 tiles.
        band_reader: Callable that returns a :class:`BandSample` for an asset
            href. Defaults to a decimated rasterio read.
        transport: httpx transport, for injecting a mock in tests.
        client: An existing :class:`httpx.AsyncClient` to reuse. If given, the
            caller owns its lifecycle and :meth:`aclose` will not close it.

    Example::

        async with SatelliteDataService() as service:
            metadata = await service.fetch_sentinel_indices(
                bbox=[2.5, 31.0, 3.5, 32.0],
                start_date="2024-06-01",
                end_date="2024-06-30",
            )
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_STAC_URL,
        collection: str = DEFAULT_COLLECTION,
        max_cloud_cover: float = 20.0,
        limit: int = 10,
        timeout: float = 30.0,
        max_sample_pixels: int = 256,
        band_reader: BandReader | None = None,
        scl_reader: BandReader | None = None,
        min_valid_coverage: float = DEFAULT_MIN_VALID_COVERAGE,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        transport: httpx.AsyncBaseTransport | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not 0.0 <= min_valid_coverage <= 1.0:
            raise SatelliteServiceError(
                f"min_valid_coverage must be within [0, 1], got {min_valid_coverage}"
            )
        if max_candidates < 1:
            raise SatelliteServiceError(f"max_candidates must be >= 1, got {max_candidates}")
        if not 0.0 <= max_cloud_cover <= 100.0:
            raise SatelliteServiceError(
                f"max_cloud_cover must be within [0, 100], got {max_cloud_cover}"
            )
        if limit < 1:
            raise SatelliteServiceError(f"limit must be >= 1, got {limit}")
        if max_sample_pixels < 1:
            raise SatelliteServiceError(
                f"max_sample_pixels must be >= 1, got {max_sample_pixels}"
            )

        self.base_url = base_url.rstrip("/")
        self.collection = collection
        self.max_cloud_cover = max_cloud_cover
        self.limit = limit
        self.timeout = timeout
        self.max_sample_pixels = max_sample_pixels
        # Injected readers take an href only and serve every raster (tests use an
        # in-memory reader). Production readers are bound to the requested AOI per
        # call in _readers(), so only that window of the scene is read.
        self._band_reader_override: BandReader | None = band_reader
        self._scl_reader_override: BandReader | None = scl_reader or band_reader
        self.min_valid_coverage = min_valid_coverage
        self.max_candidates = max_candidates
        self._transport = transport
        self._external_client = client
        self._client = client

    # --- Lifecycle ---------------------------------------------------------

    async def __aenter__(self) -> SatelliteDataService:
        """Create the HTTP client lazily on entry."""
        self._get_client()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the HTTP client if this service owns it."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close the HTTP client, unless it was supplied by the caller."""
        if self._client is not None and self._client is not self._external_client:
            await self._client.aclose()
        if self._external_client is None:
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return the shared client, constructing it on first use."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self._transport,
                headers={"Accept": "application/geo+json"},
            )
        return self._client

    # --- Public API --------------------------------------------------------

    async def fetch_sentinel_indices(
        self, bbox: list, start_date: str, end_date: str
    ) -> dict:
        """Retrieve the least-cloudy Sentinel-2 scene and its vegetation index.

        Searches the CDSE STAC API for ``self.collection`` scenes intersecting
        ``bbox`` within ``[start_date, end_date]``, discards scenes at or above
        ``max_cloud_cover`` percent cloud, picks the clearest remaining scene,
        then samples its B04, B08 and SCL layers. NDVI is computed from the
        offset-corrected reflectance ``(DN + offset) / 10000`` over pixels the
        scene classification layer does not mark as cloud, shadow, cirrus, defective
        or no-data. The clearest few scenes are tried in turn until one reaches
        ``min_valid_coverage``.

        Args:
            bbox: ``[lon_min, lat_min, lon_max, lat_max]`` in decimal degrees,
                EPSG:4326. Validated by :class:`SpatialQueryRequest`.
            start_date: Inclusive window start, ISO 8601.
            end_date: Inclusive window end, ISO 8601.

        Returns:
            A dictionary that validates as
            :class:`~app.core.schemas.SatelliteLayerMetadata`. On any
            operational failure it is a sentinel-valued metadata document
            (``stac_id == UNAVAILABLE_STAC_ID``) rather than an exception.

        Raises:
            pydantic.ValidationError: If ``bbox`` or the dates are malformed.
                Bad input is a caller bug, not an operational failure, so it is
                surfaced rather than masked.
        """
        window = SpatialQueryRequest(
            query=_VALIDATION_QUERY, bbox=list(bbox), start_date=start_date, end_date=end_date
        )
        logger.debug(
            "searching %s over %s for %s",
            self.collection,
            window.bbox_wkt,
            f"{start_date}..{end_date}",
        )

        try:
            items = await self.search_items(window)
        except httpx.HTTPError as exc:
            logger.warning(
                "STAC search failed for %s: %s: %s",
                self.collection,
                type(exc).__name__,
                exc,
            )
            return self._degraded_metadata(f"STAC request failed: {exc}")
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.warning("STAC returned an unusable response: %s", exc)
            return self._degraded_metadata(f"malformed STAC response: {exc}")

        candidates = self._filter_by_cloud_cover(items)
        if not candidates:
            reason = (
                "no scene found"
                if not items
                else f"all {len(items)} scene(s) exceeded {self.max_cloud_cover}% cloud cover"
            )
            logger.warning("%s over %s", reason, window.bbox_wkt)
            return self._degraded_metadata(reason)

        # Try the clearest few scenes. Keep the first whose usable coverage meets
        # the threshold; otherwise report the best attempt, flagged as no-data.
        best: tuple[dict[str, Any], NdviResult] | None = None
        for item in candidates[: self.max_candidates]:
            stac_id = str(item.get("id") or UNAVAILABLE_STAC_ID)
            band_links = self._resolve_bands(item)
            baseline, offset, reason = resolve_reflectance_offset(item)
            if reason is not None:
                logger.warning("scene %s: %s; NDVI withheld", stac_id, reason)
            ndvi = await self._safe_ndvi(
                band_links, self._resolve_scl(item), stac_id, offset, window.bbox
            )
            coverage = ndvi.valid_coverage
            logger.info(
                "candidate %s: cloud=%.1f%% baseline=%s offset=%s ndvi=%s coverage=%s",
                stac_id,
                self._cloud_cover(item),
                f"{baseline:.2f}" if baseline is not None else "unknown",
                offset,
                ndvi.mean,
                f"{coverage:.3f}" if coverage is not None else "unknown",
            )
            if ndvi.mean is not None and coverage is not None and coverage >= self.min_valid_coverage:
                return self._layer_metadata(
                    item, stac_id, band_links, ndvi.mean, coverage, baseline, offset
                ).model_dump()
            if best is None or (coverage or 0.0) > (best[1].valid_coverage or 0.0):
                best = (item, ndvi)

        assert best is not None  # candidates is non-empty here
        item, ndvi = best
        stac_id = str(item.get("id") or UNAVAILABLE_STAC_ID)
        logger.warning(
            "no candidate reached the %.0f%% valid-coverage threshold; NDVI withheld "
            "(best coverage %s)",
            self.min_valid_coverage * 100,
            f"{ndvi.valid_coverage:.1%}" if ndvi.valid_coverage is not None else "unknown",
        )
        baseline, offset, _ = resolve_reflectance_offset(item)
        return self._layer_metadata(
            item,
            stac_id,
            self._resolve_bands(item),
            None,
            ndvi.valid_coverage,
            baseline,
            offset,
        ).model_dump()

    def _layer_metadata(
        self,
        item: dict[str, Any],
        stac_id: str,
        band_links: dict[str, str],
        ndvi_mean: float | None,
        valid_coverage: float | None,
        baseline: float | None,
        offset: float | None,
    ) -> SatelliteLayerMetadata:
        """Assemble the layer document for one scene.

        ``ndvi_mean`` of ``None`` is written as ``NDVI_NO_DATA``, so the value is
        never a sentinel that could be mistaken for measured bare ground.
        """
        return SatelliteLayerMetadata(
            stac_id=stac_id,
            cloud_cover=self._cloud_cover(item),
            ndvi_mean=ndvi_mean if ndvi_mean is not None else NDVI_NO_DATA,
            lst_celsius_max=LST_NOT_AVAILABLE,
            band_links=band_links,
            ndvi_valid_coverage=valid_coverage,
            reflectance_offset=offset,
            processing_baseline=f"{baseline:05.2f}" if baseline is not None else None,
        )

    def _degraded_metadata(self, reason: str) -> dict:
        """Build a sentinel-valued metadata document for a failed retrieval.

        The result still validates as
        :class:`~app.core.schemas.SatelliteLayerMetadata`; the failure is
        signalled by ``stac_id == UNAVAILABLE_STAC_ID`` rather than by raising,
        so one dead upstream cannot take an agent graph down.

        Args:
            reason: Human-readable cause, logged at ``DEBUG`` for tracing. The
                call site logs the operational detail at ``WARNING``.

        Returns:
            A dictionary that validates as
            :class:`~app.core.schemas.SatelliteLayerMetadata`.
        """
        logger.debug("degraded metadata returned: %s", reason)
        return SatelliteLayerMetadata(
            stac_id=UNAVAILABLE_STAC_ID,
            cloud_cover=CLOUD_COVER_UNKNOWN,
            ndvi_mean=NDVI_NO_DATA,
            lst_celsius_max=LST_NOT_AVAILABLE,
            band_links={},
            ndvi_valid_coverage=None,
            reflectance_offset=None,
            processing_baseline=None,
        ).model_dump()

    async def search_items(self, window: SpatialQueryRequest) -> list[dict[str, Any]]:
        """Query the STAC API and return raw item dictionaries.

        Args:
            window: A validated spatio-temporal window.

        Returns:
            The ``features`` array from the STAC response, possibly empty.

        Raises:
            httpx.HTTPError: On transport failures and non-2xx responses.
            ValueError: If the response is not a STAC FeatureCollection.
        """
        params = {
            "collections": self.collection,
            "bbox": ",".join(f"{value:.6f}" for value in window.bbox),
            "datetime": (
                f"{_rfc3339(window.start_datetime)}/{_rfc3339(window.end_datetime)}"
            ),
            "limit": str(self.limit),
            # Server-side narrowing. Also enforced client-side in
            # _filter_by_cloud_cover, because not every STAC implementation
            # honours the query extension.
            "query": json.dumps({"eo:cloud_cover": {"lt": self.max_cloud_cover}}),
        }

        client = self._get_client()
        response = await client.get("/search", params=params)
        response.raise_for_status()

        try:
            payload = response.json()
        except ValueError as exc:  # httpx raises json.JSONDecodeError (a ValueError)
            raise ValueError(f"STAC response was not valid JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"STAC response was a {type(payload).__name__}, expected an object")

        features = payload.get("features")
        if features is None:
            raise ValueError("STAC response has no 'features' member")
        if not isinstance(features, list):
            raise ValueError(
                f"STAC 'features' was a {type(features).__name__}, expected a list"
            )
        return [feature for feature in features if isinstance(feature, dict)]

    # --- Item selection ----------------------------------------------------

    def _filter_by_cloud_cover(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep scenes under the threshold, sorted clearest-first.

        Scenes that do not advertise a cloud cover are kept but sorted last: an
        unknown value is weaker evidence than a known clear one, but dropping it
        outright would throw away usable imagery.
        """

        def sort_key(item: dict[str, Any]) -> tuple[int, float]:
            cover = self._cloud_cover(item, default=None)
            if cover is None:
                return (1, 0.0)
            return (0, cover)

        return sorted(
            (
                item
                for item in items
                if (cover := self._cloud_cover(item, default=None)) is None
                or cover < self.max_cloud_cover
            ),
            key=sort_key,
        )

    @staticmethod
    def _cloud_cover(
        item: dict[str, Any], default: float | None = CLOUD_COVER_UNKNOWN
    ) -> float | None:
        """Extract scene cloud cover, tolerating differing property names.

        Args:
            item: A STAC item.
            default: Value returned when no usable cloud cover is advertised.
                Pass ``None`` to distinguish "unknown" from a real percentage.

        Returns:
            Cloud cover percentage clamped to [0, 100], or ``default``.
        """
        properties = item.get("properties")
        if not isinstance(properties, dict):
            return default
        for key in CLOUD_COVER_KEYS:
            value = properties.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(value):
                continue
            return min(100.0, max(0.0, float(value)))
        return default

    @staticmethod
    def _resolve_bands(item: dict[str, Any]) -> dict[str, str]:
        """Locate the red and NIR asset hrefs on a STAC item.

        Resolution order per band: exact asset key, case-insensitive key, an
        ``eo:bands`` name match, then a filename match. This tolerates catalogs
        that name assets differently from the canonical Sentinel-2 keys.

        Args:
            item: A STAC item.

        Returns:
            Mapping of canonical band name to href. Empty when the item exposes
            no usable red/NIR asset.
        """
        assets = item.get("assets")
        if not isinstance(assets, dict):
            return {}

        links: dict[str, str] = {}
        for band, keys in ((RED_BAND, RED_BAND_KEYS), (NIR_BAND, NIR_BAND_KEYS)):
            href = SatelliteDataService._find_asset_href(assets, band, keys)
            if href is not None:
                links[band] = href
        return links

    @staticmethod
    def _find_asset_href(
        assets: dict[str, Any], band: str, keys: tuple[str, ...]
    ) -> str | None:
        """Return the href for one band, or ``None`` if it cannot be located."""

        def href_of(asset: Any) -> str | None:
            if not isinstance(asset, dict):
                return None
            href = asset.get("href")
            if isinstance(href, str) and href.strip():
                return href
            # Some catalogs nest a secondary location under 'alternate'.
            alternate = asset.get("alternate")
            if isinstance(alternate, dict):
                for variant in alternate.values():
                    if isinstance(variant, dict):
                        nested = variant.get("href")
                        if isinstance(nested, str) and nested.strip():
                            return nested
            return None

        for key in keys:  # 1. known key, then 2. case-insensitive key
            if key in assets:
                href = href_of(assets[key])
                if href is not None:
                    return href
        lowered = {str(k).lower(): v for k, v in assets.items()}
        for key in keys:
            if key.lower() in lowered:
                href = href_of(lowered[key.lower()])
                if href is not None:
                    return href

        for key, asset in assets.items():  # 3. eo:bands name, 4. filename
            if not isinstance(asset, dict):
                continue
            eo_bands = asset.get("eo:bands")
            if isinstance(eo_bands, list):
                for entry in eo_bands:
                    if isinstance(entry, dict) and str(entry.get("name", "")).upper() == band:
                        href = href_of(asset)
                        if href is not None:
                            return href
            href = href_of(asset)
            if href and any(marker in href.upper() for marker in (f"{band}.", f"{band}/", f"{band}_")):
                return href
        return None

    # --- NDVI --------------------------------------------------------------

    @staticmethod
    def _resolve_scl(item: dict[str, Any]) -> str | None:
        """Locate the scene classification layer href, or ``None``."""
        assets = item.get("assets")
        if not isinstance(assets, dict):
            return None
        return SatelliteDataService._find_asset_href(assets, SCL_BAND, SCL_KEYS)

    def _readers(self, bbox: Sequence[float] | None) -> tuple[BandReader, BandReader]:
        """Return the ``(band, SCL)`` readers for one requested area.

        Injected readers are used as given. Production readers are bound to
        ``bbox`` so that only the requested window of each raster is read.
        """
        if self._band_reader_override is not None:
            band_reader = self._band_reader_override
        else:
            band_reader = partial(
                read_band_sample, max_sample_pixels=self.max_sample_pixels, bbox=bbox
            )
        if self._scl_reader_override is not None:
            scl_reader = self._scl_reader_override
        else:
            scl_reader = partial(read_scl_sample, bbox=bbox)
        return band_reader, scl_reader

    async def _safe_ndvi(
        self,
        band_links: dict[str, str],
        scl_href: str | None,
        stac_id: str,
        offset: float | None,
        bbox: Sequence[float] | None = None,
    ) -> NdviResult:
        """Compute coverage-aware NDVI, converting every failure into a withheld result.

        Refuses to compute NDVI without a cloud mask or a determinable reflectance
        offset, because either absence would make the result unsupported. A scene
        that does not intersect the requested area reports coverage ``0.0``.
        """
        missing = [band for band in (RED_BAND, NIR_BAND) if band not in band_links]
        if missing:
            logger.warning(
                "scene %s is missing band(s) %s; NDVI not computed",
                stac_id,
                ", ".join(missing),
            )
            return NdviResult(None, None)
        if scl_href is None:
            logger.warning(
                "scene %s has no scene classification layer; cloud mask unavailable, "
                "NDVI not computed",
                stac_id,
            )
            return NdviResult(None, None)
        if offset is None:
            logger.warning(
                "scene %s: reflectance offset cannot be determined; NDVI not computed",
                stac_id,
            )
            return NdviResult(None, None)

        try:
            # rasterio is blocking and GDAL does network I/O, so keep it off the
            # event loop.
            return await asyncio.to_thread(
                self._ndvi,
                band_links[RED_BAND],
                band_links[NIR_BAND],
                scl_href,
                offset,
                bbox,
            )
        except OutsideSceneError as exc:
            logger.warning("scene %s does not cover the requested area: %s", stac_id, exc)
            return NdviResult(None, 0.0)
        except Exception as exc:  # noqa: BLE001 - any raster failure is non-fatal
            logger.warning(
                "could not read bands for scene %s: %s: %s",
                stac_id,
                type(exc).__name__,
                exc,
            )
            return NdviResult(None, None)

    def _ndvi(
        self,
        red_href: str,
        nir_href: str,
        scl_href: str,
        offset: float,
        bbox: Sequence[float] | None = None,
    ) -> NdviResult:
        """Compute coverage-aware NDVI from band hrefs. Blocking; runs in a thread.

        Args:
            red_href: Asset href for the red band.
            nir_href: Asset href for the NIR band.
            scl_href: Asset href for the scene classification layer.
            offset: BOA additive offset to apply to DN before scaling.
            bbox: Requested area in EPSG:4326, or ``None`` for the whole scene.

        Returns:
            The :class:`NdviResult` for the requested area.

        Raises:
            Exception: Propagates rasterio/GDAL errors to :meth:`_safe_ndvi`.
        """
        band_reader, scl_reader = self._readers(bbox)
        red = band_reader(red_href)
        nir = band_reader(nir_href)
        scl = scl_reader(scl_href)
        invalid = scl_invalid_mask(scl.values, np.shape(red.values))
        return compute_ndvi(
            red,
            nir,
            reflectance_offset=offset,
            quantification_value=QUANTIFICATION_VALUE,
            scl_invalid=invalid,
        )

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def processing_baseline(item: dict[str, Any]) -> float | None:
    """Read the Sentinel-2 processing baseline of a STAC item, e.g. ``5.10``.

    Looks at ``s2:processing_baseline`` first, then at the ``_Nxxyy_`` token in the
    product identifier. Returns ``None`` when neither is present or parseable.
    """
    properties = item.get("properties")
    if isinstance(properties, dict):
        value = properties.get("s2:processing_baseline")
        parsed = _parse_baseline(value)
        if parsed is not None:
            return parsed
    return _baseline_from_id(item)


def _baseline_from_id(item: dict[str, Any]) -> float | None:
    """Baseline encoded in the product identifier token ``_Nxxyy_``, or ``None``."""
    match = re.search(r"_N(\d{2})(\d{2})_", str(item.get("id") or ""))
    if match:
        return int(match.group(1)) + int(match.group(2)) / 100.0
    return None


def _parse_baseline(value: Any) -> float | None:
    """Parse ``"05.10"`` or ``5.1`` style baselines; ``None`` if malformed."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) and parsed > 0 else None
    return None


def boa_add_offset(baseline: float | None) -> float | None:
    """BOA additive offset for a processing baseline.

    Returns:
        ``-1000`` from baseline 04.00 onward, ``0`` for earlier baselines, and
        ``None`` when the baseline is unknown (the caller must then not compute NDVI).
    """
    if baseline is None:
        return None
    return BOA_ADD_OFFSET_PB04 if baseline >= BOA_OFFSET_FIRST_BASELINE - 1e-9 else 0.0


def resolve_reflectance_offset(
    item: dict[str, Any],
) -> tuple[float | None, float | None, str | None]:
    """Decide the BOA offset for an item, or withhold it with a reason.

    The offset is derived from the processing baseline, matching the
    ``BOA_ADD_OFFSET_VALUES_LIST`` convention of the Sentinel-2 L2A product
    metadata (GDAL and satpy both read that list). Fail-closed checks:

    * the baseline in ``s2:processing_baseline`` must agree with the ``_Nxxyy_``
      token in the product identifier, when both are present;
    * the baseline must be known;
    * if an asset declares ``raster:bands`` scale and offset, they must equal the
      L2A convention ``(DN + offset) / 10000``. A collection that already applied
      the offset, or stores raw DN, would otherwise be double-corrected or
      under-corrected.

    Args:
        item: A STAC item.

    Returns:
        ``(baseline, offset, reason)``. When ``reason`` is not ``None`` the offset
        is ``None`` and NDVI must not be computed from this item.
    """
    properties = item.get("properties")
    declared = (
        _parse_baseline(properties.get("s2:processing_baseline"))
        if isinstance(properties, dict)
        else None
    )
    from_name = _baseline_from_id(item)
    if declared is not None and from_name is not None and abs(declared - from_name) > 1e-9:
        return (
            declared,
            None,
            (
                f"processing baseline is inconsistent (properties {declared:05.2f}, "
                f"product name {from_name:05.2f})"
            ),
        )
    baseline = declared if declared is not None else from_name
    offset = boa_add_offset(baseline)
    if offset is None:
        return None, None, "processing baseline is unknown"
    mismatch = _declared_scaling_mismatch(item, offset)
    if mismatch is not None:
        return baseline, None, mismatch
    return baseline, offset, None


def _declared_scaling_mismatch(item: dict[str, Any], offset: float) -> str | None:
    """Describe a ``raster:bands`` scale/offset that contradicts the L2A convention."""
    assets = item.get("assets")
    if not isinstance(assets, dict):
        return None
    expected_scale = 1.0 / QUANTIFICATION_VALUE
    expected_offset = offset / QUANTIFICATION_VALUE
    for band, keys in ((RED_BAND, RED_BAND_KEYS), (NIR_BAND, NIR_BAND_KEYS)):
        href = SatelliteDataService._find_asset_href(assets, band, keys)
        if href is None:
            continue
        asset = next(
            (a for a in assets.values() if isinstance(a, dict) and a.get("href") == href),
            None,
        )
        if asset is None:
            continue
        entries = asset.get("raster:bands")
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or "scale" not in entry or "offset" not in entry:
                continue
            scale, declared_offset = entry["scale"], entry["offset"]
            if not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in (scale, declared_offset)
            ):
                return f"{band} declares a non-numeric scale or offset"
            if not (
                math.isclose(scale, expected_scale, rel_tol=1e-9)
                and math.isclose(declared_offset, expected_offset, rel_tol=1e-9, abs_tol=1e-12)
            ):
                return (
                    f"{band} declares scale {scale} and offset {declared_offset}, which do "
                    f"not match (DN + {offset:g}) / {QUANTIFICATION_VALUE:g}"
                )
    return None


def _resample_any(mask: np.ndarray, axis: int, target: int) -> np.ndarray:
    """Reduce one axis of a boolean mask to ``target`` cells, OR-ing each window.

    A cell is flagged if *any* native pixel under it is flagged. Using "any" rather
    than nearest-neighbour or mode means a single cloud pixel cannot be dropped.
    """
    size = mask.shape[axis]
    if size == target:
        return mask
    if size > target:
        starts = (np.arange(target) * size) // target
        return np.logical_or.reduceat(mask, starts, axis=axis)
    # Upsampling: each target cell maps to one source cell.
    index = ((np.arange(target) + 0.5) * size // target).astype(int)
    return np.take(mask, index, axis=axis)


def scl_invalid_mask(scl: Any, target_shape: tuple[int, ...]) -> np.ndarray:
    """Boolean mask of pixels to exclude, resampled to the band grid.

    Args:
        scl: Scene classification values, at native (or any) resolution.
        target_shape: Shape of the band samples the mask must align with.

    Returns:
        ``True`` where the pixel is unusable for NDVI.
    """
    import numpy as np

    classes = np.asarray(scl)
    if classes.ndim != 2 or len(target_shape) != 2:
        raise SatelliteServiceError("scene classification and band samples must be 2-D")
    excluded = SCL_INVALID_CLASSES | SCL_NON_VEGETATION_CLASSES
    invalid = np.isin(classes, np.fromiter(excluded, dtype=classes.dtype))
    invalid = _resample_any(invalid, 0, target_shape[0])
    return _resample_any(invalid, 1, target_shape[1])


def compute_ndvi(
    red: BandSample,
    nir: BandSample,
    *,
    reflectance_offset: float,
    quantification_value: float,
    scl_invalid: np.ndarray | None,
) -> NdviResult:
    """NDVI over a footprint, with explicit usable-pixel coverage.

    Nodata and non-finite DN, plus SCL-invalid pixels, are excluded. The remaining
    DN are converted to reflectance as ``(DN + offset) / quantification_value``
    before the ratio. Coverage is the usable fraction of *all* footprint pixels, so
    out-of-footprint area reduces it.

    Args:
        red: Sampled red band (DN).
        nir: Sampled NIR band (DN).
        reflectance_offset: BOA additive offset (``0`` for pre-04.00 baselines).
        quantification_value: DN divisor, normally 10000.
        scl_invalid: Mask of SCL-invalid pixels, or ``None`` if not available.

    Returns:
        The :class:`NdviResult`. A shape mismatch reports ``NdviResult(None, None)``.
    """
    import numpy as np

    red_dn = np.asarray(red.values, dtype="float64")
    nir_dn = np.asarray(nir.values, dtype="float64")
    if red_dn.shape != nir_dn.shape:
        logger.warning(
            "band shapes differ (red %s vs nir %s); NDVI not computed",
            red_dn.shape,
            nir_dn.shape,
        )
        return NdviResult(None, None)
    if scl_invalid is not None and np.shape(scl_invalid) != red_dn.shape:
        logger.warning("cloud mask shape %s does not match bands; NDVI not computed", np.shape(scl_invalid))
        return NdviResult(None, None)

    total = red_dn.size
    footprint = min(red.footprint_fraction, nir.footprint_fraction)
    usable = np.isfinite(red_dn) & np.isfinite(nir_dn)
    if red.nodata is not None:
        usable &= red_dn != float(red.nodata)
    if nir.nodata is not None:
        usable &= nir_dn != float(nir.nodata)
    if scl_invalid is not None:
        usable &= ~np.asarray(scl_invalid, dtype=bool)

    red_r = (red_dn + reflectance_offset) / quantification_value
    nir_r = (nir_dn + reflectance_offset) / quantification_value
    usable &= (nir_r + red_r) != 0
    count = int(usable.sum())
    coverage = (count / total) * footprint if total else 0.0
    if count == 0:
        logger.warning("no usable pixel in footprint; NDVI not computed")
        return NdviResult(None, coverage)

    ndvi = (nir_r[usable] - red_r[usable]) / (nir_r[usable] + red_r[usable])
    return NdviResult(float(np.clip(ndvi.mean(), -1.0, 1.0)), coverage)


def _edge(value: float) -> int:
    """Nearest pixel boundary for a window edge.

    A pixel is included only when at least half of it lies in the requested area.
    Rounding outward instead would pull in a neighbouring column whenever the
    EPSG:4326 -> raster round trip overshoots a boundary by a fraction of a pixel
    (a lon/lat rectangle becomes a slightly larger rectangle in UTM).
    """
    return round(value)


def _aoi_window(source: Any, bbox: Sequence[float]) -> tuple[Any, float]:
    """Pixel window of ``source`` covering ``bbox``, and the share of ``bbox`` inside it.

    ``bbox`` is EPSG:4326 and is transformed to the raster CRS (Sentinel-2 tiles
    are UTM). Window edges are rounded to the nearest pixel boundary and clipped
    to the raster. The returned fraction is the share of the requested area that
    lies inside the raster, computed before rounding.

    Raises:
        OutsideSceneError: If no part of ``bbox`` lies inside the raster.
    """
    from rasterio.errors import WindowError
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds

    left, bottom, right, top = transform_bounds("EPSG:4326", source.crs, *bbox)
    requested = from_bounds(left, bottom, right, top, transform=source.transform)
    requested_area = requested.width * requested.height
    if requested_area <= 0:
        raise OutsideSceneError("requested area has no extent in the raster CRS")
    try:
        inside = requested.intersection(Window(0, 0, source.width, source.height))
    except WindowError as exc:
        raise OutsideSceneError("requested area lies outside the scene") from exc
    fraction = (inside.width * inside.height) / requested_area

    col0 = max(0, _edge(inside.col_off))
    row0 = max(0, _edge(inside.row_off))
    col1 = min(source.width, _edge(inside.col_off + inside.width))
    row1 = min(source.height, _edge(inside.row_off + inside.height))
    if col1 <= col0 or row1 <= row0:
        raise OutsideSceneError("requested area contains no whole pixel of the scene")
    return Window(col0, row0, col1 - col0, row1 - row0), fraction


def read_scl_sample(href: str, *, bbox: Sequence[float] | None = None) -> BandSample:
    """Read the scene classification layer at native resolution. Blocking.

    Never decimated: resampling would average class codes into meaningless values.

    Args:
        href: Asset href or local path.
        bbox: Optional EPSG:4326 area; only that window is read.

    Returns:
        A :class:`BandSample` of class codes.
    """
    import rasterio

    with rasterio.open(href) as source:
        if source.height <= 0 or source.width <= 0:
            raise SatelliteServiceError(f"SCL {href!r} reported an empty extent")
        if bbox is None:
            window, fraction = None, 1.0
        else:
            window, fraction = _aoi_window(source, bbox)
        values = source.read(1, window=window, masked=False)
        return BandSample(values=values, nodata=None, footprint_fraction=fraction)


def mean_ndvi(red: BandSample, nir: BandSample) -> float | None:
    """Compute mean NDVI from two band samples.

    NDVI is ``(NIR - RED) / (NIR + RED)``. Pixels are excluded when either band
    is nodata, non-finite, or when ``NIR + RED == 0`` (which would divide by
    zero).

    Args:
        red: Sampled red band.
        nir: Sampled NIR band.

    Returns:
        The mean NDVI clamped to [-1, 1], or ``None`` when no pixel is usable
        or the bands do not share a shape.
    """
    import numpy as np

    red_values = np.asarray(red.values, dtype="float64")
    nir_values = np.asarray(nir.values, dtype="float64")
    if red_values.shape != nir_values.shape:
        logger.warning(
            "band shapes differ (red %s vs nir %s); NDVI not computed",
            red_values.shape,
            nir_values.shape,
        )
        return None

    valid = np.isfinite(red_values) & np.isfinite(nir_values)
    if red.nodata is not None:
        valid &= red_values != float(red.nodata)
    if nir.nodata is not None:
        valid &= nir_values != float(nir.nodata)
    if not valid.any():
        logger.warning("every pixel was masked as nodata; NDVI not computed")
        return None

    red_valid = red_values[valid]
    nir_valid = nir_values[valid]
    denominator = nir_valid + red_valid
    usable = denominator != 0
    if not usable.any():
        logger.warning("NIR + RED was zero for every pixel; NDVI not computed")
        return None

    ndvi = (nir_valid[usable] - red_valid[usable]) / denominator[usable]
    return float(np.clip(ndvi.mean(), -1.0, 1.0))


def read_band_sample(
    href: str,
    *,
    max_sample_pixels: int = 256,
    bbox: Sequence[float] | None = None,
) -> BandSample:
    """Read a decimated sample of one band over the requested area. Blocking.

    Full-resolution Sentinel-2 10 m tiles are 10980x10980 and cover about
    110 km, far larger than a city. When ``bbox`` is given only its window is read,
    then decimated by an integer factor that brings it under ``max_sample_pixels``
    using average resampling. The result is an estimate over a coarsened grid,
    not a full-resolution statistic.

    Args:
        href: Asset href or local path.
        max_sample_pixels: Upper bound on the number of pixels returned.
        bbox: Optional EPSG:4326 area. ``None`` reads the whole raster.

    Returns:
        A :class:`BandSample` of the first band, with its ``footprint_fraction``.

    Raises:
        OutsideSceneError: If ``bbox`` does not intersect the raster.
        rasterio.errors.RasterioError: If the asset cannot be opened or read.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window

    with rasterio.open(href) as source:
        if source.height <= 0 or source.width <= 0:
            raise SatelliteServiceError(f"band {href!r} reported an empty extent")
        if bbox is None:
            window, fraction = Window(0, 0, source.width, source.height), 1.0
        else:
            window, fraction = _aoi_window(source, bbox)
        height, width = int(window.height), int(window.width)
        # Integer decimation factor that brings the window under the pixel budget.
        factor = max(1, math.ceil(math.sqrt((height * width) / max(1, max_sample_pixels))))
        out_shape = (max(1, height // factor), max(1, width // factor))
        data = source.read(
            1,
            window=window,
            out_shape=out_shape,
            resampling=Resampling.average,
            masked=False,
        )
        return BandSample(values=data, nodata=source.nodata, footprint_fraction=fraction)


def _rfc3339(moment: datetime) -> str:
    """Render a datetime as an RFC 3339 UTC timestamp with a ``Z`` suffix.

    Offsets are converted rather than passed through, so the query is canonical
    regardless of how the caller expressed the window. A naive datetime is read
    as UTC, matching :attr:`SpatialQueryRequest.start_datetime`.

    Args:
        moment: The timestamp to render.

    Returns:
        An RFC 3339 string ending in ``Z``.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
