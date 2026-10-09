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

Known limitations
-----------------
* Sentinel-2 carries **no thermal band**, so ``lst_celsius_max`` cannot come
  from this collection and is always ``LST_NOT_AVAILABLE`` here. Land-surface
  temperature needs Sentinel-3 SLSTR or Landsat TIRS.
* Mean NDVI requires reading actual pixels. Bands are decimated to at most
  ``max_sample_pixels`` pixels with average resampling, so the result is an
  estimate over a coarsened grid, not a full-resolution statistic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import Any

import httpx

from app.core.schemas import SatelliteLayerMetadata, SpatialQueryRequest

logger = logging.getLogger(__name__)

__all__ = [
    "NDVI_NO_DATA",
    "NIR_BAND",
    "RED_BAND",
    "UNAVAILABLE_STAC_ID",
    "BandSample",
    "SatelliteDataService",
    "SatelliteServiceError",
    "mean_ndvi",
    "read_band_sample",
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
RED_BAND_KEYS = ("B04", "b04", "red", "B4")
NIR_BAND_KEYS = ("B08", "b08", "nir", "B8")

#: Property keys that may carry scene cloud cover, in preferred order.
CLOUD_COVER_KEYS = ("eo:cloud_cover", "cloudCover", "cloud_cover", "s2:cloud_cover")

#: ``stac_id`` returned when the retrieval failed entirely.
UNAVAILABLE_STAC_ID = "unavailable"

#: ``ndvi_mean`` returned when NDVI could not be computed. It is a legal NDVI
#: value (bare/water), so pair it with the log record rather than trusting it.
NDVI_NO_DATA = -1.0

#: ``cloud_cover`` returned when the retrieval failed entirely: "fully opaque",
#: i.e. nothing usable was observed.
CLOUD_COVER_UNKNOWN = 100.0

#: ``lst_celsius_max`` placeholder — Sentinel-2 has no thermal band.
LST_NOT_AVAILABLE = 0.0

#: Query text used only to reuse :class:`SpatialQueryRequest` validation.
_VALIDATION_QUERY = "sentinel-2 index retrieval"

#: Type for the pluggable band reader.
BandReader = Callable[[str], "BandSample"]


class SatelliteServiceError(RuntimeError):
    """Raised only for programming errors, never for operational failures."""


@dataclass(frozen=True)
class BandSample:
    """Decimated pixel samples of a single band.

    Attributes:
        values: 2-D array of reflectance samples, already flattened of its band
            axis.
        nodata: The source's nodata sentinel, or ``None`` if it declares none.
    """

    values: Any  # numpy.ndarray; typed loosely to avoid a hard numpy import here
    nodata: float | None = None


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
        transport: httpx.AsyncBaseTransport | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
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
        self._band_reader = band_reader or partial(
            read_band_sample, max_sample_pixels=max_sample_pixels
        )
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
        then samples its B04 and B08 bands to compute mean NDVI as
        ``(NIR - RED) / (NIR + RED)``.

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

        item = candidates[0]
        stac_id = str(item.get("id") or UNAVAILABLE_STAC_ID)
        cloud_cover = self._cloud_cover(item)
        band_links = self._resolve_bands(item)
        logger.info(
            "selected scene %s (cloud cover %.1f%%) with bands %s",
            stac_id,
            cloud_cover,
            sorted(band_links) or "none",
        )

        ndvi = await self._safe_mean_ndvi(band_links, stac_id)
        metadata = SatelliteLayerMetadata(
            stac_id=stac_id,
            cloud_cover=cloud_cover,
            ndvi_mean=ndvi if ndvi is not None else NDVI_NO_DATA,
            lst_celsius_max=LST_NOT_AVAILABLE,
            band_links=band_links,
        )
        return metadata.model_dump()

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
            if href and (f"{band}." in href.upper() or f"{band}/" in href.upper()):
                return href
        return None

    # --- NDVI --------------------------------------------------------------

    async def _safe_mean_ndvi(self, band_links: dict[str, str], stac_id: str) -> float | None:
        """Compute mean NDVI, converting every failure into ``None``.

        Args:
            band_links: Resolved band hrefs for the scene.
            stac_id: Scene identifier, for logging.

        Returns:
            The mean NDVI, or ``None`` if it could not be computed.
        """
        missing = [band for band in (RED_BAND, NIR_BAND) if band not in band_links]
        if missing:
            logger.warning(
                "scene %s is missing band(s) %s; NDVI not computed", stac_id, ", ".join(missing)
            )
            return None

        try:
            # rasterio is blocking and GDAL does network I/O, so keep it off the
            # event loop.
            return await asyncio.to_thread(
                self._mean_ndvi, band_links[RED_BAND], band_links[NIR_BAND]
            )
        except Exception as exc:  # noqa: BLE001 - any raster failure is non-fatal
            logger.warning(
                "could not read bands for scene %s: %s: %s",
                stac_id,
                type(exc).__name__,
                exc,
            )
            return None

    def _mean_ndvi(self, red_href: str, nir_href: str) -> float | None:
        """Compute mean NDVI from two band hrefs. Blocking; runs in a thread.

        Args:
            red_href: Asset href for the red band.
            nir_href: Asset href for the NIR band.

        Returns:
            Mean NDVI over valid pixels, or ``None`` if no pixel is usable.

        Raises:
            Exception: Propagates rasterio/GDAL errors to
                :meth:`_safe_mean_ndvi`.
        """
        red = self._band_reader(red_href)
        nir = self._band_reader(nir_href)
        return mean_ndvi(red, nir)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


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


def read_band_sample(href: str, *, max_sample_pixels: int = 256) -> BandSample:
    """Read a decimated sample of one band. Blocking; runs in a thread.

    Full-resolution Sentinel-2 10 m tiles are 10980x10980, so the band is
    decimated by an integer factor that brings it under ``max_sample_pixels``
    using average resampling. That keeps the estimate representative while
    bounding memory and bandwidth.

    Args:
        href: Asset href or local path.
        max_sample_pixels: Upper bound on the number of pixels returned.

    Returns:
        A :class:`BandSample` of the first band.

    Raises:
        rasterio.errors.RasterioError: If the asset cannot be opened or read.
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(href) as source:
        height, width = source.height, source.width
        if height <= 0 or width <= 0:
            raise SatelliteServiceError(f"band {href!r} reported an empty extent")
        # Integer decimation factor that brings the tile under the pixel budget.
        factor = max(1, math.ceil(math.sqrt((height * width) / max(1, max_sample_pixels))))
        out_shape = (max(1, height // factor), max(1, width // factor))
        data = source.read(
            1, out_shape=out_shape, resampling=Resampling.average, masked=False
        )
        return BandSample(values=data, nodata=source.nodata)


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
