"""Shared fixtures for the GeoSentinel-AI test suite.

Nothing here touches the network or GDAL's remote drivers: HTTP is stubbed with
:class:`httpx.MockTransport` and rasters are real GeoTIFFs written into a
temporary directory, so the suite runs offline and deterministically.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from app.services.satellite_service import BandSample, SatelliteDataService

STAC_SEARCH_PATH = "/search"


# ---------------------------------------------------------------------------
# STAC fixtures
# ---------------------------------------------------------------------------


def make_item(
    item_id: str = "S2B_MSIL2A_20240615T103029_N0510_R108_T31SET",
    cloud_cover: float | None = 5.0,
    cloud_key: str = "eo:cloud_cover",
    red_href: str | None = "https://eodata.dataspace.copernicus.eu/B04.jp2",
    nir_href: str | None = "https://eodata.dataspace.copernicus.eu/B08.jp2",
    asset_style: str = "canonical",
    extra_assets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a STAC item resembling a Copernicus Data Space Sentinel-2 scene.

    Args:
        item_id: Scene identifier.
        cloud_cover: Cloud cover percentage, or ``None`` to omit the property.
        cloud_key: Property name carrying the cloud cover.
        red_href: Red band href, or ``None`` to omit the asset.
        nir_href: NIR band href, or ``None`` to omit the asset.
        asset_style: How band assets are keyed — ``canonical`` (``B04``),
            ``lowercase`` (``b04``), ``eo_bands`` (matched via ``eo:bands``),
            ``filename`` (matched from the href), or ``none``.
        extra_assets: Additional assets to merge in.

    Returns:
        A STAC item dictionary.
    """
    properties: dict[str, Any] = {"datetime": "2024-06-15T10:30:29Z"}
    if cloud_cover is not None:
        properties[cloud_key] = cloud_cover

    assets: dict[str, Any] = {"PRODUCT": {"href": "https://example.org/product.zip"}}
    if asset_style == "canonical":
        if red_href:
            assets["B04"] = {"href": red_href, "title": "Band 4 (red)"}
        if nir_href:
            assets["B08"] = {"href": nir_href, "title": "Band 8 (NIR)"}
    elif asset_style == "lowercase":
        if red_href:
            assets["b04"] = {"href": red_href}
        if nir_href:
            assets["b08"] = {"href": nir_href}
    elif asset_style == "eo_bands":
        if red_href:
            assets["red_band"] = {"href": red_href, "eo:bands": [{"name": "B04"}]}
        if nir_href:
            assets["nir_band"] = {"href": nir_href, "eo:bands": [{"name": "B08"}]}
    elif asset_style == "filename":
        if red_href:
            assets["tile_0"] = {"href": red_href}
        if nir_href:
            assets["tile_1"] = {"href": nir_href}
    elif asset_style != "none":
        raise AssertionError(f"unknown asset_style {asset_style!r}")

    if extra_assets:
        assets.update(extra_assets)

    return {
        "type": "Feature",
        "id": item_id,
        "bbox": [2.5, 31.0, 3.5, 32.0],
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[2.5, 31.0], [3.5, 31.0], [3.5, 32.0], [2.5, 32.0], [2.5, 31.0]]],
        },
        "properties": properties,
        "assets": assets,
    }


def stac_response(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap items in a STAC FeatureCollection envelope."""
    return {
        "type": "FeatureCollection",
        "features": features,
        "numberMatched": len(features),
        "numberReturned": len(features),
    }


def json_handler(payload: Any, status_code: int = 200) -> Callable[[httpx.Request], httpx.Response]:
    """Return a MockTransport handler that always answers with ``payload``."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload)

    return handler


def raising_handler(exc: Exception) -> Callable[[httpx.Request], httpx.Response]:
    """Return a MockTransport handler that raises ``exc``."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


def recording_handler(
    payload: Any, status_code: int = 200
) -> tuple[Callable[[httpx.Request], httpx.Response], list[httpx.Request]]:
    """Return a handler plus the list it appends each request to."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status_code, json=payload)

    return handler, seen


# ---------------------------------------------------------------------------
# Raster fixtures
# ---------------------------------------------------------------------------


def write_geotiff(path: Path, array: np.ndarray, nodata: float | None = None) -> Path:
    """Write a single-band GeoTIFF over a 1x1 degree EPSG:4326 footprint."""
    height, width = array.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=array.dtype,
        crs="EPSG:4326",
        transform=from_bounds(2.5, 31.0, 3.5, 32.0, width, height),
        nodata=nodata,
    ) as destination:
        destination.write(array, 1)
    return path


@pytest.fixture
def band_reader() -> Callable[[str], BandSample]:
    """A band reader backed by an in-memory dict of href -> sample."""
    samples: dict[str, BandSample] = {}

    def reader(href: str) -> BandSample:
        if href not in samples:
            raise FileNotFoundError(f"no stub sample for {href}")
        return samples[href]

    reader.samples = samples  # type: ignore[attr-defined]
    return reader


@pytest.fixture
def service_factory() -> Callable[..., SatelliteDataService]:
    """Build a SatelliteDataService wired to a stub transport and reader."""

    def factory(
        payload: Any = None,
        *,
        status_code: int = 200,
        exc: Exception | None = None,
        reader: Callable[[str], BandSample] | None = None,
        **kwargs: Any,
    ) -> SatelliteDataService:
        if exc is not None:
            handler = raising_handler(exc)
        else:
            handler = json_handler(payload if payload is not None else stac_response([]), status_code)
        return SatelliteDataService(transport=httpx.MockTransport(handler), band_reader=reader, **kwargs)

    return factory


@pytest.fixture
def valid_window() -> dict[str, Any]:
    """A bounding box and time window that satisfy SpatialQueryRequest."""
    return {"bbox": [2.5, 31.0, 3.5, 32.0], "start_date": "2024-06-01", "end_date": "2024-06-30"}
