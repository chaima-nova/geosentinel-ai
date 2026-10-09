"""Integration tests against the live Copernicus Data Space STAC API.

These are **deselected by default** because they need network access to
``dataspace.copernicus.eu``. Run them explicitly::

    pytest -m integration

Each test also skips itself if the API cannot be reached, so a run on an
offline machine or in CI without egress reports skips rather than failures.

What these add over the unit suite
----------------------------------
The unit tests pin the service against a *stubbed* STAC response, so they
cannot detect drift in the real API's shape: a renamed cloud-cover property, a
different asset key, or a moved search endpoint. These tests fail if that
happens.

They deliberately do **not** assert that a scene exists for the chosen window,
because that varies with the archive. They assert only structural facts: the
search succeeds, and whatever comes back validates as
:class:`~app.core.schemas.SatelliteLayerMetadata`.
"""

from __future__ import annotations

import logging

import httpx
import pytest

from app.core.schemas import SatelliteLayerMetadata
from app.services.satellite_service import (
    DEFAULT_STAC_URL,
    NDVI_NO_DATA,
    UNAVAILABLE_STAC_ID,
    SatelliteDataService,
)

pytestmark = pytest.mark.integration

# Ouargla, Algeria: arid, frequently cloud-free, so the archive usually has a
# usable Sentinel-2 scene inside the window.
OUARGLA_BBOX = [5.2, 31.8, 5.5, 32.1]
WINDOW = {"start_date": "2024-06-01", "end_date": "2024-06-30"}


def _api_reachable(timeout: float = 10.0) -> bool:
    """Return True if the CDSE STAC root answers at all."""
    try:
        response = httpx.get(DEFAULT_STAC_URL, timeout=timeout, follow_redirects=True)
    except httpx.HTTPError:
        return False
    return response.status_code < 500


@pytest.fixture(scope="module")
def api_available() -> bool:
    """Module-scoped reachability probe, so we only pay for it once."""
    reachable = _api_reachable()
    if not reachable:
        logging.getLogger(__name__).warning(
            "%s is unreachable from this environment; integration tests will skip",
            DEFAULT_STAC_URL,
        )
    return reachable


@pytest.fixture
def require_api(api_available: bool) -> None:
    """Skip the test when the live API cannot be reached."""
    if not api_available:
        pytest.skip(f"{DEFAULT_STAC_URL} is unreachable")


async def test_stac_root_serves_the_sentinel2_collection(require_api: None) -> None:
    """The collection the service defaults to must exist."""
    async with httpx.AsyncClient(base_url=DEFAULT_STAC_URL, timeout=30.0) as client:
        response = await client.get("/collections/SENTINEL-2")
    assert response.status_code == 200, response.text[:300]
    payload = response.json()
    assert payload.get("id") == "SENTINEL-2"


async def test_search_endpoint_accepts_our_parameters(require_api: None) -> None:
    """The exact query the service builds must be accepted by the real API."""
    async with SatelliteDataService() as service:
        items = await service.search_items(_window())
    assert isinstance(items, list)


async def test_fetch_returns_valid_metadata(require_api: None) -> None:
    """Whatever the archive returns, the result must validate against the schema."""
    async with SatelliteDataService() as service:
        result = await service.fetch_sentinel_indices(bbox=OUARGLA_BBOX, **WINDOW)

    metadata = SatelliteLayerMetadata.model_validate(result)
    assert 0.0 <= metadata.cloud_cover <= 100.0
    assert -1.0 <= metadata.ndvi_mean <= 1.0
    assert metadata.stac_id  # never empty

    if metadata.stac_id == UNAVAILABLE_STAC_ID:
        # No usable scene in the window: degradation must be complete and coherent.
        assert metadata.ndvi_mean == NDVI_NO_DATA
        assert metadata.band_links == {}
    else:
        # A real scene: the index must have been attempted, and the sentinel
        # is only acceptable if the band genuinely could not be read.
        assert metadata.band_links, "a real scene should expose at least one band"


async def test_cloud_cover_threshold_is_respected(require_api: None) -> None:
    """Any scene selected must sit under the configured cloud threshold."""
    async with SatelliteDataService(max_cloud_cover=10.0) as service:
        items = await service.search_items(_window())
    candidates = service._filter_by_cloud_cover(items)
    for item in candidates:
        cover = service._cloud_cover(item, default=None)
        assert cover is None or cover < 10.0


def _window():
    """Build a validated window for direct :meth:`search_items` calls."""
    from app.core.schemas import SpatialQueryRequest

    return SpatialQueryRequest(
        query="integration probe", bbox=OUARGLA_BBOX, **WINDOW
    )
