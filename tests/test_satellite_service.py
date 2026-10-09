"""Unit tests for :mod:`app.services.satellite_service`.

These run fully offline: HTTP is stubbed with :class:`httpx.MockTransport` and
band pixels come from an injected in-memory reader, except in
``TestRasterioBandReader`` which exercises the real rasterio read path against
GeoTIFFs written to a temporary directory.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from pydantic import ValidationError

from app.core.schemas import SatelliteLayerMetadata
from app.services.satellite_service import (
    LST_NOT_AVAILABLE,
    NDVI_NO_DATA,
    UNAVAILABLE_STAC_ID,
    BandSample,
    SatelliteDataService,
    SatelliteServiceError,
    mean_ndvi,
    read_band_sample,
)
from tests.conftest import (
    json_handler,
    make_item,
    recording_handler,
    stac_response,
    write_geotiff,
)

# red / nir chosen so the expected mean NDVI is exactly 0.475 by hand:
#   (0.4/0.6 + 0.4/0.8 + 0.4/1.0 + 0.4/1.2) / 4
RED = np.array([[0.1, 0.2], [0.3, 0.4]])
NIR = np.array([[0.5, 0.6], [0.7, 0.8]])
EXPECTED_NDVI = 0.475


def stub_bands(reader: Any, red: np.ndarray = RED, nir: np.ndarray = NIR) -> None:
    """Populate an in-memory reader with the default red/NIR stub hrefs."""
    reader.samples["https://eodata.dataspace.copernicus.eu/B04.jp2"] = BandSample(red)
    reader.samples["https://eodata.dataspace.copernicus.eu/B08.jp2"] = BandSample(nir)


# ---------------------------------------------------------------------------
# NDVI mathematics
# ---------------------------------------------------------------------------


class TestMeanNdvi:
    """The index formula and its masking rules."""

    def test_matches_hand_computed_mean(self) -> None:
        result = mean_ndvi(BandSample(RED), BandSample(NIR))
        assert result == pytest.approx(EXPECTED_NDVI, abs=1e-12)

    def test_formula_is_nir_minus_red_over_sum(self) -> None:
        # A single pixel makes the formula unambiguous.
        result = mean_ndvi(BandSample(np.array([[0.2]])), BandSample(np.array([[0.6]])))
        assert result == pytest.approx((0.6 - 0.2) / (0.6 + 0.2))

    def test_full_vegetation_and_bare_soil_extremes(self) -> None:
        assert mean_ndvi(BandSample(np.array([[0.0]])), BandSample(np.array([[1.0]]))) == pytest.approx(1.0)
        assert mean_ndvi(BandSample(np.array([[1.0]])), BandSample(np.array([[0.0]]))) == pytest.approx(-1.0)

    def test_nodata_pixels_are_excluded(self) -> None:
        red = RED.copy()
        red[1, 0] = -9999.0
        result = mean_ndvi(BandSample(red, nodata=-9999.0), BandSample(NIR))
        # Remaining three pixels: 0.666667, 0.5, 0.333333 -> mean 0.5
        assert result == pytest.approx(0.5, abs=1e-12)

    def test_non_finite_pixels_are_excluded(self) -> None:
        red = RED.copy()
        red[0, 0] = np.nan
        result = mean_ndvi(BandSample(red), BandSample(NIR))
        assert result == pytest.approx((0.5 + 0.4 + 1 / 3) / 3, abs=1e-12)

    def test_zero_denominator_returns_none(self) -> None:
        zeros = np.array([[0.0]])
        assert mean_ndvi(BandSample(zeros), BandSample(zeros)) is None

    def test_zero_denominator_pixel_skipped_others_kept(self) -> None:
        red = np.array([[0.0, 0.2]])
        nir = np.array([[0.0, 0.6]])
        assert mean_ndvi(BandSample(red), BandSample(nir)) == pytest.approx(0.5)

    def test_all_nodata_returns_none(self) -> None:
        red = np.full((2, 2), -9999.0)
        assert mean_ndvi(BandSample(red, nodata=-9999.0), BandSample(NIR)) is None

    def test_out_of_range_result_is_clamped(self) -> None:
        # Negative reflectance pushes the raw ratio below -1.
        result = mean_ndvi(BandSample(np.array([[-0.2]])), BandSample(np.array([[0.1]])))
        assert result == -1.0

    def test_mismatched_shapes_return_none(self) -> None:
        assert mean_ndvi(BandSample(np.zeros((2, 2))), BandSample(np.zeros((3, 3)))) is None

    def test_integer_input_is_accepted(self) -> None:
        result = mean_ndvi(
            BandSample(np.array([[1, 2]], dtype="uint16")),
            BandSample(np.array([[3, 4]], dtype="uint16")),
        )
        assert result == pytest.approx(((3 - 1) / 4 + (4 - 2) / 6) / 2)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestFetchSentinelIndices:
    """End-to-end behaviour of the public method."""

    async def test_returns_valid_metadata(self, service_factory, band_reader, valid_window) -> None:
        stub_bands(band_reader)
        service = service_factory(stac_response([make_item(cloud_cover=5.0)]), reader=band_reader)

        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)

        # The contract: it must validate as SatelliteLayerMetadata, not merely look like one.
        metadata = SatelliteLayerMetadata.model_validate(result)
        assert metadata.stac_id == "S2B_MSIL2A_20240615T103029_N0510_R108_T31SET"
        assert metadata.cloud_cover == 5.0
        assert metadata.ndvi_mean == pytest.approx(EXPECTED_NDVI)
        assert metadata.band_links == {
            "B04": "https://eodata.dataspace.copernicus.eu/B04.jp2",
            "B08": "https://eodata.dataspace.copernicus.eu/B08.jp2",
        }

    async def test_result_is_a_plain_dict(self, service_factory, band_reader, valid_window) -> None:
        stub_bands(band_reader)
        service = service_factory(stac_response([make_item()]), reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert isinstance(result, dict)
        assert set(result) == {"stac_id", "cloud_cover", "ndvi_mean", "lst_celsius_max", "band_links"}

    async def test_lst_is_placeholder_because_sentinel2_has_no_thermal_band(
        self, service_factory, band_reader, valid_window
    ) -> None:
        stub_bands(band_reader)
        service = service_factory(stac_response([make_item()]), reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["lst_celsius_max"] == LST_NOT_AVAILABLE

    async def test_selects_the_least_cloudy_scene(self, service_factory, band_reader, valid_window) -> None:
        stub_bands(band_reader)
        items = [
            make_item("scene_cloudy", cloud_cover=18.0),
            make_item("scene_clear", cloud_cover=1.5),
            make_item("scene_medium", cloud_cover=9.0),
        ]
        service = service_factory(stac_response(items), reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["stac_id"] == "scene_clear"
        assert result["cloud_cover"] == 1.5

    async def test_cloud_cover_at_threshold_is_rejected(self, service_factory, band_reader, valid_window) -> None:
        stub_bands(band_reader)
        service = service_factory(
            stac_response([make_item("at_threshold", cloud_cover=20.0)]), reader=band_reader
        )
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["stac_id"] == UNAVAILABLE_STAC_ID

    async def test_custom_threshold_is_honoured(self, service_factory, band_reader, valid_window) -> None:
        stub_bands(band_reader)
        items = [make_item("a", cloud_cover=35.0), make_item("b", cloud_cover=45.0)]
        service = service_factory(
            stac_response(items), reader=band_reader, max_cloud_cover=40.0
        )
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["stac_id"] == "a"

    async def test_unknown_cloud_cover_is_kept_but_sorted_last(
        self, service_factory, band_reader, valid_window
    ) -> None:
        stub_bands(band_reader)
        items = [make_item("no_cloud_info", cloud_cover=None), make_item("known", cloud_cover=15.0)]
        service = service_factory(stac_response(items), reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["stac_id"] == "known"

    async def test_unknown_cloud_cover_falls_back_when_it_is_all_there_is(
        self, service_factory, band_reader, valid_window
    ) -> None:
        stub_bands(band_reader)
        service = service_factory(
            stac_response([make_item("no_cloud_info", cloud_cover=None)]), reader=band_reader
        )
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["stac_id"] == "no_cloud_info"
        assert result["cloud_cover"] == 100.0  # CLOUD_COVER_UNKNOWN sentinel


# ---------------------------------------------------------------------------
# STAC request construction
# ---------------------------------------------------------------------------


class TestStacRequest:
    """The query sent to the Copernicus Data Space API."""

    async def test_query_parameters(self, service_factory, band_reader, valid_window) -> None:
        stub_bands(band_reader)
        handler, seen = recording_handler(stac_response([make_item()]))
        service = SatelliteDataService(transport=httpx.MockTransport(handler), band_reader=band_reader)
        async with service:
            await service.fetch_sentinel_indices(**valid_window)

        assert len(seen) == 1
        request = seen[0]
        # httpx merges base_url's /stac path with /search, giving the real
        # Copernicus Data Space endpoint.
        assert str(request.url).startswith("https://dataspace.copernicus.eu/stac/search")
        params = dict(request.url.params)
        assert params["collections"] == "SENTINEL-2"
        assert params["bbox"] == "2.500000,31.000000,3.500000,32.000000"
        assert params["datetime"] == "2024-06-01T00:00:00Z/2024-06-30T00:00:00Z"
        assert params["limit"] == "10"
        assert json.loads(params["query"]) == {"eo:cloud_cover": {"lt": 20.0}}

    async def test_timezone_offsets_are_normalised_to_utc(self, service_factory, band_reader) -> None:
        stub_bands(band_reader)
        handler, seen = recording_handler(stac_response([]))
        service = SatelliteDataService(transport=httpx.MockTransport(handler), band_reader=band_reader)
        async with service:
            await service.fetch_sentinel_indices(
                bbox=[0.0, 0.0, 1.0, 1.0],
                start_date="2024-06-01T02:00:00+02:00",
                end_date="2024-06-02T00:00:00Z",
            )
        assert dict(seen[0].url.params)["datetime"] == "2024-06-01T00:00:00Z/2024-06-02T00:00:00Z"

    async def test_limit_and_base_url_are_configurable(self, band_reader, valid_window) -> None:
        stub_bands(band_reader)
        handler, seen = recording_handler(stac_response([]))
        service = SatelliteDataService(
            base_url="https://stac.example.org/api/",
            transport=httpx.MockTransport(handler),
            band_reader=band_reader,
            limit=3,
        )
        assert service.base_url == "https://stac.example.org/api"  # trailing slash stripped
        async with service:
            await service.fetch_sentinel_indices(**valid_window)
        assert dict(seen[0].url.params)["limit"] == "3"
        assert str(seen[0].url).startswith("https://stac.example.org/api/search")

    async def test_accepts_geojson_content_type(self, band_reader, valid_window) -> None:
        stub_bands(band_reader)
        handler, seen = recording_handler(stac_response([]))
        service = SatelliteDataService(transport=httpx.MockTransport(handler), band_reader=band_reader)
        async with service:
            await service.fetch_sentinel_indices(**valid_window)
        assert seen[0].headers["accept"] == "application/geo+json"


# ---------------------------------------------------------------------------
# Cloud cover parsing
# ---------------------------------------------------------------------------


class TestCloudCoverParsing:
    """Tolerance for catalogs that name the property differently."""

    @pytest.mark.parametrize("key", ["eo:cloud_cover", "cloudCover", "cloud_cover", "s2:cloud_cover"])
    async def test_recognised_property_names(self, service_factory, band_reader, valid_window, key) -> None:
        stub_bands(band_reader)
        service = service_factory(stac_response([make_item(cloud_cover=7.5, cloud_key=key)]), reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["cloud_cover"] == 7.5

    async def test_out_of_range_value_is_clamped(self, service_factory, band_reader, valid_window) -> None:
        stub_bands(band_reader)
        service = service_factory(stac_response([make_item(cloud_cover=145.0)]), reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["cloud_cover"] == 100.0

    async def test_non_numeric_value_is_ignored(self, service_factory, band_reader, valid_window) -> None:
        stub_bands(band_reader)
        service = service_factory(
            stac_response([make_item(cloud_cover=None, extra_assets={})]), reader=band_reader
        )
        item = make_item(cloud_cover=None)
        item["properties"]["eo:cloud_cover"] = "not a number"
        service = service_factory(stac_response([item]), reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["cloud_cover"] == 100.0

    async def test_boolean_value_is_ignored(self, service_factory, band_reader, valid_window) -> None:
        stub_bands(band_reader)
        item = make_item(cloud_cover=None)
        item["properties"]["eo:cloud_cover"] = True
        service = service_factory(stac_response([item]), reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["cloud_cover"] == 100.0

    async def test_missing_properties_block_is_tolerated(self, service_factory, band_reader, valid_window) -> None:
        stub_bands(band_reader)
        item = make_item()
        del item["properties"]
        service = service_factory(stac_response([item]), reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["cloud_cover"] == 100.0


# ---------------------------------------------------------------------------
# Band resolution
# ---------------------------------------------------------------------------


class TestBandResolution:
    """Locating red/NIR assets across differing catalog conventions."""

    @pytest.mark.parametrize("style", ["canonical", "lowercase", "eo_bands", "filename"])
    async def test_resolves_alternative_asset_naming(
        self, service_factory, band_reader, valid_window, style
    ) -> None:
        stub_bands(band_reader)
        service = service_factory(stac_response([make_item(asset_style=style)]), reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["band_links"] == {
            "B04": "https://eodata.dataspace.copernicus.eu/B04.jp2",
            "B08": "https://eodata.dataspace.copernicus.eu/B08.jp2",
        }
        assert result["ndvi_mean"] == pytest.approx(EXPECTED_NDVI)

    async def test_alternate_href_is_used_when_primary_missing(self, service_factory, band_reader, valid_window) -> None:
        stub_bands(band_reader)
        item = make_item(asset_style="none", extra_assets={
            "B04": {"alternate": {"s3": {"href": "https://eodata.dataspace.copernicus.eu/B04.jp2"}}},
            "B08": {"alternate": {"s3": {"href": "https://eodata.dataspace.copernicus.eu/B08.jp2"}}},
        })
        service = service_factory(stac_response([item]), reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert set(result["band_links"]) == {"B04", "B08"}
        assert result["ndvi_mean"] == pytest.approx(EXPECTED_NDVI)

    @pytest.mark.parametrize("missing", ["B04", "B08"])
    async def test_missing_band_degrades_to_no_data(
        self, service_factory, band_reader, valid_window, missing
    ) -> None:
        stub_bands(band_reader)
        item = make_item(red_href=None if missing == "B04" else "https://eodata.dataspace.copernicus.eu/B04.jp2",
                         nir_href=None if missing == "B08" else "https://eodata.dataspace.copernicus.eu/B08.jp2")
        service = service_factory(stac_response([item]), reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)

        metadata = SatelliteLayerMetadata.model_validate(result)  # still valid
        assert metadata.ndvi_mean == NDVI_NO_DATA
        assert missing not in metadata.band_links
        assert metadata.stac_id != UNAVAILABLE_STAC_ID  # scene was found, only NDVI failed

    async def test_missing_assets_block_degrades(self, service_factory, band_reader, valid_window) -> None:
        stub_bands(band_reader)
        item = make_item()
        del item["assets"]
        service = service_factory(stac_response([item]), reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["ndvi_mean"] == NDVI_NO_DATA
        assert result["band_links"] == {}

    async def test_asset_without_href_is_skipped(self, service_factory, band_reader, valid_window) -> None:
        stub_bands(band_reader)
        item = make_item(asset_style="none", extra_assets={"B04": {"title": "no href"}, "B08": {"href": ""}})
        service = service_factory(stac_response([item]), reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["band_links"] == {}
        assert result["ndvi_mean"] == NDVI_NO_DATA


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


class TestFailureHandling:
    """Operational failures degrade; they never raise."""

    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ConnectError("connection refused"),
            httpx.ConnectTimeout("timed out"),
            httpx.ReadTimeout("read timed out"),
            httpx.RemoteProtocolError("peer closed connection"),
        ],
    )
    async def test_transport_failures_degrade(self, service_factory, band_reader, valid_window, exc) -> None:
        stub_bands(band_reader)
        service = service_factory(exc=exc, reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)

        metadata = SatelliteLayerMetadata.model_validate(result)
        assert metadata.stac_id == UNAVAILABLE_STAC_ID
        assert metadata.ndvi_mean == NDVI_NO_DATA
        assert metadata.band_links == {}

    @pytest.mark.parametrize("status_code", [400, 401, 403, 404, 500, 502, 503])
    async def test_http_error_status_degrades(
        self, service_factory, band_reader, valid_window, status_code
    ) -> None:
        stub_bands(band_reader)
        service = service_factory(stac_response([]), status_code=status_code, reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["stac_id"] == UNAVAILABLE_STAC_ID

    async def test_non_json_body_degrades(self, band_reader, valid_window) -> None:
        stub_bands(band_reader)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>not json</html>")

        service = SatelliteDataService(
            transport=httpx.MockTransport(handler), band_reader=band_reader
        )
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["stac_id"] == UNAVAILABLE_STAC_ID

    @pytest.mark.parametrize(
        "payload",
        [
            [],  # JSON array, not an object
            {"type": "FeatureCollection"},  # no 'features'
            {"features": "not a list"},
            {"features": {"a": 1}},
        ],
    )
    async def test_malformed_stac_envelope_degrades(
        self, service_factory, band_reader, valid_window, payload
    ) -> None:
        stub_bands(band_reader)
        service = service_factory(payload, reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["stac_id"] == UNAVAILABLE_STAC_ID

    async def test_non_dict_features_are_dropped(self, service_factory, band_reader, valid_window) -> None:
        stub_bands(band_reader)
        payload = {"type": "FeatureCollection", "features": ["junk", 42, make_item("good")]}
        service = service_factory(payload, reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["stac_id"] == "good"

    async def test_empty_result_set_degrades(self, service_factory, band_reader, valid_window) -> None:
        stub_bands(band_reader)
        service = service_factory(stac_response([]), reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["stac_id"] == UNAVAILABLE_STAC_ID
        assert result["cloud_cover"] == 100.0

    async def test_band_reader_failure_degrades_ndvi_only(
        self, service_factory, band_reader, valid_window
    ) -> None:
        # Reader is empty, so it raises FileNotFoundError for both hrefs.
        service = service_factory(stac_response([make_item()]), reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)

        metadata = SatelliteLayerMetadata.model_validate(result)
        assert metadata.ndvi_mean == NDVI_NO_DATA
        assert metadata.stac_id != UNAVAILABLE_STAC_ID
        assert metadata.cloud_cover == 5.0  # scene metadata survived

    async def test_shape_mismatch_degrades_ndvi_only(self, service_factory, band_reader, valid_window) -> None:
        band_reader.samples["https://eodata.dataspace.copernicus.eu/B04.jp2"] = BandSample(np.zeros((2, 2)))
        band_reader.samples["https://eodata.dataspace.copernicus.eu/B08.jp2"] = BandSample(np.zeros((3, 3)))
        service = service_factory(stac_response([make_item()]), reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["ndvi_mean"] == NDVI_NO_DATA
        assert result["stac_id"] != UNAVAILABLE_STAC_ID

    async def test_item_without_id_falls_back(self, service_factory, band_reader, valid_window) -> None:
        stub_bands(band_reader)
        item = make_item()
        del item["id"]
        service = service_factory(stac_response([item]), reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["stac_id"] == UNAVAILABLE_STAC_ID
        assert result["ndvi_mean"] == pytest.approx(EXPECTED_NDVI)  # NDVI still computed


class TestInputValidation:
    """Bad caller input is surfaced, not masked."""

    @pytest.mark.parametrize(
        "bbox",
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0], [-200.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 95.0],
         [10.0, 0.0, 5.0, 10.0], []],
    )
    async def test_invalid_bbox_raises(self, service_factory, band_reader, bbox) -> None:
        service = service_factory(stac_response([]), reader=band_reader)
        async with service:
            with pytest.raises(ValidationError):
                await service.fetch_sentinel_indices(bbox=bbox, start_date="2024-06-01", end_date="2024-06-30")

    async def test_backwards_window_raises(self, service_factory, band_reader) -> None:
        service = service_factory(stac_response([]), reader=band_reader)
        async with service:
            with pytest.raises(ValidationError):
                await service.fetch_sentinel_indices(
                    bbox=[0.0, 0.0, 1.0, 1.0], start_date="2024-06-30", end_date="2024-06-01"
                )

    async def test_unparseable_date_raises(self, service_factory, band_reader) -> None:
        service = service_factory(stac_response([]), reader=band_reader)
        async with service:
            with pytest.raises(ValidationError):
                await service.fetch_sentinel_indices(
                    bbox=[0.0, 0.0, 1.0, 1.0], start_date="01/06/2024", end_date="2024-06-30"
                )

    async def test_no_request_is_made_when_input_is_invalid(self, band_reader) -> None:
        handler, seen = recording_handler(stac_response([]))
        service = SatelliteDataService(transport=httpx.MockTransport(handler), band_reader=band_reader)
        async with service:
            with pytest.raises(ValidationError):
                await service.fetch_sentinel_indices(
                    bbox=[0.0, 0.0, 1.0], start_date="2024-06-01", end_date="2024-06-30"
                )
        assert seen == []


class TestConstructionGuards:
    """Invalid service configuration fails immediately."""

    @pytest.mark.parametrize("cloud", [-1.0, 100.1, math.nan])
    def test_bad_cloud_cover_rejected(self, cloud: float) -> None:
        with pytest.raises(SatelliteServiceError):
            SatelliteDataService(max_cloud_cover=cloud)

    @pytest.mark.parametrize("limit", [0, -5])
    def test_bad_limit_rejected(self, limit: int) -> None:
        with pytest.raises(SatelliteServiceError):
            SatelliteDataService(limit=limit)

    def test_bad_max_sample_pixels_rejected(self) -> None:
        with pytest.raises(SatelliteServiceError):
            SatelliteDataService(max_sample_pixels=0)

    def test_cloud_cover_boundaries_accepted(self) -> None:
        assert SatelliteDataService(max_cloud_cover=0.0).max_cloud_cover == 0.0
        assert SatelliteDataService(max_cloud_cover=100.0).max_cloud_cover == 100.0


# ---------------------------------------------------------------------------
# Client lifecycle
# ---------------------------------------------------------------------------


class TestClientLifecycle:
    """The HTTP client is created lazily and closed correctly."""

    async def test_context_manager_creates_and_closes_client(self, service_factory, band_reader) -> None:
        service = service_factory(stac_response([]), reader=band_reader)
        assert service._client is None
        async with service:
            assert service._client is not None
        assert service._client is None

    async def test_explicit_aclose(self, service_factory, band_reader) -> None:
        service = service_factory(stac_response([]), reader=band_reader)
        await service.__aenter__()
        client = service._client
        assert client is not None
        await service.aclose()
        assert client.is_closed

    async def test_externally_supplied_client_is_not_closed(self, band_reader) -> None:
        client = httpx.AsyncClient(
            base_url="https://dataspace.copernicus.eu/stac",
            transport=httpx.MockTransport(json_handler(stac_response([]))),
        )
        service = SatelliteDataService(client=client, band_reader=band_reader)
        async with service:
            await service.fetch_sentinel_indices(
                bbox=[0.0, 0.0, 1.0, 1.0], start_date="2024-06-01", end_date="2024-06-30"
            )
        assert not client.is_closed  # caller owns it
        await client.aclose()
        assert client.is_closed

    async def test_client_is_reused_across_calls(self, band_reader, valid_window) -> None:
        handler, seen = recording_handler(stac_response([]))
        service = SatelliteDataService(transport=httpx.MockTransport(handler), band_reader=band_reader)
        async with service:
            first = service._get_client()
            await service.fetch_sentinel_indices(**valid_window)
            second = service._get_client()
            await service.fetch_sentinel_indices(**valid_window)
        assert first is second
        assert len(seen) == 2


# ---------------------------------------------------------------------------
# Real rasterio read path (no network, real GeoTIFF files)
# ---------------------------------------------------------------------------


class TestRasterioBandReader:
    """The default reader against real rasters written by rasterio."""

    def test_reads_array_and_nodata(self, tmp_path: Path) -> None:
        path = write_geotiff(tmp_path / "b04.tif", RED.astype("float32"), nodata=-9999.0)
        sample = read_band_sample(str(path), max_sample_pixels=256)
        assert sample.values.shape == (2, 2)
        assert sample.nodata == -9999.0
        assert np.allclose(sample.values, RED, atol=1e-6)

    def test_decimation_respects_pixel_budget(self, tmp_path: Path) -> None:
        big = np.linspace(0.1, 0.9, 200 * 200, dtype="float32").reshape(200, 200)
        path = write_geotiff(tmp_path / "big.tif", big)
        sample = read_band_sample(str(path), max_sample_pixels=100)
        assert sample.values.size <= 100
        assert sample.values.size > 0

    def test_small_raster_is_never_upsampled(self, tmp_path: Path) -> None:
        path = write_geotiff(tmp_path / "tiny.tif", RED.astype("float32"))
        # Within budget: returned as-is.
        assert read_band_sample(str(path), max_sample_pixels=256).values.shape == (2, 2)
        # Below budget: decimated, and never larger than the source.
        tight = read_band_sample(str(path), max_sample_pixels=1)
        assert tight.values.shape == (1, 1)
        assert tight.values.size <= RED.size

    async def test_end_to_end_with_real_rasters(self, service_factory, tmp_path: Path, valid_window) -> None:
        """Full pipeline: mocked STAC, real GeoTIFFs, no injected reader."""
        red_path = write_geotiff(tmp_path / "b04.tif", RED.astype("float32"), nodata=-9999.0)
        nir_path = write_geotiff(tmp_path / "b08.tif", NIR.astype("float32"), nodata=-9999.0)
        item = make_item(red_href=str(red_path), nir_href=str(nir_path))

        service = service_factory(stac_response([item]))  # default rasterio reader
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)

        metadata = SatelliteLayerMetadata.model_validate(result)
        assert metadata.ndvi_mean == pytest.approx(EXPECTED_NDVI, abs=1e-6)

    async def test_missing_raster_file_degrades_ndvi(self, service_factory, tmp_path: Path, valid_window) -> None:
        item = make_item(
            red_href=str(tmp_path / "absent.tif"), nir_href=str(tmp_path / "also_absent.tif")
        )
        service = service_factory(stac_response([item]))
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["ndvi_mean"] == NDVI_NO_DATA
        assert result["stac_id"] != UNAVAILABLE_STAC_ID

    async def test_nodata_masking_through_the_real_reader(
        self, service_factory, tmp_path: Path, valid_window
    ) -> None:
        red = RED.astype("float32").copy()
        red[1, 0] = -9999.0
        red_path = write_geotiff(tmp_path / "b04.tif", red, nodata=-9999.0)
        nir_path = write_geotiff(tmp_path / "b08.tif", NIR.astype("float32"), nodata=-9999.0)
        item = make_item(red_href=str(red_path), nir_href=str(nir_path))

        service = service_factory(stac_response([item]))
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert result["ndvi_mean"] == pytest.approx(0.5, abs=1e-6)


class TestRealTransportFailure:
    """Degradation against a genuine transport failure, not a stub.

    The rest of the failure tests raise from a MockTransport. This one lets
    httpx fail for real against an unresolvable host, proving the handling
    holds on the actual transport path. Needs no egress, so it runs by default.
    """

    async def test_unresolvable_host_degrades_instead_of_raising(self) -> None:
        service = SatelliteDataService(
            base_url="https://geosentinel-nonexistent-host.invalid/stac", timeout=5.0
        )
        async with service:
            result = await service.fetch_sentinel_indices(
                bbox=[5.2, 31.8, 5.5, 32.1],
                start_date="2024-06-01",
                end_date="2024-06-30",
            )

        metadata = SatelliteLayerMetadata.model_validate(result)
        assert metadata.stac_id == UNAVAILABLE_STAC_ID
        assert metadata.ndvi_mean == NDVI_NO_DATA
        assert metadata.band_links == {}


# ---------------------------------------------------------------------------
# Defensive branches
# ---------------------------------------------------------------------------


class TestDefensiveBranches:
    """Edge cases that would otherwise be silently untested."""

    @pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
    def test_non_finite_cloud_cover_falls_back(self, bad: float) -> None:
        """A NaN/Inf cloud cover must not become a NaN percentage."""
        assert SatelliteDataService._cloud_cover(
            {"properties": {"eo:cloud_cover": bad}}, default=None
        ) is None
        assert SatelliteDataService._cloud_cover({"properties": {"eo:cloud_cover": bad}}) == 100.0

    async def test_non_finite_cloud_cover_end_to_end(self, band_reader, valid_window) -> None:
        """Same, through the full pipeline.

        The body is passed as raw text because httpx's ``json=`` encoder rejects
        non-finite floats with ``allow_nan=False``; ``response.json()`` accepts
        them, which is exactly how a broken upstream could deliver one.
        """
        stub_bands(band_reader)
        item = make_item(cloud_cover=None)
        item["properties"]["eo:cloud_cover"] = float("inf")
        body = json.dumps(stac_response([item]))
        assert "Infinity" in body  # guard: the stub really carries a non-finite value

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body, headers={"content-type": "application/json"})

        service = SatelliteDataService(
            transport=httpx.MockTransport(handler), band_reader=band_reader
        )
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)

        # Must be the scene path, not the degraded path.
        assert result["stac_id"] != UNAVAILABLE_STAC_ID
        assert result["cloud_cover"] == 100.0
        assert result["ndvi_mean"] == pytest.approx(EXPECTED_NDVI)

    async def test_non_dict_asset_entry_is_skipped(
        self, service_factory, band_reader, valid_window
    ) -> None:
        stub_bands(band_reader)
        item = make_item(
            asset_style="none",
            extra_assets={
                "B04": "not-an-asset-object",
                "B08": ["also", "wrong"],
                "thumbnail": {"href": "https://eodata.dataspace.copernicus.eu/TCI.jp2"},
            },
        )
        service = service_factory(stac_response([item]), reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        # The scene was still selected; only band resolution failed. Asserting
        # stac_id keeps this from passing via the degraded path by accident.
        assert result["stac_id"] != UNAVAILABLE_STAC_ID
        assert result["band_links"] == {}
        assert result["ndvi_mean"] == NDVI_NO_DATA

    async def test_case_insensitive_key_fallback(self, service_factory, band_reader, valid_window) -> None:
        """'RED'/'NIR' match the 'red'/'nir' keys only via the case-insensitive pass."""
        stub_bands(band_reader)
        item = make_item(
            asset_style="none",
            extra_assets={
                "RED": {"href": "https://eodata.dataspace.copernicus.eu/B04.jp2"},
                "NIR": {"href": "https://eodata.dataspace.copernicus.eu/B08.jp2"},
            },
        )
        service = service_factory(stac_response([item]), reader=band_reader)
        async with service:
            result = await service.fetch_sentinel_indices(**valid_window)
        assert set(result["band_links"]) == {"B04", "B08"}
        assert result["ndvi_mean"] == pytest.approx(EXPECTED_NDVI)

    def test_rfc3339_treats_naive_datetime_as_utc(self) -> None:
        from datetime import datetime

        from app.services.satellite_service import _rfc3339

        assert _rfc3339(datetime(2024, 6, 1, 12, 0, 0)) == "2024-06-01T12:00:00Z"

    def test_empty_extent_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeSource:
            height = 0
            width = 0
            nodata = None

            def __enter__(self) -> "FakeSource":
                return self

            def __exit__(self, *exc: object) -> None:
                return None

        import rasterio

        monkeypatch.setattr(rasterio, "open", lambda href: FakeSource())
        with pytest.raises(SatelliteServiceError, match="empty extent"):
            read_band_sample("https://example.org/B04.jp2")
