"""Reflectance-encoding and asset-naming regression tests for Sentinel-2 L2A.

Two kinds of input are used, and they are kept apart on purpose:

* ``tests/fixtures/sentinel2_l2a_S2B_20240825_T13TDE.json`` is a trimmed **real**
  Sentinel-2 L2A STAC item (metadata only, no raster). It pins the asset keys and
  processing baseline that a real product carries.
* Everything else is **synthetic** metadata, labelled as such, used to exercise
  each encoding rule and each fail-closed branch.

No test here touches the network.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from app.services.satellite_service import (
    NDVI_NO_DATA,
    NIR_BAND,
    RED_BAND,
    BandSample,
    SatelliteDataService,
    boa_add_offset,
    processing_baseline,
    resolve_reflectance_offset,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sentinel2_l2a_S2B_20240825_T13TDE.json"


def _real_item() -> dict[str, Any]:
    """The real Sentinel-2 L2A item (baseline 05.11), as a fresh copy."""
    return copy.deepcopy(json.loads(FIXTURE.read_text(encoding="utf-8")))


def _synthetic_item(
    *,
    baseline: str | None,
    product_id: str = "S2B_MSIL2A_20240825T173909_N0511_R098_T13TDE_20240825T215747",
) -> dict[str, Any]:
    """Synthetic item. ``baseline`` is the ``s2:processing_baseline`` string, if any."""
    properties: dict[str, Any] = {"eo:cloud_cover": 5.0}
    if baseline is not None:
        properties["s2:processing_baseline"] = baseline
    return {
        "id": product_id,
        "properties": properties,
        "assets": {
            "B04": {"href": "https://example.test/B04.tif"},
            "B08": {"href": "https://example.test/B08.tif"},
            "SCL": {"href": "https://example.test/SCL.tif"},
        },
    }


# ---------------------------------------------------------------------------
# Real item metadata
# ---------------------------------------------------------------------------


class TestRealItemMetadata:
    def test_real_item_baseline_and_offset(self) -> None:
        baseline, offset, reason = resolve_reflectance_offset(_real_item())
        assert baseline == pytest.approx(5.11)
        assert offset == -1000.0
        assert reason is None

    def test_real_item_asset_keys_resolve(self) -> None:
        item = _real_item()
        links = SatelliteDataService._resolve_bands(item)
        assert set(links) == {RED_BAND, NIR_BAND}
        assert links[RED_BAND].endswith("T13TDE_20240825T173909_B04_10m.tif")
        assert links[NIR_BAND].endswith("T13TDE_20240825T173909_B08_10m.tif")
        assert SatelliteDataService._resolve_scl(item) is not None
        assert SatelliteDataService._resolve_scl(item).endswith("_SCL_20m.tif")

    def test_real_item_baseline_agrees_with_product_uri(self) -> None:
        # The real item's id has no _Nxxyy_ token; the baseline is carried by the
        # property and by the product URI (.SAFE name), which must agree.
        item = _real_item()
        assert processing_baseline(item) == pytest.approx(5.11)
        assert item["properties"]["s2:processing_baseline"] == "05.11"
        assert "_N0511_" in item["properties"]["s2:product_uri"]


# ---------------------------------------------------------------------------
# Collection naming (CDSE asset keys)
# ---------------------------------------------------------------------------


class TestCdseAssetNaming:
    """CDSE names bands ``B04_10m``, ``B08_10m``, ``SCL_20m`` (synthetic item).

    The mapping is taken from the MIT-licensed mapchete-eo CDSE asset-name table.
    Before this fix the resolver missed these keys, so NDVI was never computed
    against the default CDSE endpoint.
    """

    def test_cdse_keys_resolve(self) -> None:
        item = {
            "id": "S2A_MSIL2A_20240601T100000_N0511_R022_T31SGA_20240601T120000",
            "assets": {
                "B04_10m": {"href": "https://eodata.example/T31SGA_B04_10m.jp2"},
                "B08_10m": {"href": "https://eodata.example/T31SGA_B08_10m.jp2"},
                "SCL_20m": {"href": "https://eodata.example/T31SGA_SCL_20m.jp2"},
            },
        }
        links = SatelliteDataService._resolve_bands(item)
        assert links[RED_BAND] == "https://eodata.example/T31SGA_B04_10m.jp2"
        assert links[NIR_BAND] == "https://eodata.example/T31SGA_B08_10m.jp2"
        assert SatelliteDataService._resolve_scl(item) == "https://eodata.example/T31SGA_SCL_20m.jp2"

    def test_filename_fallback_matches_cdse_style_href(self) -> None:
        item = {
            "id": "x",
            "assets": {"band_a": {"href": "https://eodata.example/T31SGA_B04_10m.jp2"}},
        }
        links = SatelliteDataService._resolve_bands(item)
        assert links.get(RED_BAND) == "https://eodata.example/T31SGA_B04_10m.jp2"


# ---------------------------------------------------------------------------
# Offset per processing baseline
# ---------------------------------------------------------------------------


class TestOffsetPerBaseline:
    @pytest.mark.parametrize(
        ("baseline", "expected_offset"),
        [
            ("02.07", 0.0),  # pre-04.00: DN carry no offset
            ("03.99", 0.0),  # just below the boundary
            ("04.00", -1000.0),  # first baseline with BOA_ADD_OFFSET = -1000
            ("04.01", -1000.0),
            ("05.11", -1000.0),
        ],
    )
    def test_offset_by_baseline(self, baseline: str, expected_offset: float) -> None:
        item = _synthetic_item(baseline=baseline, product_id="no-token-in-id")
        _, offset, reason = resolve_reflectance_offset(item)
        assert reason is None
        assert offset == expected_offset

    def test_boa_add_offset_boundary_is_inclusive(self) -> None:
        assert boa_add_offset(4.0) == -1000.0
        assert boa_add_offset(3.99) == 0.0

    def test_baseline_taken_from_product_name_when_property_absent(self) -> None:
        item = _synthetic_item(baseline=None)  # id carries _N0511_
        baseline, offset, reason = resolve_reflectance_offset(item)
        assert (baseline, offset, reason) == (pytest.approx(5.11), -1000.0, None)


# ---------------------------------------------------------------------------
# Fail-closed cases
# ---------------------------------------------------------------------------


class TestFailClosed:
    def test_unknown_baseline_withholds_offset(self) -> None:
        item = {"id": "no-token-here", "properties": {}, "assets": {}}
        baseline, offset, reason = resolve_reflectance_offset(item)
        assert baseline is None
        assert offset is None
        assert reason == "processing baseline is unknown"

    def test_unparseable_baseline_and_no_token_is_unknown(self) -> None:
        item = _synthetic_item(baseline="five-point-one", product_id="no-token")
        _, offset, reason = resolve_reflectance_offset(item)
        assert offset is None
        assert reason == "processing baseline is unknown"

    def test_inconsistent_baseline_is_withheld(self) -> None:
        # Property says 05.11 but the product name says N0510.
        item = _synthetic_item(
            baseline="05.11",
            product_id="S2A_MSIL2A_20240601_N0510_R022_T31SGA_20240601",
        )
        _, offset, reason = resolve_reflectance_offset(item)
        assert offset is None
        assert reason is not None and "inconsistent" in reason

    def test_declared_already_applied_offset_is_withheld_for_pb04(self) -> None:
        # Synthetic: scale 1e-4 with offset 0 says the offset is already applied
        # (a harmonised encoding). Applying -1000 again would double-correct.
        item = _synthetic_item(baseline="05.11")
        item["assets"]["B04"]["raster:bands"] = [{"scale": 0.0001, "offset": 0.0}]
        item["assets"]["B08"]["raster:bands"] = [{"scale": 0.0001, "offset": 0.0}]
        _, offset, reason = resolve_reflectance_offset(item)
        assert offset is None
        assert reason is not None and "B04" in reason

    def test_declared_raw_dn_scale_is_withheld(self) -> None:
        # Synthetic: scale 1 says the raster is unscaled DN, which is not reflectance.
        item = _synthetic_item(baseline="05.11")
        item["assets"]["B04"]["raster:bands"] = [{"scale": 1.0, "offset": -1000.0}]
        _, offset, reason = resolve_reflectance_offset(item)
        assert offset is None
        assert reason is not None and "scale 1.0" in reason

    def test_declared_offset_for_pre_pb04_is_withheld(self) -> None:
        # Synthetic: a pre-04.00 product must carry no offset.
        item = _synthetic_item(baseline="03.00", product_id="x_N0300_y")
        item["assets"]["B08"]["raster:bands"] = [{"scale": 0.0001, "offset": -0.1}]
        _, offset, reason = resolve_reflectance_offset(item)
        assert offset is None
        assert reason is not None and "B08" in reason

    def test_declared_non_numeric_scaling_is_withheld(self) -> None:
        item = _synthetic_item(baseline="05.11")
        item["assets"]["B04"]["raster:bands"] = [{"scale": "0.0001", "offset": -0.1}]
        _, offset, reason = resolve_reflectance_offset(item)
        assert offset is None
        assert reason is not None and "non-numeric" in reason

    def test_declared_scaling_matching_convention_is_accepted(self) -> None:
        # Synthetic: (DN - 1000) / 10000 written as scale and offset.
        item = _synthetic_item(baseline="05.11")
        item["assets"]["B04"]["raster:bands"] = [{"scale": 0.0001, "offset": -0.1}]
        item["assets"]["B08"]["raster:bands"] = [{"scale": 0.0001, "offset": -0.1}]
        _, offset, reason = resolve_reflectance_offset(item)
        assert (offset, reason) == (-1000.0, None)

    def test_declared_entries_without_offset_key_are_not_checked(self) -> None:
        item = _synthetic_item(baseline="05.11")
        item["assets"]["B04"]["raster:bands"] = [{"nodata": 0}]
        _, offset, reason = resolve_reflectance_offset(item)
        assert (offset, reason) == (-1000.0, None)


# ---------------------------------------------------------------------------
# End-to-end through the service, with stubbed reads
# ---------------------------------------------------------------------------


class _RecordingReader:
    """Band reader that records every href it is asked for."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, href: str) -> Any:  # pragma: no cover - must not be reached
        self.calls.append(href)
        raise AssertionError("band reader must not be called for a withheld offset")


def _service_for(item: dict[str, Any], reader: Any) -> SatelliteDataService:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"type": "FeatureCollection", "features": [item]})

    return SatelliteDataService(
        transport=httpx.MockTransport(handler),
        band_reader=reader,
        min_valid_coverage=0.9,
    )


async def test_inconsistent_metadata_never_reads_pixels() -> None:
    item = _synthetic_item(
        baseline="05.11",
        product_id="S2A_MSIL2A_20240601_N0510_R022_T31SGA_20240601",
    )
    reader = _RecordingReader()
    service = _service_for(item, reader)
    async with service:
        layer = await service.fetch_sentinel_indices(
            bbox=[5.2, 31.8, 5.5, 32.1], start_date="2024-06-01", end_date="2024-06-30"
        )
    assert reader.calls == []
    assert layer["ndvi_mean"] == NDVI_NO_DATA
    assert layer["ndvi_valid_coverage"] is None
    assert layer["reflectance_offset"] is None


async def test_cdse_named_item_computes_ndvi_end_to_end() -> None:
    """Regression: CDSE-named bands are read, so NDVI is no longer silently missing."""
    item: dict[str, Any] = {
        "id": "S2A_MSIL2A_20240601T100000_N0511_R022_T31SGA_20240601T120000",
        "properties": {"eo:cloud_cover": 5.0},
        "assets": {
        "B04_10m": {"href": "https://eodata.example/B04_10m.jp2"},
        "B08_10m": {"href": "https://eodata.example/B08_10m.jp2"},
        "SCL_20m": {"href": "https://eodata.example/SCL_20m.jp2"},
        },
    }

    from app.services.satellite_service import BandSample

    samples = {
        "https://eodata.example/B04_10m.jp2": BandSample(np.full((4, 4), 1000 + 0.1 * 10000)),
        "https://eodata.example/B08_10m.jp2": BandSample(np.full((4, 4), 1000 + 0.3 * 10000)),
        "https://eodata.example/SCL_20m.jp2": BandSample(np.full((4, 4), 4, dtype="uint8")),
    }

    def reader(href: str) -> BandSample:
        return samples[href]

    service = _service_for(item, reader)
    async with service:
        layer = await service.fetch_sentinel_indices(
            bbox=[5.2, 31.8, 5.5, 32.1], start_date="2024-06-01", end_date="2024-06-30"
        )
    # reflectance red 0.1, nir 0.3 -> NDVI 0.5 over a fully usable footprint.
    assert layer["ndvi_mean"] == pytest.approx(0.5)
    assert layer["ndvi_valid_coverage"] == pytest.approx(1.0)
    assert layer["reflectance_offset"] == -1000.0
    assert layer["processing_baseline"] == "05.11"
