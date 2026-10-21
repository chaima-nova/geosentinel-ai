"""Offline tests for the WorldPop ingestion script's validation logic.

The script is loaded from its path, since ``tools/`` is not a package. No network
access is made: the tests use synthetic local rasters only.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "vulnerability" / "fetch_worldpop.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_worldpop", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raster(path: Path, *, crs: str = "EPSG:4326", res: float = 1 / 1200, count: int = 1) -> Path:
    data = np.full((count, 24, 24), 2.0, dtype="float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=24,
        width=24,
        count=count,
        dtype="float32",
        crs=crs,
        transform=from_origin(5.0, 31.92, res, res),
    ) as destination:
        destination.write(data)
    return path


def test_valid_100m_wgs84_raster_passes(script: ModuleType, tmp_path: Path) -> None:
    assert script.validate_raster(_raster(tmp_path / "ok.tif")) == []


@pytest.mark.parametrize(
    ("kwargs", "expected_fragment"),
    [
        ({"crs": "EPSG:32631"}, "expected EPSG:4326"),
        ({"res": 1 / 120}, "pixel size"),
        ({"count": 2}, "expected 1 band"),
    ],
)
def test_invalid_rasters_are_reported(
    script: ModuleType, tmp_path: Path, kwargs: dict, expected_fragment: str
) -> None:
    problems = script.validate_raster(_raster(tmp_path / "bad.tif", **kwargs))
    assert problems
    assert any(expected_fragment in problem for problem in problems)


def test_unreadable_file_is_a_validation_failure(script: ModuleType, tmp_path: Path) -> None:
    junk = tmp_path / "junk.tif"
    junk.write_bytes(b"not a raster")
    problems = script.validate_raster(junk)
    assert problems and "could not be opened" in problems[0]


def test_sha256_matches_hashlib(script: ModuleType, tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "blob.bin"
    path.write_bytes(b"geosentinel" * 1000)
    assert script.sha256_of(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_sidecar_records_licence_and_unverified_checksum(script: ModuleType, tmp_path: Path) -> None:
    path = _raster(tmp_path / "ok.tif")
    sidecar = script.build_sidecar(
        source_url="local:ok.tif",
        file_path=path,
        sha256=script.sha256_of(path),
        expected_sha256=None,
    )
    assert sidecar["licence"].startswith("CC BY 4.0")
    assert sidecar["sha256_verified_against_publisher"] is False
    assert "not a social vulnerability index" in str(sidecar["content"])
