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


# --- Manifest-driven areas and coverage (added with the London pilot) -----------

LONDON_BOX = [-0.51, 51.28, 0.34, 51.7]


def _extent_raster(path: Path, bounds: tuple[float, float, float, float]) -> Path:
    west, south, east, north = bounds
    res = 1 / 1200
    width = round((east - west) / res)
    height = round((north - south) / res)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(west, north, res, res),
    ) as destination:
        destination.write(np.full((1, height, width), 3.0, dtype="float32"))
    return path


def test_validate_raster_accepts_a_raster_that_contains_the_area(script, tmp_path: Path) -> None:
    raster = _extent_raster(tmp_path / "gb.tif", (-1.0, 51.0, 1.0, 52.0))
    assert script.validate_raster(raster, area_bbox=LONDON_BOX) == []


def test_validate_raster_rejects_a_raster_that_misses_part_of_the_area(script, tmp_path: Path) -> None:
    raster = _extent_raster(tmp_path / "east.tif", (-0.2, 51.0, 1.0, 52.0))
    problems = script.validate_raster(raster, area_bbox=LONDON_BOX)
    assert any("does not contain" in problem for problem in problems)


def test_validate_raster_rejects_a_raster_from_another_country(script, tmp_path: Path) -> None:
    raster = _extent_raster(tmp_path / "dz.tif", (5.0, 31.5, 5.8, 32.4))
    problems = script.validate_raster(raster, area_bbox=LONDON_BOX)
    assert any("does not contain" in problem for problem in problems)


def test_resolve_area_maps_london_and_refuses_unverified_areas(script) -> None:
    _key, _dataset, entry, study = script.resolve_area("london")
    assert entry["iso3"] == "GBR"
    assert study["bbox"] == LONDON_BOX
    with pytest.raises(ValueError):
        script.resolve_area("amsterdam")


def test_cli_refuses_an_unverified_area_with_exit_one(script, tmp_path: Path) -> None:
    assert script.main(["--area", "rotterdam", "--dest", str(tmp_path)]) == 1
    assert not any(tmp_path.iterdir())


def test_cli_download_failure_exits_one_and_writes_nothing(script, tmp_path: Path, monkeypatch) -> None:
    def fail(url: str, target: Path) -> None:
        raise OSError("simulated outage")

    monkeypatch.setattr(script, "download", fail)
    assert script.main(["--area", "london", "--dest", str(tmp_path)]) == 1
    assert not any(tmp_path.iterdir())


def test_cli_accepts_a_local_london_raster_and_records_its_provenance(
    script, tmp_path: Path
) -> None:
    import json

    raster = _extent_raster(tmp_path / "gbr.tif", (-1.0, 51.0, 1.0, 52.0))
    assert script.main(["--area", "london", "--from-local", str(raster)]) == 0
    sidecar = json.loads(raster.with_suffix(".source.json").read_text(encoding="utf-8"))
    assert sidecar["study_area"] == "london"
    assert sidecar["study_area_bbox"] == LONDON_BOX
    assert sidecar["licence"] == "CC BY 4.0"
    assert sidecar["reference_year"] == 2020
    assert "Global2" in sidecar["dataset"]
    assert sidecar["url_verified"] is False
    assert sidecar["sha256_verified_against_publisher"] is False
    assert sidecar["content"].startswith("population exposure only")


def test_cli_keeps_a_local_raster_that_does_not_cover_london_but_refuses_it(
    script, tmp_path: Path
) -> None:
    raster = _extent_raster(tmp_path / "dz.tif", (5.0, 31.5, 5.8, 32.4))
    assert script.main(["--area", "london", "--from-local", str(raster)]) == 3
    assert raster.exists()  # the user's file is never deleted by --from-local
    assert not raster.with_suffix(".source.json").exists()


def test_cli_removes_a_downloaded_raster_that_fails_validation(script, tmp_path: Path, monkeypatch) -> None:
    def fake_download(url: str, target: Path) -> None:
        _extent_raster(target, (5.0, 31.5, 5.8, 32.4))  # wrong country

    monkeypatch.setattr(script, "download", fake_download)
    assert script.main(["--area", "london", "--dest", str(tmp_path)]) == 3
    assert not (tmp_path / "gbr_pop_2020_CN_100m_R2025A_v1.tif").exists()
