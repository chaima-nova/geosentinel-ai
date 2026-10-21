"""Smoke test for the synthetic NDVI comparison harness.

Locks in the ordering the harness is meant to demonstrate on a scene with known
truth: each correction moves the estimate closer to truth. It does not validate
against live Copernicus data.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "tools" / "satellite_comparison" / "run_comparison.py"


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("run_comparison", HARNESS)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corrections_move_the_estimate_toward_truth(harness, tmp_path: Path) -> None:
    result = harness.run(size=120, seed=7, workdir=tmp_path)
    methods = result["methods"]
    legacy = methods["legacy_raw_dn"]["abs_error"]
    offset_only = methods["offset_only"]["abs_error"]
    corrected = methods["corrected_min_coverage_0.8"]["abs_error"]
    assert legacy is not None and offset_only is not None and corrected is not None
    assert offset_only < legacy
    assert corrected < offset_only
    assert result["label"].startswith("SYNTHETIC")


def test_default_gate_withholds_when_coverage_is_insufficient(harness, tmp_path: Path) -> None:
    result = harness.run(size=120, seed=7, workdir=tmp_path)
    row = result["methods"]["corrected"]
    assert row["withheld_by_coverage_gate"] is True
    assert row["ndvi"] is None


def test_output_is_deterministic_for_a_seed(harness, tmp_path: Path) -> None:
    first = harness.run(size=80, seed=3, workdir=tmp_path / "a")
    second = harness.run(size=80, seed=3, workdir=tmp_path / "b")
    assert first == second
