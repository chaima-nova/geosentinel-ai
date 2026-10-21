"""Reproducible comparison of Sentinel-2 NDVI methods on a synthetic scene.

Writes a synthetic Sentinel-2-like scene (known surface reflectance, clouds,
cloud shadow and a no-data strip) to GeoTIFFs, then estimates the footprint's
mean NDVI with three methods and compares each to the known truth:

* ``legacy_raw_dn``   - the original behaviour: mean NDVI from raw digital numbers,
  no BOA offset, no scene-classification mask (``app.services.satellite_service.mean_ndvi``).
* ``offset_only``     - the corrected reflectance scaling, but no SCL mask.
* ``corrected``       - the full corrected service path: STAC stub, real GeoTIFF
  reads, offset, SCL mask and coverage gate (``SatelliteDataService``).

IMPORTANT: the scene is synthetic. Results show how each method behaves on data
whose truth is known. They are NOT a validation against live Copernicus products,
and they do not compare against odc-stac, which was not integrated (no live STAC
access from the environment that produced these results).

Usage::

    python tools/satellite_comparison/run_comparison.py --size 200 --seed 7 \
        --out tools/satellite_comparison/results
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import rasterio
from rasterio.transform import from_bounds

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.satellite_service import (
    SCL_INVALID_CLASSES,
    SatelliteDataService,
    compute_ndvi,
    mean_ndvi,
    read_band_sample,
    read_scl_sample,
    scl_invalid_mask,
)

QUANTIFICATION = 10000.0
PROCESSING_BASELINE_PB05 = -1000.0  # BOA_ADD_OFFSET for baseline >= 04.00
SYNTHETIC_LABEL = "SYNTHETIC SCENE - not a Copernicus product"


def build_scene(size: int, seed: int) -> dict[str, np.ndarray]:
    """Create a deterministic synthetic scene with known reflectance and classes."""
    rng = np.random.default_rng(seed)
    red = 0.05 + 0.15 * rng.random((size, size))
    nir = red + 0.10 + 0.40 * rng.random((size, size))  # vegetated: NIR > red
    scl = np.full((size, size), 4, dtype="uint8")  # vegetation everywhere by default

    # Cloud: bright in both bands, so it would drag NDVI toward zero if included.
    c0, c1 = size // 5, size // 3
    red[c0:c1, c0:c1] = 0.35
    nir[c0:c1, c0:c1] = 0.38
    scl[c0:c1, c0:c1] = 9

    # Cloud shadow: dark, classed 3.
    s0, s1 = size // 2, size // 2 + size // 6
    red[s0:s1, size // 2 : size // 2 + size // 6] = 0.02
    nir[s0:s1, size // 2 : size // 2 + size // 6] = 0.03
    scl[s0:s1, size // 2 : size // 2 + size // 6] = 3

    # Out-of-swath no-data strip along the top: DN 0 and SCL 0.
    strip = max(1, size // 25)
    scl[:strip, :] = 0
    red[:strip, :] = 0.0  # encoded as DN 0 below
    nir[:strip, :] = 0.0

    # Clip to valid reflectance range, then keep the truth on clear pixels only.
    red = np.clip(red, 0.0, 1.0)
    nir = np.clip(nir, 0.0, 1.0)
    return {"red": red, "nir": nir, "scl": scl}


def truth_ndvi(scene: dict[str, np.ndarray]) -> float:
    """Mean NDVI over clear, valid pixels at native resolution (the known truth)."""
    red, nir, scl = scene["red"], scene["nir"], scene["scl"]
    invalid = np.isin(scl, sorted(SCL_INVALID_CLASSES))
    clear = ~invalid & ((nir + red) > 0)
    ndvi = (nir[clear] - red[clear]) / (nir[clear] + red[clear])
    return float(ndvi.mean())


def encode_dn(reflectance: np.ndarray, scl: np.ndarray) -> np.ndarray:
    """L2A DN for baseline 05.10: DN = rho * 10000 - offset; no-data stays 0."""
    dn = np.rint(reflectance * QUANTIFICATION - PROCESSING_BASELINE_PB05).astype("uint16")
    dn[scl == 0] = 0
    return dn


def write_tif(path: Path, array: np.ndarray, nodata: float | None = None) -> Path:
    height, width = array.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype=array.dtype, crs="EPSG:4326",
        transform=from_bounds(2.5, 31.0, 3.5, 32.0, width, height), nodata=nodata,
    ) as dst:
        dst.write(array, 1)
    return path


def stac_payload(tag: str, cloud_cover: float) -> dict[str, Any]:
    item = {
        "type": "Feature",
        "id": f"SYNTHETIC_S2B_MSIL2A_20240615T103029_N0510_R108_T31SET_{tag}",
        "bbox": [2.5, 31.0, 3.5, 32.0],
        "geometry": {"type": "Polygon", "coordinates": [[[2.5, 31.0], [3.5, 31.0], [3.5, 32.0], [2.5, 32.0], [2.5, 31.0]]]},
        "properties": {"datetime": "2024-06-15T10:30:29Z", "eo:cloud_cover": cloud_cover,
                        "s2:processing_baseline": "05.10", "description": SYNTHETIC_LABEL},
        "assets": {
            "B04": {"href": f"{tag}_B04.tif"},
            "B08": {"href": f"{tag}_B08.tif"},
            "SCL": {"href": f"{tag}_SCL.tif"},
        },
    }
    return {"type": "FeatureCollection", "features": [item]}


def run(size: int = 200, seed: int = 7, workdir: Path | None = None) -> dict[str, Any]:
    scene = build_scene(size, seed)
    truth = truth_ndvi(scene)
    tmp = workdir or Path(tempfile.mkdtemp(prefix="satcmp-"))
    tmp.mkdir(parents=True, exist_ok=True)

    dn_red = encode_dn(scene["red"], scene["scl"])
    dn_nir = encode_dn(scene["nir"], scene["scl"])
    red_path = write_tif(tmp / "scene_B04.tif", dn_red, nodata=None)
    nir_path = write_tif(tmp / "scene_B08.tif", dn_nir, nodata=None)
    scl_path = write_tif(tmp / "scene_SCL.tif", scene["scl"])
    cloud_pct = 100.0 * float(np.isin(scene["scl"], [8, 9]).mean())

    # Method 1: legacy, raw DN, no offset, no mask (the original code path).
    legacy = mean_ndvi(
        read_band_sample(str(red_path), max_sample_pixels=256),
        read_band_sample(str(nir_path), max_sample_pixels=256),
    )

    # Method 2: offset-corrected reflectance, no SCL mask.
    red_s = read_band_sample(str(red_path), max_sample_pixels=256)
    nir_s = read_band_sample(str(nir_path), max_sample_pixels=256)
    offset_only = compute_ndvi(
        red_s, nir_s, reflectance_offset=PROCESSING_BASELINE_PB05,
        quantification_value=QUANTIFICATION, scl_invalid=None,
    )

    # Method 3: full corrected service path over real GeoTIFF reads.
    payload = stac_payload("scene", cloud_pct)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async def corrected_run(min_valid_coverage: float) -> dict[str, Any]:
        service = SatelliteDataService(
            transport=httpx.MockTransport(handler), min_valid_coverage=min_valid_coverage
        )
        # Asset hrefs are local paths; the stub transport returns them unchanged.
        async with service:
            return await service.fetch_sentinel_indices(
                bbox=[2.5, 31.0, 3.5, 32.0], start_date="2024-06-01", end_date="2024-06-30"
            )

    def resolve_local_hrefs(item_payload: dict[str, Any]) -> dict[str, Any]:
        for asset in item_payload["features"][0]["assets"].values():
            asset["href"] = str(tmp / asset["href"])
        return item_payload

    payload = resolve_local_hrefs(payload)
    corrected_layer = asyncio.run(corrected_run(0.9))
    corrected_relaxed = asyncio.run(corrected_run(0.8))

    def err(value: float | None) -> float | None:
        return None if value is None else abs(value - truth)

    def corrected_row(layer: dict[str, Any], threshold: float) -> dict[str, Any]:
        withheld = layer["ndvi_mean"] == -1.0
        value = None if withheld else layer["ndvi_mean"]
        return {
            "ndvi": _r(value), "abs_error": _r(err(value)),
            "valid_coverage": _r(layer["ndvi_valid_coverage"]),
            "offset_applied": layer["reflectance_offset"], "mask": "SCL",
            "min_valid_coverage": threshold, "withheld_by_coverage_gate": withheld,
        }

    scl_mask = scl_invalid_mask(read_scl_sample(str(scl_path)).values, np.shape(red_s.values))
    return {
        "label": SYNTHETIC_LABEL,
        "size_px": size,
        "seed": seed,
        "truth_ndvi_native": round(truth, 6),
        "scene_cloud_percent_scl": round(cloud_pct, 2),
        "scl_masked_fraction_on_sample_grid": round(float(scl_mask.mean()), 6),
        "methods": {
            "legacy_raw_dn": {
                "ndvi": _r(legacy), "abs_error": _r(err(legacy)),
                "valid_coverage": None, "offset_applied": 0.0, "mask": "none",
            },
            "offset_only": {
                "ndvi": _r(offset_only.mean), "abs_error": _r(err(offset_only.mean)),
                "valid_coverage": _r(offset_only.valid_coverage), "offset_applied": PROCESSING_BASELINE_PB05, "mask": "none",
            },
            "corrected": corrected_row(corrected_layer, 0.9),
            "corrected_min_coverage_0.8": corrected_row(corrected_relaxed, 0.8),
        },
        "notes": [
            "Truth is the mean NDVI over clear native-resolution pixels.",
            "Methods average on the decimated sample grid, so a small gap to truth is expected even when correct.",
            "legacy_raw_dn and offset_only have no mask: the no-data strip (DN 0) and cloud/shadow are averaged in.",
            "The coverage gate withholds NDVI below min_valid_coverage; the 0.8 row shows the estimate behind that decision.",
            "No live Copernicus or Earth Search data was used; no odc-stac comparison was run.",
        ],
    }


def _r(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def to_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Satellite NDVI comparison (synthetic)",
        "",
        (
            f"**{result['label']}.** Results show behaviour against a known truth, not validation "
            "against live products."
        ),
        "",
        f"- Scene: {result['size_px']} px, seed {result['seed']}",
        f"- Truth NDVI (clear native pixels): {result['truth_ndvi_native']}",
        f"- Scene cloud (SCL 8-9): {result['scene_cloud_percent_scl']} %",
        "",
        "| Method | NDVI | abs. error | valid coverage | offset | mask | withheld |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, row in result["methods"].items():
        lines.append(
            f"| {name} | {row['ndvi']} | {row['abs_error']} | {row['valid_coverage']} | "
            f"{row['offset_applied']} | {row['mask']} | {row.get('withheld_by_coverage_gate', '-')} |"
        )
    lines += ["", "## Notes", ""]
    lines += [f"- {note}" for note in result["notes"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    result = run(size=args.size, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "results.md").write_text(to_markdown(result))
    print(to_markdown(result))


if __name__ == "__main__":
    main()
