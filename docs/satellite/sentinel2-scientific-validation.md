# Sentinel-2 NDVI: encoding, masking and coverage policy

Status: **implemented and tested offline. Live real-scene validation has not been
performed** (see [Real-scene validation status](#real-scene-validation-status)).

This document records what the code assumes about Sentinel-2 L2A data, why, and
what it does when an assumption cannot be confirmed. Code: `app/services/satellite_service.py`.

## 1. Supported encodings

| Encoding | Where it occurs | Handling |
| --- | --- | --- |
| L2A digital numbers with BOA additive offset, processing baseline ≥ 04.00 | CDSE `SENTINEL-2` (the default), Element84 `sentinel-2-l2a` as observed in the real item fixture | `reflectance = (DN + BOA_ADD_OFFSET) / 10000`, `BOA_ADD_OFFSET = -1000` |
| L2A DN, processing baseline < 04.00 | Older products | `reflectance = DN / 10000`, offset 0 |
| Any other encoding (already-applied offset, raw DN, unknown scale) | Harmonised or re-encoded collections | **Fail closed.** NDVI is withheld if an asset declares `raster:bands` scale/offset that contradicts the convention |
| Unknown or inconsistent processing baseline | Metadata gaps or conflicts | **Fail closed.** NDVI withheld with a stated reason |

**Why the offset is derived from the baseline.** The Copernicus L2A product
guide and the Sentinel-2 data quality reports state that from processing
baseline 04.00 (25 January 2022) the product carries `BOA_ADD_OFFSET = -1000` and
`QUANTIFICATION_VALUE = 10000`, and that DN 0 is no-data. Earlier baselines carry
no offset. Two independent readers implement the same rule from the product-level
`BOA_ADD_OFFSET_VALUES_LIST` in `MTD_MSIL2A.xml`:

* GDAL, `frmts/sentinel2/sentinel2dataset.cpp`, reads `BOA_ADD_OFFSET` per band
  from that list. The GitHub API reports no SPDX licence for the repository
  (`NOASSERTION`), so its licence was not verified.
* satpy, `satpy/readers/msi_safe.py`, computes `(data + offset) / quantification`
  from the same list. The GitHub API reports `Apache-2.0` for the repository.

Both were read only to confirm the convention. **No code was copied from either.**

The STAC item does not carry that list. The real item fixture has no
`raster:bands` on its B04, B08 or SCL assets (see §5), so the offset cannot be
read from the item itself. The baseline-derived value is therefore the rule the
code applies, and the metadata checks in §3 guard against contradicting
metadata. The code does not apply `-1000` to a collection that declares it
already applied.

**Baseline sources.** `s2:processing_baseline` is used when present, and it must
agree with the `_Nxxyy_` token in the product identifier when that token is present.
Disagreement withholds NDVI.

## 2. Asset names

The resolver accepts the canonical keys (`B04`, `B08`, `SCL`), the Earth Search
style keys, and the CDSE keys `B04_10m`, `B08_10m` and `SCL_20m`. The CDSE names
come from the CDSE asset-name mapping in mapchete-eo (MIT licence,
`mapchete_eo/platforms/sentinel2/preconfigured_sources/item_mappers.py`).
**Before this change the resolver did not match the CDSE names, so NDVI was never
computed against the default CDSE endpoint.** The CDSE keys are therefore the
highest-priority fix in this change. They are unverified against a live CDSE
response (§6).

## 3. Metadata checks (before any pixel is read)

Implemented in `resolve_reflectance_offset`. Each failure withholds NDVI and is
logged with its reason. Tests are in `tests/test_satellite_encoding.py`.

* `s2:processing_baseline` and the product-identifier baseline must agree.
* The baseline must be known.
* A `raster:bands` scale and offset, when declared on B04 or B08, must equal
  `1/10000` and `offset/10000` for the expected offset. A declared offset of 0
  for a 04.00+ product, or a scale of 1, is withheld.
* Non-numeric declared scaling is withheld.

## 4. Area of interest, footprint and coverage

* Only the requested bounding box is read, not the whole ~110 km tile. The earlier
  code averaged the entire tile, so NDVI described the scene, not the city. This
  was a confirmed defect.
* The bbox is transformed to the raster CRS (UTM). Edges are rounded to the
  nearest pixel boundary. A pixel is included only when at least half of it lies in
  the requested area. A lon/lat rectangle becomes a slightly larger rectangle in
  UTM, so outward rounding would add a neighbouring column.
* `footprint_fraction` is the share of the requested area inside the scene.
  `ndvi_valid_coverage = usable_fraction × footprint_fraction`. An area that
  extends past the scene edge loses coverage rather than being scored as usable.
* An area that does not intersect the scene reports coverage 0.0.
* The 0.9 coverage threshold (`min_valid_coverage`) is a **policy choice**, not a
  scientifically established universal value. It is configurable on
  `SatelliteDataService`. Below it, NDVI is reported as `NDVI_NO_DATA` and the
  evidence auditor treats the satellite evidence as `INVALID`.

## 5. Scene Classification Layer (SCL) policy

| SCL | Class | Treatment | Reason |
| --- | --- | --- | --- |
| 0 | No data | Excluded | Not an observation |
| 1 | Saturated or defective | Excluded | Unreliable radiometry |
| 2 | Dark area pixels | Excluded | Shadow or low-signal ambiguity |
| 3 | Cloud shadow | Excluded | Contaminated surface |
| 4 | Vegetation | **Included** | Target class |
| 5 | Not vegetated (bare soil) | **Included** | Desert and urban bare ground are the main Ouargla surfaces. Low NDVI here is measurement, not missing data. |
| 6 | Water | **Excluded (policy)** | Negative NDVI biases a vegetation mean. Excluding it lowers coverage, so a water-dominated AOI is withheld. |
| 7 | Unclassified | Included, **flagged as uncertain** | Could be bare or built surface, could be a mis-classified cloud edge. Kept because excluding it would remove most urban pixels. |
| 8 | Cloud, medium probability | Excluded | Cloud |
| 9 | Cloud, high probability | Excluded | Cloud |
| 10 | Thin cirrus | Excluded | Cirrus contaminates reflectance |
| 11 | Snow or ice | **Excluded (policy)** | Not vegetation and reflectance-bright |

The SCL is read at native 20 m resolution and reduced to the band grid with an
"any excluded" rule, so a single cloud pixel is never averaged away. The
nearest-neighbour alternative was rejected because it can drop single cloud pixels.

The water and snow exclusion is an explicit policy decision. It is documented
here and enforced in `SCL_NON_VEGETATION_CLASSES`, so changing it is a visible code
change, not a silent threshold.

## 6. Real-scene validation status

**Not performed in this environment.** The build sandbox could not reach
`dataspace.copernicus.eu`, `earth-search.aws.element84.com` or
`planetarycomputer.microsoft.com` (connection failures). Only PyPI and GitHub were
reachable.

What was verified from real data, offline:

* A real Sentinel-2 L2A STAC item was used as a metadata regression fixture
  (`tests/fixtures/sentinel2_l2a_S2B_20240825_T13TDE.json`). It is trimmed to B04,
  B08 and SCL and carries no raster data. Source: `stac-utils/rustac`
  (`crates/core/data/two-sentinel-2-items.json`, Apache-2.0). The underlying data is
  Copernicus Sentinel-2 open data.
  * Item `S2B_MSIL2A_20240825T173909_R098_T13TDE_20240825T215747`, collection
    `sentinel-2-l2a`, platform Sentinel-2B, EPSG:32613, B04 and B08 at 10 m
    (10980 × 10980), SCL at 20 m, `s2:processing_baseline` `05.11`, product URI
    `…_N0511_…`, eo:cloud_cover 48.7 %, no `raster:bands` declared on B04, B08 or SCL.
  * The baseline, offset (−1000), asset resolution and encoding checks pass on this item.

**Manual real-scene procedure** (to run where the endpoints are reachable):

1. Pick a Sentinel-2 L2A scene over Ouargla (for example, June 2024, cloud cover under 10 %).
2. Record `id`, `properties.s2:processing_baseline`, `properties.proj:epsg`,
   each asset's `href` and `raster:bands`, and the GDAL `nodata` of B04 and B08.
3. Call `SatelliteDataService().fetch_sentinel_indices(bbox, start, end)` with the
   Ouargla preset bbox `[5.1, 31.7, 5.6, 32.2]`. Confirm `reflectance_offset`
   matches the baseline convention and that `ndvi_valid_coverage` is plausible.
4. Compare the NDVI with an independent reading of the same scene (for example,
   SNAP or GDAL on the same window). Agreement should be within rounding.
5. Run `pytest -m integration` against the live endpoints.

## 7. Limitations

* NDVI here is a **heuristic vegetation indicator** for a coarse AOI mean, not a validated measurement.
* The mean uses a decimated window (default budget 256 px, about 16 × 16 cells
  over a city-sized area). It is an estimate over a coarse grid.
* Sentinel-2 has no thermal band. `lst_celsius_max` is always `LST_NOT_AVAILABLE`.
  Land-surface temperature needs Landsat thermal or Sentinel-3 SLSTR, which is not
  implemented.
* Harmonised collections cannot be identified reliably from the item alone. They
  are rejected only when they declare contradicting scaling. Operators must not
  point `collection` at a harmonised product.
* odc-stac was not integrated. It was not compared on live data, so there is no
  evidence it performs better here.
