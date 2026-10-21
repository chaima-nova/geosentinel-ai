# Real-data validation gaps: status report

Status vocabulary: `validated`, `implemented but not validated`, `blocked`.
Nothing in this repository is currently `validated`. Each gap below lists the exact
commands run and their outcome. Entries are dated 2026-10-21, from the build sandbox.

The build sandbox could not reach the official data hosts. The results below reflect
that network state, not the quality of the data sources.

## Network reachability (2026-10-21)

| Host | Result |
|---|---|
| data.worldpop.org | no HTTP response (curl exit 000); project script: `TLS/SSL connection has been closed (EOF)` |
| dataspace.copernicus.eu (CDSE STAC) | unreachable; integration tests skip with reason |
| assets.publishing.service.gov.uk (GOV.UK IoD2025) | no HTTP response (000) |
| opendata.cbs.nl (CBS) | no HTTP response (000) |
| services1.arcgis.com (ONS boundary service) | no HTTP response (000) |
| github.com, pypi.org, registry.npmjs.org | reachable |

Re-tested 2026-10-21 in a later session with the same result: data.worldpop.org, dataspace.copernicus.eu,
assets.publishing.service.gov.uk, opendata.cbs.nl, geoportal.statistics.gov.uk, services1.arcgis.com,
data.gov.uk, landsatlook.usgs.gov, planetarycomputer.microsoft.com, earth-search.aws.element84.com,
sentinel-cogs.s3.us-west-2.amazonaws.com, raw.githubusercontent.com, huggingface.co and zenodo.org all returned
HTTP 000 (no response). files.pythonhosted.org returned 404 on its root, which means it is reachable.

## Gap status

### Gap 1: Greater London boundary — **blocked**
- No official boundary has been obtained. No polygon is bundled or invented.
- Implemented: `app/services/boundary_service.py` (load, validate, bbox containment, grid masking),
  `tests/test_boundary.py` (18 tests, all synthetic polygons labelled as such).
- Registry: `config/study_areas.json` records `boundary.status = not_obtained` for London,
  Amsterdam and Rotterdam.
- To unblock: download the ONS/GLA Greater London boundary on a network-enabled machine,
  record its source URL, licence and SHA-256 in the registry, then run
  `load_boundary(path, expected_sha256=...)`.
- Not wired into the WorldPop provider: there is no real boundary to mask against yet.

### Gap 2: WorldPop Global2 GBR exposure raster — **blocked**
- Command: `python tools/vulnerability/fetch_worldpop.py --area london --dest /tmp/wp_attempt`
- Outcome: exit 1, `error: download failed: ConnectError: TLS/SSL connection has been closed (EOF)`.
- The manifest entry keeps `sha256: null` and `validation_status: not_downloaded`. No checksum was invented.
- Licence: CC BY 4.0 (WorldPop Global2 R2025A, 100 m, constrained, 2020).
- To unblock: run the same command with network access; then verify the sidecar, the CRS and the coverage check.

### Gap 3: Sentinel-2 scene search (Copernicus Data Space) — **blocked**
- Command: `pytest -m integration tests/test_satellite_service_integration.py -rA`
- Outcome: 4 skipped, reason `https://dataspace.copernicus.eu/stac is unreachable`. **Skipped, not passed.**
- No scene, cloud-cover or NDVI result has been obtained for London.
- To unblock: run the integration tests on a machine that can reach the CDSE STAC endpoint, and record the scene IDs.

### Gap 4: Landsat Collection 2 ST_B10 land-surface temperature — **implemented but not validated**
- Implemented: `app/services/landsat_lst_service.py`. Scale 0.00341802, offset 149.0 K,
  fill 0, valid DN ≥ 293 (USGS Landsat C2 L2 scale-factor guidance).
- Tests: `tests/test_landsat_lst.py` (13 tests). They check the USGS worked example
  (DN 44947 → 302.6 K) and the synthetic fill, Collection 1 and coverage cases.
- Limits: no real Landsat scene has been read. QA_PIXEL cloud/shadow masking is **not**
  implemented, so every result reports `qa_mask_applied = False` and states that clouds are unscreened.
  The module is not wired into the risk score.
- To validate: read a real ST_B10 scene for London, apply QA_PIXEL masking (bit positions to be confirmed
  against the USGS Landsat C2 L2 product guide), and compare against a reference.

### Gap 5: England Indices of Deprivation 2025 (sensitivity) — **blocked**
- GOV.UK URL unreachable (000). Not integrated. No sensitivity or adaptive-capacity value is computed.
- Licence (candidate): OGL v3, LSOA level. The composite HEPS score stays withheld.

### Gap 6: Browser verification of the dashboard — **partially verified**
Browser: Chromium 153 from `@sparticuz/chromium` (third-party build; Playwright's own download was blocked).
Run with `LD_LIBRARY_PATH` pointing at the extracted AL2023 libraries. Dev servers: Vite on 5173, FastAPI on 8000.

Verified (`/tmp/browser/ui_check.mjs`, exit 0, 2026-10-21):
- London is the default selection.
- Amsterdam and Rotterdam options are disabled with the reason "unavailable".
- Editing a bbox value switches the selector to "Custom bounding box" (the select value is empty), and the
  evidence header reads "Last query: London, United Kingdom", not a London label for the new bbox.
- Per-source rows show UNAVAILABLE with "No data for this source. Not treated as zero."
- The HEPS score shows "withheld" with the insufficient-evidence reason.
- Mobile at 390 px: no horizontal overflow (0 px).
- Loading (`loading.mjs`): submit button disabled while the request is pending.
- Error (`error.mjs`): the real server message is shown in the feed (test-intercepted HTTP 500, labelled as such).

Re-verified 2026-10-21 after the zoom fix (zoom control moved to top-right in `frontend/src/components/HeatMap.tsx`):
- The map title no longer overlaps the zoom control (measured: title x 489–851 px, zoom x 1299–1333 px at 1440 px width).
  Screenshot `/tmp/shots/E_map_zoom_fixed.png`, not committed.
- All other checks above were re-run with the same results.

Defects still open:
- CARTO basemap tiles fail with `ERR_CONNECTION_CLOSED` in this sandbox. This is a network limit, not a fix.
  The map shows no tiles here, so it is not visually verified with a basemap.
- `/tmp/browser/states.mjs` is a stale combined script that fails with a top-level-await error (exit 13).
  Its loading and error checks were re-run as the separate scripts above.

Screenshots are kept outside Git in the sandbox at `/tmp/shots/`. None are committed.

## Commands that reproduce the backend checks

```
env -i PATH=/usr/bin:/bin HOME=$HOME python -m pytest -p no:cacheprovider -q
ruff check app/services/landsat_lst_service.py app/services/boundary_service.py tests/test_landsat_lst.py tests/test_boundary.py
cd frontend && npx tsc --noEmit && npm run build
```
