# Satellite NDVI comparison (synthetic)

**SYNTHETIC SCENE - not a Copernicus product.** Results show behaviour against a known truth, not validation against live products.

- Scene: 200 px, seed 7
- Truth NDVI (clear native pixels): 0.537069
- Scene cloud (SCL 8-9): 1.69 %

| Method | NDVI | abs. error | valid coverage | offset | mask | withheld |
|---|---|---|---|---|---|---|
| legacy_raw_dn | 0.385152 | 0.151917 | None | 0.0 | none | - |
| offset_only | 0.573299 | 0.03623 | 1.0 | -1000.0 | none | - |
| corrected | None | None | 0.875556 | -1000.0 | SCL | True |
| corrected_min_coverage_0.8 | 0.544301 | 0.007232 | 0.875556 | -1000.0 | SCL | False |

## Notes

- Truth is the mean NDVI over clear native-resolution pixels.
- Methods average on the decimated sample grid, so a small gap to truth is expected even when correct.
- legacy_raw_dn and offset_only have no mask: the no-data strip (DN 0) and cloud/shadow are averaged in.
- The coverage gate withholds NDVI below min_valid_coverage; the 0.8 row shows the estimate behind that decision.
- No live Copernicus or Earth Search data was used; no odc-stac comparison was run.
