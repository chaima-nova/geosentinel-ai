/**
 * Deterministic heat-grid synthesis for the interactive map.
 *
 * {@link buildHeatField} spreads one composite HEPS score (returned by the
 * backend for the whole query area) into a grid of per-cell scores so the
 * dashboard can render a plausible heat surface. For a given ``bbox`` +
 * ``baseHeps`` + ``seed`` the output is fully deterministic, so re-rendering
 * the same query never flickers.
 *
 * Coordinate conventions:
 * - ``bbox`` is ``[lon_min, lat_min, lon_max, lat_max]`` (see ``types.ts``).
 * - Cell rings are GeoJSON positions: ``[lon, lat]``.
 * - Hotspot centers are Leaflet points: ``[lat, lon]``.
 */

/** Query bounding box: ``[lon_min, lat_min, lon_max, lat_max]``. */
export type BBox = [number, number, number, number];

/** One grid cell of the synthetic heat surface. */
export interface HeatCell {
  /** Closed GeoJSON linear ring (``[lon, lat]`` positions, first == last). */
  ring: [number, number][];
  /** Synthetic HEPS score for the cell, in [0, 1]. */
  heps: number;
}

/** A local maximum of the heat surface, rendered as a marker. */
export interface HeatSpot {
  /** Leaflet marker position: ``[lat, lon]``. */
  center: [number, number];
  /** HEPS score at the hotspot, in [0, 1]. */
  heps: number;
}

/** The complete synthetic heat surface for one query. */
export interface HeatField {
  /** Grid cells covering the bbox. */
  cells: HeatCell[];
  /** Top scoring cells, hottest first. */
  hotspots: HeatSpot[];
  /** Lowest cell score in the field. */
  min: number;
  /** Highest cell score in the field. */
  max: number;
}

/** Optional tuning knobs for {@link buildHeatField}. */
export interface HeatFieldOptions {
  /** Grid columns (lon direction). Default 8. */
  cols?: number;
  /** Grid rows (lat direction). Default 6. */
  rows?: number;
  /** Max deviation from ``baseHeps`` introduced by the noise, in [0, 1]. Default 0.35. */
  amplitude?: number;
  /** Number of hotspot markers to extract. Default 3. */
  hotspotCount?: number;
}

function clamp01(value: number): number {
  if (Number.isNaN(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

/** xmur3-style string hash: folds a seed string into a 32-bit integer. */
function hashSeed(seed: string): number {
  let h = 1779033703 ^ seed.length;
  for (let i = 0; i < seed.length; i += 1) {
    h = Math.imul(h ^ seed.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  h = Math.imul(h ^ (h >>> 16), 2246822507);
  h = Math.imul(h ^ (h >>> 13), 3266489909);
  return (h ^= h >>> 16) >>> 0;
}

/** mulberry32: tiny deterministic PRNG returning floats in [0, 1). */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), state | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function smoothstep(t: number): number {
  return t * t * (3 - 2 * t);
}

/** Bilinear interpolation of a value-noise lattice at (u, v) in [0, 1]². */
function sampleLattice(lattice: number[][], u: number, v: number): number {
  const rows = lattice.length - 1;
  const cols = lattice[0].length - 1;
  const x = u * cols;
  const y = v * rows;
  const x0 = Math.min(Math.floor(x), cols - 1);
  const y0 = Math.min(Math.floor(y), rows - 1);
  const tx = smoothstep(x - x0);
  const ty = smoothstep(y - y0);
  const c00 = lattice[y0][x0];
  const c10 = lattice[y0][x0 + 1];
  const c01 = lattice[y0 + 1][x0];
  const c11 = lattice[y0 + 1][x0 + 1];
  const top = c00 + (c10 - c00) * tx;
  const bottom = c01 + (c11 - c01) * tx;
  return top + (bottom - top) * ty;
}

/**
 * Build a deterministic synthetic heat field for a query area.
 *
 * Each grid cell gets ``baseHeps`` perturbed by seeded value noise plus a
 * couple of gaussian hot blobs, then clamped to [0, 1]. The mean cell score
 * stays close to ``baseHeps``, so the surface never contradicts the backend.
 *
 * @param bbox - Query area ``[lon_min, lat_min, lon_max, lat_max]``.
 * @param baseHeps - The composite HEPS score reported by the backend.
 * @param seed - Anything that identifies the query (e.g. query text + bbox).
 * @param options - Optional grid size / amplitude / hotspot tuning.
 * @returns A {@link HeatField} with cell polygons and hotspot markers.
 */
export function buildHeatField(
  bbox: BBox,
  baseHeps: number,
  seed: string,
  options: HeatFieldOptions = {},
): HeatField {
  const cols = Math.max(2, options.cols ?? 8);
  const rows = Math.max(2, options.rows ?? 6);
  const amplitude = clamp01(options.amplitude ?? 0.35);
  const hotspotCount = Math.max(0, options.hotspotCount ?? 3);

  const [lonMin, latMin, lonMax, latMax] = bbox;
  const base = clamp01(baseHeps);
  const rand = mulberry32(hashSeed(seed));

  // Value-noise lattice over the grid corners.
  const lattice: number[][] = [];
  for (let r = 0; r <= rows; r += 1) {
    const row: number[] = [];
    for (let c = 0; c <= cols; c += 1) row.push(rand());
    lattice.push(row);
  }

  // A few gaussian hot blobs for organic structure.
  const blobs = Array.from({ length: 3 }, () => ({
    u: rand(),
    v: rand(),
    radius: 0.18 + rand() * 0.3,
    strength: 0.5 + rand() * 0.5,
  }));

  const dLon = (lonMax - lonMin) / cols;
  const dLat = (latMax - latMin) / rows;

  const cells: HeatCell[] = [];
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const u = (c + 0.5) / cols;
      const v = (r + 0.5) / rows;

      const noise = sampleLattice(lattice, u, v);
      let blobSum = 0;
      for (const blob of blobs) {
        const du = u - blob.u;
        const dv = v - blob.v;
        blobSum += blob.strength * Math.exp(-(du * du + dv * dv) / (2 * blob.radius * blob.radius));
      }

      // Blend: centered noise (−0.5..0.5) weighted against the blob field.
      const field = 0.6 * (noise - 0.5) + 0.4 * (Math.min(blobSum, 1) - 0.5);
      const heps = clamp01(base + amplitude * 2 * field);

      const lon0 = lonMin + c * dLon;
      const lat0 = latMin + r * dLat;
      const lon1 = lon0 + dLon;
      const lat1 = lat0 + dLat;

      cells.push({
        ring: [
          [lon0, lat0],
          [lon1, lat0],
          [lon1, lat1],
          [lon0, lat1],
          [lon0, lat0],
        ],
        heps,
      });
    }
  }

  const scored = cells
    .map((cell, index) => ({ cell, index }))
    .sort((a, b) => b.cell.heps - a.cell.heps);

  const hotspots: HeatSpot[] = scored.slice(0, hotspotCount).map(({ cell, index }) => {
    const row = Math.floor(index / cols);
    const col = index % cols;
    const lat = latMin + (row + 0.5) * dLat;
    const lon = lonMin + (col + 0.5) * dLon;
    return { center: [lat, lon], heps: cell.heps };
  });

  return {
    cells,
    hotspots,
    min: scored.length > 0 ? scored[scored.length - 1].cell.heps : base,
    max: scored.length > 0 ? scored[0].cell.heps : base,
  };
}
