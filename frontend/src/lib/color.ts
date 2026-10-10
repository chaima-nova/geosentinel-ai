/**
 * Color helpers for rendering HEPS (Heat Exposure & Population Stress) scores.
 *
 * The scale mirrors the backend ``StressThresholds`` (see
 * ``app/services/risk_service.py``): moderate >= 0.30, high >= 0.55,
 * extreme >= 0.75, and reuses the dashboard palette (emerald → amber →
 * orange → fuchsia, cf. ``MetricsHeader``).
 */

/** One anchor point of the HEPS color ramp. */
export interface ColorStop {
  /** HEPS score position in [0, 1]. */
  t: number;
  /** CSS hex color at that position. */
  hex: string;
}

/**
 * Anchor stops of the HEPS color ramp, aligned with the backend stress
 * thresholds so a cell's color matches its ``grid_stress_level``.
 */
export const HEPS_COLOR_STOPS: readonly ColorStop[] = [
  { t: 0.0, hex: "#10b981" }, // emerald-500  — low
  { t: 0.3, hex: "#f59e0b" }, // amber-500    — moderate
  { t: 0.55, hex: "#f97316" }, // orange-500   — high
  { t: 0.75, hex: "#d946ef" }, // fuchsia-500  — extreme
  { t: 1.0, hex: "#a21caf" }, // fuchsia-700  — deep extreme
];

/** Clamp any number into the [0, 1] range used by HEPS scores. */
export function clamp01(value: number): number {
  if (Number.isNaN(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

/** Map the backend stress vocabulary onto a HEPS score. */
export function hepsToStressLevel(heps: number): "low" | "moderate" | "high" | "extreme" {
  const score = clamp01(heps);
  if (score >= 0.75) return "extreme";
  if (score >= 0.55) return "high";
  if (score >= 0.3) return "moderate";
  return "low";
}

function hexToRgb(hex: string): [number, number, number] {
  const value = hex.replace("#", "");
  return [
    Number.parseInt(value.slice(0, 2), 16),
    Number.parseInt(value.slice(2, 4), 16),
    Number.parseInt(value.slice(4, 6), 16),
  ];
}

function toHexChannel(channel: number): string {
  return Math.round(channel).toString(16).padStart(2, "0");
}

function rgbToHex(r: number, g: number, b: number): string {
  return `#${toHexChannel(r)}${toHexChannel(g)}${toHexChannel(b)}`;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Map a HEPS score to a display color by interpolating between the anchor
 * stops in {@link HEPS_COLOR_STOPS}.
 *
 * @param heps - A HEPS score; values outside [0, 1] are clamped.
 * @returns A CSS hex color string, e.g. `"#f97316"`.
 */
export function hepsToColor(heps: number): string {
  const score = clamp01(heps);
  const stops = HEPS_COLOR_STOPS;

  if (score <= stops[0].t) return stops[0].hex;
  const last = stops[stops.length - 1];
  if (score >= last.t) return last.hex;

  for (let i = 0; i < stops.length - 1; i += 1) {
    const lower = stops[i];
    const upper = stops[i + 1];
    if (score >= lower.t && score <= upper.t) {
      const span = upper.t - lower.t || 1;
      const t = (score - lower.t) / span;
      const [r1, g1, b1] = hexToRgb(lower.hex);
      const [r2, g2, b2] = hexToRgb(upper.hex);
      return rgbToHex(lerp(r1, r2, t), lerp(g1, g2, t), lerp(b1, b2, t));
    }
  }
  return last.hex;
}

/**
 * Build a CSS ``linear-gradient`` string sampling the HEPS ramp, used for
 * map legends and score bars.
 *
 * @param direction - Any CSS gradient direction, e.g. ``"to bottom"``
 *   (default, for a vertical legend with 1.0 on top) or ``"to right"``.
 * @returns A CSS value such as
 *   ``"linear-gradient(to bottom, #10b981 0%, #f59e0b 30%, ...)"``.
 */
export function scaleCssGradient(direction: string = "to bottom"): string {
  const stops = HEPS_COLOR_STOPS.map(
    (stop) => `${stop.hex} ${(stop.t * 100).toFixed(0)}%`,
  ).join(", ");
  return `linear-gradient(${direction}, ${stops})`;
}
