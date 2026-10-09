import { useEffect, useMemo } from "react";
import { CircleMarker, GeoJSON, MapContainer, TileLayer, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import type { RiskAnalysisResult } from "../types";
import { hepsToColor, scaleCssGradient } from "../lib/color";
import { buildHeatField, type HeatField } from "../lib/heatgrid";

type BBox = [number, number, number, number];

function FitBounds({ bbox }: { bbox: BBox }) {
  const map = useMap();
  useEffect(() => {
    map.fitBounds(
      [
        [bbox[1], bbox[0]],
        [bbox[3], bbox[2]],
      ],
      { padding: [24, 24] },
    );
  }, [bbox, map]);
  return null;
}

export default function HeatMap({
  result,
  bbox,
  seed,
}: {
  result: RiskAnalysisResult | null;
  bbox: BBox;
  seed: string;
}) {
  const baseHeps = result?.heps_score ?? 0;

  const field: HeatField | null = useMemo(
    () => (result ? buildHeatField(bbox, baseHeps, seed) : null),
    [result, bbox, baseHeps, seed],
  );

  const collection = useMemo(
    () =>
      field && {
        type: "FeatureCollection" as const,
        features: field.cells.map((cell) => ({
          type: "Feature" as const,
          geometry: { type: "Polygon" as const, coordinates: [cell.ring] },
          properties: { heps: cell.heps },
        })),
      },
    [field],
  );

  const center: [number, number] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2];

  return (
    <section className="relative h-full w-full overflow-hidden rounded-2xl border border-fuchsia-500/20 bg-zinc-950/70">
      <h2 className="absolute left-4 top-3 z-[1000] text-xs font-semibold uppercase tracking-widest text-zinc-300">
        Interactive Leaflet Map — heat exposure
      </h2>

      <MapContainer
        center={center}
        zoom={10}
        className="h-full w-full"
        scrollWheelZoom
        attributionControl
      >
        <FitBounds bbox={bbox} />
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />

        {collection && (
          <GeoJSON
            key={`${seed}-${baseHeps.toFixed(3)}`}
            data={collection}
            style={(feature) => ({
              color: "#0b0f0e",
              weight: 0.6,
              fillColor: hepsToColor((feature?.properties?.heps as number) ?? 0),
              fillOpacity: 0.55,
            })}
            onEachFeature={(feature, layer) => {
              const heps = (feature.properties?.heps as number) ?? 0;
              layer.bindTooltip(
                `HEPS ${heps.toFixed(2)} — ${heps >= 0.6 ? "high exposure" : heps >= 0.3 ? "moderate" : "low"}`,
                { sticky: true },
              );
            }}
          />
        )}

        {field?.hotspots.map((spot, i) => (
          <CircleMarker
            key={i}
            center={spot.center}
            radius={7}
            pathOptions={{
              color: "#ffffff",
              weight: 1.5,
              fillColor: hepsToColor(spot.heps),
              fillOpacity: 0.95,
            }}
          >
          </CircleMarker>
        ))}
      </MapContainer>

      {/* Legend */}
      <div className="absolute bottom-6 right-4 z-[1000] flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-950/80 p-2">
        <span className="text-[10px] text-zinc-400">0.0</span>
        <div className="h-24 w-2 rounded-full" style={{ background: scaleCssGradient() }} />
        <span className="text-[10px] text-zinc-400">1.0</span>
        <span className="ml-1 rotate-180 text-[10px] tracking-wide text-zinc-500 [writing-mode:vertical-rl]">
          heps_score
        </span>
      </div>

      {!result && (
        <div className="absolute inset-x-0 top-12 z-[1000] mx-auto w-fit rounded-full border border-zinc-800 bg-zinc-950/80 px-4 py-1 text-xs text-zinc-400">
          Run a query to render the heat overlay
        </div>
      )}
    </section>
  );
}
