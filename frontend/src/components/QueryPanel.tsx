import { useState } from "react";
import type { SpatialQueryRequest } from "../types";
import { type BBox, STUDY_AREAS, defaultArea, roleLabel } from "../lib/studyAreas";

// Presets come from config/study_areas.json. Areas without a verified bbox are listed
// as unavailable, with their reason, and cannot be submitted.
const PRESETS = STUDY_AREAS;

const inputCls =
  "w-full rounded-lg border border-zinc-800 bg-zinc-900/80 px-2.5 py-1.5 text-sm text-zinc-100 " +
  "placeholder-zinc-500 focus:border-fuchsia-500 focus:outline-none";

const labelCls = "mb-1 block text-[10px] font-semibold uppercase tracking-widest text-zinc-500";

export default function QueryPanel({
  onSubmit,
  loading,
}: {
  onSubmit: (request: SpatialQueryRequest) => void;
  loading: boolean;
}) {
  const [query, setQuery] = useState(
    "Assess heat-island exposure and energy-grid load for industrial corridors during extreme weather",
  );
  const initial = defaultArea();
  const [areaId, setAreaId] = useState<string>(initial.id);
  const [bbox, setBbox] = useState<BBox>(initial.bbox as BBox);
  const [start, setStart] = useState("2026-06-01");
  const [end, setEnd] = useState("2026-08-31");

  const setCoord = (index: number, value: number) => {
    setAreaId(""); // a hand-edited box is no longer the named study area
    setBbox((prev) => {
      const next = [...prev] as BBox;
      next[index] = value;
      return next;
    });
  };

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({ query, bbox, start_date: start, end_date: end });
  };

  return (
    <form
      onSubmit={submit}
      className="flex flex-col gap-4 rounded-2xl border border-fuchsia-500/20 bg-zinc-950/70 p-4"
    >
      <h2 className="text-xs font-semibold uppercase tracking-widest text-zinc-400">
        Sidebar Query Panel
      </h2>

      <div>
        <label className={labelCls} htmlFor="nlq">Natural-language query</label>
        <textarea
          id="nlq"
          rows={3}
          className={inputCls + " resize-none"}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Describe the climate risk to assess…"
        />
      </div>

      <div>
        <label className={labelCls} htmlFor="preset">Bounding box coordinates</label>
        <select
          id="preset"
          className={inputCls}
          value={areaId}
          onChange={(e) => {
            const preset = PRESETS.find((p) => p.id === e.target.value);
            if (preset && preset.bbox !== null) {
              setAreaId(preset.id);
              setBbox(preset.bbox);
            }
          }}
        >
          {areaId === "" && (
            <option value="" disabled>
              Custom bounding box
            </option>
          )}
          {PRESETS.map((p) => (
            <option key={p.id} value={p.id} disabled={p.bbox === null}>
              {p.label}
              {p.bbox === null ? " — unavailable" : ""}
            </option>
          ))}
        </select>
        {(() => {
          const selected = PRESETS.find((p) => p.id === areaId);
          if (!selected) return null;
          return (
            <p className="mt-1 text-[11px] leading-snug text-zinc-500">
              {roleLabel(selected.role)}.{" "}
              {selected.bbox === null
                ? selected.unavailable_reason
                : selected.bbox_basis ?? ""}
            </p>
          );
        })()}

        <div className="mt-2 grid grid-cols-2 gap-2">
          <div>
            <label className={labelCls} htmlFor="lonMin">Lon Min</label>
            <input id="lonMin" type="number" step="0.0001" className={inputCls}
              value={bbox[0]} onChange={(e) => setCoord(0, Number(e.target.value))} />
          </div>
          <div>
            <label className={labelCls} htmlFor="latMin">Lat Min</label>
            <input id="latMin" type="number" step="0.0001" className={inputCls}
              value={bbox[1]} onChange={(e) => setCoord(1, Number(e.target.value))} />
          </div>
          <div>
            <label className={labelCls} htmlFor="lonMax">Lon Max</label>
            <input id="lonMax" type="number" step="0.0001" className={inputCls}
              value={bbox[2]} onChange={(e) => setCoord(2, Number(e.target.value))} />
          </div>
          <div>
            <label className={labelCls} htmlFor="latMax">Lat Max</label>
            <input id="latMax" type="number" step="0.0001" className={inputCls}
              value={bbox[3]} onChange={(e) => setCoord(3, Number(e.target.value))} />
          </div>
        </div>
      </div>

      <div>
        <label className={labelCls}>Date range</label>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className={labelCls} htmlFor="start">Start</label>
            <input id="start" type="date" className={inputCls} value={start}
              onChange={(e) => setStart(e.target.value)} />
          </div>
          <div>
            <label className={labelCls} htmlFor="end">End</label>
            <input id="end" type="date" className={inputCls} value={end}
              onChange={(e) => setEnd(e.target.value)} />
          </div>
        </div>
      </div>

      <button
        type="submit"
        disabled={loading}
        className="rounded-xl bg-gradient-to-r from-fuchsia-600 to-emerald-500 px-4 py-2.5 text-sm font-bold uppercase tracking-wide text-white
          transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {loading ? "Analyzing…" : "Analyze Risk"}
      </button>
    </form>
  );
}
