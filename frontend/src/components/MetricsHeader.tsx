import type { RiskAnalysisResult } from "../types";

const STRESS_STYLES: Record<string, string> = {
  low: "bg-emerald-500/15 text-emerald-300 ring-emerald-500/40",
  moderate: "bg-amber-500/15 text-amber-300 ring-amber-500/40",
  high: "bg-orange-500/15 text-orange-300 ring-orange-500/40",
  extreme: "bg-fuchsia-500/15 text-fuchsia-300 ring-fuchsia-500/40",
};

const AUDIT_STYLES: Record<string, string> = {
  passed: "bg-emerald-500/15 text-emerald-300 ring-emerald-500/40",
  approved: "bg-emerald-500/15 text-emerald-300 ring-emerald-500/40",
  escalated: "bg-amber-500/15 text-amber-300 ring-amber-500/40",
  failed: "bg-rose-500/15 text-rose-300 ring-rose-500/40",
  pending: "bg-zinc-500/15 text-zinc-300 ring-zinc-500/40",
};

function Badge({ label, value, styles }: { label: string; value: string; styles: string }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-zinc-800 bg-zinc-900/60 px-4 py-3">
      <span className="text-sm font-medium text-zinc-400">{label}</span>
      <span
        className={`rounded-full px-3 py-1 text-xs font-bold uppercase tracking-wide ring-1 ${styles}`}
      >
        {value}
      </span>
    </div>
  );
}

export default function MetricsHeader({
  result,
  loading,
}: {
  result: RiskAnalysisResult | null;
  loading: boolean;
}) {
  const heps = result?.heps_score;
  const stress = result?.grid_stress_level ?? "—";
  const audit = result?.audit_status ?? "—";

  return (
    <header className="flex flex-wrap items-stretch gap-4 rounded-2xl border border-fuchsia-500/20 bg-zinc-950/70 p-4 backdrop-blur">
      <div className="flex min-w-48 flex-1 flex-col justify-center rounded-xl border border-zinc-800 bg-zinc-900/60 px-4 py-3">
        <div className="flex items-baseline gap-2">
          <span className="text-sm font-medium text-zinc-400">HEPS Score:</span>
          <span
            className={`text-3xl font-extrabold tabular-nums ${
              loading ? "animate-pulse text-fuchsia-300" : "text-fuchsia-400"
            }`}
          >
            {heps === undefined ? "–.–" : heps.toFixed(2)}
          </span>
          <span className="text-sm text-zinc-500">/ 1.00</span>
        </div>
        <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
          <div
            className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-fuchsia-500 transition-all duration-700"
            style={{ width: `${Math.round((heps ?? 0) * 100)}%` }}
          />
        </div>
      </div>

      <Badge
        label="Grid Stress:"
        value={stress}
        styles={STRESS_STYLES[stress] ?? "bg-zinc-500/15 text-zinc-300 ring-zinc-500/40"}
      />
      <Badge
        label="Audit Status:"
        value={audit}
        styles={AUDIT_STYLES[audit] ?? "bg-zinc-500/15 text-zinc-300 ring-zinc-500/40"}
      />
    </header>
  );
}
