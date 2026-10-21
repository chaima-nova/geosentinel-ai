import type { EvidenceItem, RiskAnalysisResult } from "../types";
import { areaForBbox, roleLabel, validationLabel } from "../lib/studyAreas";

// Each provenance state is labelled with what it means for a score. Only VERIFIED
// evidence can support a result; everything else is shown as unavailable or unusable.
const PROVENANCE: Record<string, { label: string; styles: string; meaning: string }> = {
  verified: {
    label: "VERIFIED",
    styles: "bg-emerald-500/15 text-emerald-300 ring-emerald-500/40",
    meaning: "Measured from a source that passed validation.",
  },
  missing: {
    label: "UNAVAILABLE",
    styles: "bg-zinc-500/15 text-zinc-300 ring-zinc-500/40",
    meaning: "No data for this source. Not treated as zero.",
  },
  placeholder: {
    label: "PLACEHOLDER",
    styles: "bg-amber-500/15 text-amber-300 ring-amber-500/40",
    meaning: "A stand-in value. Not evidence; cannot support a score.",
  },
  synthetic: {
    label: "SYNTHETIC",
    styles: "bg-amber-500/15 text-amber-300 ring-amber-500/40",
    meaning: "Generated for testing. Not evidence; cannot support a score.",
  },
  stale: {
    label: "STALE",
    styles: "bg-rose-500/15 text-rose-300 ring-rose-500/40",
    meaning: "Too old for the requested window. Cannot support a score.",
  },
  invalid: {
    label: "INVALID",
    styles: "bg-rose-500/15 text-rose-300 ring-rose-500/40",
    meaning: "Failed a validation check (coverage, CRS, units or nodata).",
  },
  failed: {
    label: "FAILED",
    styles: "bg-rose-500/15 text-rose-300 ring-rose-500/40",
    meaning: "The source could not be queried.",
  },
};

function ProvenanceBadge({ provenance }: { provenance: string }) {
  const entry = PROVENANCE[provenance] ?? {
    label: provenance.toUpperCase(),
    styles: "bg-zinc-500/15 text-zinc-300 ring-zinc-500/40",
    meaning: "Unrecognised provenance; treated as unusable.",
  };
  return (
    <span
      className={`shrink-0 rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wide ring-1 ${entry.styles}`}
      title={entry.meaning}
    >
      {entry.label}
    </span>
  );
}

function EvidenceRow({ item }: { item: EvidenceItem }) {
  const meaning = (PROVENANCE[item.provenance] ?? PROVENANCE.invalid).meaning;
  return (
    <li className="flex flex-col gap-1 rounded-lg border border-zinc-800 bg-zinc-900/50 px-3 py-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium capitalize text-zinc-200">
          {item.source.replace(/_/g, " ")}
        </span>
        <ProvenanceBadge provenance={item.provenance} />
      </div>
      {item.detail && <p className="text-xs leading-snug text-zinc-400">{item.detail}</p>}
      <p className="text-[11px] text-zinc-500">{meaning}</p>
    </li>
  );
}

export default function EvidencePanel({
  result,
  bbox,
}: {
  result: RiskAnalysisResult | null;
  bbox: number[];
}) {
  const area = areaForBbox(bbox);
  const scoreWithheld = result !== null && result.heps_score === null;

  return (
    <section
      aria-label="Evidence provenance"
      className="flex flex-col gap-3 rounded-2xl border border-fuchsia-500/20 bg-zinc-950/70 p-4"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-zinc-400">
          Study area and evidence
        </h2>
        {area ? (
          <span className="text-xs text-zinc-400">
            Last query: {area.label} · {roleLabel(area.role)} · {validationLabel(area.validation)}
          </span>
        ) : (
          <span className="text-xs text-zinc-400">
            Last query: custom bounding box · not a configured study area, so no pilot validation applies
          </span>
        )}
      </div>

      {area && area.validation_blockers.length > 0 && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2">
          <p className="text-[11px] font-semibold uppercase tracking-wide text-amber-300">
            Blocking validation for this area
          </p>
          <ul className="mt-1 list-disc space-y-0.5 pl-4 text-xs text-amber-100/80">
            {area.validation_blockers.map((blocker) => (
              <li key={blocker}>{blocker}</li>
            ))}
          </ul>
        </div>
      )}

      {result === null ? (
        <p className="text-xs text-zinc-500">
          Run a query to see the provenance of each evidence source for this area.
        </p>
      ) : (
        <>
          {scoreWithheld && (
            <p className="rounded-lg border border-zinc-700 bg-zinc-900/60 px-3 py-2 text-xs text-zinc-300">
              <span className="font-semibold text-zinc-100">HEPS score withheld.</span>{" "}
              Evidence status is <span className="font-mono">{result.evidence_status ?? "unknown"}</span>
              , so no composite score is reported. A score needs every required input to pass
              validation.
            </p>
          )}
          {result.evidence && result.evidence.length > 0 ? (
            <ul className="flex flex-col gap-2">
              {result.evidence.map((item) => (
                <EvidenceRow key={`${item.source}-${item.provenance}`} item={item} />
              ))}
            </ul>
          ) : (
            <p className="text-xs text-zinc-500">
              This response carries no per-source evidence records.
            </p>
          )}
        </>
      )}

      <p className="text-[11px] leading-snug text-zinc-500">
        Population counts describe exposure only and are not social vulnerability. The composite
        HEPS score also needs validated hazard and sensitivity inputs, and it stays withheld until
        each one passes validation.
      </p>
    </section>
  );
}
