import { useEffect, useRef } from "react";

export default function ExecutionFeed({ lines }: { lines: string[] }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    ref.current?.scrollTo({ top: ref.current.scrollHeight, behavior: "smooth" });
  }, [lines]);

  return (
    <section className="flex min-h-0 flex-1 flex-col rounded-2xl border border-fuchsia-500/20 bg-zinc-950/70 p-4">
      <h2 className="mb-2 text-xs font-semibold uppercase tracking-widest text-zinc-400">
        Execution Stream Feed
      </h2>
      <div
        ref={ref}
        className="h-40 flex-1 overflow-y-auto rounded-lg border border-zinc-800 bg-black/50 p-2 font-mono text-[11px] leading-relaxed text-emerald-300/90"
      >
        {lines.length === 0 ? (
          <p className="text-zinc-600">awaiting first query…</p>
        ) : (
          lines.map((line, i) => <p key={i}>{line}</p>)
        )}
      </div>
    </section>
  );
}
