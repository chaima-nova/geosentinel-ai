import type { DagState, NodeId, NodeStatus } from "../types";

const NODE_ORDER: NodeId[] = ["coordinator", "ingestion", "reasoning", "auditor"];

const NODE_TITLES: Record<NodeId, string> = {
  coordinator: "Meta-Coordinator",
  ingestion: "Spatial Ingestion Agent",
  reasoning: "Risk Reasoning Agent",
  auditor: "Sentinel Auditor Agent",
};

const STATUS_TEXT: Record<NodeStatus, string> = {
  pending: "PENDING",
  running: "COMPUTING…",
  success: "SUCCESS",
  warning: "NEEDS REVIEW",
  failed: "DEGRADED",
};

const LIGHT_STYLES: Record<NodeStatus, string> = {
  pending: "bg-zinc-600",
  running: "bg-amber-400 node-pulse",
  success: "bg-emerald-400",
  warning: "bg-amber-400",
  failed: "bg-rose-500",
};

const CARD_STYLES: Record<NodeStatus, string> = {
  pending: "border-zinc-800",
  running: "border-amber-400/60 shadow-[0_0_18px_rgba(251,191,36,0.35)]",
  success: "border-emerald-500/50",
  warning: "border-amber-400/50",
  failed: "border-rose-500/60 shadow-[0_0_18px_rgba(244,63,94,0.3)]",
};

function Connector({ active, done }: { active: boolean; done: boolean }) {
  return (
    <div className="relative hidden flex-1 items-center self-center md:flex" aria-hidden>
      <div className={`dag-link ${active ? "dag-link-active" : done ? "dag-link-done" : ""}`} />
      <div className={`dag-arrow ${active ? "text-fuchsia-400" : done ? "text-emerald-400" : "text-zinc-700"}`}>
        ▸
      </div>
    </div>
  );
}

export default function AgentWorkflowDAG({
  dag,
  notes,
}: {
  dag: DagState;
  notes?: Partial<Record<NodeId, string>>;
}) {
  return (
    <section className="rounded-2xl border border-fuchsia-500/20 bg-zinc-950/70 p-4">
      <h2 className="mb-3 text-xs font-semibold uppercase tracking-widest text-zinc-400">
        Agent Workflow Pipeline
      </h2>
      <div className="flex flex-col items-stretch gap-2 md:flex-row md:items-center">
        {NODE_ORDER.map((id, idx) => {
          const status = dag[id];
          const prevDone = idx > 0 && dag[NODE_ORDER[idx - 1]] === "success";
          const linkActive = prevDone && status === "running";
          return (
            <div key={id} className="contents">
              {idx > 0 && <Connector active={linkActive} done={prevDone && status !== "pending"} />}
              <div
                className={`flex items-center gap-3 rounded-xl border bg-zinc-900/70 px-3 py-3 transition-all duration-300 ${CARD_STYLES[status]}`}
              >
                <span className={`h-3.5 w-3.5 shrink-0 rounded-full ${LIGHT_STYLES[status]}`} />
                <div className="min-w-0">
                  <div className="truncate text-xs font-semibold text-zinc-100">
                    {NODE_TITLES[id]}
                  </div>
                  <div className="truncate text-[10px] uppercase tracking-wide text-zinc-500">
                    {notes?.[id] ?? STATUS_TEXT[status]}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
