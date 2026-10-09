import { useCallback, useRef, useState } from "react";
import AgentWorkflowDAG from "./components/AgentWorkflowDAG";
import ExecutionFeed from "./components/ExecutionFeed";
import HeatMap from "./components/HeatMap";
import MetricsHeader from "./components/MetricsHeader";
import QueryPanel from "./components/QueryPanel";
import { ApiError, describeValidation, runQuery } from "./api";
import { INITIAL_DAG, type DagState, type NodeId, type RiskAnalysisResult, type SpatialQueryRequest } from "./types";

type BBox = [number, number, number, number];

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const ts = () => new Date().toTimeString().slice(0, 8);

export default function App() {
  const [result, setResult] = useState<RiskAnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [dag, setDag] = useState<DagState>(INITIAL_DAG);
  const [notes, setNotes] = useState<Partial<Record<NodeId, string>>>({});
  const [feed, setFeed] = useState<string[]>([]);
  const [bbox, setBbox] = useState<BBox>([6.6, 33.2, 7.1, 33.7]);
  const [seed, setSeed] = useState("initial");
  const busy = useRef(false);

  const push = useCallback((line: string) => {
    setFeed((prev) => [...prev, line]);
  }, []);

  const handleSubmit = useCallback(
    async (request: SpatialQueryRequest) => {
      if (busy.current) return;
      busy.current = true;
      setLoading(true);
      setResult(null);
      setFeed([]);
      setNotes({});
      setBbox(request.bbox);
      setSeed(`${request.query}|${request.bbox.join(",")}`);
      setDag({ ...INITIAL_DAG, coordinator: "running" });
      push(`[${ts()}] meta-coordinator: parsing spatial query`);

      await sleep(400);
      setDag((d) => ({ ...d, coordinator: "success", ingestion: "running" }));
      push(`[${ts()}] spatial-ingestion: querying Sentinel-2 STAC (cloud < 20%)`);

      try {
        const res = await runQuery(request);
        const props = res.geojson_features?.features?.[0]?.properties;
        const degraded = props?.stac_id === "unavailable" || props?.error !== undefined;

        setDag((d) => ({ ...d, ingestion: degraded ? "failed" : "success", reasoning: "running" }));
        setNotes((n) => ({
          ...n,
          coordinator: "success",
          ingestion: degraded ? "degraded — no scene" : "Sentinel-2 data fetched",
        }));
        push(
          degraded
            ? `[${ts()}] WARNING ingestion: no usable scene — scoring from demographics`
            : `[${ts()}] ingestion: scene ${props?.stac_id} fetched (ndvi ${
                props?.ndvi_mean === undefined ? "n/a" : props.ndvi_mean.toFixed(2)
              })`,
        );

        await sleep(500);
        setDag((d) => ({ ...d, reasoning: "success", auditor: "running" }));
        push(`[${ts()}] risk-reasoning: HEPS=${res.heps_score.toFixed(2)} stress=${res.grid_stress_level}`);

        await sleep(400);
        const auditNode =
          res.audit_status === "passed" ? "success" : res.audit_status === "escalated" ? "warning" : "failed";
        setDag((d) => ({ ...d, auditor: auditNode }));
        setNotes((n) => ({ ...n, reasoning: "HEPS computed", auditor: res.audit_status }));
        push(`[${ts()}] sentinel-auditor: verdict=${res.audit_status}`);
        for (const note of res.geojson_features?.features?.[0]?.properties?.notes ?? []) {
          push(`[${ts()}]   note: ${note}`);
        }

        setResult(res);
      } catch (err) {
        setDag((d) => ({
          coordinator: d.coordinator === "success" ? "success" : "failed",
          ingestion: "failed",
          reasoning: "failed",
          auditor: "failed",
        }));
        push(
          `[${ts()}] ERROR: ${
            err instanceof ApiError
              ? `HTTP ${err.status} — ${describeValidation(err.detail)}`
              : err instanceof TypeError
                ? "backend unreachable — is uvicorn running on :8000?"
                : String(err)
          }`,
        );
      } finally {
        setLoading(false);
        busy.current = false;
      }
    },
    [push],
  );

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="mx-auto flex max-w-7xl flex-col gap-4 p-4">
        <header className="flex items-center justify-between rounded-2xl border border-fuchsia-500/30 bg-zinc-950/70 px-4 py-3">
          <div className="flex items-center gap-3">
            <div className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-fuchsia-600 to-emerald-500 text-sm font-black">
              ⬡
            </div>
            <h1 className="text-lg font-bold">GeoSentinel-AI</h1>
          </div>
          <span className="text-xs text-zinc-500">climate risk · multi-agent orchestration</span>
        </header>

        <AgentWorkflowDAG dag={dag} notes={notes} />

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[360px_1fr]">
          <div className="flex flex-col gap-4">
            <QueryPanel onSubmit={handleSubmit} loading={loading} />
            <ExecutionFeed lines={feed} />
          </div>

          <div className="flex flex-col gap-4">
            <MetricsHeader result={result} loading={loading} />
            <div className="h-[520px]">
              <HeatMap result={result} bbox={bbox} seed={seed} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
