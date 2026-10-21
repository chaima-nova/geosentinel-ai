// Client-side mirrors of the backend Pydantic contracts (app/core/schemas.py).
// Kept structural so the JSON returned by POST /api/v1/query validates against them.

export interface SpatialQueryRequest {
  query: string;
  bbox: [number, number, number, number]; // [lon_min, lat_min, lon_max, lat_max]
  start_date: string;
  end_date: string;
}

export interface RiskFactorProps {
  heat_exposure: number | null;
  cooling_deficit: number | null;
  vulnerability: number;
}

export interface HeatFeatureProperties {
  query?: string;
  start_date?: string;
  end_date?: string;
  heps_score?: number | null;
  grid_stress_level?: string | null;
  data_complete?: boolean;
  evidence_status?: EvidenceStatus | null;
  notes?: string[];
  stac_id?: string;
  cloud_cover?: number;
  ndvi_mean?: number;
  lst_celsius_max?: number;
  factors?: RiskFactorProps;
  error?: string;
  [key: string]: unknown;
}

export interface GeoJsonFeature {
  type: "Feature";
  geometry: {
    type: string;
    coordinates: unknown;
  };
  properties: HeatFeatureProperties;
}

export interface GeoJsonFeatureCollection {
  type: "FeatureCollection";
  features: GeoJsonFeature[];
}

// Mirrors app.core.schemas.EvidenceStatus. "insufficient" = no defensible score.
export type EvidenceStatus = "complete" | "partial" | "insufficient" | "failed";

// Mirrors app.core.schemas.EvidenceItem. provenance is one of
// verified | synthetic | placeholder | missing | failed | stale | invalid.
export interface EvidenceItem {
  source: string;
  provenance: string;
  detail: string | null;
}

export interface RiskAnalysisResult {
  // null when no defensible score exists: never render it as 0.
  heps_score: number | null;
  grid_stress_level: string | null;
  geojson_features: GeoJsonFeatureCollection;
  audit_status: string;
  evidence_status?: EvidenceStatus | null;
  evidence?: EvidenceItem[];
}

export interface HealthResponse {
  status: string;
  app: { name: string; version: string; environment: string; python?: string };
  credentials: { configured: boolean; missing: string[] };
  database: { url: string; extensions?: string[] };
  cors_origins: string[];
}

// --- Pipeline / DAG visual state ---

export type NodeStatus =
  | "pending" // gray
  | "running" // yellow, pulsing
  | "success" // green
  | "warning" // amber: usable but needs review (escalated)
  | "failed"; // red: failed or degraded

export type NodeId = "coordinator" | "ingestion" | "reasoning" | "auditor";

export interface DagState {
  coordinator: NodeStatus;
  ingestion: NodeStatus;
  reasoning: NodeStatus;
  auditor: NodeStatus;
}

export const INITIAL_DAG: DagState = {
  coordinator: "pending",
  ingestion: "pending",
  reasoning: "pending",
  auditor: "pending",
};
