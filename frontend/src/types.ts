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
  heps_score?: number;
  grid_stress_level?: string;
  data_complete?: boolean;
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

export interface RiskAnalysisResult {
  heps_score: number;
  grid_stress_level: string;
  geojson_features: GeoJsonFeatureCollection;
  audit_status: string;
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
