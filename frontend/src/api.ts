import type { HealthResponse, RiskAnalysisResult, SpatialQueryRequest } from "./types";

// Relative paths only: the Vite dev server (and the production preview) proxy
// /api and /health to the FastAPI backend. The browser never addresses
// localhost, so the sandboxed preview keeps working.
const QUERY_ENDPOINT = "/api/v1/query";
const HEALTH_ENDPOINT = "/health";

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, detail: unknown) {
    super(`API request failed with status ${status}`);
    this.status = status;
    this.detail = detail;
  }
}

/**
 * POST /api/v1/query — run the risk pipeline for a spatio-temporal request.
 *
 * @throws ApiError on a non-2xx response. A 422 carries the FastAPI validation
 *   detail; callers can surface the offending field.
 */
export async function runQuery(request: SpatialQueryRequest): Promise<RiskAnalysisResult> {
  const response = await fetch(QUERY_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  const body = await safeJson(response);
  if (!response.ok) {
    throw new ApiError(response.status, body);
  }
  return body as RiskAnalysisResult;
}

/** GET /health — process status and configuration summary. */
export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(HEALTH_ENDPOINT);
  if (!response.ok) {
    throw new ApiError(response.status, await safeJson(response));
  }
  return (await response.json()) as HealthResponse;
}

/** Human-readable summary of a FastAPI 422 for display in the feed. */
export function describeValidation(detail: unknown): string {
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        const loc = Array.isArray(item?.loc) ? item.loc.slice(1).join(".") : "body";
        return `${loc}: ${item?.msg ?? "invalid"}`;
      })
      .join("; ");
  }
  return String(detail);
}

async function safeJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}
