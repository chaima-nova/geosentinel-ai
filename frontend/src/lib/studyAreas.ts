// Study-area registry for the dashboard. The source of truth is
// config/study_areas.json at the repository root, shared with the backend tests.
// An area with a null bbox is shown as unavailable with its reason and is never queried.

import registryJson from "../../../config/study_areas.json";

export type BBox = [number, number, number, number]; // [lon_min, lat_min, lon_max, lat_max]

export type StudyAreaRole =
  | "primary_pilot"
  | "secondary_pilot"
  | "expansion_candidate"
  | "demo_legacy";

export interface StudyArea {
  id: string;
  label: string;
  country_iso3: string;
  role: StudyAreaRole;
  bbox: BBox | null;
  bbox_basis?: string;
  unavailable_reason?: string;
  validation: string;
  validation_blockers: string[];
}

interface StudyAreaRegistry {
  default_area: string;
  areas: StudyArea[];
}

const registry = registryJson as unknown as StudyAreaRegistry;

export const STUDY_AREAS: StudyArea[] = registry.areas;

export const DEFAULT_AREA_ID: string = registry.default_area;

export function defaultArea(): StudyArea {
  const area = STUDY_AREAS.find((a) => a.id === DEFAULT_AREA_ID);
  if (!area || area.bbox === null) {
    throw new Error(`default study area '${DEFAULT_AREA_ID}' has no bounding box`);
  }
  return area;
}

export function roleLabel(role: StudyAreaRole): string {
  switch (role) {
    case "primary_pilot":
      return "Primary pilot";
    case "secondary_pilot":
      return "Secondary pilot";
    case "expansion_candidate":
      return "Expansion candidate";
    case "demo_legacy":
      return "Demo preset (not a pilot)";
  }
}

export function validationLabel(validation: string): string {
  switch (validation) {
    case "validated":
      return "Validated";
    case "not_started":
      return "Not started";
    default:
      return "Not validated";
  }
}

/** The configured area whose bbox exactly matches ``bbox``, or ``null`` for a custom box. */
export function areaForBbox(bbox: number[]): StudyArea | null {
  return (
    STUDY_AREAS.find(
      (a) =>
        a.bbox !== null &&
        a.bbox.length === bbox.length &&
        a.bbox.every((v, i) => v === bbox[i]),
    ) ?? null
  );
}
