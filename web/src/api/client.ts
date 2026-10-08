// Typed access to the lattice viewer API (eicvibe.api, /api/v1).
//
// Everything below a lattice digest is immutable, so each URL is fetched at
// most once per page and shared by every view that needs it. The browser's
// HTTP cache revalidates with ETags, so a reload costs a 304 per request.

import type { components } from "./schema";

type Schemas = components["schemas"];
export type LatticeEntry = Schemas["LatticeEntry"];
export type LatticeSummary = Schemas["LatticeSummary"];
export type BranchSummary = Schemas["BranchSummary"];
export type ForkSummary = Schemas["ForkSummary"];
export type BranchData = Schemas["BranchData"];
export type Column = Schemas["Column"];
export type OccurrenceDetail = Schemas["OccurrenceDetail"];
export type Pose = Schemas["Pose"];

const BASE = "/api/v1";

/** The response schema this client reads (eicvibe.api.schemas.SCHEMA_VERSION). */
export const SCHEMA_VERSION = 3;

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(BASE + path, { signal, headers: { Accept: "application/json" } });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = ((await response.json()) as { detail?: string }).detail ?? detail;
    } catch {
      // not JSON
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

const immutable = new Map<string, Promise<unknown>>();

/** A response that never changes for its URL, fetched once per page. */
function once<T>(path: string): Promise<T> {
  let found = immutable.get(path);
  if (!found) {
    found = getJson<T>(path);
    found.catch(() => immutable.delete(path)); // let a failed request be retried
    immutable.set(path, found);
  }
  return found as Promise<T>;
}

const enc = encodeURIComponent;

export const api = {
  /** The lattices on the server, at their current revisions. Not cached: revisions change. */
  lattices: (signal?: AbortSignal) => getJson<LatticeEntry[]>("/lattices", signal),
  summary: (digest: string) => once<LatticeSummary>(`/lattices/${enc(digest)}`),
  branch: (digest: string, branch: string) =>
    once<BranchData>(`/lattices/${enc(digest)}/branches/${enc(branch)}`),
  occurrence: (digest: string, occurrence: string) =>
    once<OccurrenceDetail>(`/lattices/${enc(digest)}/occurrences/${enc(occurrence)}`),
  downloadUrl: (digest: string) => `${BASE}/lattices/${enc(digest)}/lattice.json`,
};
