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
export type EngineInfo = Schemas["EngineInfo"];
export type BranchOptics = Schemas["BranchOptics"];
export type RunEntry = Schemas["RunEntry"];
export type OrbitFrames = Schemas["OrbitFrames"];
export type CaptureEntry = Schemas["CaptureEntry"];
export type RunCheck = Schemas["RunCheck"];
export type PlaneCheck = Schemas["PlaneCheck"];
export type Detuning = Schemas["Detuning"];

const BASE = "/api/v1";

/** The response schema this client reads (eicvibe.api.schemas.SCHEMA_VERSION). */
export const SCHEMA_VERSION = 5;

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
  /** Engines that compute optics; their versions fill in once they have run. */
  engines: () => getJson<EngineInfo[]>("/engines"),
  /** Design optics of a branch; the first request for a revision may take a while (Julia starts). */
  /** Virtual rings' runs the server reads; not cached: runs come and go on. */
  runs: (signal?: AbortSignal) => getJson<RunEntry[]>("/runs", signal),
  /** ``count`` orbits ``every`` turns apart, the newest at turn ``end - 1`` or the latest. */
  orbits: (run: string, every: number, count: number, end?: number, signal?: AbortSignal, source = "buffer") =>
    getJson<OrbitFrames>(
      `/runs/${enc(run)}/orbits?every=${every}&count=${count}${end === undefined ? "" : `&end=${end}`}` +
        (source === "buffer" ? "" : `&source=${enc(source)}`),
      signal,
    ),
  /** A ring run's captures: its BPM buffer frozen by a kick, injection, loss or request. */
  captures: (run: string, signal?: AbortSignal) => getJson<CaptureEntry[]>(`/runs/${enc(run)}/captures`, signal),
  /** A ring run's tracking against the model: tunes, phases, beta and orbit at its BPMs (engine work: seconds). */
  check: (
    run: string,
    options: { source: string; turns: number; first?: number; engine?: string; detuning?: boolean },
    signal?: AbortSignal,
  ) =>
    getJson<RunCheck>(
      `/runs/${enc(run)}/check?source=${enc(options.source)}&turns=${options.turns}` +
        (options.first === undefined ? "" : `&first=${options.first}`) +
        (options.engine ? `&engine=${enc(options.engine)}` : "") +
        (options.detuning ? "&detuning=true" : ""),
      signal,
    ),
  optics: (digest: string, branch: string, engine: string, fidelity: string) =>
    once<BranchOptics>(
      `/lattices/${enc(digest)}/branches/${enc(branch)}/optics?engine=${enc(engine)}&fidelity=${enc(fidelity)}`,
    ),
};
