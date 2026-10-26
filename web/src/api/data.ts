// What the views along s draw, wherever it comes from (ADR 0016): the page asks
// the server (eicvibe.api), a notebook's widget asks its kernel. The answers are
// the same JSON either way: the kernel builds them with the server's own code.

import { api, type BranchData, type BranchOptics, type CaptureEntry, type EngineInfo, type OrbitFrames } from "./client";

export interface ViewerData {
  /** Engines that compute optics; their versions fill in once they have run. */
  engines(): Promise<EngineInfo[]>;
  /** A branch's data: its table and floor path. */
  branch(branch: string): Promise<BranchData>;
  /** A branch's (or a route's, ``route:<name>``) optics, with the lattice's settings or an operating mode's. */
  optics(branch: string, engine: string, fidelity: string, mode: string): Promise<BranchOptics>;
  /** ``count`` orbits of a run ``every`` turns apart, the newest at ``end - 1`` or the latest. */
  orbits(run: string, every: number, count: number, end: number | undefined, signal: AbortSignal | undefined, source: string): Promise<OrbitFrames>;
  /** A ring run's captures: its BPM buffer frozen by a kick, injection, loss or request. */
  captures(run: string, signal?: AbortSignal): Promise<CaptureEntry[]>;
}

/** A revision's data from the server. */
export function serverData(digest: string): ViewerData {
  return {
    engines: api.engines,
    branch: (branch) => api.branch(digest, branch),
    optics: (branch, engine, fidelity, mode) => api.optics(digest, branch, engine, fidelity, mode),
    orbits: api.orbits,
    captures: api.captures,
  };
}
