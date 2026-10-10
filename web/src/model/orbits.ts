// BPM orbits of a virtual machine's run (ADR 0007), prepared for the orbit
// view: which runs belong to the branch or route shown, how the orbits fade
// with age, the range the plotted readings span, and the BPM nearest a point.

import type { OrbitFrames, RunEntry } from "../api/client";

/** Opacity of orbit ``k`` of ``count``, oldest first: the newest is opaque, older ones fade. */
export function fade(k: number, count: number): number {
  return count <= 1 ? 1 : 0.12 + (0.88 * (k + 1)) / count;
}

/**
 * The runs to show for a branch or route (its id is ``route:<name>``): of this
 * revision first, then of the same lattice and names. A ring's runs are kept
 * apart from a route's, which may share its name (the EIN's BAR ring and route).
 */
export function runsFor(
  runs: readonly RunEntry[],
  digest: string,
  latticeName: string,
  view: { id: string; name: string },
): { run: RunEntry; sameRevision: boolean }[] {
  const kind = view.id.startsWith("route:") ? "route" : "ring";
  const same = runs.filter((r) => r.lattice_digest === digest && r.branch_id === view.id);
  const named = runs.filter(
    (r) => !same.includes(r) && r.kind === kind && r.lattice === latticeName && r.branch === view.name,
  );
  return [...same.map((run) => ({ run, sameRevision: true })), ...named.map((run) => ({ run, sameRevision: false }))];
}

/** Lowest and highest reading of a plane's orbits at BPMs with s in [s0, s1]; null if none. */
export function readingExtent(frames: OrbitFrames, plane: "x" | "y", s0: number, s1: number): [number, number] | null {
  let lo = Infinity, hi = -Infinity;
  frames.s.forEach((s, i) => {
    if (s < s0 || s > s1) return;
    for (const orbit of frames[plane]) {
      const v = orbit[i];
      if (v === null || v === undefined) continue;
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
  });
  return Number.isFinite(lo) ? [lo, hi] : null;
}

/** Index of the BPM nearest ``s`` within ``tolerance`` metres, or null. */
export function nearestBpm(positions: readonly number[], s: number, tolerance: number): number | null {
  let best: number | null = null, distance = tolerance;
  positions.forEach((p, i) => {
    const d = Math.abs(p - s);
    if (d <= distance) {
      distance = d;
      best = i;
    }
  });
  return best;
}
