import { describe, expect, it } from "vitest";

import type { OrbitFrames, RunEntry } from "../api/client";
import { fade, nearestBpm, readingExtent, runsFor } from "./orbits";

const run = (id: string, digest: string, branchId: string, branch = "ring", lattice = "demo"): RunEntry => ({
  id,
  kind: branchId.startsWith("route:") ? "route" : "ring",
  unit: branchId.startsWith("route:") ? "shot" : "turn",
  lattice,
  lattice_digest: digest,
  branch,
  branch_id: branchId,
  engine: "xsuite",
  fidelity: "fast",
  start: "2026-10-08T00:00:00+00:00",
  revolution_period: 1e-6,
  bpms: 3,
  first_turn: 0,
  end_turn: 100,
  state: "running",
  updated: null,
  buffer_turns: 0,
  slow_period: 0,
  captures: 0,
});

describe("BPM orbits", () => {
  it("fades older orbits, keeping the newest opaque", () => {
    const alphas = [0, 1, 2, 3, 4].map((k) => fade(k, 5));
    expect(alphas[4]).toBe(1);
    expect(alphas.every((a, k) => k === 0 || a > alphas[k - 1]!)).toBe(true);
    expect(alphas[0]).toBeGreaterThan(0.1);
    expect(fade(0, 1)).toBe(1);
  });

  it("prefers a run of the revision shown, then one of the same names", () => {
    const runs = [run("old", "other", "br_9"), run("now", "abc", "br_1")];
    const ring = { id: "br_1", name: "ring" };
    expect(runsFor(runs, "abc", "demo", ring)).toEqual([
      { run: runs[1], sameRevision: true },
      { run: runs[0], sameRevision: false },
    ]);
    expect(runsFor(runs, "abc", "demo", { id: "br_2", name: "line" })).toEqual([]);
  });

  it("keeps a ring's runs apart from a route's of the same name", () => {
    const runs = [run("ring", "old", "br_9", "bar"), run("route", "old", "route:bar", "bar")];
    expect(runsFor(runs, "abc", "demo", { id: "br_1", name: "bar" }).map((c) => c.run.id)).toEqual(["ring"]);
    expect(runsFor(runs, "abc", "demo", { id: "route:bar", name: "bar" }).map((c) => c.run.id)).toEqual(["route"]);
  });

  it("spans the readings in view and finds the BPM under the pointer", () => {
    const frames = { s: [1, 2, 3], x: [[0.5, null, -2], [1, 4, -1]], y: [[0, 0, 0], [0, 0, 0]] } as unknown as OrbitFrames;
    expect(readingExtent(frames, "x", 0, 10)).toEqual([-2, 4]);
    expect(readingExtent(frames, "x", 0, 1.5)).toEqual([0.5, 1]);
    expect(readingExtent(frames, "x", 5, 6)).toBeNull();
    expect(nearestBpm([1, 2, 3], 2.1, 0.2)).toBe(1);
    expect(nearestBpm([1, 2, 3], 2.5, 0.2)).toBeNull();
  });
});
