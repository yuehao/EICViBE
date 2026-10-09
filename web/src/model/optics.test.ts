import { describe, expect, it } from "vitest";

import type { BranchOptics } from "../api/client";
import { summaryText } from "../components/OpticsView";
import { extent, niceAxis, opticsModel, pointAt, valueAt } from "./optics";
import { createZoomHistory } from "./viewport";

// A quadrupole, then a 3 m drift with a waist inside it: beta 4 m and alpha 1 at its entrance,
// so gamma = 0.5 /m and beta(t) = 4 - 2 t + 0.5 t^2, smallest (2 m) 2 m in.
function optics(overrides: Partial<BranchOptics> = {}): BranchOptics {
  return {
    schema_version: 4,
    branch: "br_1",
    engine: "trackpad",
    version: "0.2.0",
    fidelity: "standard",
    method: "TPSA",
    periodic: false,
    available: true,
    origin: "initial conditions",
    occurrences: ["q", "d"],
    s: [0, 0.5, 3.5],
    values: {
      beta_x: [5, 4, 2.5],
      beta_y: [3, 3.5, 6],
      alpha_x: [1.5, 1, -0.5],
      alpha_y: [0, 0, -1],
      eta_x: [0.2, 0.3, 0.9],
      eta_y: [0, 0, 0],
      etap_x: [0.1, 0.2, 0.2],
      etap_y: [0, 0, 0],
      mu_x: [0, 0.01, 0.25],
      mu_y: [0, 0.02, 0.1],
      x: [0, 0, 0],
      y: [0, 0, 0],
    },
    columns: [],
    tune_x: null,
    tune_y: null,
    chroma_x: null,
    chroma_y: null,
    momentum_compaction: null,
    profile: null,
    ...overrides,
  };
}

const kinds = ["Quadrupole", "Drift"];

describe("optics model", () => {
  it("finds points by s and draws a drift's beta as its exact parabola", () => {
    const model = opticsModel(optics());
    expect(pointAt(model.points, 0.2)).toBe(0);
    expect(pointAt(model.points, 0.5)).toBe(1);
    expect(pointAt(model.points, 9)).toBe(2);
    expect(valueAt(model.points, "beta_x", 0.25, kinds)).toBeCloseTo(4.5); // straight inside the quadrupole
    expect(valueAt(model.points, "beta_x", 2.5, kinds)).toBeCloseTo(2); // the waist, 2 m into the drift
    expect(valueAt(model.points, "beta_x", 3.5, kinds)).toBeCloseTo(2.5); // and its exit, as the engine says
    expect(valueAt(model.points, "eta_x", 1.5, kinds)).toBeCloseTo(0.5); // dispersion: straight, with its slope
  });

  it("scales axes over what is in view, waists included", () => {
    const model = opticsModel(optics());
    expect(extent(model.points, ["beta_x"], 1, 3.5, kinds)).toEqual([2, 5]);
    expect(extent(model.points, [], 0, 3.5, kinds)).toBeNull();
    expect(niceAxis(2, 5, 4)).toEqual({ min: 0, max: 6, step: 2 });
    expect(niceAxis(2, 5, 5)).toEqual({ min: 0, max: 5, step: 1 });
    expect(niceAxis(-0.13, 0.9, 4)).toEqual({ min: -0.5, max: 1, step: 0.5 });
    expect(niceAxis(0, 0, 4).max).toBeGreaterThan(0);
  });

  it("draws through an engine's samples inside magnets, and still exact in drifts", () => {
    // The quadrupole sampled at two steps: beta 5 -> 4.4 -> 4, not the straight 5 -> 4.
    const values = optics().values;
    const pick = (key: string, inside: number) => [values[key]![0]!, inside, values[key]![1]!, values[key]![2]!];
    const model = opticsModel(
      optics({
        profile: {
          s: [0, 0.25, 0.5, 3.5],
          index: [0, 0, 1],
          values: {
            beta_x: pick("beta_x", 4.4),
            alpha_x: pick("alpha_x", 1.2),
            beta_y: pick("beta_y", 3.2),
            alpha_y: pick("alpha_y", 0),
            eta_x: pick("eta_x", 0.25),
            etap_x: pick("etap_x", 0.15),
            eta_y: pick("eta_y", 0),
            etap_y: pick("etap_y", 0),
            mu_x: pick("mu_x", 0.005),
            mu_y: pick("mu_y", 0.01),
          },
        },
      }),
    );
    expect(model.sampled).toBe(true);
    expect(model.exits.s.length).toBe(3);
    expect(valueAt(model.points, "beta_x", 0.25, kinds)).toBeCloseTo(4.4);
    expect(valueAt(model.points, "beta_x", 0.375, kinds)).toBeCloseTo(4.2); // straight between samples
    expect(valueAt(model.points, "beta_x", 2.5, kinds)).toBeCloseTo(2); // the drift's waist, as before
    expect(() =>
      opticsModel(optics({ profile: { s: [0, 1], index: [0, 0], values: {} } })),
    ).toThrow(/2 points for 2 pieces/);
  });

  it("refuses columns that do not match the occurrences", () => {
    expect(() => opticsModel(optics({ s: [0, 1] }))).toThrow(/2 points for 2 occurrences/);
  });

  it("summarizes a ring by its tunes and chromaticities, a line by where it starts", () => {
    const ring = opticsModel(
      optics({ periodic: true, origin: "periodic solution", tune_x: 2.4214099, tune_y: 1.4746565, chroma_x: -0.06127, chroma_y: -0.08477, momentum_compaction: 0.0793755 }),
    );
    expect(summaryText(ring)).toBe("Q 2.42141, 1.47466 · ξ -0.06127, -0.08477 · αc 0.07938");
    expect(summaryText(opticsModel(optics({ origin: "fork f in ltr" })))).toBe("from fork f in ltr · μ 0.25, 0.1 (2π)");
    expect(summaryText(opticsModel(optics({ available: false, occurrences: [], s: [], values: {} })))).toBe("");
  });
});

describe("zoom history", () => {
  it("undoes zooms newest first, skips repeats and forgets beyond its limit", () => {
    const history = createZoomHistory(3);
    expect(history.undo()).toBeNull();
    for (const s1 of [100, 50, 50, 20, 10]) history.push({ s0: 0, s1 });
    expect(history.size()).toBe(3); // 100 forgotten; the repeated 50 kept once
    expect(history.undo()).toEqual({ s0: 0, s1: 10 });
    expect(history.undo()).toEqual({ s0: 0, s1: 20 });
    history.clear();
    expect(history.undo()).toBeNull();
  });
});
