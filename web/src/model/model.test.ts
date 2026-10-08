import { describe, expect, it } from "vitest";

import type { BranchData, BranchSummary } from "../api/client";
import { branchModel, branchOrder, occurrenceAt, strengthText } from "./branch";
import { formatNumber, formatTick } from "./format";
import { SegmentIndex, segmentDistance } from "./spatial";
import { createStore, sameTarget, type Target } from "./store";
import { formatHash, parseHash } from "./urlState";
import { clampRange, fit, niceStep, pan, toScreen, toWorld, zoomAt, zoomRange } from "./viewport";

const summary: BranchSummary = {
  id: "br_1",
  name: "line",
  kind: "linac",
  length: 4,
  occurrences: 4,
  placed_by: null,
  bounds: { x_min: 0, x_max: 0, z_min: 0, z_max: 4 },
  reference: null,
};

const data: BranchData = {
  schema_version: 1,
  branch: "br_1",
  table: {
    id: ["a", "b", "c", "d"],
    name: ["qf", "d1", "qd", "bpm"],
    eic_name: ["line-q1", "line-dt3", "line-q4", null],
    kind: ["Quadrupole", "Drift", "Quadrupole", "Instrument"],
    element: ["QF", "D", "QD", "BPM"],
    s: [0, 1, 3, 4],
    length: [1, 2, 1, 0],
    values: { Kn1: [0.5, null, -0.5, null], Kn1L: [0.5, null, -0.5, null] },
    columns: [
      { key: "Kn1", label: "Kn1", unit: "1/m^2", description: "" },
      { key: "Kn1L", label: "Kn1L", unit: "1/m", description: "" },
    ],
    device: ["line-q1-ps", null, "line-q2-ps", "line-bpm1"],
    fork_to: [null, null, null, null],
    component: [null, null, null, null],
  },
  floor: {
    offsets: [0, 2, 4, 6, 8],
    z: [0, 1, 1, 3, 3, 4, 4, 4],
    x: [0, 0, 0, 0, 0, 0, 0, 0],
    y: [0, 0, 0, 0, 0, 0, 0, 0],
    theta: [0, 0, 0, 0],
  },
};

describe("branch model", () => {
  const model = branchModel(data, summary, 0);

  it("shows EIC names, and the lattice name where there is none", () => {
    expect(model.labels).toEqual(["line-q1", "line-dt3", "line-q4", "bpm"]);
    expect(model.names[0]).toBe("qf");
  });

  it("puts focusing quadrupoles above the axis and defocusing ones below", () => {
    expect(Array.from(model.polarity)).toEqual([1, 0, -1, 0]);
  });

  it("finds occurrences by s and by id", () => {
    expect(occurrenceAt(model, 0)).toBe(0);
    expect(occurrenceAt(model, 2.5)).toBe(1);
    expect(occurrenceAt(model, 3)).toBe(2);
    expect(occurrenceAt(model, 99)).toBe(3);
    expect(model.index.get("c")).toBe(2);
    expect(Array.from(model.occurrenceOfPoint)).toEqual([0, 0, 1, 1, 2, 2, 3, 3]);
  });

  it("describes strength by the per-length value when there is one", () => {
    expect(strengthText(model, 0, (v) => formatNumber(v))).toBe("Kn1 = 0.5 1/m^2");
    expect(strengthText(model, 1, (v) => formatNumber(v))).toBe("");
  });

  it("hit-tests the nearest segment, and points of zero-length elements", () => {
    const hit = model.segments.nearest(3.5, 0.1, 0.3);
    expect(hit && model.occurrenceOfPoint[hit.point]).toBe(2);
    const bpm = model.segments.nearest(4.05, 0, 0.3, (point, d) => (model.kinds[model.occurrenceOfPoint[point]!] === "Instrument" ? d - 1 : d));
    expect(bpm && model.occurrenceOfPoint[bpm.point]).toBe(3);
    expect(model.segments.nearest(2, 5, 0.3)).toBeNull();
  });
});

describe("spatial index", () => {
  it("agrees with a linear scan on a random path", () => {
    const n = 2000;
    const u = new Float64Array(n), v = new Float64Array(n);
    let a = 0;
    for (let j = 0; j < n; j++) {
      a += 0.01;
      u[j] = 100 * Math.cos(a) + (j % 7) * 0.01;
      v[j] = 100 * Math.sin(a);
    }
    const offsets = Int32Array.from({ length: n / 2 + 1 }, (_, i) => 2 * i);
    const index = SegmentIndex.build(u, v, offsets);
    for (let trial = 0; trial < 200; trial++) {
      const pu = (trial * 37) % 220 - 110, pv = (trial * 53) % 220 - 110, r = 5;
      let best = Infinity;
      for (let j = 0; j < n; j += 2) best = Math.min(best, segmentDistance(pu, pv, u[j]!, v[j]!, u[j + 1]!, v[j + 1]!));
      const found = index.nearest(pu, pv, r);
      expect(found ? found.distance : Infinity).toBeCloseTo(best <= r ? best : Infinity);
    }
  });
});

describe("viewport", () => {
  it("keeps the point under the cursor fixed when zooming", () => {
    const view = { cu: 10, cv: -5, k: 2 };
    const zoomed = zoomAt(view, 3, 120, 40, 800, 600);
    const before = toWorld(view, 800, 600, 120, 40);
    const after = toWorld(zoomed, 800, 600, 120, 40);
    expect(after[0]).toBeCloseTo(before[0]);
    expect(after[1]).toBeCloseTo(before[1]);
    expect(zoomed.k).toBeCloseTo(6);
  });

  it("maps z to the right and x up, and pans with the pointer", () => {
    const view = { cu: 0, cv: 0, k: 1 };
    expect(toScreen(view, 100, 100, 10, 10)).toEqual([60, 40]);
    const moved = pan(view, 10, 0);
    expect(toScreen(moved, 100, 100, 0, 0)).toEqual([60, 50]);
  });

  it("fits a box", () => {
    const view = fit({ uMin: 0, uMax: 100, vMin: -10, vMax: 10 }, 548, 300, 24);
    expect(view.k).toBeCloseTo(5);
    expect([view.cu, view.cv]).toEqual([50, 0]);
  });

  it("zooms and clamps s ranges", () => {
    const limits = { s0: 0, s1: 100 };
    expect(zoomRange({ s0: 0, s1: 100 }, 2, 50, limits)).toEqual({ s0: 25, s1: 75 });
    expect(zoomRange({ s0: 0, s1: 100 }, 0.1, 50, limits)).toEqual({ s0: 0, s1: 100 });
    expect(clampRange({ s0: 90, s1: 120 }, limits)).toEqual({ s0: 70.6, s1: 100.6 });
  });

  it("picks round steps", () => {
    expect(niceStep(0.37)).toBe(0.5);
    expect(niceStep(73)).toBe(50);
    expect(niceStep(160)).toBe(200);
    expect(formatTick(1.5, 0.5)).toBe("1.5");
    expect(formatTick(200, 50)).toBe("200");
  });
});

describe("url state", () => {
  it("round-trips", () => {
    const state = {
      source: "ein",
      branch: "rcs",
      sel: "occ_1",
      floor: { cu: 12.5, cv: -3.25, k: 0.75 },
      range: { s0: 10, s1: 20.5 },
      q: "qf ps",
    };
    expect(parseHash(formatHash(state))).toEqual(state);
  });

  it("ignores malformed values", () => {
    expect(parseHash("#floor=1,2&s=5,1&branch=ltr")).toEqual({ branch: "ltr" });
  });
});

describe("store", () => {
  it("notifies only on change", () => {
    const store = createStore<Target | null>(null, sameTarget);
    let calls = 0;
    store.subscribe(() => calls++);
    store.set({ branch: "a", index: 1 });
    store.set({ branch: "a", index: 1 });
    store.set(null);
    expect(calls).toBe(2);
  });
});

describe("branch order", () => {
  it("follows forks from the root", () => {
    const branch = (id: string) => ({ ...summary, id, name: id });
    const fork = (from_branch: string, to_branch: string, s_from: number) => ({
      id: `${from_branch}-${to_branch}`,
      name: "",
      from_branch,
      s_from,
      to_branch,
      to_occurrence: null,
      to_side: "entrance",
      s_to: 0,
    });
    const order = branchOrder({
      root_branch: "ltr",
      branches: ["bar", "etl", "itl", "ltr", "rcs", "rte"].map(branch),
      forks: [fork("ltr", "rcs", 60), fork("ltr", "itl", 7), fork("itl", "bar", 11), fork("bar", "etl", 18), fork("rcs", "rte", 800)],
    });
    expect(order).toEqual(["ltr", "itl", "bar", "etl", "rcs", "rte"]);
  });
});

describe("format", () => {
  it("formats numbers compactly", () => {
    expect(formatNumber(0.292928013429)).toBe("0.292928");
    expect(formatNumber(1.69758e6)).toBe("1.69758e+6");
    expect(formatNumber(-2.5e-5)).toBe("-2.5e-5");
    expect(formatNumber(null)).toBe("");
    expect(formatNumber(0)).toBe("0");
  });
});
