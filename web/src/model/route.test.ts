import { describe, expect, it } from "vitest";

import type { BranchData, BranchSummary } from "../api/client";
import { branchModel, occurrenceAt, type BranchModel } from "./branch";
import { isRoute, routeModel, routeS, targetIn, type RouteModel, type RouteSummary } from "./route";

/** A straight branch of ``names``, one metre each, along z from ``z0``. */
function straight(id: string, names: string[], z0: number, order: number): BranchModel {
  const n = names.length;
  const summary: BranchSummary = {
    id,
    name: id,
    kind: "linac",
    length: n,
    occurrences: n,
    placed_by: null,
    bounds: { x_min: 0, x_max: 0, z_min: z0, z_max: z0 + n },
    reference: null,
  };
  const data: BranchData = {
    schema_version: 1,
    branch: id,
    table: {
      id: names.map((name) => `${id}.${name}`),
      name: names,
      eic_name: names.map(() => null),
      kind: names.map((name) => (name.startsWith("q") ? "Quadrupole" : "Drift")),
      element: names,
      s: names.map((_, i) => i),
      length: names.map(() => 1),
      values: { Kn1L: names.map((name) => (name.startsWith("q") ? 0.1 : null)) },
      columns: [{ key: "Kn1L", label: "Kn1L", unit: "1/m", description: "" }],
      device: names.map(() => null),
      fork_to: names.map(() => null),
      component: names.map(() => null),
    },
    floor: {
      offsets: names.map((_, i) => 2 * i).concat([2 * n]),
      z: names.flatMap((_, i) => [z0 + i, z0 + i + 1]),
      x: names.flatMap(() => [0, 0]),
      y: names.flatMap(() => [0, 0]),
      theta: names.map(() => 0),
    },
  };
  return branchModel(data, summary, order);
}

const line = straight("line", ["d1", "q1", "d2"], 0, 0);
const ring = straight("ring", ["d1", "q1", "d2", "q2"], 3, 1);
const models = new Map([
  [line.id, line],
  [ring.id, ring],
]);
const summary: RouteSummary = {
  id: "route:fill",
  name: "fill",
  branches: ["line", "ring"],
  segments: [
    { branch: "line", start: "line.d1", end: "line.d2", s_start: 0, length: 3 },
    { branch: "ring", start: "ring.q1", end: "ring.q2", s_start: 3, length: 3 },
    { branch: "ring", start: "ring.d1", end: "ring.d1", s_start: 6, length: 1 },
  ],
  length: 7,
  problems: [],
};

describe("route model", () => {
  const route = routeModel(summary, models) as RouteModel;

  it("joins its segments in beam order with s along the route", () => {
    expect(isRoute(route) && !isRoute(line)).toBe(true);
    expect(route.ids).toEqual(["line.d1", "line.q1", "line.d2", "ring.q1", "ring.d2", "ring.q2", "ring.d1"]);
    expect(Array.from(route.s)).toEqual([0, 1, 2, 3, 4, 5, 6]);
    expect(route.joins).toEqual([3, 6]);
    expect(route.values.Kn1L).toEqual([null, 0.1, null, 0.1, null, 0.1, null]);
    expect(occurrenceAt(route, 4.5)).toBe(4);
  });

  it("keeps each occurrence's floor path and where it comes from", () => {
    expect(route.offsets[route.n]).toBe(2 * route.n);
    expect(Array.from(route.z.slice(6, 8))).toEqual([4, 5]); // the ring's q1, one metre into the ring
    expect(route.occurrenceOfPoint[7]).toBe(3);
    expect(route.origin[3]).toEqual({ branch: "ring", index: 1 });
  });

  it("maps targets between a route and its branches", () => {
    const routes = new Map([[route.id, route]]);
    expect(targetIn(route, { branch: "ring", index: 1 }, models, routes)).toEqual({ branch: "route:fill", index: 3 });
    expect(targetIn(ring, { branch: "route:fill", index: 3 }, models, routes)).toEqual({ branch: "ring", index: 1 });
    expect(targetIn(line, { branch: "ring", index: 1 }, models, routes)).toBeNull();
  });

  it("places a branch's s along the route where it passes", () => {
    expect(routeS(route, "ring", 2.5)).toBe(4.5);
    expect(routeS(route, "ring", 0.5)).toBe(6.5); // once round: the ring's start comes last
    expect(routeS(route, "line", 3)).toBe(3);
    expect(routeS(route, "other", 1)).toBeNull();
  });

  it("waits for every branch it passes", () => {
    expect(routeModel(summary, new Map([[line.id, line]]))).toBeNull();
  });
});
