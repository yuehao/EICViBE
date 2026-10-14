// A route (the beam's path through several branches, routes.yaml) prepared
// as one more branch: its segments' occurrences joined in beam order, with s
// along the route. Every view along s draws it as it draws a branch; the floor
// plan draws its path over the branches it passes.

import type { LatticeSummary } from "../api/client";
import type { BranchModel } from "./branch";
import { SegmentIndex } from "./spatial";
import type { Target } from "./store";

export type RouteSummary = NonNullable<LatticeSummary["routes"]>[number];

export interface RouteModel extends BranchModel {
  route: RouteSummary;
  /** Each occurrence of the route as an occurrence of its own branch. */
  origin: Target[];
  /** s along the route where each segment after the first starts. */
  joins: number[];
  /** Each segment as a stretch of its branch: [s0, s1] in the branch's s, from ``at`` along the route. */
  spans: { branch: string; s0: number; s1: number; at: number }[];
}

export const isRoute = (model: BranchModel | null | undefined): model is RouteModel => !!model && "route" in model;

/** The route as a model, once every branch it passes has loaded; else null. */
export function routeModel(route: RouteSummary, models: ReadonlyMap<string, BranchModel>): RouteModel | null {
  const parts = route.segments.map((segment) => {
    const model = models.get(segment.branch);
    const first = model?.index.get(segment.start), last = model?.index.get(segment.end);
    return model && first !== undefined && last !== undefined && last >= first ? { segment, model, first, last } : null;
  });
  if (parts.some((p) => p === null) || !parts.length) return null;
  const head = parts[0]!.model;
  const keys = new Map<string, BranchModel["columns"][number]>();
  parts.forEach((p) => p!.model.columns.forEach((c) => keys.has(c.key) || keys.set(c.key, c)));
  const n = parts.reduce((total, p) => total + p!.last - p!.first + 1, 0);
  const ids: string[] = [], names: string[] = [], eicNames: (string | null)[] = [], labels: string[] = [];
  const kinds: string[] = [], elements: string[] = [], devices: (string | null)[] = [], forkTo: (string | null)[] = [];
  const components: (string | null)[] = [], origin: Target[] = [], joins: number[] = [];
  const spans: RouteModel["spans"] = [];
  const values: Record<string, (number | null)[]> = Object.fromEntries([...keys.keys()].map((k) => [k, []]));
  const s = new Float64Array(n), lengths = new Float64Array(n), polarity = new Int8Array(n), theta = new Float64Array(n);
  const offsets = new Int32Array(n + 1);
  const points = parts.reduce((total, p) => total + p!.model.offsets[p!.last + 1]! - p!.model.offsets[p!.first]!, 0);
  const z = new Float64Array(points), x = new Float64Array(points), occurrenceOfPoint = new Int32Array(points);
  let k = 0, point = 0;
  for (const part of parts) {
    const { segment, model, first, last } = part!;
    if (k) joins.push(segment.s_start);
    spans.push({ branch: model.id, s0: model.s[first]!, s1: model.s[last]! + model.lengths[last]!, at: segment.s_start });
    for (let i = first; i <= last; i++, k++) {
      ids.push(model.ids[i]!);
      names.push(model.names[i]!);
      eicNames.push(model.eicNames[i]!);
      labels.push(model.labels[i]!);
      kinds.push(model.kinds[i]!);
      elements.push(model.elements[i]!);
      devices.push(model.devices[i]!);
      forkTo.push(model.forkTo[i]!);
      components.push(model.components[i]!);
      origin.push({ branch: model.id, index: i });
      for (const key of keys.keys()) values[key]!.push(model.values[key]?.[i] ?? null);
      s[k] = segment.s_start + model.s[i]! - model.s[first]!;
      lengths[k] = model.lengths[i]!;
      polarity[k] = model.polarity[i]!;
      theta[k] = model.theta[i]!;
      offsets[k] = point;
      for (let j = model.offsets[i]!; j < model.offsets[i + 1]!; j++, point++) {
        z[point] = model.z[j]!;
        x[point] = model.x[j]!;
        occurrenceOfPoint[point] = k;
      }
    }
  }
  offsets[n] = point;
  // An element a route passes more than once (turns round a ring) is found at its first pass.
  const index = new Map<string, number>();
  ids.forEach((id, i) => index.has(id) || index.set(id, i));
  return {
    id: route.id,
    name: route.name,
    order: head.order,
    periodic: false,
    length: route.length,
    n,
    ids,
    names,
    eicNames,
    labels,
    kinds,
    elements,
    s,
    lengths,
    columns: [...keys.values()],
    values,
    devices,
    forkTo,
    components,
    polarity,
    offsets,
    z,
    x,
    theta,
    occurrenceOfPoint,
    index,
    segments: SegmentIndex.build(z, x, offsets),
    route,
    origin,
    joins,
    spans,
  };
}

/** Where s of a branch lies along the route (its first pass), or null where the route does not pass. */
export function routeS(route: RouteModel, branch: string, s: number): number | null {
  return routePasses(route, branch, s)[0] ?? null;
}

/** Every s along the route where it passes s of a branch: more than one where it goes round a ring. */
export function routePasses(route: RouteModel, branch: string, s: number): number[] {
  const out: number[] = [];
  for (const span of route.spans)
    if (span.branch === branch && s >= span.s0 - 1e-9 && s <= span.s1 + 1e-9) {
      const at = span.at + s - span.s0;
      if (!out.some((v) => Math.abs(v - at) < 1e-9)) out.push(at); // a span's end and the next's start
    }
  return out;
}

/** The same occurrence as a target of ``view``: a branch's occurrence on a route, or a route's in its branch. */
export function targetIn(
  view: BranchModel,
  target: Target,
  models: ReadonlyMap<string, BranchModel>,
  routes: ReadonlyMap<string, RouteModel>,
): Target | null {
  if (target.branch === view.id) return target;
  const from = models.get(target.branch) ?? routes.get(target.branch);
  const id = from?.ids[target.index];
  const index = id === undefined ? undefined : view.index.get(id);
  return index === undefined ? null : { branch: view.id, index };
}
