// A notebook view's lattice (ADR 0016): the summary its kernel gave it, and the
// branches it draws, asked of the kernel as the page asks the server: every one
// for the floor plan, those the branch or route shown passes for a view along s.

import { useEffect, useMemo, useState } from "react";

import type { LatticeSummary } from "../api/client";
import type { ViewerData } from "../api/data";
import { branchModel, branchOrder, type BranchModel } from "../model/branch";
import { routeModel, type RouteModel } from "../model/route";
import type { WidgetModel } from "./kernel";
import { useTrait } from "./scheme";

export interface Lattice {
  summary: LatticeSummary;
  /** The branch shown, by id, or route:<name>; empty: none (the floor plan's). */
  current: string;
  models: Map<string, BranchModel>;
  routes: Map<string, RouteModel>;
  currentModel: BranchModel | null;
  error: string | null;
}

export function useLattice(model: WidgetModel, data: ViewerData, every: boolean): Lattice {
  const summary = useTrait<LatticeSummary>(model, "summary");
  const current = useTrait<string>(model, "branch");
  const [models, setModels] = useState<Map<string, BranchModel>>(() => new Map());
  const [error, setError] = useState<string | null>(null);
  const order = useMemo(() => branchOrder(summary), [summary]);
  const needed = useMemo(() => {
    if (every) return order;
    const route = summary.routes?.find((r) => r.id === current);
    return route ? [...new Set(route.segments.map((s) => s.branch))] : current ? [current] : [];
  }, [every, order, summary, current]);

  useEffect(() => {
    let stop = false;
    for (const id of needed) {
      const branch = summary.branches.find((b) => b.id === id);
      if (!branch) continue;
      data.branch(id).then(
        (d) => !stop && setModels((m) => (m.has(id) ? m : new Map(m).set(id, branchModel(d, branch, order.indexOf(id))))),
        (e: Error) => !stop && setError(`Cannot load branch ${branch.name}: ${e.message}`),
      );
    }
    return () => {
      stop = true;
    };
  }, [data, needed, summary, order]);

  const routes = useMemo(() => {
    const out = new Map<string, RouteModel>();
    for (const route of summary.routes ?? []) {
      const found = route.segments.every((s) => models.has(s.branch)) ? routeModel(route, models) : null;
      if (found) out.set(route.id, found);
    }
    return out;
  }, [summary, models]);
  const currentModel = (current && (models.get(current) ?? routes.get(current))) || null;
  return { summary, current, models, routes, currentModel, error };
}
