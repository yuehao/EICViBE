// The floor plan of every branch in a notebook (ADR 0016): the page's floor plan,
// its branches asked of the kernel. Picking an element of another branch makes
// it the branch shown, which views linked to this one follow.

import { useMemo, useRef } from "react";

import type { ViewerData } from "../api/data";
import { FloorPlan, type FloorPlanHandle } from "../components/FloorPlan";
import { isRoute } from "../model/route";
import { createStore, sameTarget, type Target } from "../model/store";
import type { WidgetModel } from "./kernel";
import { useLattice } from "./lattice";
import { useNotebookScheme, useTrait } from "./scheme";
import { useSelection } from "./sync";

export function FloorWidget({ model, data }: { model: WidgetModel; data: ViewerData }) {
  const scheme = useNotebookScheme();
  const height = useTrait<number>(model, "height");
  const lattice = useLattice(model, data, true);
  const hover = useMemo(() => createStore<Target | null>(null, sameTarget), []);
  const floor = useRef<FloorPlanHandle | null>(null);
  const { selection, select } = useSelection(model, lattice, true, (t) => floor.current?.show(t));
  const { models, currentModel } = lattice;
  // A route shown is drawn over the branches it passes.
  const branches = useMemo(
    () => (isRoute(currentModel) ? new Map(models).set(currentModel.id, currentModel) : models),
    [models, currentModel],
  );

  return (
    <div className="eicvibe-widget nb-view" data-scheme={scheme} style={{ height }} data-testid="eicvibe-floor">
      {lattice.error ? <div className="banner error">{lattice.error}</div> : null}
      <div className="nb-floor">
        <FloorPlan
          summary={lattice.summary}
          branches={branches}
          current={lattice.current || null}
          selection={selection}
          hover={hover}
          scheme={scheme}
          onSelect={select}
          onViewChange={() => undefined}
          handle={floor}
        />
      </div>
    </div>
  );
}
