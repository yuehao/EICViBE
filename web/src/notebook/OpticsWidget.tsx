// A branch's or route's optics in a notebook (ADR 0016): the page's optics view
// under its lattice strip, the optics computed by the kernel's engine workers
// with the engine, fidelity and mode chosen here.

import { useCallback, useMemo, useRef } from "react";

import type { ViewerData } from "../api/data";
import { LinearView, type LinearViewHandle } from "../components/LinearView";
import { OpticsView } from "../components/OpticsView";
import { createStore, sameTarget, type Target } from "../model/store";
import type { WidgetModel } from "./kernel";
import { useLattice } from "./lattice";
import { useNotebookScheme, useTrait } from "./scheme";
import { useFollowFork, useRange, useSelection } from "./sync";

export function OpticsWidget({ model, data }: { model: WidgetModel; data: ViewerData }) {
  const scheme = useNotebookScheme();
  const height = useTrait<number>(model, "height");
  const strip = useTrait<boolean>(model, "strip");
  const lattice = useLattice(model, data, false);
  const hover = useMemo(() => createStore<Target | null>(null, sameTarget), []);
  const { range, band, history } = useRange(model, lattice.currentModel);
  const linear = useRef<LinearViewHandle | null>(null);
  const { selection, select } = useSelection(model, lattice, false, (t) => linear.current?.show(t.index));
  const onFollowFork = useFollowFork(model);
  const onChoice = useCallback(
    (c: { engine: string; fidelity: string; mode: string }) => {
      if (c.engine === model.get("engine") && c.fidelity === model.get("fidelity") && c.mode === model.get("mode")) return;
      model.set("engine", c.engine);
      model.set("fidelity", c.fidelity);
      model.set("mode", c.mode);
      model.save_changes();
    },
    [model],
  );

  return (
    <div className="eicvibe-widget nb-view" data-scheme={scheme} style={{ height }} data-testid="eicvibe-optics">
      {lattice.error ? <div className="banner error">{lattice.error}</div> : null}
      <div className="nb-panel">
        <OpticsView
          data={data}
          branch={lattice.currentModel}
          selection={selection}
          hover={hover}
          range={range}
          band={band}
          history={history}
          scheme={scheme}
          initialEngine={(model.get("engine") as string) || undefined}
          initialFidelity={(model.get("fidelity") as string) || undefined}
          modes={lattice.summary.modes ?? []}
          initialMode={(model.get("mode") as string) || undefined}
          onSelect={select}
          onChoice={onChoice}
          strip={
            strip ? (
              <LinearView
                summary={lattice.summary}
                branch={lattice.currentModel}
                selection={selection}
                hover={hover}
                range={range}
                band={band}
                history={history}
                scheme={scheme}
                onSelect={select}
                onFollowFork={onFollowFork}
                handle={linear}
              />
            ) : null
          }
        />
      </div>
    </div>
  );
}
