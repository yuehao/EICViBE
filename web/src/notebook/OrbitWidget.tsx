// BPM orbits of virtual runs in a notebook (ADR 0016): the page's orbit view
// under its lattice strip, the runs' orbits read by the kernel from their
// folders, live while a run goes on and the kernel is free.

import { useMemo, useRef } from "react";

import type { RunEntry } from "../api/client";
import type { ViewerData } from "../api/data";
import { BpmView } from "../components/BpmView";
import { LinearView, type LinearViewHandle } from "../components/LinearView";
import { createStore, sameTarget, type Target } from "../model/store";
import type { WidgetModel } from "./kernel";
import { useLattice } from "./lattice";
import { useNotebookScheme, useTrait } from "./scheme";
import { useFollowFork, useRange, useSelection } from "./sync";

export function OrbitWidget({ model, data }: { model: WidgetModel; data: ViewerData }) {
  const scheme = useNotebookScheme();
  const height = useTrait<number>(model, "height");
  const strip = useTrait<boolean>(model, "strip");
  const runs = useTrait<RunEntry[]>(model, "runs");
  const lattice = useLattice(model, data, false);
  const hover = useMemo(() => createStore<Target | null>(null, sameTarget), []);
  const { range, band, history } = useRange(model, lattice.currentModel);
  const linear = useRef<LinearViewHandle | null>(null);
  const { selection, select } = useSelection(model, lattice, false, (t) => linear.current?.show(t.index));
  const onFollowFork = useFollowFork(model);

  return (
    <div className="eicvibe-widget nb-view" data-scheme={scheme} style={{ height }} data-testid="eicvibe-orbit">
      {lattice.error ? <div className="banner error">{lattice.error}</div> : null}
      <div className="nb-panel">
        <BpmView
          data={data}
          digest={lattice.summary.digest}
          latticeName={lattice.summary.name}
          branch={lattice.currentModel}
          runs={runs}
          selection={selection}
          hover={hover}
          range={range}
          band={band}
          history={history}
          scheme={scheme}
          onSelect={select}
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
