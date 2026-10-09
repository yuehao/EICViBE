// The lattice viewer: floor plan, linear view, element table and inspector of
// one lattice revision, linked by a shared selection and hover.
//
// All of this state belongs to this browser tab. The server only answers
// reads of immutable revisions, so other users, notebooks and tabs never see
// or disturb it; the URL hash carries it, so a link reproduces the view.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, ApiError, SCHEMA_VERSION, type LatticeEntry, type LatticeSummary } from "./api/client";
import { ElementTable } from "./components/ElementTable";
import { FloorPlan, type FloorPlanHandle } from "./components/FloorPlan";
import { Inspector } from "./components/Inspector";
import { LinearView, type LinearViewHandle } from "./components/LinearView";
import { OpticsView } from "./components/OpticsView";
import { Splitter } from "./components/Splitter";
import { TopBar } from "./components/TopBar";
import { useColorScheme } from "./components/useCanvas";
import { branchModel, branchOrder, type BranchModel } from "./model/branch";
import { createStore, sameTarget, type Target } from "./model/store";
import { parseHash, updateHash } from "./model/urlState";
import { clampRange, createZoomHistory, type SRange, type View2D } from "./model/viewport";

type Origin = "floor" | "linear" | "table" | "search" | "link";

const POLL_MS = 30_000;

const sameRange = (a: SRange | null, b: SRange | null) => a === b || (!!a && !!b && a.s0 === b.s0 && a.s1 === b.s1);

export function App() {
  const scheme = useColorScheme();
  const initial = useRef(parseHash(window.location.hash)).current;
  const [entries, setEntries] = useState<LatticeEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [source, setSource] = useState<string | null>(initial.source ?? null);
  const [digest, setDigest] = useState<string | null>(null);
  const [summary, setSummary] = useState<LatticeSummary | null>(null);
  const [models, setModels] = useState<Map<string, BranchModel>>(() => new Map());
  const [current, setCurrent] = useState<string | null>(null);
  const [selection, setSelection] = useState<Target | null>(null);
  const [query, setQuery] = useState(initial.q ?? "");

  const hover = useMemo(() => createStore<Target | null>(null, sameTarget), []);
  const range = useMemo(() => createStore<SRange | null>(null, sameRange), []);
  const band = useMemo(() => createStore<SRange | null>(null, sameRange), []);
  const history = useMemo(() => createZoomHistory(), []);
  const rangeBranch = useRef<string | null>(null);
  const ranges = useRef(new Map<string, SRange>());
  const pending = useRef({ selection: initial.sel ?? null, branch: initial.branch ?? null, range: initial.range ?? null });
  const floor = useRef<FloorPlanHandle | null>(null);
  const linear = useRef<LinearViewHandle | null>(null);
  const live = useRef({ models, current });
  live.current = { models, current };

  // The lattices on the server; polled so that a new revision can be offered (never forced).
  useEffect(() => {
    let stop = false;
    const load = () =>
      api.lattices().then(
        (list) => {
          if (stop) return;
          setEntries(list);
          setError(null);
          setSource((s) => (s && list.some((e) => e.source === s) ? s : (list[0]?.source ?? null)));
        },
        (e: Error) => !stop && setError(`Cannot reach the server: ${e.message}`),
      );
    load();
    const timer = window.setInterval(load, POLL_MS);
    return () => {
      stop = true;
      window.clearInterval(timer);
    };
  }, []);

  // Which revision to show: the one in the URL if the server still has it, else the current one.
  useEffect(() => {
    if (!entries || !source) return;
    setDigest((d) => {
      if (d) return d;
      const entry = entries.find((e) => e.source === source);
      if (initial.rev && initial.source === source && initial.rev !== entry?.digest) return initial.rev;
      return entry?.digest ?? null;
    });
  }, [entries, source, initial]);

  // Load the summary, then every branch in parallel; each view fills in as its data arrives.
  useEffect(() => {
    if (!digest) return;
    let stop = false;
    setSummary(null);
    setModels(new Map());
    setSelection(null);
    api.summary(digest).then(
      (s) => {
        if (stop) return;
        if (s.schema_version !== SCHEMA_VERSION) {
          setError(
            `This page reads API version ${SCHEMA_VERSION}, but the server sends ${s.schema_version}. ` +
              "Reload the page; if that does not help, restart eicvibe-web.",
          );
          return;
        }
        setSummary(s);
        const order = branchOrder(s);
        const wanted = s.branches.find((b) => b.name === pending.current.branch)?.id;
        setCurrent(wanted ?? s.root_branch ?? order[0] ?? null);
        pending.current.branch = null;
        order.forEach((id) => {
          const branch = s.branches.find((b) => b.id === id)!;
          api.branch(digest, id).then(
            (data) => {
              if (stop) return;
              try {
                const model = branchModel(data, branch, order.indexOf(id));
                setModels((m) => new Map(m).set(id, model));
              } catch (e) {
                setError(`Cannot read branch ${branch.name} (${(e as Error).message}). Reload the page.`);
              }
            },
            (e: Error) => !stop && setError(`Cannot load branch ${branch.name}: ${e.message}`),
          );
        });
      },
      (e: Error) => {
        if (stop) return;
        const entry = entries?.find((x) => x.source === source);
        if (e instanceof ApiError && e.status === 404 && entry && entry.digest !== digest) {
          setNotice(`Revision ${digest.slice(0, 10)} is no longer on the server; showing the current one.`);
          setDigest(entry.digest);
        } else setError(`Cannot load the lattice: ${e.message}`);
      },
    );
    return () => {
      stop = true;
    };
    // entries and source are read only for the fallback; a poll must not reload the lattice
  }, [digest]);

  const order = useMemo(() => (summary ? branchOrder(summary) : []), [summary]);
  const currentModel = current ? (models.get(current) ?? null) : null;

  // Give the linear view the range of the current branch once its data is in.
  useEffect(() => {
    if (!currentModel || rangeBranch.current === currentModel.id) return;
    const whole = { s0: 0, s1: Math.max(currentModel.length, 1e-3) };
    const fromUrl = pending.current.range;
    pending.current.range = null;
    rangeBranch.current = currentModel.id;
    history.clear(); // zooms are undone within a branch
    range.set(fromUrl ? clampRange(fromUrl, whole) : (ranges.current.get(currentModel.id) ?? whole));
  }, [currentModel, range, history]);

  useEffect(
    () =>
      range.subscribe(() => {
        const value = range.get();
        if (!value || !rangeBranch.current) return;
        ranges.current.set(rangeBranch.current, value);
        updateHash({ range: value });
      }),
    [range],
  );

  // Resolve a selection from the URL once its branch has loaded.
  useEffect(() => {
    const wanted = pending.current.selection;
    if (!wanted) return;
    for (const model of models.values()) {
      const index = model.index.get(wanted);
      if (index === undefined) continue;
      pending.current.selection = null;
      setSelection({ branch: model.id, index });
      if (model.id !== live.current.current) setCurrent(model.id);
      return;
    }
  }, [models]);

  useEffect(() => {
    if (source) updateHash({ source, rev: digest && entries?.find((e) => e.source === source)?.digest !== digest ? digest : undefined });
  }, [source, digest, entries]);
  useEffect(() => {
    if (summary && current) updateHash({ branch: summary.branches.find((b) => b.id === current)?.name });
  }, [summary, current]);
  useEffect(() => {
    if (pending.current.selection) return;
    updateHash({ sel: selection ? models.get(selection.branch)?.ids[selection.index] : undefined });
  }, [selection, models]);
  useEffect(() => updateHash({ q: query || undefined }), [query]);

  const switchBranch = useCallback(
    (id: string, around?: number) => {
      const model = live.current.models.get(id);
      if (model) {
        const whole = { s0: 0, s1: Math.max(model.length, 1e-3) };
        let next = ranges.current.get(id) ?? whole;
        if (around !== undefined && (model.s[around]! < next.s0 || model.s[around]! > next.s1)) {
          const span = next.s1 - next.s0;
          next = clampRange({ s0: model.s[around]! - span / 2, s1: model.s[around]! + span / 2 }, whole);
        }
        if (id !== rangeBranch.current) history.clear();
        rangeBranch.current = id;
        range.set(next);
      }
      setCurrent(id);
    },
    [range, history],
  );

  const select = useCallback(
    (target: Target | null, origin: Origin) => {
      setSelection(target);
      if (!target) return;
      if (target.branch !== live.current.current) switchBranch(target.branch, target.index);
      const zoom = origin === "search";
      requestAnimationFrame(() => {
        if (origin !== "floor") floor.current?.show(target, zoom);
        if (origin !== "linear") linear.current?.show(target.index, zoom);
      });
    },
    [switchBranch],
  );

  const goTo = useCallback(
    (occurrence: string) => {
      for (const model of live.current.models.values()) {
        const index = model.index.get(occurrence);
        if (index !== undefined) return select({ branch: model.id, index }, "link");
      }
    },
    [select],
  );

  const onFloorSelect = useCallback((t: Target | null) => select(t, "floor"), [select]);
  const onLinearSelect = useCallback((t: Target | null) => select(t, "linear"), [select]);
  const onTableSelect = useCallback((t: Target) => select(t, "table"), [select]);
  const onFind = useCallback((t: Target) => select(t, "search"), [select]);
  const onFloorView = useCallback((view: View2D) => updateHash({ floor: view }), []);
  const onFollowFork = useCallback(
    (fork: { branch: string; occurrence: string | null; s: number }) => {
      const model = live.current.models.get(fork.branch);
      const index = fork.occurrence ? model?.index.get(fork.occurrence) : 0;
      if (model && index !== undefined) select({ branch: model.id, index }, "link");
      else switchBranch(fork.branch);
    },
    [select, switchBranch],
  );

  const branchName = useCallback((id: string) => summary?.branches.find((b) => b.id === id)?.name ?? id, [summary]);
  const describe = useCallback(
    (id: string) => {
      for (const model of models.values()) {
        const index = model.index.get(id);
        if (index !== undefined) return { name: model.names[index]!, kind: model.kinds[index]!, branch: model.name };
      }
      return null;
    },
    [models],
  );

  const selectedId = selection ? (models.get(selection.branch)?.ids[selection.index] ?? null) : null;

  return (
    <div className="app">
      <TopBar
        entries={entries}
        source={source}
        digest={digest}
        summary={summary}
        branchOrder={order}
        models={models}
        current={current}
        onSource={(s) => {
          setSource(s);
          setDigest(entries?.find((e) => e.source === s)?.digest ?? null);
          rangeBranch.current = null;
          ranges.current.clear();
        }}
        onRevision={(d) => {
          pending.current.selection = selectedId;
          pending.current.branch = branchName(current ?? "");
          rangeBranch.current = null;
          setDigest(d);
          setNotice(null);
        }}
        onBranch={(id) => switchBranch(id)}
        onFind={onFind}
      />
      {error ? <div className="banner error">{error}</div> : null}
      {notice ? (
        <div className="banner" onClick={() => setNotice(null)}>
          {notice}
        </div>
      ) : null}
      {summary && digest ? (
        <main className="workspace">
          <section className="views">
            <div className="pane floor-pane">
              <FloorPlan
                summary={summary}
                branches={models}
                current={current}
                selection={selection}
                hover={hover}
                initialView={initial.floor}
                scheme={scheme}
                onSelect={onFloorSelect}
                onViewChange={onFloorView}
                handle={floor}
              />
            </div>
            <Splitter variable="--optics-height" orientation="horizontal" sign={-1} min={200} max={900} initial={340} />
            <div className="pane optics-pane">
              <OpticsView
                digest={digest}
                branch={currentModel}
                selection={selection}
                hover={hover}
                range={range}
                band={band}
                history={history}
                scheme={scheme}
                initialEngine={initial.engine}
                initialFidelity={initial.fidelity}
                onSelect={onLinearSelect}
                strip={
                  <LinearView
                    summary={summary}
                    branch={currentModel}
                    selection={selection}
                    hover={hover}
                    range={range}
                    band={band}
                    history={history}
                    scheme={scheme}
                    onSelect={onLinearSelect}
                    onFollowFork={onFollowFork}
                    handle={linear}
                  />
                }
              />
            </div>
          </section>
          <Splitter variable="--side-width" orientation="vertical" sign={-1} min={320} max={1400} initial={620} />
          <aside className="side">
            <div className="pane table-pane">
              <ElementTable
                branch={currentModel}
                latticeName={summary.name}
                selection={selection}
                hover={hover}
                range={range}
                query={query}
                onQuery={setQuery}
                onSelect={onTableSelect}
              />
            </div>
            <Splitter variable="--inspector-height" orientation="horizontal" sign={-1} min={120} max={900} initial={320} />
            <div className="pane inspector-pane">
              <Inspector
                digest={digest}
                occurrence={selectedId}
                branchName={branchName}
                describe={describe}
                onGo={goTo}
                onGoBranch={(id) => switchBranch(id)}
              />
            </div>
          </aside>
        </main>
      ) : !error ? (
        <div className="loading">{entries && entries.length === 0 ? "The server offers no lattices." : "Loading…"}</div>
      ) : null}
    </div>
  );
}
