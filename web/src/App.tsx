// The lattice viewer: floor plan, linear view, element table and inspector of
// one lattice revision, linked by a shared selection and hover. What they show
// is a branch, or a route through several branches (routes.yaml), and under
// the lattice strip either its optics or its BPM orbits (the Optics and Orbit
// tabs).
//
// All of this state belongs to this browser tab. The server only answers
// reads of immutable revisions, so other users, notebooks and tabs never see
// or disturb it; the URL hash carries it, so a link reproduces the view.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, ApiError, SCHEMA_VERSION, type LatticeEntry, type LatticeSummary, type RunEntry } from "./api/client";
import { va, type MachineEntry } from "./api/va";
import { AccountMenu, useAuth } from "./components/Auth";
import { BpmView } from "./components/BpmView";
import { AskAi, useAi } from "./components/Ai";
import { CheckView, viewerRuns } from "./components/CheckView";
import { OperateView } from "./components/OperateView";
import { ElementTable } from "./components/ElementTable";
import { FloorPlan, type FloorPlanHandle } from "./components/FloorPlan";
import { Inspector } from "./components/Inspector";
import { LinearView, type LinearViewHandle } from "./components/LinearView";
import { MachineMenu } from "./components/MachineMenu";
import { OpticsView } from "./components/OpticsView";
import { vaEnabled } from "./components/Setpoints";
import { Splitter } from "./components/Splitter";
import { TopBar } from "./components/TopBar";
import { useColorScheme } from "./components/useCanvas";
import { branchModel, branchOrder, type BranchModel } from "./model/branch";
import { isRoute, routeModel, targetIn, type RouteModel, type RouteSummary } from "./model/route";
import { createStore, sameTarget, type Target } from "./model/store";
import { parseHash, updateHash } from "./model/urlState";
import { clampRange, createZoomHistory, type SRange, type View2D } from "./model/viewport";

type Origin = "floor" | "linear" | "table" | "search" | "link";

const POLL_MS = 30_000;
const MACHINE_KEY = "eicvibe.va.machine";

/** A per-browser convenience: absent in a private window, never needed. */
function remembered(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function remember(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // not remembered
  }
}

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
  const [runs, setRuns] = useState<RunEntry[]>([]);
  const [tab, setTab] = useState<"optics" | "orbit" | "check" | "operate">(
    initial.tab === "orbit" || initial.tab === "check" || initial.tab === "operate" ? initial.tab : "optics",
  );
  // The virtual accelerator, if this server runs one (ADR 0008): its machines, the one the viewer
  // works with (remembered by this browser), and who is logged in, asked only when changing something.
  const [vaOn, setVaOn] = useState(false);
  const [machines, setMachines] = useState<MachineEntry[]>([]);
  const [machineId, setMachineId] = useState<string | null>(() => {
    // A link to a machine's orbit (the administration page's) chooses that machine.
    if (initial.machine) remember(MACHINE_KEY, initial.machine);
    return initial.machine ?? remembered(MACHINE_KEY);
  });
  const { auth, dialog } = useAuth(vaOn);
  // The optional local model (ADR 0009): its features show only when the server has one that answers.
  const aiStatus = useAi();

  const hover = useMemo(() => createStore<Target | null>(null, sameTarget), []);
  const range = useMemo(() => createStore<SRange | null>(null, sameRange), []);
  const band = useMemo(() => createStore<SRange | null>(null, sameRange), []);
  const history = useMemo(() => createZoomHistory(), []);
  const rangeBranch = useRef<string | null>(null);
  const ranges = useRef(new Map<string, SRange>());
  const pending = useRef({ selection: initial.sel ?? null, branch: initial.branch ?? null, range: initial.range ?? null });
  const floor = useRef<FloorPlanHandle | null>(null);
  const linear = useRef<LinearViewHandle | null>(null);
  const routeCache = useRef(new Map<string, { route: RouteSummary; parts: (BranchModel | undefined)[]; model: RouteModel }>());

  // Every route of the revision, once the branches it passes have loaded; kept while they stay the same.
  const routes = useMemo(() => {
    const out = new Map<string, RouteModel>();
    for (const route of summary?.routes ?? []) {
      const parts = route.segments.map((segment) => models.get(segment.branch));
      const cached = routeCache.current.get(route.id);
      if (cached && cached.route === route && cached.parts.every((part, k) => part === parts[k])) {
        out.set(route.id, cached.model);
        continue;
      }
      const model = routeModel(route, models);
      if (!model) continue;
      routeCache.current.set(route.id, { route, parts, model });
      out.set(route.id, model);
    }
    return out;
  }, [summary, models]);

  const live = useRef({ models, routes, current });
  live.current = { models, routes, current };
  /** The branch or route with this id. */
  const viewOf = (id: string | null) => (id ? (live.current.models.get(id) ?? live.current.routes.get(id)) : undefined);

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

  // Virtual rings' runs the server reads (ADR 0007); new runs appear as they start.
  const loadRuns = useCallback(
    () =>
      api.runs().then(
        (list) => setRuns((previous) => (JSON.stringify(previous) === JSON.stringify(list) ? previous : list)),
        () => undefined,
      ),
    [],
  );
  useEffect(() => {
    loadRuns();
    const timer = window.setInterval(loadRuns, 5_000);
    return () => window.clearInterval(timer);
  }, [loadRuns]);

  useEffect(() => void vaEnabled().then(setVaOn), []);
  const loadMachines = useCallback(() => va.machines().then(setMachines, () => undefined), []);
  useEffect(() => {
    if (!vaOn) return;
    loadMachines();
    const timer = window.setInterval(loadMachines, 5_000);
    return () => window.clearInterval(timer);
  }, [vaOn, loadMachines]);
  const machine =
    machines.find((m) => m.id === machineId) ?? machines.find((m) => m.shared && m.state === "running") ?? machines[0] ?? null;
  // Each start of a machine is a new run: list it now, for the views that follow the machine.
  const machineRun = machine?.run ?? null;
  useEffect(() => {
    if (machineRun) loadRuns();
  }, [machineRun, loadRuns]);
  /** The run the views follow: the machine's, or, while it is stopped, its last (the newest listed). */
  const machineShown = machine ? (machineRun ?? runs.find((r) => r.machine === machine.id)?.id) : undefined;
  /** The branch or route id a machine tracks, in this revision. */
  const machineView = useCallback(
    (m: MachineEntry | null) =>
      !m || !summary ? null : m.kind === "route" ? `route:${m.target}` : (summary.branches.find((b) => b.name === m.target)?.id ?? null),
    [summary],
  );

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
        const name = pending.current.branch;
        const wanted = name?.startsWith("route:")
          ? s.routes?.find((r) => r.id === name)?.id
          : s.branches.find((b) => b.name === name)?.id;
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
  const currentModel = current ? (models.get(current) ?? routes.get(current) ?? null) : null;

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

  // Resolve a selection from the URL once its branch (and the route shown, if any) has loaded.
  useEffect(() => {
    const wanted = pending.current.selection;
    const shown = live.current.current;
    if (!wanted || (shown?.startsWith("route:") && !routes.has(shown))) return;
    for (const model of models.values()) {
      const index = model.index.get(wanted);
      if (index === undefined) continue;
      pending.current.selection = null;
      const view = shown ? routes.get(shown) : undefined;
      const onRoute = view ? targetIn(view, { branch: model.id, index }, models, routes) : null;
      setSelection(onRoute ?? { branch: model.id, index });
      if (!onRoute && model.id !== shown) setCurrent(model.id);
      return;
    }
  }, [models, routes]);

  useEffect(() => {
    if (source) updateHash({ source, rev: digest && entries?.find((e) => e.source === source)?.digest !== digest ? digest : undefined });
  }, [source, digest, entries]);
  useEffect(() => {
    if (summary && current)
      updateHash({ branch: current.startsWith("route:") ? current : summary.branches.find((b) => b.id === current)?.name });
  }, [summary, current]);
  useEffect(() => {
    if (pending.current.selection) return;
    const view = selection ? (models.get(selection.branch) ?? routes.get(selection.branch)) : undefined;
    updateHash({ sel: selection ? view?.ids[selection.index] : undefined });
  }, [selection, models, routes]);
  useEffect(() => updateHash({ q: query || undefined }), [query]);
  useEffect(() => updateHash({ tab: tab === "optics" ? undefined : tab }), [tab]);

  const switchBranch = useCallback(
    (id: string, around?: number) => {
      const model = live.current.models.get(id) ?? live.current.routes.get(id);
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
    (wanted: Target | null, origin: Origin) => {
      // An occurrence of a branch the route shown passes is selected on the route, which stays shown.
      const { models, routes, current } = live.current;
      const view = current ? (models.get(current) ?? routes.get(current)) : undefined;
      const target = wanted && view && wanted.branch !== view.id ? (targetIn(view, wanted, models, routes) ?? wanted) : wanted;
      setSelection(target);
      if (!target) return;
      if (target.branch !== current) switchBranch(target.branch, target.index);
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
  // Choosing a machine shows what it tracks: its ring, or its route.
  const chooseMachine = useCallback(
    (id: string) => {
      setMachineId(id);
      remember(MACHINE_KEY, id);
      const view = machineView(machines.find((m) => m.id === id) ?? null);
      if (view && view !== live.current.current) switchBranch(view);
    },
    [machines, machineView, switchBranch],
  );
  const machineTarget = machineView(machine);

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

  const selectedId = selection ? (viewOf(selection.branch)?.ids[selection.index] ?? null) : null;
  // The floor plan draws a route shown over the branches it passes.
  const floorModels = useMemo(
    () => (isRoute(currentModel) ? new Map(models).set(currentModel.id, currentModel) : models),
    [models, currentModel],
  );
  const tabs = (
    <span className="panel-tabs" role="tablist" aria-label="Panel">
      <button type="button" role="tab" aria-selected={tab === "optics"} onClick={() => setTab("optics")}>
        Optics
      </button>
      <button type="button" role="tab" aria-selected={tab === "orbit"} onClick={() => setTab("orbit")}>
        Orbit
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={tab === "check"}
        onClick={() => setTab("check")}
        title="Check the tracking: a ring run's BPM data against the model's tunes, phases, beta and orbit"
      >
        Check
      </button>
      {vaOn ? (
        <button
          type="button"
          role="tab"
          aria-selected={tab === "operate"}
          onClick={() => setTab("operate")}
          title="Orbit correction, closed bumps and tune moves on a virtual accelerator's ring: proposals a person applies"
        >
          Operate
        </button>
      ) : null}
    </span>
  );
  const strip = summary ? (
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
  ) : null;

  return (
    <div className="app">
      <TopBar
        entries={entries}
        source={source}
        digest={digest}
        summary={summary}
        branchOrder={order}
        models={models}
        routes={routes}
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
        assistant={
          aiStatus && digest && current ? (
            <AskAi
              status={aiStatus}
              context={() => {
                const shown = parseHash(window.location.hash);
                return {
                  digest,
                  branch: current,
                  tab,
                  occurrence: selectedId,
                  engine: shown.engine ?? "xsuite",
                  fidelity: shown.fidelity ?? "standard",
                  run: machine?.run ?? null,
                  machine: machine?.id ?? null,
                };
              }}
            />
          ) : null
        }
        machine={
          vaOn ? (
            <MachineMenu
              machines={machines}
              chosen={machine}
              auth={auth}
              onChoose={chooseMachine}
              onChanged={loadMachines}
              onNotice={setNotice}
            />
          ) : null
        }
        account={vaOn ? <AccountMenu auth={auth} /> : null}
      />
      {dialog}
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
                branches={floorModels}
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
              {/* Both tabs stay mounted, so each keeps its choices; the strip moves with the tab shown. */}
              <div className="tab-panel" role="tabpanel" aria-label="Optics" hidden={tab !== "optics"}>
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
                  modes={summary.modes ?? []}
                  initialMode={initial.mode}
                  onSelect={onLinearSelect}
                  tabs={tabs}
                  strip={tab === "optics" ? strip : null}
                />
              </div>
              <div className="tab-panel" role="tabpanel" aria-label="Orbit" hidden={tab !== "orbit"}>
                <BpmView
                  digest={digest}
                  latticeName={summary.name}
                  branch={currentModel}
                  runs={runs}
                  active={tab === "orbit"}
                  initialRun={initial.run}
                  preferRun={machineShown}
                  machine={vaOn ? machine : null}
                  auth={auth}
                  onShowMachine={machineTarget && machineTarget !== current ? () => switchBranch(machineTarget) : undefined}
                  selection={selection}
                  hover={hover}
                  range={range}
                  band={band}
                  history={history}
                  scheme={scheme}
                  onSelect={onLinearSelect}
                  tabs={tabs}
                  strip={tab === "orbit" ? strip : null}
                />
              </div>
              <div className="tab-panel" role="tabpanel" aria-label="Check" hidden={tab !== "check"}>
                <CheckView
                  {...viewerRuns(runs, digest, summary.name, currentModel)}
                  pageLink
                  active={tab === "check"}
                  preferRun={machineShown}
                  scheme={scheme}
                  tabs={tabs}
                  ai={aiStatus}
                />
              </div>
              {vaOn ? (
                <div className="tab-panel" role="tabpanel" aria-label="Operate" hidden={tab !== "operate"}>
                  <OperateView machine={machine} auth={auth} active={tab === "operate"} tabs={tabs} scheme={scheme} />
                </div>
              ) : null}
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
                machine={vaOn ? machine : null}
                auth={auth}
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
