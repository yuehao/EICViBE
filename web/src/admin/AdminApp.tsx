// The virtual accelerator's administration page (ADR 0008): machines to create,
// start and stop, their settings to change and put back, their errors (ADR
// 0011), the log of every change, and users. Nothing links to it; it shows nothing but a login until an
// operator or administrator logs in. Users work in the lattice viewer, which
// asks them to log in only when they change something.
//
// The server enforces every role whatever a page shows: operators start, stop,
// kick and reset shared machines; administrators also create and remove them
// and manage users.

import { useCallback, useEffect, useMemo, useState, type FormEvent, type ReactNode } from "react";

import {
  atLeast,
  orbitLink,
  ROLES,
  va,
  type ActionEntry,
  type LogEntry,
  type MachineEntry,
  type Me,
  type SettingRow,
  type SnapshotEntry,
  type TokenEntry,
  type UserEntry,
  type VaOptions,
  type WriteResult,
} from "../api/va";
import { ErrorsEditor } from "../components/ErrorsEditor";
import { AiPanel } from "./AiPanel";

type Tab = "machines" | "settings" | "errors" | "log" | "users" | "ai" | "account";

const TABS: { key: Tab; label: string; role?: (typeof ROLES)[number]; login?: boolean }[] = [
  { key: "machines", label: "Machines" },
  { key: "settings", label: "Settings" },
  { key: "errors", label: "Errors", login: true },
  { key: "log", label: "Log" },
  { key: "users", label: "Users", role: "admin" },
  { key: "ai", label: "AI", role: "admin" },
  { key: "account", label: "Account", login: true },
];

const time = (iso: string | null | undefined) => (iso ? new Date(iso).toLocaleString() : "");
const number = (v: number | null | undefined, digits = 6) => (v === null || v === undefined ? "–" : Number(v.toPrecision(digits)).toString());

/** May ``me`` control (start, stop, reset, kick) this machine? */
const mayControl = (me: Me, m: MachineEntry) => (m.shared ? atLeast(me.role, "operator") : (me.logged_in && m.owner === me.name) || me.role === "admin");
/** May ``me`` write settings of this machine? */
const mayWrite = (me: Me, m: MachineEntry) =>
  m.shared ? me.logged_in && atLeast(me.role, "physicist") : (me.logged_in && m.owner === me.name) || me.role === "admin";

function useHash(): [Record<string, string>, (update: Record<string, string | undefined>) => void] {
  const read = () => Object.fromEntries(new URLSearchParams(window.location.hash.slice(1)));
  const [state, setState] = useState(read);
  useEffect(() => {
    const changed = () => setState(read());
    window.addEventListener("hashchange", changed);
    return () => window.removeEventListener("hashchange", changed);
  }, []);
  const update = useCallback((next: Record<string, string | undefined>) => {
    const params = new URLSearchParams(window.location.hash.slice(1));
    for (const [k, v] of Object.entries(next)) if (v) params.set(k, v); else params.delete(k);
    window.history.replaceState(null, "", `#${params}`);
    setState(Object.fromEntries(params));
  }, []);
  return [state, update];
}

export function AdminApp() {
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [source, setSource] = useState<string | null>(null);
  const [me, setMe] = useState<Me | null>(null);
  const [hash, setHash] = useHash();
  const [machines, setMachines] = useState<MachineEntry[]>([]);
  const [options, setOptions] = useState<VaOptions | null>(null);
  const [error, setError] = useState<string | null>(null);
  const tab = (TABS.find((t) => t.key === hash.tab)?.key ?? "machines") as Tab;
  const selected = hash.machine ?? machines[0]?.id ?? null;

  const fail = useCallback((e: unknown) => setError((e as Error).message), []);
  const refresh = useCallback(() => va.machines().then(setMachines, fail), [fail]);
  const allowed = !!me?.logged_in && atLeast(me.role, "operator");

  useEffect(() => {
    va.status().then((s) => {
      setEnabled(s.enabled);
      setSource(s.source ?? null);
      if (s.enabled) va.me().then(setMe, fail);
    }, fail);
  }, [fail]);
  useEffect(() => {
    if (!allowed) return;
    va.options().then(setOptions, fail);
    refresh();
    const timer = window.setInterval(refresh, 2000);
    return () => window.clearInterval(timer);
  }, [allowed, refresh, fail]);

  if (enabled === false)
    return (
      <div className="admin">
        <div className="admin-empty">
          <h1>Administration</h1>
          <p>
            This server runs no virtual accelerator. Start one with <code>eicvibe-web MACHINE --va STATE</code>.
          </p>
        </div>
      </div>
    );
  if (!me) return <div className="admin"><div className="admin-empty">{error ?? "Loading…"}</div></div>;
  if (!allowed)
    return (
      <div className="admin">
        <div className="admin-empty" data-testid="admin-gate">
          <h1>EICViBE administration</h1>
          {me.logged_in ? (
            <>
              <p>This page is for operators and administrators; {me.name} is a {me.role}.</p>
              <button type="button" onClick={() => va.logout().then(() => va.me().then(setMe), fail)}>
                Log out
              </button>
            </>
          ) : (
            <Login me={me} onChange={setMe} onError={fail} />
          )}
          {error ? <p className="admin-error">{error}</p> : null}
        </div>
      </div>
    );

  return (
    <div className="admin">
      <header className="topbar admin-bar">
        <div className="brand">
          <strong>EICViBE</strong> <span>administration{source ? ` · ${source}` : ""}</span>
        </div>
        <nav className="admin-tabs" aria-label="Sections">
          {TABS.filter((t) => (!t.role || atLeast(me.role, t.role)) && (!t.login || me.logged_in)).map((t) => (
            <button key={t.key} type="button" aria-current={t.key === tab ? "page" : undefined} onClick={() => setHash({ tab: t.key })}>
              {t.label}
            </button>
          ))}
        </nav>
        <div className="spacer" />
        <a className="button" href="/">
          Lattice viewer
        </a>
        <Login me={me} onChange={(next) => { setMe(next); refresh(); }} onError={fail} />
      </header>
      {error ? (
        <div className="banner error" onClick={() => setError(null)} title="Click to dismiss">
          {error}
        </div>
      ) : null}
      <main className="admin-main">
        {tab === "machines" ? (
          <Machines me={me} machines={machines} options={options} onChanged={refresh} onError={fail} onOpen={(id, t) => setHash({ tab: t, machine: id })} />
        ) : tab === "settings" ? (
          <Settings me={me} machines={machines} machine={selected} onMachine={(id) => setHash({ machine: id })} onChanged={refresh} onError={fail} />
        ) : tab === "errors" ? (
          <Errors machines={machines} machine={selected} onMachine={(id) => setHash({ machine: id })} onChanged={refresh} onError={fail} />
        ) : tab === "log" ? (
          <Log machines={machines} machine={hash.machine} onMachine={(id) => setHash({ machine: id })} onError={fail} />
        ) : tab === "users" ? (
          <Users me={me} onError={fail} />
        ) : tab === "ai" ? (
          <AiPanel onError={fail} />
        ) : (
          <Account me={me} onError={fail} />
        )}
      </main>
    </div>
  );
}

// -- errors ---------------------------------------------------------------------------

function Errors(props: { machines: MachineEntry[]; machine: string | null; onMachine(id: string): void; onChanged(): void; onError(e: unknown): void }) {
  const { machines, machine, onMachine, onChanged, onError } = props;
  const entry = machines.find((m) => m.id === machine) ?? null;
  return (
    <section>
      <h2>
        Errors of{" "}
        <select aria-label="Machine" value={machine ?? ""} onChange={(e) => onMachine(e.target.value)}>
          {machines.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name} ({m.state})
            </option>
          ))}
        </select>
      </h2>
      <p className="muted">
        What makes the virtual machine differ from its model, as a real one does: drawn from the seed when it starts, hidden from its users, its settings and the log. A
        run keeps what it drew in its hidden.json.
      </p>
      {entry ? <ErrorsEditor machine={entry} onError={onError} onSaved={onChanged} /> : null}
    </section>
  );
}

// -- who ----------------------------------------------------------------------------

function Login({ me, onChange, onError }: { me: Me; onChange(me: Me): void; onError(e: unknown): void }) {
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  if (me.logged_in)
    return (
      <span className="admin-user">
        <strong>{me.name}</strong> <span className="muted">{me.role}</span>{" "}
        <button type="button" onClick={() => va.logout().then(() => va.me().then(onChange), onError)}>
          Log out
        </button>
      </span>
    );
  const submit = (e: FormEvent) => {
    e.preventDefault();
    va.login(name, password).then((next) => {
      setPassword("");
      onChange(next);
    }, onError);
  };
  return (
    <form className="admin-login" onSubmit={submit}>
      <input aria-label="User name" placeholder="user" value={name} onChange={(e) => setName(e.target.value)} autoComplete="username" />
      <input aria-label="Password" placeholder="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" />
      <button type="submit" disabled={!name || !password}>
        Log in
      </button>
    </form>
  );
}

// -- machines -------------------------------------------------------------------------

function Machines(props: {
  me: Me;
  machines: MachineEntry[];
  options: VaOptions | null;
  onChanged(): void;
  onError(e: unknown): void;
  onOpen(id: string, tab: Tab): void;
}) {
  const { me, machines, options, onChanged, onError, onOpen } = props;
  const act = (promise: Promise<unknown>) => promise.then(onChanged, onError);
  const [kick, setKick] = useState("0.1");
  return (
    <section>
      <h2>Machines</h2>
      <p className="muted">
        Shared machines are the ones everyone works on, in the lattice viewer; a sandbox is someone's own copy of one. A
        machine's settings are the machine files' (design and mode) with what people changed on top, which the log keeps.
      </p>
      <table className="admin-table" data-testid="machines">
        <thead>
          <tr>
            <th>Machine</th>
            <th>Tracks</th>
            <th>Engine</th>
            <th>Mode</th>
            <th>Whose</th>
            <th>State</th>
            <th>Changes</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {machines.map((m) => {
            const control = mayControl(me, m);
            const running = m.state === "running" || m.state === "starting";
            return (
              <tr key={m.id} data-machine={m.id}>
                <td>
                  <strong>{m.name}</strong>
                  <div className="muted">{m.id}</div>
                </td>
                <td>
                  {m.kind === "route" ? "route" : "ring"} {m.target}
                  {m.kind === "ring" ? (
                    <div className="muted">{m.ramp_turns ? `changes ramp over ${m.ramp_turns} turns` : "changes step"}</div>
                  ) : null}
                </td>
                <td>
                  {m.engine ?? "default"} · {m.fidelity}
                </td>
                <td>{m.mode ?? "design"}</td>
                <td>{m.shared ? "shared" : `sandbox of ${m.owner}${m.parent ? ` (from ${m.parent})` : ""}`}</td>
                <td>
                  <span className={`state ${m.state}`} title={m.error ?? ""} data-testid="machine-state">
                    {m.state}
                  </span>
                  {m.position !== null && m.position !== undefined ? (
                    <span className="muted">
                      {" "}
                      {m.kind === "route" ? "shot" : "turn"} {m.position}
                    </span>
                  ) : null}
                  {m.error ? <div className="admin-error">{m.error}</div> : null}
                  {m.warnings.map((w) => (
                    <div key={w} className="admin-error">
                      ⚠ {w}
                    </div>
                  ))}
                </td>
                <td>{m.changes}</td>
                <td className="admin-actions">
                  {control && !running && m.state !== "stopping" ? <button type="button" onClick={() => act(va.start(m.id))}>Start</button> : null}
                  {control && running ? <button type="button" onClick={() => act(va.stop(m.id))}>Stop</button> : null}
                  {control && m.kind === "ring" && m.state === "running" ? (
                    <button type="button" title={`Kick x by ${kick} mrad at the ring's start`} onClick={() => act(va.kick(m.id, Number(kick) * 1e-3, 0))}>
                      Kick
                    </button>
                  ) : null}
                  <button type="button" onClick={() => onOpen(m.id, "settings")}>Settings</button>
                  <button type="button" onClick={() => onOpen(m.id, "errors")}>Errors</button>
                  <button type="button" onClick={() => onOpen(m.id, "log")}>Log</button>
                  <a className="button" href={orbitLink(m)} target="_blank" rel="noreferrer">
                    Orbit
                  </a>
                  {me.logged_in && atLeast(me.role, "physicist") ? (
                    <button type="button" title="A sandbox of your own with this machine's settings now" onClick={() => act(va.fork(m.id))}>
                      Sandbox
                    </button>
                  ) : null}
                  {!running && m.state !== "stopping" && (m.shared ? me.role === "admin" : control) ? (
                    <button type="button" className="danger" onClick={() => window.confirm(`Remove ${m.name}? Its log stays.`) && act(va.remove(m.id))}>
                      Remove
                    </button>
                  ) : null}
                </td>
              </tr>
            );
          })}
          {machines.length === 0 ? (
            <tr>
              <td colSpan={8} className="muted">
                No machines yet.
              </td>
            </tr>
          ) : null}
        </tbody>
      </table>
      <p className="muted">
        Kick size: <input className="narrow" aria-label="Kick in mrad" value={kick} onChange={(e) => setKick(e.target.value)} /> mrad
      </p>
      {me.logged_in && atLeast(me.role, "physicist") && options ? <NewMachineForm me={me} options={options} onDone={onChanged} onError={onError} /> : null}
    </section>
  );
}

function NewMachineForm({ me, options, onDone, onError }: { me: Me; options: VaOptions; onDone(): void; onError(e: unknown): void }) {
  const admin = me.role === "admin";
  const [form, setForm] = useState({
    name: "",
    kind: options.rings.length ? "ring" : "route",
    target: "", // chosen, never assumed: a machine's name does not say what it tracks
    engine: "",
    fidelity: "standard",
    mode: "",
    shared: admin,
    particles: "50",
    rate: "",
    ramp: "1024",
    autostart: false,
  });
  const set = (patch: Partial<typeof form>) => setForm((f) => ({ ...f, ...patch }));
  const targets = form.kind === "ring" ? options.rings : options.routes;
  const submit = (e: FormEvent) => {
    e.preventDefault();
    va.create({
      name: form.name,
      kind: form.kind as "ring" | "route",
      target: form.target,
      engine: form.engine || null,
      fidelity: form.fidelity,
      mode: form.mode || null,
      shared: admin && form.shared,
      particles: Number(form.particles),
      rate: form.rate ? Number(form.rate) : null,
      ramp_turns: Number(form.ramp || 0),
      autostart: form.autostart,
      reason: "",
    }).then(() => {
      set({ name: "" });
      onDone();
    }, onError);
  };
  return (
    <form className="admin-form" onSubmit={submit} aria-label="New machine">
      <h3>New machine</h3>
      <label>
        Name <input value={form.name} onChange={(e) => set({ name: e.target.value })} required />
      </label>
      <label>
        Tracks{" "}
        <select value={form.kind} onChange={(e) => set({ kind: e.target.value, target: "" })}>
          <option value="ring" disabled={!options.rings.length}>a ring, turn after turn</option>
          <option value="route" disabled={!options.routes.length}>a route, shot after shot</option>
        </select>{" "}
        <select aria-label="Ring or route" value={form.target} onChange={(e) => set({ target: e.target.value })} required>
          <option value="" disabled>
            choose…
          </option>
          {targets.map((t) => (
            <option key={t}>{t}</option>
          ))}
        </select>
      </label>
      <label>
        Engine{" "}
        <select value={form.engine} onChange={(e) => set({ engine: e.target.value })}>
          <option value="">default (TrackPad when installed)</option>
          {options.engines.map((e) => (
            <option key={e.name} value={e.name} disabled={!e.available}>
              {e.name}
              {e.available ? "" : " (not installed)"}
            </option>
          ))}
        </select>{" "}
        <select aria-label="Fidelity" value={form.fidelity} onChange={(e) => set({ fidelity: e.target.value })}>
          {options.fidelities.filter((f) => f !== "linear").map((f) => (
            <option key={f}>{f}</option>
          ))}
        </select>
      </label>
      <label>
        Mode{" "}
        <select value={form.mode} onChange={(e) => set({ mode: e.target.value })}>
          <option value="">design (the lattice's values)</option>
          {options.modes.map((m) => (
            <option key={m}>{m}</option>
          ))}
        </select>
      </label>
      <label>
        Particles <input className="narrow" value={form.particles} onChange={(e) => set({ particles: e.target.value })} />
      </label>
      <label>
        Pace <input className="narrow" placeholder="default" value={form.rate} onChange={(e) => set({ rate: e.target.value })} /> {form.kind === "ring" ? "turns" : "shots"}/s at most
      </label>
      {form.kind === "ring" ? (
        <label title="A setting change moves smoothly (a half cosine) over this many turns, as a magnet follows its supply; 0 changes it in one step">
          Settings ramp <input className="narrow" value={form.ramp} onChange={(e) => set({ ramp: e.target.value })} /> turns
        </label>
      ) : null}
      {admin ? (
        <label>
          <input type="checkbox" checked={form.shared} onChange={(e) => set({ shared: e.target.checked })} /> shared (else a sandbox of yours)
        </label>
      ) : (
        <span className="muted">A sandbox of yours ({options.sandbox_quota} at most)</span>
      )}
      {admin && form.shared ? (
        <label>
          <input type="checkbox" checked={form.autostart} onChange={(e) => set({ autostart: e.target.checked })} /> start with the service
        </label>
      ) : null}
      <button type="submit" disabled={!form.name || !form.target}>
        Create
      </button>
    </form>
  );
}

// -- settings -----------------------------------------------------------------------

function Settings(props: {
  me: Me;
  machines: MachineEntry[];
  machine: string | null;
  onMachine(id: string): void;
  onChanged(): void;
  onError(e: unknown): void;
}) {
  const { me, machines, machine, onMachine, onChanged, onError } = props;
  const entry = machines.find((m) => m.id === machine) ?? null;
  const [rows, setRows] = useState<SettingRow[]>([]);
  const [filter, setFilter] = useState("");
  const [changedOnly, setChangedOnly] = useState(false);
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [reason, setReason] = useState("");
  const [result, setResult] = useState<WriteResult | null>(null);
  const [snapshots, setSnapshots] = useState<SnapshotEntry[]>([]);
  const [snapshotName, setSnapshotName] = useState("");

  const load = useCallback(() => {
    if (!machine) return;
    va.settings(machine).then(setRows, onError);
    va.snapshots(machine).then(setSnapshots, onError);
  }, [machine, onError]);
  useEffect(() => {
    setEdits({});
    setResult(null);
    load();
    const timer = window.setInterval(load, 4000);
    return () => window.clearInterval(timer);
  }, [load]);

  const shown = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    return rows.filter((r) => (!changedOnly || r.changed || r.channel in edits) && (!needle || r.channel.toLowerCase().includes(needle) || r.device.toLowerCase().includes(needle)));
  }, [rows, filter, changedOnly, edits]);
  const pending = Object.entries(edits).filter(([, v]) => v.trim() !== "");
  const writable = !!entry && mayWrite(me, entry);
  const control = !!entry && mayControl(me, entry);
  const after = (r: WriteResult) => {
    setResult(r);
    load();
    onChanged();
  };

  const apply = () => {
    if (!machine) return;
    const changes = pending.map(([channel, value]) => ({ channel, value: Number(value) }));
    if (changes.some((c) => !Number.isFinite(c.value))) return onError(new Error("every new value must be a number"));
    va.write(machine, changes, reason).then((r) => {
      if (r.applied) {
        setEdits({});
        setReason("");
      }
      after(r);
    }, onError);
  };

  return (
    <section>
      <h2>
        Settings of{" "}
        <select aria-label="Machine" value={machine ?? ""} onChange={(e) => onMachine(e.target.value)}>
          {machines.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name} ({m.state})
            </option>
          ))}
        </select>
      </h2>
      <p className="muted">
        "Original" is what the machine files give (design{entry?.mode ? `, mode ${entry.mode}` : ""}); "now" is the machine's value. A change applies to the
        running engine between turns (or shots) and goes into the log with the value it replaced. The machine files never change.
      </p>
      <div className="admin-row">
        <input type="search" placeholder="Filter by channel or device…" aria-label="Filter settings" value={filter} onChange={(e) => setFilter(e.target.value)} />
        <label>
          <input type="checkbox" checked={changedOnly} onChange={(e) => setChangedOnly(e.target.checked)} /> changed only
        </label>
        <span className="muted">
          {shown.length} of {rows.length} channels, {rows.filter((r) => r.changed).length} changed
        </span>
        <div className="spacer" />
        {control && rows.some((r) => r.changed) ? (
          <button type="button" className="danger" onClick={() => machine && window.confirm("Put every changed setting back to its original value?") && va.reset(machine, null, reason || "reset to original").then(after, onError)}>
            Reset all to original
          </button>
        ) : null}
      </div>
      <div className="admin-scroll">
        <table className="admin-table settings" data-testid="settings">
          <thead>
            <tr>
              <th>Channel</th>
              <th>Unit</th>
              <th className="num">Original</th>
              <th className="num">Now</th>
              {writable ? <th>New value</th> : null}
              <th>Changed</th>
            </tr>
          </thead>
          <tbody>
            {shown.slice(0, 500).map((r) => (
              <tr key={r.channel} className={r.changed ? "changed" : ""}>
                <td>
                  <code>{r.channel}</code>
                  {r.limits ? <div className="muted">limits {r.limits[0]} – {r.limits[1]}</div> : null}
                </td>
                <td>{r.unit}</td>
                <td className="num">{number(r.original)}</td>
                <td className="num">
                  <strong>{number(r.value)}</strong>
                </td>
                {writable ? (
                  <td>
                    <input
                      className="narrow"
                      aria-label={`New value of ${r.channel}`}
                      value={edits[r.channel] ?? ""}
                      placeholder={number(r.value)}
                      onChange={(e) => setEdits((x) => ({ ...x, [r.channel]: e.target.value }))}
                    />
                  </td>
                ) : null}
                <td>
                  {r.changed ? (
                    <>
                      {r.by} · {time(r.at)}{" "}
                      {writable ? (
                        <button type="button" title="Back to the original value" onClick={() => machine && va.reset(machine, [r.channel], reason).then(after, onError)}>
                          ↺
                        </button>
                      ) : null}
                    </>
                  ) : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {shown.length > 500 ? <p className="muted">Showing 500; filter to find the others.</p> : null}
      </div>
      {writable ? (
        <div className="admin-apply">
          <span>
            {pending.length} change{pending.length === 1 ? "" : "s"}
          </span>
          <input placeholder="Why (goes into the log)" aria-label="Reason" value={reason} onChange={(e) => setReason(e.target.value)} />
          <button type="button" disabled={!pending.length} onClick={apply}>
            Apply as one change
          </button>
          <button type="button" disabled={!pending.length} onClick={() => setEdits({})}>
            Discard
          </button>
        </div>
      ) : (
        <p className="muted">{me.logged_in ? "Your role cannot change this machine's settings." : "Log in to change settings."}</p>
      )}
      {result ? <WriteOutcome result={result} /> : null}
      <h3>Snapshots</h3>
      <div className="admin-row">
        {me.logged_in && atLeast(me.role, "physicist") ? (
          <>
            <input placeholder="Snapshot name" aria-label="Snapshot name" value={snapshotName} onChange={(e) => setSnapshotName(e.target.value)} />
            <button type="button" disabled={!snapshotName || !machine} onClick={() => machine && va.snapshot(machine, snapshotName).then(() => { setSnapshotName(""); load(); }, onError)}>
              Take a snapshot
            </button>
          </>
        ) : null}
      </div>
      <table className="admin-table">
        <tbody>
          {snapshots.map((s) => (
            <tr key={s.id}>
              <td>{s.name}</td>
              <td className="muted">
                {s.user} · {time(s.time)}
              </td>
              <td>{s.changes} changed settings</td>
              <td>
                {control ? (
                  <button type="button" onClick={() => machine && va.restore(machine, s.id, reason).then(after, onError)}>
                    Restore
                  </button>
                ) : null}
              </td>
            </tr>
          ))}
          {snapshots.length === 0 ? (
            <tr>
              <td className="muted">No snapshots of this machine.</td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </section>
  );
}

function WriteOutcome({ result }: { result: WriteResult }) {
  return (
    <div className={`admin-outcome ${result.applied ? "ok" : "refused"}`} data-testid="write-outcome">
      <strong>{result.applied ? "Applied" : "Refused"}</strong>
      {result.correlation ? <span className="muted"> · change {result.correlation}</span> : null}
      <ul className="plain">
        {result.writes.map((w) => (
          <li key={w.channel}>
            <code>{w.channel}</code> {number(w.old)} → {number(w.new)} {w.unit}
            {w.refusal ? <span className="admin-error"> {w.refusal}</span> : null}
          </li>
        ))}
      </ul>
    </div>
  );
}

// -- the log --------------------------------------------------------------------------

function Log(props: { machines: MachineEntry[]; machine?: string; onMachine(id: string): void; onError(e: unknown): void }) {
  const { machines, machine, onMachine, onError } = props;
  const [kind, setKind] = useState<"writes" | "actions">("writes");
  const [user, setUser] = useState("");
  const [correlation, setCorrelation] = useState("");
  const [writes, setWrites] = useState<LogEntry[]>([]);
  const [actions, setActions] = useState<ActionEntry[]>([]);
  const [older, setOlder] = useState(false);

  const load = useCallback(
    (before?: number) => {
      const filters = { machine: machine || undefined, user: user || undefined, before, limit: 100 };
      if (kind === "writes")
        va.log({ ...filters, correlation: correlation || undefined }).then((rows) => setWrites((w) => (before ? [...w, ...rows] : rows)), onError);
      else va.actions(filters).then((rows) => setActions((a) => (before ? [...a, ...rows] : rows)), onError);
      setOlder(!!before);
    },
    [kind, machine, user, correlation, onError],
  );
  useEffect(() => {
    load();
    const timer = window.setInterval(() => !older && load(), 3000);
    return () => window.clearInterval(timer);
  }, [load, older]);

  const rows: { id: number }[] = kind === "writes" ? writes : actions;
  return (
    <section>
      <h2>Log</h2>
      <p className="muted">Every change, kept as it was made: nothing in it is ever edited or removed.</p>
      <div className="admin-row">
        <select aria-label="Log" value={kind} onChange={(e) => setKind(e.target.value as "writes" | "actions")}>
          <option value="writes">setting changes</option>
          <option value="actions">starts, stops, logins, users, AI…</option>
        </select>
        <select aria-label="Machine filter" value={machine ?? ""} onChange={(e) => onMachine(e.target.value)}>
          <option value="">every machine</option>
          {machines.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name}
            </option>
          ))}
        </select>
        <input placeholder="user" aria-label="User filter" className="narrow" value={user} onChange={(e) => setUser(e.target.value)} />
        {correlation ? (
          <button type="button" onClick={() => setCorrelation("")}>
            change {correlation} ✕
          </button>
        ) : null}
      </div>
      <div className="admin-scroll">
        {kind === "writes" ? (
          <table className="admin-table" data-testid="log">
            <thead>
              <tr>
                <th>When</th>
                <th>Who</th>
                <th>Machine</th>
                <th>Channel</th>
                <th className="num">Old → new</th>
                <th>Result</th>
                <th>Why</th>
                <th>Change</th>
              </tr>
            </thead>
            <tbody>
              {writes.map((w) => (
                <tr key={w.id} className={w.applied ? "" : "refused"}>
                  <td>{time(w.time)}</td>
                  <td>{w.user}</td>
                  <td>{w.machine}</td>
                  <td>
                    <code>{w.channel}</code>
                  </td>
                  <td className="num">
                    {number(w.old)} → {number(w.new)} {w.unit}
                  </td>
                  <td>{w.applied ? "applied" : <span className="admin-error">{w.refusal}</span>}</td>
                  <td>{w.reason}</td>
                  <td>
                    <button type="button" className="link" title="Every write of this change" onClick={() => setCorrelation(w.correlation)}>
                      {w.correlation}
                    </button>
                    {w.turn !== null && w.turn !== undefined ? <span className="muted"> @ {w.turn}</span> : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <table className="admin-table" data-testid="actions">
            <thead>
              <tr>
                <th>When</th>
                <th>Who</th>
                <th>Machine</th>
                <th>What</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody>
              {actions.map((a) => (
                <tr key={a.id}>
                  <td>{time(a.time)}</td>
                  <td>{a.user}</td>
                  <td>{a.machine}</td>
                  <td>{a.action}</td>
                  <td className="muted">
                    {Object.entries(a.detail)
                      .map(([k, v]) => `${k} ${typeof v === "object" ? JSON.stringify(v) : String(v)}`)
                      .join(", ")}
                    {a.reason ? ` — ${a.reason}` : ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      {rows.length >= 100 ? (
        <button type="button" onClick={() => load(Math.min(...rows.map((r) => r.id)))}>
          Older
        </button>
      ) : null}
    </section>
  );
}

// -- users and the account ----------------------------------------------------------------

function Users({ me, onError }: { me: Me; onError(e: unknown): void }) {
  const [users, setUsers] = useState<UserEntry[]>([]);
  const [name, setName] = useState("");
  const [role, setRole] = useState("physicist");
  const [shown, setShown] = useState<ReactNode>(null);
  const load = useCallback(() => va.users().then(setUsers, onError), [onError]);
  useEffect(() => void load(), [load]);
  const secret = (who: string, password: string) =>
    setShown(
      <div className="admin-outcome ok" data-testid="new-password">
        Password of <strong>{who}</strong>: <code>{password}</code> (shown once; give it to them, and they can change it under Account)
      </div>,
    );
  return (
    <section>
      <h2>Users</h2>
      {shown}
      <table className="admin-table" data-testid="users">
        <tbody>
          {users.map((u) => (
            <tr key={u.name} className={u.disabled ? "refused" : ""}>
              <td>
                <strong>{u.name}</strong>
              </td>
              <td>
                <select aria-label={`Role of ${u.name}`} value={u.role} disabled={u.name === me.name} onChange={(e) => va.changeUser(u.name, { role: e.target.value }).then(load, onError)}>
                  {ROLES.map((r) => (
                    <option key={r}>{r}</option>
                  ))}
                </select>
              </td>
              <td className="muted">since {time(u.created)}</td>
              <td>
                <label>
                  <input type="checkbox" checked={u.disabled} disabled={u.name === me.name} onChange={(e) => va.changeUser(u.name, { disabled: e.target.checked }).then(load, onError)} /> disabled
                </label>
              </td>
              <td>
                <button type="button" onClick={() => va.resetPassword(u.name).then((c) => secret(c.name, c.password), onError)}>
                  New password
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <form
        className="admin-row"
        aria-label="New user"
        onSubmit={(e) => {
          e.preventDefault();
          va.addUser(name, role).then((c) => {
            secret(c.name, c.password);
            setName("");
            load();
          }, onError);
        }}
      >
        <input placeholder="new user name" aria-label="New user name" value={name} onChange={(e) => setName(e.target.value)} />
        <select aria-label="New user's role" value={role} onChange={(e) => setRole(e.target.value)}>
          {ROLES.map((r) => (
            <option key={r}>{r}</option>
          ))}
        </select>
        <button type="submit" disabled={!name}>
          Add user
        </button>
      </form>
      <p className="muted">
        Viewers read; physicists change settings of shared machines and run sandboxes; operators also start, stop, kick and reset shared machines;
        administrators also create machines and manage users.
      </p>
    </section>
  );
}

function Account({ me, onError }: { me: Me; onError(e: unknown): void }) {
  const [old, setOld] = useState("");
  const [next, setNext] = useState("");
  const [done, setDone] = useState<string | null>(null);
  const [tokens, setTokens] = useState<TokenEntry[]>([]);
  const [label, setLabel] = useState("");
  const [issued, setIssued] = useState<string | null>(null);
  const load = useCallback(() => va.tokens().then(setTokens, onError), [onError]);
  useEffect(() => void load(), [load]);
  return (
    <section>
      <h2>{me.name}</h2>
      <h3>Password</h3>
      <form
        className="admin-row"
        onSubmit={(e) => {
          e.preventDefault();
          va.changePassword(old, next).then(() => {
            setOld("");
            setNext("");
            setDone("Password changed.");
          }, onError);
        }}
      >
        <input type="password" placeholder="current" aria-label="Current password" value={old} onChange={(e) => setOld(e.target.value)} autoComplete="current-password" />
        <input type="password" placeholder="new, 8 characters or more" aria-label="New password" value={next} onChange={(e) => setNext(e.target.value)} autoComplete="new-password" />
        <button type="submit" disabled={!old || next.length < 8}>
          Change
        </button>
        {done ? <span className="muted">{done}</span> : null}
      </form>
      <h3>Tokens for notebooks</h3>
      <p className="muted">A notebook writes as you with a token, and the log says so.</p>
      <ul className="plain">
        {tokens.map((t) => (
          <li key={t.created}>
            {t.label} <span className="muted">since {time(t.created)}</span>
          </li>
        ))}
      </ul>
      <form
        className="admin-row"
        onSubmit={(e) => {
          e.preventDefault();
          va.newToken(label).then((t) => {
            setIssued(t.token);
            setLabel("");
            load();
          }, onError);
        }}
      >
        <input placeholder="what it is for, e.g. orbit notebook" aria-label="Token label" value={label} onChange={(e) => setLabel(e.target.value)} />
        <button type="submit" disabled={!label}>
          New token
        </button>
      </form>
      {issued ? (
        <div className="admin-outcome ok">
          Shown once:
          <pre>{`from eicvibe.va.client import Client\nva = Client("${window.location.origin}", token="${issued}")\nva.set("MACHINE", {"CHANNEL": 1.0}, reason="...")`}</pre>
        </div>
      ) : null}
    </section>
  );
}
