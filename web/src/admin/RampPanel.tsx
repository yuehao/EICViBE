// The admin page's Ramp tab (ADR 0013): a ring's ramp, where the ring is in it, and the control that
// starts it. A synchrotron with a ramp profile holds its beam at injection until an operator ramps it;
// then its energy and RF follow the profile turn by turn, the bunch is extracted after the flattop, and
// the ring is at injection again with a fresh bunch. The tab charts the profile's programs with where
// the ring is now, and lists the ramps of the machine's run with the bunch each extracted.

import { useCallback, useEffect, useState } from "react";

import { orbitLink, va, type MachineEntry, type RampView } from "../api/va";
import { Chart, type Marker, type Series } from "../components/Chart";
import { useColorScheme } from "../components/useCanvas";
import { formatNumber } from "../model/format";

const LINE = { color: "#1f5fbf", darkColor: "#6ea8ff" };
const GREY = { color: "#6b7280", darkColor: "#9ca3af" };
const STATES: Record<string, string> = { injection: "at injection", ramping: "ramping", flattop: "flattop" };

interface Props {
  machines: MachineEntry[];
  machine: string | null;
  mayControl(machine: MachineEntry): boolean;
  onMachine(id: string): void;
  onChanged(): void;
  onError(e: unknown): void;
}

export function RampPanel(props: Props) {
  const { machines, mayControl, onMachine, onChanged, onError } = props;
  const rings = machines.filter((m) => m.ramp);
  const entry = rings.find((m) => m.id === props.machine) ?? rings[0] ?? null;
  const id = entry?.id ?? null;
  const [view, setView] = useState<RampView | null>(null);
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const scheme = useColorScheme();

  const load = useCallback(() => {
    if (id) va.rampView(id).then(setView, onError);
  }, [id, onError]);
  useEffect(() => {
    setView(null);
    load();
    const timer = window.setInterval(load, 1000);
    return () => window.clearInterval(timer);
  }, [load]);

  if (!rings.length)
    return (
      <section>
        <h2>Ramp</h2>
        <p className="muted">No machine here ramps: a ring ramps when its machine folder gives it a ramp profile (ramps/, ADR 0013).</p>
      </section>
    );
  const control = entry !== null && mayControl(entry);
  const ramp = () => {
    if (!entry) return;
    setBusy(true);
    va.ramp(entry.id, reason || "ramp from the admin page")
      .then(() => {
        setReason("");
        load();
        onChanged();
      }, onError)
      .finally(() => setBusy(false));
  };

  return (
    <section className="ramp-panel" data-testid="ramp-panel">
      <h2>
        Ramp of{" "}
        <select aria-label="Machine" value={entry?.id ?? ""} onChange={(e) => onMachine(e.target.value)}>
          {rings.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name} · {m.target} ({m.state})
            </option>
          ))}
        </select>
      </h2>
      <p className="muted">
        The ring holds its beam at injection until it is ramped. Then its energy and RF follow the profile turn by turn, its magnets keeping their normalized
        strengths; after the flattop the bunch is extracted, and the ring is at injection again with a fresh bunch. Its ramp down has no beam and is not tracked.
      </p>
      {view ? (
        <>
          <Status view={view} control={control} busy={busy} reason={reason} onReason={setReason} onRamp={ramp} />
          <Programs view={view} scheme={scheme} />
          <div className="ramp-tables">
            <Profile view={view} />
            <Ramps view={view} machine={entry} />
          </div>
        </>
      ) : (
        <p className="muted">Loading…</p>
      )}
    </section>
  );
}

function Status(props: { view: RampView; control: boolean; busy: boolean; reason: string; onReason(v: string): void; onRamp(): void }) {
  const { view, control, busy } = props;
  const cycle = view.cycle;
  const state = String(cycle?.state ?? "");
  const turn = Number(cycle?.turn ?? 0);
  const fraction = state === "ramping" || state === "flattop" ? Math.min(1, turn / view.extraction_turn) : 0;
  return (
    <div className="ramp-status" data-testid="ramp-status">
      <div className="ramp-state">
        {cycle ? (
          <>
            <span className={`ramp-badge ${state}`}>{STATES[state] ?? state}</span>
            <strong>{formatNumber(Number(cycle.energy) / 1e9, 5)} GeV</strong>
            {state === "ramping" || state === "flattop" ? (
              <span className="muted">
                turn {turn.toLocaleString()} of {view.extraction_turn.toLocaleString()} · {formatNumber((turn * view.period) * 1e3, 4)} ms
              </span>
            ) : null}
          </>
        ) : (
          <span className="muted">not running</span>
        )}
        <span className="muted">· engine {view.engine ?? "default"}</span>
      </div>
      <div className="ramp-progress" role="progressbar" aria-label="Ramp progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(fraction * 100)}>
        <i style={{ width: `${(view.flattop_turn / view.extraction_turn) * 100}%` }} className="ramp-progress-ramp" />
        <b style={{ width: `${fraction * 100}%` }} />
      </div>
      {control ? (
        <div className="ramp-control">
          <input aria-label="Reason" placeholder="Why (goes into the log)" value={props.reason} onChange={(e) => props.onReason(e.target.value)} />
          <button type="button" className="primary" disabled={busy || view.refusal !== null} onClick={props.onRamp} title={view.refusal ?? "Ramp the ring to extraction"}>
            Start ramp
          </button>
          {view.refusal ? <span className="muted" data-testid="ramp-refusal">{view.refusal}</span> : null}
        </div>
      ) : (
        <p className="muted">Ramping a shared machine is for operators; a sandbox, for its owner.</p>
      )}
    </div>
  );
}

function Programs({ view, scheme }: { view: RampView; scheme: string }) {
  const program = view.program;
  const ms = view.period * 1e3;
  const domain: [number, number] = [0, view.extraction_turn * ms];
  const flattop = view.flattop_turn * ms;
  const state = String(view.cycle?.state ?? "");
  const now = state === "ramping" || state === "flattop" ? Number(view.cycle?.turn ?? 0) * ms : null;
  // The flattop's label gives way to now's where the two would overlap.
  const near = now !== null && Math.abs(now - flattop) < 0.12 * domain[1];
  const markers: Marker[] = [{ x: flattop, label: near ? undefined : "flattop", ...GREY, dash: [4, 3] }];
  if (now !== null) markers.push({ x: now, label: "now", ...GREY });
  const line = (label: string, y: readonly number[]): Series[] => [{ label, ...LINE, x: program.time, y, style: "line" }];
  const charts = [
    { title: "Energy", y: "energy (GeV)", series: line("energy", program.energy), testId: "ramp-energy" },
    { title: "RF voltage, all cavities", y: "voltage (MV)", series: line("voltage", program.voltage), testId: "ramp-voltage" },
    { title: "Synchronous phase, from the zero crossing", y: "phase (deg)", series: line("phase", program.phase), testId: "ramp-phase" },
  ];
  return (
    <div className="check-charts ramp-charts">
      {charts.map((c) => (
        <Chart key={c.testId} title={c.title} testId={c.testId} xLabel="time from the ramp's start (ms)" yLabel={c.y} series={c.series} markers={markers} xDomain={domain} scheme={scheme} />
      ))}
    </div>
  );
}

function Profile({ view }: { view: RampView }) {
  const e = view.energy as Record<string, number>;
  const ms = view.period * 1e3;
  const program = view.program;
  const source = view.source as Record<string, unknown>;
  const file = typeof source.path === "string" ? source.path.split("/").pop() : null;
  const rows: [string, string][] = [
    ["energy", `${formatNumber(e.injection! / 1e9, 5)} → ${formatNumber(e.extraction! / 1e9, 5)} GeV`],
    ["linear ramp", `${formatNumber(e.rate! / 1e9, 5)} GeV/s to ${formatNumber(e.linear_end! / 1e9, 5)} GeV, after a smooth start of ${e.smooth_turns} turns`],
    ["ramp", `${formatNumber(view.flattop_turn * ms, 5)} ms, ${view.flattop_turn.toLocaleString()} turns`],
    ["flattop", `${formatNumber(view.flattop * 1e3, 4)} ms; extraction at turn ${view.extraction_turn.toLocaleString()}`],
    ["RF voltage", `${formatNumber(program.voltage[0]!, 4)} → ${formatNumber(Math.max(...program.voltage), 4)} MV`],
    ["synchronous phase", `up to ${formatNumber(Math.max(...program.phase), 3)} degrees`],
    ["the designers' cycle", `${formatNumber(view.cycle_time, 4)} s, ${formatNumber(view.porch * 1e3, 3)} ms porch`],
    ["source", [file, Array.isArray(source.sheets) ? source.sheets.join(", ") : null].filter(Boolean).join(": ") || "–"],
  ];
  return (
    <div>
      <h3>
        Profile <code>{view.ramp}</code> <span className="muted">of {view.branch}</span>
      </h3>
      <table className="kv" data-testid="ramp-profile">
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k}>
              <th>{k}</th>
              <td>{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Ramps({ view, machine }: { view: RampView; machine: MachineEntry | null }) {
  const ms = view.period * 1e3;
  return (
    <div>
      <h3>Ramps of this run</h3>
      <p className="muted">
        The BPMs read every turn of a ramp, its flattop and extraction; the readings are kept whole as a capture, saved at
        extraction{machine ? (
          <>
            {" "}
            (the viewer's <a href={orbitLink(machine)}>orbit</a>, its data: the ramp's capture)
          </>
        ) : null}
        .
      </p>
      {view.ramps.length === 0 ? (
        <p className="muted">None yet.</p>
      ) : (
        <table className="admin-table" data-testid="ramp-history">
          <thead>
            <tr>
              <th>Began</th>
              <th className="num">Bunch</th>
              <th className="num">Energy (GeV)</th>
              <th className="num">εx (nm)</th>
              <th className="num">εy (pm)</th>
              <th className="num">σδ (10⁻³)</th>
              <th className="num">σz (mm)</th>
            </tr>
          </thead>
          <tbody>
            {view.ramps.map((r) => (
              <tr key={r.turn}>
                <td>
                  turn {r.turn.toLocaleString()}
                  <div className="muted">{r.wall ? new Date(r.wall).toLocaleTimeString() : ""}</div>
                  <div className="muted" data-testid="ramp-capture" title={r.capture ?? undefined}>
                    {r.capture
                      ? `BPMs: ${(r.capture_turns ?? 0).toLocaleString()} turns kept`
                      : r.extracted == null
                        ? "BPMs: every turn, kept at extraction"
                        : "BPMs: no capture"}
                  </div>
                </td>
                {r.extracted == null ? (
                  <td colSpan={6} className="muted">
                    {r.flattop != null ? "at the flattop" : "ramping"}: the ramp and the flattop take {formatNumber(view.extraction_turn * ms, 4)} ms
                  </td>
                ) : (
                  <>
                    <td className="num" title="particles extracted of those ramped">
                      {r.particles ?? "–"} / {r.injected ?? "–"}
                    </td>
                    <td className="num">{r.energy != null ? formatNumber(r.energy / 1e9, 5) : "–"}</td>
                    <td className="num">{r.emittance_x != null ? formatNumber(r.emittance_x * 1e9, 3) : "–"}</td>
                    <td className="num">{r.emittance_y != null ? formatNumber(r.emittance_y * 1e12, 3) : "–"}</td>
                    <td className="num">{r.sigma_delta != null ? formatNumber(r.sigma_delta * 1e3, 3) : "–"}</td>
                    <td className="num">{r.sigma_zeta != null ? formatNumber(r.sigma_zeta * 1e3, 3) : "–"}</td>
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
