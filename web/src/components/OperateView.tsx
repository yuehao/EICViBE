// The Operate tab (ADR 0010): orbit correction, closed bumps and tune moves, as
// accelerator physicists do them, on a virtual accelerator's ring.
//
// EICViBE computes each operation on the machine as it is set and as its BPMs
// read, and shows it as a proposal: the setting changes, what the model
// predicts, the checks it passed, the method and its references. Nothing is
// written until a person applies it (an agent's token can propose here but
// not apply); once the settings have ramped, Verify compares what the BPMs or
// the tune monitor show with the prediction. Proposals an agent makes appear
// here for a person to review.

import { useCallback, useEffect, useState, type ReactNode } from "react";

import { va, type MachineEntry, type ProposalEntry, type ProposalRequest, type TuneReading } from "../api/va";
import { formatNumber } from "../model/format";
import { CURVES } from "../model/optics";
import type { Auth } from "./Auth";
import { Chart } from "./Chart";

const X = CURVES[0]!;
const Y = CURVES[1]!;
const GREY = { color: "#6b7280", darkColor: "#9ca3af" };

type Operation = ProposalRequest["operation"];

const OPERATIONS: { key: Operation; label: string; title: string }[] = [
  { key: "orbit_correction", label: "Orbit correction", title: "SVD (damped) or MICADO on the model's orbit response matrix, from the BPMs' latest slow orbit" },
  { key: "local_bump", label: "Local bump", title: "A closed bump: three correctors for an offset, four for an offset and an angle" },
  { key: "tune_move", label: "Tune move", title: "The quadrupole knob that moves the tunes with the least beta-beating and dispersion change" },
];

const STATES: Record<string, string> = {
  proposed: "proposed",
  applied: "applied",
  verified: "verified",
  discarded: "discarded",
  stale: "stale: the machine changed",
};

interface Props {
  machine: MachineEntry | null;
  auth: Auth | null;
  active: boolean;
  tabs: ReactNode;
  scheme: string;
}

export function OperateView(props: Props) {
  const machine = props.machine;
  const [operation, setOperation] = useState<Operation>("orbit_correction");
  const [plane, setPlane] = useState<"x" | "y">("x");
  const [source, setSource] = useState<"measured" | "model">("measured");
  const [method, setMethod] = useState<"svd" | "micado">("svd");
  const [count, setCount] = useState("3");
  const [at, setAt] = useState("");
  const [amplitude, setAmplitude] = useState("1");
  const [angle, setAngle] = useState("");
  const [dqx, setDqx] = useState("0.01");
  const [dqy, setDqy] = useState("0");
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [proposals, setProposals] = useState<ProposalEntry[]>([]);
  const [tunes, setTunes] = useState<TuneReading | null>(null);
  const [open, setOpen] = useState<string | null>(null);

  const id = machine?.id ?? null;
  const refresh = useCallback(() => {
    if (!id) return;
    va.proposals(id).then(setProposals, () => undefined);
    va.tunes(id).then(setTunes, () => setTunes(null));
  }, [id]);

  useEffect(() => {
    setProposals([]);
    setTunes(null);
    if (!props.active || !id) return;
    refresh();
    const timer = window.setInterval(refresh, 2000); // an agent's proposals appear as it makes them
    return () => window.clearInterval(timer);
  }, [props.active, id, refresh]);

  const act = async (why: string, action: () => Promise<unknown>) => {
    setError(null);
    setBusy(true);
    try {
      await (props.auth ? props.auth.withLogin(why, action) : action());
      refresh();
    } catch (e) {
      if (!(e instanceof Error && e.message === "login cancelled")) setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const propose = () => {
    if (!machine) return;
    const request: Partial<ProposalRequest> & Pick<ProposalRequest, "operation"> = { operation, plane, reason };
    if (operation === "orbit_correction") Object.assign(request, { source, method, count: Number(count) || 3 });
    if (operation === "local_bump")
      Object.assign(request, { at: at.trim(), amplitude_mm: Number(amplitude), angle_mrad: angle.trim() === "" ? null : Number(angle) });
    if (operation === "tune_move") Object.assign(request, { dqx: Number(dqx) || 0, dqy: Number(dqy) || 0 });
    void act(`To compute a proposal on ${machine.name}, log in.`, async () => {
      const made = await va.propose(machine.id, request);
      setOpen(made.id);
    });
  };

  const bar = (
    <div className="optics-bar bpm-bar">
      {props.tabs}
      {machine ? <span className="muted">{machine.name}</span> : null}
      {tunes ? (
        <span className="muted" data-testid="tune-monitor" title="The ring's tune monitor: the beam's own motion, fractional tunes folded into 0 to 0.5">
          tune monitor {tunes.x.toFixed(4)}, {tunes.y.toFixed(4)}
          {tunes.settled ? "" : " (ramping)"}
        </span>
      ) : null}
    </div>
  );
  if (!machine || machine.kind !== "ring")
    return (
      <div className="bpm check">
        {bar}
        <div className="check-body">
          <div className="optics-message">
            <p>
              {machine
                ? `${machine.name} runs a route; operations are for rings so far.`
                : "Operations work on a virtual accelerator's ring: choose a machine in the top bar."}
            </p>
          </div>
        </div>
      </div>
    );

  return (
    <div className="bpm check" data-testid="operate">
      {bar}
      <div className="check-body operate-body">
        <form
          className="operate-form"
          aria-label="New proposal"
          onSubmit={(e) => {
            e.preventDefault();
            propose();
          }}
        >
          <select aria-label="Operation" value={operation} onChange={(e) => setOperation(e.target.value as Operation)}>
            {OPERATIONS.map((o) => (
              <option key={o.key} value={o.key} title={o.title}>
                {o.label}
              </option>
            ))}
          </select>
          {operation !== "tune_move" ? (
            <select aria-label="Plane" value={plane} onChange={(e) => setPlane(e.target.value as "x" | "y")}>
              <option value="x">horizontal</option>
              <option value="y">vertical</option>
            </select>
          ) : null}
          {operation === "orbit_correction" ? (
            <>
              <select aria-label="Orbit from" value={source} onChange={(e) => setSource(e.target.value as "measured" | "model")}>
                <option value="measured">from the BPMs</option>
                <option value="model">from the model</option>
              </select>
              <select aria-label="Method" value={method} onChange={(e) => setMethod(e.target.value as "svd" | "micado")}>
                <option value="svd">SVD, damped</option>
                <option value="micado">MICADO</option>
              </select>
              {method === "micado" ? (
                <label>
                  <input type="number" min={1} max={50} aria-label="Correctors" value={count} onChange={(e) => setCount(e.target.value)} /> correctors
                </label>
              ) : null}
            </>
          ) : null}
          {operation === "local_bump" ? (
            <>
              <label>
                at <input aria-label="Bump at" placeholder="a BPM's name" value={at} onChange={(e) => setAt(e.target.value)} />
              </label>
              <label>
                <input type="number" step="any" aria-label="Amplitude" value={amplitude} onChange={(e) => setAmplitude(e.target.value)} /> mm
              </label>
              <label title="Leave empty for an offset alone (three correctors); an angle takes four">
                <input type="number" step="any" aria-label="Angle" placeholder="angle" value={angle} onChange={(e) => setAngle(e.target.value)} /> mrad
              </label>
            </>
          ) : null}
          {operation === "tune_move" ? (
            <>
              <label>
                ΔQx <input type="number" step="any" aria-label="Change of Qx" value={dqx} onChange={(e) => setDqx(e.target.value)} />
              </label>
              <label>
                ΔQy <input type="number" step="any" aria-label="Change of Qy" value={dqy} onChange={(e) => setDqy(e.target.value)} />
              </label>
            </>
          ) : null}
          <input aria-label="Reason" placeholder="why (for the log)" value={reason} onChange={(e) => setReason(e.target.value)} />
          <button type="submit" disabled={busy || (operation === "local_bump" && !at.trim())}>
            {busy ? "Computing…" : "Propose"}
          </button>
        </form>
        {error ? <p className="admin-error" role="alert">{error}</p> : null}
        {proposals.length === 0 ? (
          <p className="muted">No proposals yet. Nothing is written until a person applies one.</p>
        ) : null}
        <ul className="proposals">
          {proposals.map((p) => (
            <ProposalCard
              key={p.id}
              entry={p}
              open={open === p.id}
              busy={busy}
              scheme={props.scheme}
              onToggle={() => setOpen((o) => (o === p.id ? null : p.id))}
              onApply={() => {
                const failed = p.proposal.checks.filter((c) => !c.ok).map((c) => c.name);
                if (failed.length && !window.confirm(`Checks not passed: ${failed.join(", ")}. Apply anyway?`)) return;
                void act(`To apply proposal ${p.id} to ${machine.name}, log in.`, () => va.apply(p.id, reason));
              }}
              onVerify={() => void act(`To verify proposal ${p.id}, log in.`, () => va.verify(p.id).then((v) => setProposals((all) => all.map((q) => (q.id === v.id ? v : q)))))}
              onDiscard={() => void act(`To discard proposal ${p.id}, log in.`, () => va.discard(p.id, reason))}
            />
          ))}
        </ul>
      </div>
    </div>
  );
}

function ProposalCard(props: {
  entry: ProposalEntry;
  open: boolean;
  busy: boolean;
  scheme: string;
  onToggle(): void;
  onApply(): void;
  onVerify(): void;
  onDiscard(): void;
}) {
  const { entry } = props;
  const p = entry.proposal;
  const ok = p.checks.every((c) => c.ok);
  const verification = entry.verification as Record<string, unknown> | null;
  return (
    <li className={`proposal state-${entry.state}`} data-testid="proposal">
      <div className="proposal-head">
        <button type="button" className="link" aria-expanded={props.open} onClick={props.onToggle}>
          {props.open ? "▾" : "▸"}
        </button>
        <span className={`proposal-state ${ok ? "ok" : "bad"}`}>{STATES[entry.state] ?? entry.state}</span>
        <span className="proposal-summary">{p.summary}</span>
        <span className="muted">
          {entry.agent ? "✦ agent of " : ""}
          {entry.user}, {new Date(entry.created).toLocaleTimeString()}
        </span>
        <span className="spacer" />
        {entry.state === "proposed" ? (
          <>
            <button type="button" disabled={props.busy} onClick={props.onApply} title="Write these changes to the machine (logged, ramped)">
              Apply
            </button>
            <button type="button" disabled={props.busy} onClick={props.onDiscard}>
              Discard
            </button>
          </>
        ) : null}
        {entry.state === "applied" || entry.state === "verified" ? (
          <button type="button" disabled={props.busy} onClick={props.onVerify} title="What the BPMs or the tune monitor show since, against the prediction">
            Verify
          </button>
        ) : null}
      </div>
      <ul className="proposal-checks">
        {p.checks.map((c) => (
          <li key={c.name} className={c.ok ? "ok" : "bad"} title={c.note}>
            {c.ok ? "✓" : "✗"} {c.name}
            {c.value === null || c.value === undefined ? "" : `: ${formatNumber(c.value)}`}
            {c.limit === null || c.limit === undefined ? "" : ` (limit ${formatNumber(c.limit)})`}
          </li>
        ))}
      </ul>
      {verification ? <Verification result={verification} /> : null}
      {props.open ? (
        <div className="proposal-detail">
          <Prediction entry={entry} scheme={props.scheme} />
          <table className="proposal-changes">
            <thead>
              <tr>
                <th>device</th>
                <th>channel</th>
                <th>now</th>
                <th>proposed</th>
                <th>change</th>
              </tr>
            </thead>
            <tbody>
              {p.changes.map((c) => (
                <tr key={c.channel}>
                  <td>{c.device}</td>
                  <td>{c.channel}</td>
                  <td>
                    {formatNumber(c.before)} {c.unit}
                  </td>
                  <td>
                    {formatNumber(c.after)} {c.unit}
                  </td>
                  <td>
                    {formatNumber(c.after - c.before)} {c.unit}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted">
            {p.method}; computed with {p.engine}.
          </p>
          <details>
            <summary>References</summary>
            <ul>
              {p.references.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          </details>
        </div>
      ) : null}
    </li>
  );
}

function Prediction({ entry, scheme }: { entry: ProposalEntry; scheme: string }) {
  const predicted = entry.proposal.predicted as Record<string, unknown>;
  if (entry.operation === "tune_move") {
    const pair = (v: unknown, i: number, digits: number) => {
      const row = (v as number[][])[i] ?? [];
      return `(${(row[0] ?? NaN).toFixed(digits)}, ${(row[1] ?? NaN).toFixed(digits)})`;
    };
    const beat = (predicted.beta_beating as number[]).map((b) => `${(100 * b).toFixed(2)} %`);
    return (
      <p>
        Tunes {pair(predicted.tunes, 0, 4)} → {pair(predicted.tunes, 1, 4)}; beta-beating at most {beat.join(" and ")}; dispersion changed by at most{" "}
        {formatNumber(predicted.dispersion_change_mm as number)} mm; chromaticity {pair(predicted.chromaticity, 0, 2)} → {pair(predicted.chromaticity, 1, 2)}.
      </p>
    );
  }
  const names = predicted.bpms as string[];
  const index = names.map((_, i) => i);
  const curve = predicted.plane === "y" ? Y : X;
  const series =
    entry.operation === "orbit_correction"
      ? [
          { label: "before", ...GREY, x: index, y: predicted.before as (number | null)[], style: "both" as const },
          { label: "after", color: curve.color, darkColor: curve.darkColor, x: index, y: predicted.after as (number | null)[], style: "both" as const },
        ]
      : [{ label: "change", color: curve.color, darkColor: curve.darkColor, x: index, y: predicted.change as number[], style: "both" as const }];
  return (
    <Chart
      title={entry.operation === "orbit_correction" ? "Orbit at the BPMs, predicted" : "Orbit change at the BPMs, predicted (closed outside the bump)"}
      xLabel="BPM, in beam order"
      yLabel={`${predicted.plane} (mm)`}
      series={series}
      names={names}
      scheme={scheme}
      testId="proposal-chart"
    />
  );
}

function Verification({ result }: { result: Record<string, unknown> }) {
  if (!result.ready)
    return (
      <p className="muted" data-testid="verification">
        Not measured yet: the settings ramp and the machine measures until turn {String(result.until)}; verify again then.
      </p>
    );
  const good = Boolean(result.ok);
  const planes = result.planes as Record<string, Record<string, number>> | undefined;
  return (
    <p className={good ? "verified ok" : "verified bad"} data-testid="verification">
      {good ? "✓ Verified" : "✗ Not as predicted"}
      {planes
        ? `: tune change measured ${Object.entries(planes)
            .map(([plane, v]) => `${plane} ${formatNumber(v.measured_change!)} (predicted ${formatNumber(v.predicted_change!)})`)
            .join(", ")}`
        : result.rms_difference_mm !== undefined
          ? `: the orbit change measured differs from the prediction by ${formatNumber(result.rms_difference_mm as number)} mm rms (predicted change ${formatNumber(
              result.rms_predicted_change_mm as number,
            )} mm rms)`
          : `: ${String(result.note ?? "")}`}
    </p>
  );
}
