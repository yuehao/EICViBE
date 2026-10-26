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
//
// The tab is for the next operation, not for history: it shows only what still
// waits for a person (proposed, or applied and not yet verified), and the
// proposal verified here last, until the next is proposed. What was done is in
// the log.
//
// The operations and their forms come from the virtual accelerator's catalog
// (GET /operations, ADR 0017): each operation's parameters as a JSON Schema,
// with x-form hints for how a field is shown. A new operation needs no change
// here.

import { useCallback, useEffect, useState, type ReactNode } from "react";

import { va, type MachineEntry, type OperationEntry, type ProposalEntry, type ProposalRequest, type TuneReading } from "../api/va";
import { formatNumber } from "../model/format";
import { CURVES } from "../model/optics";
import type { Auth } from "./Auth";
import { Chart } from "./Chart";

const X = CURVES[0]!;
const Y = CURVES[1]!;
const GREY = { color: "#6b7280", darkColor: "#9ca3af" };

/** How a parameter is shown: text before and after its input, an enumeration's names, when it is shown. */
interface FormHints {
  prefix?: string;
  unit?: string;
  labels?: Record<string, string>;
  placeholder?: string;
  when?: Record<string, string>;
  advanced?: boolean;
}

/** A parameter's JSON Schema, as pydantic writes it, so far as the form reads it. */
interface Parameter {
  type?: string;
  enum?: string[];
  anyOf?: Parameter[];
  title?: string;
  description?: string;
  default?: unknown;
  examples?: unknown[];
  minimum?: number;
  maximum?: number;
  "x-form"?: FormHints;
}

interface Parameters {
  properties: Record<string, Parameter>;
  required?: string[];
}

const kindOf = (p: Parameter): string => (p.enum ? "enum" : (p.type ?? p.anyOf?.find((a) => a.type !== "null")?.type ?? "string"));
const nullable = (p: Parameter) => Boolean(p.anyOf?.some((a) => a.type === "null"));
const initial = (p: Parameter) => {
  const value = p.examples?.[0] ?? p.default;
  return value === null || value === undefined ? "" : String(value);
};

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
  const [operations, setOperations] = useState<OperationEntry[]>([]);
  const [chosen, setChosen] = useState("");
  const [values, setValues] = useState<Record<string, string>>({}); // by parameter, kept across operations
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [proposals, setProposals] = useState<ProposalEntry[]>([]); // those waiting for a person
  const [finished, setFinished] = useState<ProposalEntry | null>(null); // verified here, until the next proposal
  const [tunes, setTunes] = useState<TuneReading | null>(null);
  const [open, setOpen] = useState<string | null>(null);

  const id = machine?.id ?? null;
  const running = machine?.state === "running";
  const refresh = useCallback(() => {
    if (!id) return;
    va.proposals(id, true).then(setProposals, () => undefined);
  }, [id]);

  useEffect(() => {
    setProposals([]);
    setFinished(null);
    if (!props.active || !id) return;
    refresh();
    const timer = window.setInterval(refresh, 2000); // an agent's proposals appear as it makes them
    return () => window.clearInterval(timer);
  }, [props.active, id, refresh]);

  // The tune monitor is a running ring's: a stopped one has none to ask for. Its reading changes every
  // 1024 turns, so it is asked for less often than the proposals.
  useEffect(() => {
    setTunes(null);
    if (!props.active || !id || !running) return;
    const read = () => va.tunes(id).then(setTunes, () => setTunes(null));
    read();
    const timer = window.setInterval(read, 5000);
    return () => window.clearInterval(timer);
  }, [props.active, id, running]);

  useEffect(() => {
    if (props.active && operations.length === 0) va.operations().then(setOperations, (e) => setError(String(e)));
  }, [props.active, operations.length]);

  const usable = operations.filter((o) => machine && o.machines.includes(machine.kind));
  const operation = usable.find((o) => o.name === chosen) ?? usable[0];
  const parameters = (operation?.parameters ?? { properties: {} }) as unknown as Parameters;
  const valueOf = (name: string) => values[name] ?? initial(parameters.properties[name] ?? {});
  const shown = Object.entries(parameters.properties).filter(
    ([, p]) => !p["x-form"]?.advanced && kindOf(p) !== "array" && Object.entries(p["x-form"]?.when ?? {}).every(([k, v]) => valueOf(k) === v),
  );
  const missing = (parameters.required ?? []).some((name) => shown.some(([n]) => n === name) && !valueOf(name).trim());

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
    if (!machine || !operation) return;
    // The fields shown, as their types say; the others take their defaults.
    const request: ProposalRequest = { operation: operation.name, reason };
    for (const [name, p] of shown) {
      const text = valueOf(name).trim();
      if (text === "") {
        if (nullable(p)) request[name] = null;
        continue;
      }
      request[name] = kindOf(p) === "number" || kindOf(p) === "integer" ? Number(text) : text;
    }
    void act(`To compute a proposal on ${machine.name}, log in.`, async () => {
      const made = await va.propose(machine.id, request);
      setFinished(null);
      setOpen(made.id);
    });
  };

  const listed = [...proposals.filter((p) => p.id !== finished?.id), ...(finished ? [finished] : [])].sort((a, b) =>
    b.created.localeCompare(a.created),
  );

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
  if (!machine || (operations.length > 0 && usable.length === 0))
    return (
      <div className="bpm check">
        {bar}
        <div className="check-body">
          <div className="optics-message">
            <p>
              {machine
                ? `${machine.name} runs a ${machine.kind}; no operation runs on one so far.`
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
          <select aria-label="Operation" value={operation?.name ?? ""} onChange={(e) => setChosen(e.target.value)}>
            {usable.map((o) => (
              <option key={o.name} value={o.name} title={o.summary}>
                {o.label}
              </option>
            ))}
          </select>
          {shown.map(([name, p]) => (
            <ParameterField key={name} parameter={p} value={valueOf(name)} onChange={(v) => setValues((all) => ({ ...all, [name]: v }))} />
          ))}
          <input aria-label="Reason" placeholder="why (for the log)" value={reason} onChange={(e) => setReason(e.target.value)} />
          <button type="submit" disabled={busy || !operation || missing}>
            {busy ? "Computing…" : "Propose"}
          </button>
        </form>
        {error ? <p className="admin-error" role="alert">{error}</p> : null}
        {listed.length === 0 ? (
          <p className="muted">Nothing waiting. Propose an operation: nothing is written until a person applies it.</p>
        ) : null}
        <ul className="proposals">
          {listed.map((p) => (
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
              onVerify={() =>
                void act(`To verify proposal ${p.id}, log in.`, () =>
                  va.verify(p.id).then((v) => (v.state === "verified" ? setFinished(v) : setProposals((all) => all.map((q) => (q.id === v.id ? v : q))))),
                )
              }
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

/** One parameter's input, as its schema and x-form hints say: a select for an enumeration, else a number or text. */
function ParameterField({ parameter: p, value, onChange }: { parameter: Parameter; value: string; onChange(value: string): void }) {
  const hints = p["x-form"] ?? {};
  const label = p.title ?? "";
  const kind = kindOf(p);
  const input = p.enum ? (
    <select aria-label={label} value={value} onChange={(e) => onChange(e.target.value)}>
      {p.enum.map((v) => (
        <option key={v} value={v}>
          {hints.labels?.[v] ?? v}
        </option>
      ))}
    </select>
  ) : (
    <input
      type={kind === "number" || kind === "integer" ? "number" : "text"}
      step={kind === "number" ? "any" : undefined}
      min={p.minimum}
      max={p.maximum}
      aria-label={label}
      placeholder={hints.placeholder}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    />
  );
  if (!hints.prefix && !hints.unit) return input;
  return (
    <label title={p.description}>
      {hints.prefix ? `${hints.prefix} ` : null}
      {input}
      {hints.unit ? ` ${hints.unit}` : null}
    </label>
  );
}

/** What a proposal predicts, by what it holds: tunes, or the orbit at the BPMs (before and after, or its change). */
function Prediction({ entry, scheme }: { entry: ProposalEntry; scheme: string }) {
  const predicted = entry.proposal.predicted as Record<string, unknown>;
  if (Array.isArray(predicted.tunes)) {
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
  if (!Array.isArray(predicted.bpms)) return null;
  const names = predicted.bpms as string[];
  const index = names.map((_, i) => i);
  const curve = predicted.plane === "y" ? Y : X;
  const change = Array.isArray(predicted.change);
  const series = change
    ? [{ label: "change", color: curve.color, darkColor: curve.darkColor, x: index, y: predicted.change as number[], style: "both" as const }]
    : [
        { label: "before", ...GREY, x: index, y: predicted.before as (number | null)[], style: "both" as const },
        { label: "after", color: curve.color, darkColor: curve.darkColor, x: index, y: predicted.after as (number | null)[], style: "both" as const },
      ];
  return (
    <Chart
      title={change ? "Orbit change at the BPMs, predicted" : "Orbit at the BPMs, predicted"}
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
