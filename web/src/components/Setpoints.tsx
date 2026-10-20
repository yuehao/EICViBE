// The selected element's setpoints on the machine the viewer works with (ADR
// 0008): the value now and, where someone changed it, the original from the
// machine files and who. Anyone sees them; setting one asks for a login the
// first time, and the change goes through the same checked, logged path as
// every other.
//
// Only a setting that drives this element alone is changed here (its own
// supply, or its trim). One that drives several (a string of magnets in
// series, an RF station's cavities) belongs to its power supply: here it is
// shown, and changed in the supply's own window, which lists every magnet the
// change moves.

import { useCallback, useEffect, useState } from "react";

import { va, VaError, type MachineEntry, type SettingRow, type WriteResult } from "../api/va";
import { formatNumber } from "../model/format";
import { Cancelled, type Auth } from "./Auth";
import { BeyondLimits, outside, SourceBadge } from "./SupplyMarks";

let enabled: Promise<boolean> | null = null;
/** Whether this server runs a virtual accelerator; asked once per page. */
export function vaEnabled(): Promise<boolean> {
  enabled ??= va.status().then(
    (s) => s.enabled,
    () => false,
  );
  return enabled;
}

interface Props {
  occurrence: string;
  machine: MachineEntry | null;
  auth: Auth;
  /** The name of an occurrence, for the magnets a supply drives. */
  describe(id: string): { name: string } | null;
}

/** The outcome of a write, or why it did not happen. */
type Outcome = WriteResult | string | null;

export function Setpoints({ occurrence, machine, auth, describe }: Props) {
  const [rows, setRows] = useState<SettingRow[]>([]);
  const [values, setValues] = useState<Record<string, string>>({});
  const [reason, setReason] = useState("");
  const [outcome, setOutcome] = useState<Outcome>(null);
  const [supply, setSupply] = useState<SettingRow | null>(null);
  const id = machine?.id ?? null;

  const load = useCallback(() => {
    if (id) va.settings(id).then(setRows, () => undefined);
  }, [id]);
  useEffect(() => {
    load();
    const timer = window.setInterval(load, 5000);
    return () => window.clearInterval(timer);
  }, [load]);
  useEffect(() => {
    setValues({});
    setOutcome(null);
    setSupply(null);
  }, [occurrence, id]);

  if (!machine) return null;
  const mine = rows.filter((r) => r.occurrences.includes(occurrence));
  const write = (channel: string, value: number, why: string): Promise<Outcome> =>
    auth.withLogin(`To set ${channel} on ${machine.name}, log in.`, () => va.write(machine.id, [{ channel, value }], why)).then(
      (r) => {
        load();
        return r;
      },
      (e: Error) => {
        if (e instanceof Cancelled) return null;
        return e instanceof VaError && e.status === 403 ? `You cannot change ${machine.name}: ${e.message}` : e.message;
      },
    );
  const set = (channel: string) => {
    const value = Number(values[channel]);
    if (!Number.isFinite(value)) return setOutcome(`${values[channel]} is not a number`);
    void write(channel, value, reason).then((r) => {
      setOutcome(r);
      if (r && typeof r !== "string" && r.applied) setValues((v) => ({ ...v, [channel]: "" }));
    });
  };

  return (
    <section className="setpoints" data-testid="setpoints">
      <h3>
        Setpoints on {machine.name} <span className="muted">({machine.state})</span>
      </h3>
      {mine.length === 0 ? <p className="muted">No setting of this machine drives this element.</p> : null}
      {mine.length ? (
        <table className="kv channels">
          <tbody>
            {mine.map((r) => (
              <tr key={r.channel} className={r.changed ? "changed" : ""}>
                <th>
                  <code>{r.channel}</code>
                </th>
                <td>
                  <Value row={r} />
                  {r.occurrences.length > 1 ? (
                    <div className="setpoint-shared" data-testid="setpoint-shared">
                      <span className="muted">
                        drives {r.occurrences.length} {r.kind === "circuit" ? "magnets in series" : "elements"}: set it on the supply, not on one
                        of them
                      </span>{" "}
                      <button type="button" aria-label={`Power supply ${r.device}`} onClick={() => setSupply(r)}>
                        Power supply…
                      </button>
                    </div>
                  ) : (
                    <div className="setpoint-edit">
                      <input
                        aria-label={`New value of ${r.channel}`}
                        placeholder={formatNumber(r.value, 6)}
                        value={values[r.channel] ?? ""}
                        onChange={(e) => setValues((v) => ({ ...v, [r.channel]: e.target.value }))}
                        onKeyDown={(e) => e.key === "Enter" && values[r.channel] && set(r.channel)}
                      />
                      <button type="button" aria-label={`Set ${r.channel}`} disabled={!values[r.channel]} onClick={() => set(r.channel)}>
                        Set
                      </button>
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      {mine.length && Object.values(values).some(Boolean) ? (
        <input className="setpoint-reason" placeholder="Why (goes into the log)" aria-label="Reason" value={reason} onChange={(e) => setReason(e.target.value)} />
      ) : null}
      <OutcomeLine outcome={outcome} machine={machine} />
      {supply ? (
        <SupplyWindow
          row={rows.find((r) => r.channel === supply.channel) ?? supply}
          machine={machine}
          describe={describe}
          write={write}
          onClose={() => setSupply(null)}
        />
      ) : null}
    </section>
  );
}

function Value({ row }: { row: SettingRow }) {
  return (
    <>
      <strong className={outside(row.value, row.limits) ? "beyond" : undefined}>{formatNumber(row.value, 6)}</strong> {row.unit}
      <SourceBadge source={row.source} />
      {row.changed ? (
        <span className="muted">
          {" "}
          (original {formatNumber(row.original, 6)}; set by {row.by})
        </span>
      ) : null}
      {row.beyond && row.limits ? <BeyondLimits current={row.original} limits={row.limits} unit={row.unit} /> : null}
    </>
  );
}

function OutcomeLine({ outcome, machine }: { outcome: Outcome; machine: MachineEntry }) {
  if (typeof outcome === "string") return <p className="admin-error">{outcome}</p>;
  if (!outcome) return null;
  return (
    <p className={outcome.applied ? "muted" : "admin-error"} data-testid="setpoint-outcome">
      {outcome.applied
        ? `Set, logged as change ${outcome.correlation}.${machine.kind === "ring" && machine.ramp_turns ? ` It ramps in over ${machine.ramp_turns} turns.` : ""}`
        : `Refused: ${outcome.writes
            .map((w) => w.refusal)
            .filter(Boolean)
            .join("; ")}`}
    </p>
  );
}

/** A power supply's own window: its setting, and every magnet (or element) a change of it moves. */
function SupplyWindow(props: {
  row: SettingRow;
  machine: MachineEntry;
  describe(id: string): { name: string } | null;
  write(channel: string, value: number, why: string): Promise<Outcome>;
  onClose(): void;
}) {
  const { row, machine } = props;
  const [value, setValue] = useState("");
  const [reason, setReason] = useState("");
  const [outcome, setOutcome] = useState<Outcome>(null);
  const count = row.occurrences.length;
  const names = row.occurrences.map((o) => props.describe(o)?.name ?? o).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  const submit = () => {
    const number = Number(value);
    if (value.trim() === "" || !Number.isFinite(number)) return setOutcome(`${value} is not a number`);
    void props.write(row.channel, number, reason).then((r) => {
      setOutcome(r);
      if (r && typeof r !== "string" && r.applied) setValue("");
    });
  };
  return (
    <div className="dialog-backdrop" onMouseDown={(e) => e.target === e.currentTarget && props.onClose()}>
      <form
        className="dialog supply-window"
        role="dialog"
        aria-label={`Power supply ${row.device}`}
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
        onKeyDown={(e) => e.key === "Escape" && props.onClose()}
      >
        <h3>
          {row.device} <span className="muted">power supply on {machine.name}</span>
        </h3>
        <p>
          <code>{row.channel}</code>: <Value row={row} />
          {row.limits ? (
            <span className="muted">
              {" "}
              · limits {formatNumber(row.limits[0]!, 6)} … {formatNumber(row.limits[1]!, 6)} {row.unit}
            </span>
          ) : null}
        </p>
        <p>
          It drives <strong>{count}</strong> {row.kind === "circuit" ? "magnets in series" : "elements"}; a change moves them all:
        </p>
        <div className="supply-magnets" data-testid="supply-magnets">
          {names.join(", ")}
        </div>
        <label>
          New value ({row.unit}) <input autoFocus aria-label={`New value of ${row.channel}`} placeholder={formatNumber(row.value, 6)} value={value} onChange={(e) => setValue(e.target.value)} />
        </label>
        <label>
          Why <input aria-label="Reason for the supply" placeholder="goes into the log" value={reason} onChange={(e) => setReason(e.target.value)} />
        </label>
        <OutcomeLine outcome={outcome} machine={machine} />
        <div className="dialog-buttons">
          <button type="button" onClick={props.onClose}>
            Close
          </button>
          <button type="submit" disabled={!value.trim()}>
            Set all {count}
          </button>
        </div>
      </form>
    </div>
  );
}
