// The selected element's setpoints on the machine the viewer works with (ADR
// 0008): the value now and, where someone changed it, the original from the
// machine files and who. Anyone sees them; setting one asks for a login the
// first time, and the change goes through the same checked, logged path as
// every other.

import { useCallback, useEffect, useState } from "react";

import { va, VaError, type MachineEntry, type SettingRow, type WriteResult } from "../api/va";
import { formatNumber } from "../model/format";
import { Cancelled, type Auth } from "./Auth";

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
}

export function Setpoints({ occurrence, machine, auth }: Props) {
  const [rows, setRows] = useState<SettingRow[]>([]);
  const [values, setValues] = useState<Record<string, string>>({});
  const [reason, setReason] = useState("");
  const [outcome, setOutcome] = useState<WriteResult | string | null>(null);
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
  }, [occurrence, id]);

  if (!machine) return null;
  const mine = rows.filter((r) => r.occurrences.includes(occurrence));
  const set = (channel: string) => {
    const value = Number(values[channel]);
    if (!Number.isFinite(value)) return setOutcome(`${values[channel]} is not a number`);
    auth
      .withLogin(`To set ${channel} on ${machine.name}, log in.`, () => va.write(machine.id, [{ channel, value }], reason))
      .then(
        (r) => {
          setOutcome(r);
          if (r.applied) setValues((v) => ({ ...v, [channel]: "" }));
          load();
        },
        (e: Error) => {
          if (e instanceof Cancelled) return;
          setOutcome(e instanceof VaError && e.status === 403 ? `You cannot change ${machine.name}: ${e.message}` : e.message);
        },
      );
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
                  <strong>{formatNumber(r.value, 6)}</strong> {r.unit}
                  {r.changed ? (
                    <span className="muted">
                      {" "}
                      (original {formatNumber(r.original, 6)}; set by {r.by})
                    </span>
                  ) : null}
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
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      {mine.length && Object.values(values).some(Boolean) ? (
        <input className="setpoint-reason" placeholder="Why (goes into the log)" aria-label="Reason" value={reason} onChange={(e) => setReason(e.target.value)} />
      ) : null}
      {typeof outcome === "string" ? <p className="admin-error">{outcome}</p> : null}
      {outcome && typeof outcome !== "string" ? (
        <p className={outcome.applied ? "muted" : "admin-error"} data-testid="setpoint-outcome">
          {outcome.applied
            ? `Set, logged as change ${outcome.correlation}.${
                machine.kind === "ring" && machine.ramp_turns ? ` It ramps in over ${machine.ramp_turns} turns.` : ""
              }`
            : `Refused: ${outcome.writes
                .map((w) => w.refusal)
                .filter(Boolean)
                .join("; ")}`}
        </p>
      ) : null}
    </section>
  );
}
