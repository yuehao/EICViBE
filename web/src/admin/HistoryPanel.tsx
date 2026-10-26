// The administration page's History section: what the virtual accelerator keeps
// besides its logs, and deleting it. Every run's BPM history stays on disk
// (the state folder's runs/) and every proposal in the service's database until
// an administrator deletes them here. A running machine's run is kept, and so
// are the proposals still waiting for a person unless asked. The write and
// action logs are append-only (ADR 0008): they keep what was done, these
// deletions included.

import { useCallback, useEffect, useState } from "react";

import { va, type HistoryView } from "../api/va";

const FINISHED = ["verified", "discarded", "stale"] as const;
const many = (n: number, what: string) => `${n} ${what}${n === 1 ? "" : "s"}`;
const WAITING = ["proposed", "applied"] as const;

/** Bytes, as a disk's size is read: 1.2 GB. */
export function size(bytes: number): string {
  const units = ["B", "kB", "MB", "GB", "TB"];
  let value = bytes;
  let k = 0;
  while (value >= 1000 && k < units.length - 1) {
    value /= 1000;
    k += 1;
  }
  return `${k === 0 ? value : value.toFixed(value < 10 ? 1 : 0)} ${units[k]}`;
}

export function HistoryPanel({ onError }: { onError(e: unknown): void }) {
  const [history, setHistory] = useState<HistoryView | null>(null);
  const [reason, setReason] = useState("");
  const [alsoWaiting, setAlsoWaiting] = useState(false);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState<string | null>(null);

  const load = useCallback(() => va.history().then(setHistory, onError), [onError]);
  useEffect(() => {
    void load();
  }, [load]);

  const act = (action: () => Promise<string>) => {
    setBusy(true);
    setDone(null);
    action()
      .then(setDone, onError)
      .finally(() => {
        setBusy(false);
        void load();
      });
  };

  if (!history) return <section>Loading…</section>;
  const past = history.runs.filter((r) => !r.in_use);
  const total = history.runs.reduce((sum, r) => sum + r.bytes, 0);
  const pastBytes = past.reduce((sum, r) => sum + r.bytes, 0);
  const count = (states: readonly string[]) => states.reduce((n, s) => n + (history.proposals[s] ?? 0), 0);
  const finished = count(FINISHED);
  const waiting = count(WAITING);
  const doomed = finished + (alsoWaiting ? waiting : 0);
  const states = (list: readonly string[]) => list.map((s) => `${s} ${history.proposals[s] ?? 0}`).join(", ");

  return (
    <section data-testid="history">
      <h2>History</h2>
      <p className="muted">
        What the virtual accelerator keeps besides its logs: each run's BPM history on disk, and the proposals of the Operate tab. Deleting them frees
        the disk and clears the lists. The write and action logs are append-only: they keep what was done, these deletions included.
      </p>
      <input placeholder="Why (goes into the log)" aria-label="Reason" value={reason} onChange={(e) => setReason(e.target.value)} />
      {done ? (
        <p data-testid="history-done" role="status">
          {done}
        </p>
      ) : null}

      <h3>Runs</h3>
      <p data-testid="runs-summary">
        {many(history.runs.length, "run")}, {size(total)} on disk; {past.length} past ({size(pastBytes)}), {history.runs.length - past.length} in use by
        running machines, which are kept.
      </p>
      {history.runs.length ? (
        <table className="admin-table" data-testid="runs">
          <thead>
            <tr>
              <th>run</th>
              <th>machine</th>
              <th>engine</th>
              <th>started</th>
              <th>size</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {history.runs.map((r) => (
              <tr key={r.id}>
                <td>{r.id}</td>
                <td>{r.machine ?? "–"}</td>
                <td>{r.engine ?? "–"}</td>
                <td>{r.started ? new Date(r.started).toLocaleString() : "–"}</td>
                <td>{size(r.bytes)}</td>
                <td className="admin-actions">
                  {r.in_use ? (
                    <span className="muted">in use</span>
                  ) : (
                    <button
                      type="button"
                      className="danger"
                      disabled={busy}
                      onClick={() =>
                        window.confirm(`Delete run ${r.id} (${size(r.bytes)})? Its BPM history goes; the logs stay.`) &&
                        act(() => va.deleteRun(r.id, reason).then((d) => `Deleted ${d.deleted.join(", ")}, ${size(d.bytes)}.`))
                      }
                    >
                      Delete
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      <button
        type="button"
        className="danger"
        disabled={busy || past.length === 0}
        onClick={() =>
          window.confirm(`Delete ${many(past.length, "past run")} (${size(pastBytes)})? Their BPM histories go; the logs stay.`) &&
          act(() => va.deleteRuns(reason).then((d) => `Deleted ${many(d.deleted.length, "run")}, ${size(d.bytes)}.`))
        }
      >
        Delete past runs
      </button>

      <h3>Proposals</h3>
      <p data-testid="proposals-summary">
        Finished: {states(FINISHED)}. Waiting for a person: {states(WAITING)}.
      </p>
      <label>
        <input type="checkbox" checked={alsoWaiting} onChange={(e) => setAlsoWaiting(e.target.checked)} /> also those still waiting for a person
      </label>{" "}
      <button
        type="button"
        className="danger"
        disabled={busy || doomed === 0}
        onClick={() =>
          window.confirm(`Delete ${many(doomed, "proposal")}? The logs keep what was proposed and done.`) &&
          act(() => va.deleteProposals(alsoWaiting, reason).then((d) => `Deleted ${many(d.deleted, "proposal")}.`))
        }
      >
        Delete {many(doomed, "proposal")}
      </button>
    </section>
  );
}
