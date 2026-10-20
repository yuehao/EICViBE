// Which machine of the virtual accelerator the viewer works with (ADR 0008):
// the inspector's setpoints and the Orbit tab follow it. A user may make a
// sandbox of their own from it, to try changes without disturbing anyone, and
// start, stop or remove their sandboxes, or give them other errors (ADR 0011);
// doing so asks them to log in. A ring with a ramp profile shows where it is in
// its ramp, and an operator ramps it from here (ADR 0013).

import { useState } from "react";

import { atLeast, cycleText, va, VaError, type MachineEntry } from "../api/va";
import { Cancelled, type Auth } from "./Auth";
import { ErrorsEditor } from "./ErrorsEditor";

interface Props {
  machines: MachineEntry[];
  chosen: MachineEntry | null;
  auth: Auth;
  onChoose(id: string): void;
  onChanged(): void;
  onNotice(message: string): void;
}

const tracks = (m: MachineEntry) => `the ${m.kind === "route" ? "route" : "ring"} ${m.target}`;

export function MachineMenu({ machines, chosen, auth, onChoose, onChanged, onNotice }: Props) {
  const [errors, setErrors] = useState(false);
  if (!machines.length) return null;
  const me = auth.me;
  const mine = !!chosen && !chosen.shared && !!me?.logged_in && chosen.owner === me.name;
  const fail = (e: unknown) => {
    if (e instanceof Cancelled) return;
    onNotice(e instanceof VaError && e.status === 403 ? `You cannot do that: ${e.message}` : (e as Error).message);
  };
  const sandbox = () =>
    chosen &&
    auth
      .withLogin(`To make a sandbox of ${chosen.name}, log in.`, async () => {
        const made = await va.fork(chosen.id);
        await va.start(made.id, "a sandbox to try changes");
        return made;
      })
      .then((made) => {
        onChoose(made.id);
        onChanged();
      }, fail);
  const control = (action: () => Promise<unknown>) =>
    auth.withLogin("To change your sandbox, log in.", action).then(onChanged, fail);
  const running = chosen?.state === "running" || chosen?.state === "starting";
  const ramp = () =>
    chosen &&
    auth.withLogin(`To ramp ${chosen.name}, log in.`, () => va.ramp(chosen.id, "ramp from the viewer")).then(onChanged, fail);

  return (
    <span className="machine-menu">
      {chosen ? <i className={`state-dot ${chosen.state}`} title={`${chosen.name} is ${chosen.state}`} /> : null}
      {chosen?.warnings.length ? (
        <span
          className="admin-error"
          title={`Machine ${chosen.name}, which tracks ${tracks(chosen)}:\n${chosen.warnings.join("\n")}`}
          aria-label={`${chosen.name} has warnings`}
        >
          ⚠
        </span>
      ) : null}
      <select aria-label="Machine" value={chosen?.id ?? ""} onChange={(e) => onChoose(e.target.value)} title="The machine whose setpoints and orbit the viewer shows">
        {machines.map((m) => (
          <option key={m.id} value={m.id}>
            {m.name} · {m.target}
            {m.shared ? "" : ` (${m.owner}'s sandbox)`}
          </option>
        ))}
      </select>
      {chosen && cycleText(chosen) ? (
        <span className="muted" data-testid="machine-cycle" title={`${chosen.name}'s ramp ${chosen.ramp}`}>
          {cycleText(chosen)}
        </span>
      ) : null}
      {chosen?.ramp && chosen.cycle?.state === "injection" && (!me?.logged_in || atLeast(me.role, "operator")) ? (
        <button type="button" onClick={ramp} title={`Ramp ${chosen.target} as its ramp ${chosen.ramp} says: to extraction, then a fresh bunch at injection`}>
          Ramp
        </button>
      ) : null}
      {chosen && (!me?.logged_in || atLeast(me.role, "physicist")) ? (
        <button type="button" onClick={sandbox} title={`A machine of your own with ${chosen.name}'s settings now, to try changes without disturbing anyone`}>
          Sandbox
        </button>
      ) : null}
      {mine && chosen ? (
        <>
          <button type="button" onClick={() => control(() => (running ? va.stop(chosen.id) : va.start(chosen.id)))}>
            {running ? "Stop" : "Start"}
          </button>
          <button type="button" title={`${chosen.name}'s magnet, supply and BPM errors`} onClick={() => setErrors(true)}>
            Errors…
          </button>
          {!running ? (
            <button type="button" className="danger" onClick={() => window.confirm(`Remove ${chosen.name}? Its log stays.`) && control(() => va.remove(chosen.id))}>
              Remove
            </button>
          ) : null}
        </>
      ) : null}
      {errors && mine && chosen ? (
        <div className="dialog-backdrop" onMouseDown={(e) => e.target === e.currentTarget && setErrors(false)}>
          <div className="dialog errors-dialog" role="dialog" aria-label={`Errors of ${chosen.name}`} onKeyDown={(e) => e.key === "Escape" && setErrors(false)}>
            <h3>
              Errors of {chosen.name} <span className="muted">({chosen.state})</span>
            </h3>
            <ErrorsEditor machine={chosen} onError={fail} onSaved={onChanged} />
            <div className="dialog-buttons">
              <button type="button" onClick={() => setErrors(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </span>
  );
}
