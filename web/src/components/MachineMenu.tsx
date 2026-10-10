// Which machine of the virtual accelerator the viewer works with (ADR 0008):
// the inspector's setpoints and the Orbit tab follow it. A user may make a
// sandbox of their own from it, to try changes without disturbing anyone, and
// start, stop or remove their sandboxes; doing so asks them to log in.

import { atLeast, va, VaError, type MachineEntry } from "../api/va";
import { Cancelled, type Auth } from "./Auth";

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
          {!running ? (
            <button type="button" className="danger" onClick={() => window.confirm(`Remove ${chosen.name}? Its log stays.`) && control(() => va.remove(chosen.id))}>
              Remove
            </button>
          ) : null}
        </>
      ) : null}
    </span>
  );
}
