// Who is changing the machine, asked only when someone changes it (ADR 0008).
//
// Viewing needs nothing. The first time a user sets a value, or makes a
// sandbox, the viewer asks them to log in, then goes on with what they asked
// for; the browser keeps the login (a session cookie) for the day. Once logged
// in, the top bar shows who they are, with a menu to log out, make a token for
// notebooks or change the password.

import { useCallback, useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";

import { va, VaError, type Me } from "../api/va";

export interface Auth {
  /** Who is logged in; null until known (or without a virtual accelerator). */
  me: Me | null;
  /** The user, once logged in: asks for a login if needed, saying ``why``; rejects if they cancel. */
  ensure(why: string): Promise<Me>;
  /** Run ``action`` as a logged-in user, asking again once if the login has expired meanwhile. */
  withLogin<T>(why: string, action: () => Promise<T>): Promise<T>;
  logout(): Promise<void>;
}

interface Pending {
  why: string;
  resolve(me: Me): void;
  reject(error: Error): void;
}

export class Cancelled extends Error {
  constructor() {
    super("login cancelled");
  }
}

export function useAuth(enabled: boolean): { auth: Auth; dialog: ReactNode } {
  const [me, setMe] = useState<Me | null>(null);
  const [pending, setPending] = useState<Pending | null>(null);
  const meRef = useRef(me);
  meRef.current = me;

  useEffect(() => {
    if (enabled) va.me().then(setMe, () => undefined);
  }, [enabled]);

  const ensure = useCallback(
    (why: string) =>
      meRef.current?.logged_in
        ? Promise.resolve(meRef.current)
        : new Promise<Me>((resolve, reject) => setPending({ why, resolve, reject })),
    [],
  );
  const withLogin = useCallback(
    async <T,>(why: string, action: () => Promise<T>): Promise<T> => {
      await ensure(why);
      try {
        return await action();
      } catch (e) {
        if (!(e instanceof VaError) || e.status !== 401) throw e;
        setMe((m) => (m ? { ...m, logged_in: false, name: "anonymous", role: "viewer" } : m));
        meRef.current = null;
        await ensure(`Your login has expired. ${why}`);
        return action();
      }
    },
    [ensure],
  );
  const logout = useCallback(async () => {
    await va.logout();
    setMe(await va.me());
  }, []);

  const dialog = pending ? (
    <LoginDialog
      why={pending.why}
      onDone={(next) => {
        setMe(next);
        meRef.current = next;
        pending.resolve(next);
        setPending(null);
      }}
      onCancel={() => {
        pending.reject(new Cancelled());
        setPending(null);
      }}
    />
  ) : null;
  return { auth: { me, ensure, withLogin, logout }, dialog };
}

function LoginDialog({ why, onDone, onCancel }: { why: string; onDone(me: Me): void; onCancel(): void }) {
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const submit = (e: FormEvent) => {
    e.preventDefault();
    va.login(name, password).then(onDone, (err: Error) => setError(err.message));
  };
  return (
    <div className="dialog-backdrop login-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onCancel()}>
      <form className="dialog" role="dialog" aria-label="Log in" onSubmit={submit} onKeyDown={(e) => e.key === "Escape" && onCancel()}>
        <h3>Log in</h3>
        <p className="muted">{why} Changes to the machine are logged with who made them.</p>
        <label>
          User <input autoFocus aria-label="User name" value={name} onChange={(e) => setName(e.target.value)} autoComplete="username" />
        </label>
        <label>
          Password{" "}
          <input aria-label="Password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" />
        </label>
        {error ? <p className="admin-error">{error}</p> : null}
        <div className="dialog-buttons">
          <button type="button" onClick={onCancel}>
            Cancel
          </button>
          <button type="submit" disabled={!name || !password}>
            Log in
          </button>
        </div>
      </form>
    </div>
  );
}

/** Who is logged in, with a menu; nothing for someone who only views. */
export function AccountMenu({ auth }: { auth: Auth }) {
  const [open, setOpen] = useState(false);
  const [token, setToken] = useState<string | null>(null);
  const [label, setLabel] = useState("");
  const [kind, setKind] = useState<"api" | "agent">("api");
  const [old, setOld] = useState("");
  const [next, setNext] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const me = auth.me;
  if (!me?.logged_in) return null;
  return (
    <div className="notes-wrap">
      <button type="button" aria-expanded={open} onClick={() => setOpen((o) => !o)} title={`Logged in as ${me.name} (${me.role})`}>
        {me.name}
      </button>
      {open ? (
        <div className="popover account" role="dialog" aria-label="Account">
          <p>
            <strong>{me.name}</strong> <span className="muted">{me.role}</span>
          </p>
          <h3>Token for a notebook or an agent</h3>
          <form
            className="admin-row"
            onSubmit={(e) => {
              e.preventDefault();
              va.newToken(label, kind).then((t) => {
                setToken(t.token);
                setLabel("");
              }, (err: Error) => setMessage(err.message));
            }}
          >
            <input placeholder="what for" aria-label="Token label" value={label} onChange={(e) => setLabel(e.target.value)} />
            <select
              aria-label="Token kind"
              value={kind}
              onChange={(e) => setKind(e.target.value as "api" | "agent")}
              title="A notebook's token has your rights; an agent's (a model's harness) only proposes on shared machines, and a person applies"
            >
              <option value="api">notebook</option>
              <option value="agent">agent</option>
            </select>
            <button type="submit" disabled={!label}>
              Make
            </button>
          </form>
          {token ? (
            kind === "agent" ? (
              <pre className="token">{`Authorization: Bearer ${token}\n(an agent's token: it proposes on shared machines; a person applies)`}</pre>
            ) : (
              <pre className="token">{`from eicvibe.va.client import Client\nva = Client("${window.location.origin}", token="${token}")`}</pre>
            )
          ) : null}
          <h3>Password</h3>
          <form
            className="admin-row"
            onSubmit={(e) => {
              e.preventDefault();
              va.changePassword(old, next).then(() => {
                setOld("");
                setNext("");
                setMessage("Password changed.");
              }, (err: Error) => setMessage(err.message));
            }}
          >
            <input type="password" placeholder="current" aria-label="Current password" value={old} onChange={(e) => setOld(e.target.value)} />
            <input type="password" placeholder="new (8 or more)" aria-label="New password" value={next} onChange={(e) => setNext(e.target.value)} />
            <button type="submit" disabled={!old || next.length < 8}>
              Change
            </button>
          </form>
          {message ? <p className="muted">{message}</p> : null}
          <p>
            <button type="button" onClick={() => auth.logout().then(() => setOpen(false))}>
              Log out
            </button>
          </p>
        </div>
      ) : null}
    </div>
  );
}
