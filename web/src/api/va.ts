// The virtual accelerator's API (eicvibe.va.api, /api/v1/va; ADR 0008), for the admin page.
//
// The browser logs in once and then carries a session cookie, so every call
// here is a plain same-origin fetch. Nothing is cached: machines, settings and
// the log change while the page is open.

import type { components } from "./schema";

type Schemas = components["schemas"];
export type Me = Schemas["Me"];
export type VaStatus = Schemas["VaStatus"];
export type VaOptions = Schemas["VaOptions"];
export type MachineEntry = Schemas["MachineEntry"];
export type NewMachine = Schemas["NewMachine"];
export type SettingRow = Schemas["SettingRow"];
export type WriteResult = Schemas["WriteResult"];
export type SnapshotEntry = Schemas["SnapshotEntry"];
export type LogEntry = Schemas["LogEntry"];
export type ActionEntry = Schemas["ActionEntry"];
export type UserEntry = Schemas["UserEntry"];
export type UserCreated = Schemas["UserCreated"];
export type TokenIssued = Schemas["TokenIssued"];
export type TokenEntry = Schemas["TokenEntry"];
export type AiSettings = Schemas["AiSettings"];
export type AiChange = Schemas["AiChange"];
export type AiTest = Schemas["AiTest"];
export type ProposalRequest = Schemas["ProposalRequest"];
export type ProposalEntry = Schemas["ProposalEntry"];
export type TuneReading = Schemas["TuneReading"];
export type RampView = Schemas["RampView"];
export type ErrorSpec = Schemas["ErrorSpec"];
export type MagnetErrors = Schemas["MagnetErrors"];
export type SupplyNoise = Schemas["SupplyNoise"];
export type ErrorsEntry = Schemas["ErrorsEntry"];

const BASE = "/api/v1/va";

export class VaError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function call<T>(method: string, path: string, body?: unknown, query?: Record<string, unknown>): Promise<T> {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query ?? {})) if (value !== undefined && value !== null && value !== "") params.set(key, String(value));
  const url = BASE + path + (params.size ? `?${params}` : "");
  const response = await fetch(url, {
    method,
    headers: body === undefined ? { Accept: "application/json" } : { Accept: "application/json", "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const parsed = (await response.json()) as { detail?: unknown };
      detail = typeof parsed.detail === "string" ? parsed.detail : JSON.stringify(parsed.detail ?? detail);
    } catch {
      // not JSON
    }
    throw new VaError(response.status, detail);
  }
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T);
}

const enc = encodeURIComponent;

export const va = {
  status: () => call<VaStatus>("GET", "/status"),
  me: () => call<Me>("GET", "/me"),
  login: (name: string, password: string) => call<Me>("POST", "/login", { name, password }),
  logout: () => call<void>("POST", "/logout"),
  changePassword: (old: string, next: string) => call<void>("POST", "/me/password", { old, new: next }),
  tokens: () => call<TokenEntry[]>("GET", "/me/tokens"),
  newToken: (label: string, kind: "api" | "agent" = "api") => call<TokenIssued>("POST", "/me/tokens", { label, kind }),

  options: () => call<VaOptions>("GET", "/options"),
  machines: () => call<MachineEntry[]>("GET", "/machines"),
  create: (machine: NewMachine) => call<MachineEntry>("POST", "/machines", machine),
  start: (id: string, reason = "") => call<MachineEntry>("POST", `/machines/${enc(id)}/start`, { reason }),
  stop: (id: string, reason = "") => call<MachineEntry>("POST", `/machines/${enc(id)}/stop`, { reason }),
  fork: (id: string, name?: string) => call<MachineEntry>("POST", `/machines/${enc(id)}/fork`, { name: name || null, reason: "" }),
  remove: (id: string) => call<void>("DELETE", `/machines/${enc(id)}`),
  kick: (id: string, x: number, y: number, reason = "") =>
    call<void>("POST", `/machines/${enc(id)}/beam`, { action: "kick", x, y, reason }),
  inject: (id: string) => call<void>("POST", `/machines/${enc(id)}/beam`, { action: "inject", reason: "" }),
  /** Ramp a ring as its ramp profile says, from injection to extraction (ADR 0013). */
  ramp: (id: string, reason = "") => call<void>("POST", `/machines/${enc(id)}/beam`, { action: "ramp", reason }),
  /** A ring's ramp: its profile and programs, where the ring is in it, and its run's ramps. */
  rampView: (id: string) => call<RampView>("GET", `/machines/${enc(id)}/ramp`),
  acquire: (id: string) => call<void>("POST", `/machines/${enc(id)}/acquire`, { reason: "" }),

  settings: (id: string, changed = false) =>
    call<SettingRow[]>("GET", `/machines/${enc(id)}/settings`, undefined, { changed }),
  write: (id: string, changes: { channel: string; value: number }[], reason: string) =>
    call<WriteResult>("POST", `/machines/${enc(id)}/settings`, { changes, reason, atomic: true }),
  reset: (id: string, channels: string[] | null, reason: string) =>
    call<WriteResult>("POST", `/machines/${enc(id)}/reset`, { channels, reason }),
  snapshots: (id: string) => call<SnapshotEntry[]>("GET", `/machines/${enc(id)}/snapshots`),
  snapshot: (id: string, name: string) => call<SnapshotEntry>("POST", `/machines/${enc(id)}/snapshots`, { name }),
  restore: (id: string, snapshot: number, reason = "") =>
    call<WriteResult>("POST", `/machines/${enc(id)}/snapshots/${snapshot}/restore`, { reason }),

  /** Operations (ADR 0010): proposals computed on a machine as set, applied by a person, verified. */
  propose: (id: string, request: Partial<ProposalRequest> & Pick<ProposalRequest, "operation">) =>
    call<ProposalEntry>("POST", `/machines/${enc(id)}/proposals`, request),
  proposals: (id: string) => call<ProposalEntry[]>("GET", `/machines/${enc(id)}/proposals`),
  apply: (proposal: string, reason = "") => call<ProposalEntry>("POST", `/proposals/${enc(proposal)}/apply`, { reason }),
  verify: (proposal: string) => call<ProposalEntry>("POST", `/proposals/${enc(proposal)}/verify`),
  discard: (proposal: string, reason = "") => call<ProposalEntry>("POST", `/proposals/${enc(proposal)}/discard`, { reason }),
  tunes: (id: string) => call<TuneReading>("GET", `/machines/${enc(id)}/tunes`),
  errors: (id: string) => call<ErrorsEntry>("GET", `/machines/${enc(id)}/errors`),
  checkErrors: (id: string, errors: ErrorSpec) => call<ErrorsEntry>("POST", `/machines/${enc(id)}/errors/check`, { errors }),
  setErrors: (id: string, errors: ErrorSpec, reason = "") => call<ErrorsEntry>("PUT", `/machines/${enc(id)}/errors`, { errors, reason }),

  log: (filters: { machine?: string; user?: string; correlation?: string; before?: number; limit?: number }) =>
    call<LogEntry[]>("GET", "/log", undefined, filters),
  actions: (filters: { machine?: string; user?: string; before?: number; limit?: number }) =>
    call<ActionEntry[]>("GET", "/actions", undefined, filters),

  users: () => call<UserEntry[]>("GET", "/users"),
  addUser: (name: string, role: string) => call<UserCreated>("POST", "/users", { name, role }),
  changeUser: (name: string, change: { role?: string; disabled?: boolean }) =>
    call<UserEntry>("PATCH", `/users/${enc(name)}`, change),
  resetPassword: (name: string) => call<UserCreated>("POST", `/users/${enc(name)}/password`),

  /** The AI features' model (administrators; ADR 0009). Keys go in, never out. */
  ai: () => call<AiSettings>("GET", "/ai"),
  setAi: (change: AiChange) => call<AiSettings>("PUT", "/ai", change),
  testAi: (change: AiChange) => call<AiTest>("POST", "/ai/test", change),
};

export const ROLES = ["viewer", "physicist", "operator", "admin"] as const;

/** Whether ``role`` is ``needed`` or above it. */
export function atLeast(role: string, needed: (typeof ROLES)[number]): boolean {
  return ROLES.indexOf(role as (typeof ROLES)[number]) >= ROLES.indexOf(needed);
}

/** Where a ring with a ramp is in it, e.g. "ramping · 3.21 GeV"; empty for one without. */
export function cycleText(machine: Pick<MachineEntry, "cycle">): string {
  const cycle = machine.cycle;
  if (!cycle) return "";
  const state = cycle.state === "injection" ? "at injection" : String(cycle.state);
  return `${state} · ${(Number(cycle.energy) / 1e9).toFixed(2)} GeV`;
}

/**
 * The viewer's address showing a machine's orbit: its ring, or its route, in the Orbit tab, with
 * the machine chosen, so that it follows the machine's runs (a new one each time it starts).
 */
export function orbitLink(machine: Pick<MachineEntry, "id" | "kind" | "target">): string {
  const branch = machine.kind === "route" ? `route:${machine.target}` : machine.target;
  return `/#branch=${enc(branch)}&tab=orbit&machine=${enc(machine.id)}`;
}
