// What a browser is looking at, kept in the URL hash. Every browser has its
// own view; copying the URL hands the same view to a colleague.

import type { SRange, View2D } from "./viewport";

export interface UrlState {
  /** Catalog name of the lattice. */
  source?: string;
  /** Revision digest; absent means the current revision. */
  rev?: string;
  /** Branch name, or route:<name>. */
  branch?: string;
  /** Selected occurrence id. */
  sel?: string;
  floor?: View2D;
  range?: SRange;
  /** Element table filter text. */
  q?: string;
  /** Optics engine and fidelity. */
  engine?: string;
  fidelity?: string;
  /** The lower panel's tab: optics (absent), orbit or check. */
  tab?: string;
  /** The run the Orbit tab shows first (a link from the admin page); read once. */
  run?: string;
}

const numbers = (text: string | null, count: number): number[] | null => {
  if (!text) return null;
  const values = text.split(",").map(Number);
  return values.length === count && values.every(Number.isFinite) ? values : null;
};

export function parseHash(hash: string): UrlState {
  const params = new URLSearchParams(hash.replace(/^#/, ""));
  const state: UrlState = {};
  for (const key of ["source", "rev", "branch", "sel", "q", "engine", "fidelity", "tab", "run"] as const) {
    const value = params.get(key);
    if (value) state[key] = value;
  }
  const floor = numbers(params.get("floor"), 3);
  if (floor && floor[2]! > 0) state.floor = { cu: floor[0]!, cv: floor[1]!, k: floor[2]! };
  const range = numbers(params.get("s"), 2);
  if (range && range[1]! > range[0]!) state.range = { s0: range[0]!, s1: range[1]! };
  return state;
}

const round = (value: number, digits: number) => String(Number(value.toPrecision(digits)));

export function formatHash(state: UrlState): string {
  const params = new URLSearchParams();
  for (const key of ["source", "rev", "branch", "sel"] as const) {
    const value = state[key];
    if (value) params.set(key, value);
  }
  if (state.floor) {
    const { cu, cv, k } = state.floor;
    params.set("floor", [round(cu, 8), round(cv, 8), round(k, 5)].join(","));
  }
  if (state.range) params.set("s", [round(state.range.s0, 8), round(state.range.s1, 8)].join(","));
  if (state.q) params.set("q", state.q);
  if (state.engine) params.set("engine", state.engine);
  if (state.fidelity) params.set("fidelity", state.fidelity);
  if (state.tab) params.set("tab", state.tab);
  return "#" + params.toString().replaceAll("%2C", ",");
}

let pending: UrlState | null = null;
let timer: number | undefined;

/** Merge ``update`` into the URL hash, at most a few times a second, without adding history entries. */
export function updateHash(update: Partial<UrlState>): void {
  pending = { ...(pending ?? parseHash(window.location.hash)), ...update };
  if (timer !== undefined) return;
  timer = window.setTimeout(() => {
    timer = undefined;
    const next = formatHash(pending!);
    pending = null;
    if (next !== window.location.hash) window.history.replaceState(null, "", next);
  }, 200);
}
