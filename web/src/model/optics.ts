// Optics of a branch prepared for plotting: the API's columns as typed arrays,
// values at any s, and axis ranges.
//
// The engines report optics at the branch start and at the exit of every
// occurrence, and some (TrackPad) also inside elements, at their integrator
// steps. Between points the plot draws straight segments, except in
// field-free elements, where beta is the parabola that its value and alpha
// at the entrance determine and dispersion the straight line: exact there.

import type { BranchOptics, Column } from "../api/client";

/** Optics along s: points, and the occurrence each segment lies in. */
export interface Points {
  s: Float64Array;
  values: Record<string, Float64Array>;
  /** Occurrence (index in the branch) of the segment from point k to k + 1. */
  owner: Int32Array;
}

export interface OpticsModel {
  branch: string;
  engine: string;
  version: string | null;
  fidelity: string;
  method: string | null;
  periodic: boolean;
  available: boolean;
  /** How the optics start (periodic solution, initial conditions, a fork), or why there are none. */
  origin: string;
  /** At the start and at the exit of every occurrence: n + 1 points. */
  exits: Points;
  /** What to draw: the engine's samples inside elements where it has them, else the exits. */
  points: Points;
  sampled: boolean;
  columns: Map<string, Column>;
  occurrences: string[];
  tuneX: number | null;
  tuneY: number | null;
  chromaX: number | null;
  chromaY: number | null;
  momentumCompaction: number | null;
}

const typed = (columns: Record<string, number[]>) => {
  const out: Record<string, Float64Array> = {};
  for (const [key, column] of Object.entries(columns)) out[key] = Float64Array.from(column);
  return out;
};

export function opticsModel(data: BranchOptics): OpticsModel {
  const n = data.occurrences.length;
  if (data.available && data.s.length !== n + 1)
    throw new Error(`optics has ${data.s.length} points for ${n} occurrences`);
  const exits: Points = {
    s: Float64Array.from(data.s),
    values: typed(data.values),
    owner: Int32Array.from({ length: n }, (_, k) => k),
  };
  const profile = data.profile;
  if (profile && profile.s.length !== profile.index.length + 1)
    throw new Error(`optics profile has ${profile.s.length} points for ${profile.index.length} pieces`);
  const points: Points = profile
    ? { s: Float64Array.from(profile.s), values: typed(profile.values), owner: Int32Array.from(profile.index) }
    : exits;
  return {
    branch: data.branch,
    engine: data.engine,
    version: data.version ?? null,
    fidelity: data.fidelity,
    method: data.method ?? null,
    periodic: data.periodic,
    available: data.available,
    origin: data.origin,
    exits,
    points,
    sampled: !!profile,
    columns: new Map(data.columns.map((c) => [c.key, c])),
    occurrences: data.occurrences,
    tuneX: data.tune_x ?? null,
    tuneY: data.tune_y ?? null,
    chromaX: data.chroma_x ?? null,
    chromaY: data.chroma_y ?? null,
    momentumCompaction: data.momentum_compaction ?? null,
  };
}

/** Kinds without fields in the optics (RF cavities are drifts in transverse optics, but kept straight here). */
const FIELD_FREE = new Set(["Drift", "Instrument", "Marker", "BeginningEle", "Fork"]);

export function isFieldFree(kind: string | undefined): boolean {
  return kind !== undefined && FIELD_FREE.has(kind);
}

/** Index of the last point at or before s (binary search); 0 before the start. */
export function pointAt(points: Points, s: number): number {
  let lo = 0;
  let hi = points.s.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (points.s[mid]! <= s) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

const BETA_ALPHA: Record<string, string> = { beta_x: "alpha_x", beta_y: "alpha_y" };
const ETA_PRIME: Record<string, string> = { eta_x: "etap_x", eta_y: "etap_y" };

/** Kind of the element segment k lies in, given the branch's kinds by occurrence. */
export const segmentKind = (points: Points, k: number, kinds: readonly string[]) => kinds[points.owner[k]!];

/**
 * A column's value ``t`` metres into segment ``k`` (from point k to k + 1): exact in field-free
 * elements, linear elsewhere.
 */
export function segmentValue(points: Points, key: string, k: number, t: number, kinds: readonly string[]): number {
  const v = points.values[key]!;
  const length = points.s[k + 1]! - points.s[k]!;
  if (length <= 0) return v[k + 1]!;
  if (isFieldFree(segmentKind(points, k, kinds))) {
    const alpha = BETA_ALPHA[key];
    if (alpha) {
      const beta = v[k]!, a = points.values[alpha]![k]!;
      return beta - 2 * a * t + ((1 + a * a) / beta) * t * t;
    }
    const prime = ETA_PRIME[key];
    if (prime) return v[k]! + points.values[prime]![k]! * t;
  }
  return v[k]! + ((v[k + 1]! - v[k]!) * t) / length;
}

/** A column's value at s, given the kinds of the branch's occurrences. */
export function valueAt(points: Points, key: string, s: number, kinds: readonly string[]): number {
  const n = points.s.length - 1;
  if (s <= points.s[0]!) return points.values[key]![0]!;
  if (s >= points.s[n]!) return points.values[key]![n]!;
  let k = pointAt(points, s);
  while (k < n - 1 && points.s[k + 1]! <= s) k++; // past zero-length pieces at s
  return segmentValue(points, key, k, s - points.s[k]!, kinds);
}

/** Lowest and highest values of ``keys`` on [s0, s1], including the points just outside; null if none. */
export function extent(
  points: Points,
  keys: readonly string[],
  s0: number,
  s1: number,
  kinds: readonly string[],
): [number, number] | null {
  const n = points.s.length - 1;
  if (n < 0 || keys.length === 0) return null;
  const first = Math.max(pointAt(points, s0) - 1, 0);
  const last = Math.min(pointAt(points, s1) + 1, n);
  let lo = Infinity, hi = -Infinity;
  for (const key of keys) {
    const v = points.values[key];
    if (!v) continue;
    for (let k = first; k <= last; k++) {
      lo = Math.min(lo, v[k]!);
      hi = Math.max(hi, v[k]!);
      // A drift's beta can dip below both ends: include its waist.
      const alpha = BETA_ALPHA[key];
      if (alpha && k < n && isFieldFree(segmentKind(points, k, kinds))) {
        const a = points.values[alpha]![k]!, beta = v[k]!, length = points.s[k + 1]! - points.s[k]!;
        const waist = (a * beta) / (1 + a * a);
        if (waist > 0 && waist < length) lo = Math.min(lo, beta / (1 + a * a));
      }
    }
  }
  return Number.isFinite(lo) ? [lo, hi] : null;
}

export interface Axis {
  min: number;
  max: number;
  step: number;
}

/** Bounds and tick step covering [lo, hi] with about ``count`` ticks; zero included when ``withZero``. */
export function niceAxis(lo: number, hi: number, count = 5, withZero = true): Axis {
  if (withZero) {
    lo = Math.min(lo, 0);
    hi = Math.max(hi, 0);
  }
  if (!(hi > lo)) {
    const pad = Math.abs(hi) * 0.1 || 1;
    lo -= pad;
    hi += pad;
  }
  const raw = (hi - lo) / Math.max(count, 1);
  const power = 10 ** Math.floor(Math.log10(raw));
  const mantissa = raw / power;
  const step = (mantissa <= 1 ? 1 : mantissa <= 2 ? 2 : mantissa <= 5 ? 5 : 10) * power;
  return { min: Math.floor(lo / step + 1e-9) * step, max: Math.ceil(hi / step - 1e-9) * step, step };
}

export interface Curve {
  key: string;
  label: string;
  axis: "left" | "right";
  color: string;
  darkColor: string;
  dash: number[];
}

/** What the plot can draw: beta functions on the left axis, dispersion on the right. */
export const CURVES: Curve[] = [
  { key: "beta_x", label: "βx", axis: "left", color: "#1f5fbf", darkColor: "#6ea8ff", dash: [] },
  { key: "beta_y", label: "βy", axis: "left", color: "#c7362f", darkColor: "#ff7b72", dash: [] },
  { key: "eta_x", label: "ηx", axis: "right", color: "#2b8a3e", darkColor: "#69db7c", dash: [] },
  { key: "eta_y", label: "ηy", axis: "right", color: "#8a5bd6", darkColor: "#c4a7ff", dash: [5, 3] },
];
