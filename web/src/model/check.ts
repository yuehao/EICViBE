// The Check tab's arithmetic: what a run's BPMs measured against the model (ADR 0007).

import type { PlaneCheck, RunCheck } from "../api/client";

/** Where a tune's line shows in a spectrum of BPM readings: q and 1 - q look alike, 0 to 0.5. */
export const lineOf = (tune: number | null | undefined): number | null => {
  if (tune === null || tune === undefined) return null;
  const q = ((tune % 1) + 1) % 1;
  return Math.min(q, 1 - q);
};

/** Measured minus model fractional tune, wrapped to the nearest: positive when the beam's tune is higher. */
export function tuneDifference(plane: PlaneCheck): number | null {
  if (plane.measured_full === null || plane.measured_full === undefined) return null;
  if (plane.model_tune === null || plane.model_tune === undefined) return null;
  return plane.measured_full - plane.model_tune;
}

/** Whether the measured tune agrees with the model: within 1e-3 of a turn, about a spectrum's resolution. */
export const agrees = (difference: number | null) => difference !== null && Math.abs(difference) < 1e-3;

/** Decades for a log axis covering the positive values: [lowest power of ten, highest]. */
export function decades(values: readonly (number | null | undefined)[], span = 4): [number, number] {
  const positive = values.filter((v): v is number => typeof v === "number" && v > 0);
  if (!positive.length) return [-3, 0];
  const top = Math.ceil(Math.log10(Math.max(...positive)) + 1e-9);
  return [top - span, top];
}

/** The measured point on the detuning plot: the amplitude right after the kick, the first stretch's tune. */
export function measuredPoint(check: RunCheck, plane: "x" | "y"): { amplitude: number; tune: number } | null {
  const p = check[plane];
  const tune = p.drift_tune[0];
  const amplitude = check[check.detuning?.plane === "y" ? "y" : "x"].initial_amplitude;
  if (tune === null || tune === undefined || amplitude === null || amplitude === undefined) return null;
  return { amplitude, tune };
}

/** "0.2525 (28.2525)": a tune with its integer part where known. */
export function tuneText(fraction: number | null | undefined, full?: number | null): string {
  if (fraction === null || fraction === undefined) return "—";
  return full === null || full === undefined ? fraction.toFixed(4) : `${fraction.toFixed(4)} (${full.toFixed(4)})`;
}
