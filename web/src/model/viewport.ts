// View transforms. The floor plan looks down on the machine (from +y): global
// z points right and x points up, so a positive bend angle turns clockwise.
// Zooming and panning only change the transform, never the geometry.

export interface View2D {
  /** World point (z, x) at the centre of the canvas, m. */
  cu: number;
  cv: number;
  /** Pixels per metre. */
  k: number;
}

export interface Box {
  uMin: number;
  uMax: number;
  vMin: number;
  vMax: number;
}

export const MIN_SCALE = 1e-3;
export const MAX_SCALE = 5e3;

export function clampScale(k: number): number {
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, k));
}

/** The view that shows ``box`` in a ``width`` x ``height`` canvas with ``margin`` pixels around it. */
export function fit(box: Box, width: number, height: number, margin = 24): View2D {
  const du = Math.max(box.uMax - box.uMin, 1e-3);
  const dv = Math.max(box.vMax - box.vMin, 1e-3);
  const k = clampScale(Math.min((width - 2 * margin) / du, (height - 2 * margin) / dv));
  return { cu: (box.uMin + box.uMax) / 2, cv: (box.vMin + box.vMax) / 2, k: Number.isFinite(k) && k > 0 ? k : 1 };
}

export function toScreen(view: View2D, width: number, height: number, u: number, v: number): [number, number] {
  return [(u - view.cu) * view.k + width / 2, height / 2 - (v - view.cv) * view.k];
}

export function toWorld(view: View2D, width: number, height: number, sx: number, sy: number): [number, number] {
  return [view.cu + (sx - width / 2) / view.k, view.cv - (sy - height / 2) / view.k];
}

/** Zoom by ``factor`` keeping the world point under screen point (sx, sy) where it is. */
export function zoomAt(view: View2D, factor: number, sx: number, sy: number, width: number, height: number): View2D {
  const [u, v] = toWorld(view, width, height, sx, sy);
  const k = clampScale(view.k * factor);
  return { k, cu: u - (sx - width / 2) / k, cv: v + (sy - height / 2) / k };
}

/** Move the view so the scene follows a drag of (dx, dy) pixels. */
export function pan(view: View2D, dx: number, dy: number): View2D {
  return { ...view, cu: view.cu - dx / view.k, cv: view.cv + dy / view.k };
}

/** Union of boxes; null for none. */
export function union(boxes: Iterable<Box>): Box | null {
  let out: Box | null = null;
  for (const b of boxes)
    out = out
      ? {
          uMin: Math.min(out.uMin, b.uMin),
          uMax: Math.max(out.uMax, b.uMax),
          vMin: Math.min(out.vMin, b.vMin),
          vMax: Math.max(out.vMax, b.vMax),
        }
      : { ...b };
  return out;
}

/** A 1, 2 or 5 times a power of ten close to ``target``, for scale bars and axis ticks. */
export function niceStep(target: number): number {
  const power = 10 ** Math.floor(Math.log10(target));
  const mantissa = target / power;
  return (mantissa < 1.5 ? 1 : mantissa < 3.5 ? 2 : mantissa < 7.5 ? 5 : 10) * power;
}

/** Visible s range of the linear view. */
export interface SRange {
  s0: number;
  s1: number;
}

export function zoomRange(range: SRange, factor: number, at: number, limits: SRange, minSpan = 0.05): SRange {
  const span = Math.min(Math.max((range.s1 - range.s0) / factor, minSpan), limits.s1 - limits.s0 || minSpan);
  const fraction = (at - range.s0) / (range.s1 - range.s0 || 1);
  return clampRange({ s0: at - fraction * span, s1: at - fraction * span + span }, limits);
}

export function clampRange(range: SRange, limits: SRange): SRange {
  const span = range.s1 - range.s0;
  const margin = span * 0.02;
  let s0 = range.s0;
  if (s0 < limits.s0 - margin) s0 = limits.s0 - margin;
  if (s0 + span > limits.s1 + margin) s0 = limits.s1 + margin - span;
  return { s0, s1: s0 + span };
}
