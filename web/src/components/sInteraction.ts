// Mouse and wheel along s, shared by the lattice strip and the optics plot.
//
// Drag selects an s range and zooms to it; shift-drag (or the middle button)
// pans; the wheel zooms about the pointer, or pans when scrolled sideways;
// double-click shows the whole branch; right click undoes the last zoom.
// The range being selected is a shared store, so every view along s shows it.

import type { Store } from "../model/store";
import { clampRange, S_MARGIN, sScale, zoomRange, type SRange, type ZoomHistory } from "../model/viewport";

/** A selection narrower than this, px, is a click. */
const CLICK = 4;
/** Wheel events closer together than this, ms, are one zoom for undoing. */
const GESTURE = 600;

export interface SInteraction {
  canvas: HTMLCanvasElement;
  range: Store<SRange | null>;
  band: Store<SRange | null>;
  history: ZoomHistory;
  /** The current branch's whole range, or null without a branch. */
  limits(): SRange | null;
  /** The pointer moved with no button down. */
  hover(x: number, y: number, s: number): void;
  leave(): void;
  click(x: number, y: number, s: number): void;
  /** A zoom or pan started: hide tooltips. */
  busy(): void;
}

export function attachSInteraction(o: SInteraction): () => void {
  const { canvas, range, band, history } = o;
  let drag: { mode: "select" | "pan"; x: number; s: number; range: SRange } | null = null;
  let moved = false;
  let lastWheel = 0;

  const at = (event: { clientX: number; clientY: number }) => {
    const rect = canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  };
  const sAt = (x: number, shown = range.get()) =>
    shown ? shown.s0 + (x - S_MARGIN) / sScale(shown, canvas.getBoundingClientRect().width) : 0;

  const zoomTo = (next: SRange) => {
    const shown = range.get();
    const whole = o.limits();
    if (!shown || !whole) return;
    history.push(shown);
    range.set(clampRange(next, whole));
  };

  const down = (event: PointerEvent) => {
    const shown = range.get();
    if (!shown || !o.limits() || (event.button !== 0 && event.button !== 1)) return;
    canvas.setPointerCapture(event.pointerId);
    const x = at(event).x;
    drag = { mode: event.button === 1 || event.shiftKey ? "pan" : "select", x, s: sAt(x, shown), range: shown };
    moved = false;
  };
  const move = (event: PointerEvent) => {
    const p = at(event);
    if (!drag) {
      if (range.get()) o.hover(p.x, p.y, sAt(p.x));
      return;
    }
    const whole = o.limits();
    if (!whole) return;
    const dx = p.x - drag.x;
    if (!moved && Math.abs(dx) <= CLICK) return;
    if (!moved) o.busy();
    moved = true;
    if (drag.mode === "pan") {
      canvas.style.cursor = "grabbing";
      const ds = dx / sScale(drag.range, canvas.getBoundingClientRect().width);
      range.set(clampRange({ s0: drag.range.s0 - ds, s1: drag.range.s1 - ds }, whole));
    } else {
      const s = Math.min(Math.max(sAt(p.x, drag.range), whole.s0), whole.s1);
      band.set({ s0: Math.min(drag.s, s), s1: Math.max(drag.s, s) });
    }
  };
  const up = (event: PointerEvent) => {
    const current = drag;
    drag = null;
    canvas.style.cursor = "";
    if (!current) return;
    if (!moved) {
      o.click(at(event).x, at(event).y, sAt(at(event).x));
      return;
    }
    if (current.mode === "pan") {
      history.push(current.range);
      return;
    }
    const selected = band.get();
    band.set(null);
    if (selected && selected.s1 - selected.s0 > 1e-6) {
      const span = Math.max(selected.s1 - selected.s0, 0.05);
      const middle = (selected.s0 + selected.s1) / 2;
      zoomTo({ s0: middle - span / 2, s1: middle + span / 2 });
    }
  };
  const cancel = () => {
    drag = null;
    band.set(null);
    canvas.style.cursor = "";
  };
  const wheel = (event: WheelEvent) => {
    event.preventDefault();
    const shown = range.get();
    const whole = o.limits();
    if (!shown || !whole) return;
    if (event.timeStamp - lastWheel > GESTURE) history.push(shown);
    lastWheel = event.timeStamp;
    const scale = event.deltaMode === 1 ? 0.05 : 0.0015;
    if (Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
      const ds = event.deltaX * scale * 0.5 * (shown.s1 - shown.s0);
      range.set(clampRange({ s0: shown.s0 + ds, s1: shown.s1 + ds }, whole));
    } else {
      range.set(zoomRange(shown, Math.exp(-event.deltaY * scale), sAt(at(event).x), whole));
    }
    o.busy();
  };
  const doubleClick = () => {
    const whole = o.limits();
    if (whole) zoomTo(whole);
  };
  const contextMenu = (event: MouseEvent) => {
    event.preventDefault();
    const previous = history.undo();
    const whole = o.limits();
    if (previous && whole) range.set(clampRange(previous, whole));
  };
  const leave = () => {
    if (!drag) o.leave();
  };

  canvas.addEventListener("pointerdown", down);
  canvas.addEventListener("pointermove", move);
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", cancel);
  canvas.addEventListener("pointerleave", leave);
  canvas.addEventListener("wheel", wheel, { passive: false });
  canvas.addEventListener("dblclick", doubleClick);
  canvas.addEventListener("contextmenu", contextMenu);
  return () => {
    canvas.removeEventListener("pointerdown", down);
    canvas.removeEventListener("pointermove", move);
    canvas.removeEventListener("pointerup", up);
    canvas.removeEventListener("pointercancel", cancel);
    canvas.removeEventListener("pointerleave", leave);
    canvas.removeEventListener("wheel", wheel);
    canvas.removeEventListener("dblclick", doubleClick);
    canvas.removeEventListener("contextmenu", contextMenu);
  };
}
