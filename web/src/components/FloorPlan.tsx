// Floor plan of every branch, seen from above (z to the right, x up).
//
// Each branch's geometry becomes a few Path2D objects in world coordinates
// (one per element kind) when it loads; panning and zooming only change the
// canvas transform, so a frame costs a few dozen stroke calls however large
// the machine is. Hover uses a spatial grid; tooltips and the coordinate
// readout are written to the DOM directly, so moving the mouse never
// re-renders React.

import { useCallback, useEffect, useImperativeHandle, useRef, useState, type Ref } from "react";

import type { LatticeSummary } from "../api/client";
import { isOpen, strengthText, type BranchModel } from "../model/branch";
import { formatMetres, formatNumber } from "../model/format";
import { BRANCH_COLORS, isDrift, kindColor, kindStyle, swatch } from "../model/kinds";
import { sameTarget, type Store, type Target } from "../model/store";
import {
  fit,
  niceStep,
  pan,
  toScreen,
  toWorld,
  union,
  zoomAt,
  type Box,
  type View2D,
} from "../model/viewport";
import { readTheme, useCanvas, type Theme } from "./useCanvas";

interface Layers {
  centre: Path2D;
  kinds: [string, Path2D][];
  markers: number[];
  box: Box;
}

const layerCache = new WeakMap<BranchModel, Layers>();

function layers(branch: BranchModel): Layers {
  const cached = layerCache.get(branch);
  if (cached) return cached;
  const { n, offsets, z, x, kinds, lengths } = branch;
  const centre = new Path2D();
  const byKind = new Map<string, Path2D>();
  const markers: number[] = [];
  let uMin = Infinity, uMax = -Infinity, vMin = Infinity, vMax = -Infinity;
  for (let j = 0; j < z.length; j++) {
    uMin = Math.min(uMin, z[j]!);
    uMax = Math.max(uMax, z[j]!);
    vMin = Math.min(vMin, x[j]!);
    vMax = Math.max(vMax, x[j]!);
  }
  if (z.length) centre.moveTo(z[0]!, x[0]!);
  for (let i = 0; i < n; i++) {
    const first = offsets[i]!, stop = offsets[i + 1]!;
    for (let j = first + 1; j < stop; j++) centre.lineTo(z[j]!, x[j]!);
    const style = kindStyle(kinds[i]!);
    if (style.floorWidth > 0 && lengths[i]! > 0) {
      let path = byKind.get(kinds[i]!);
      if (!path) byKind.set(kinds[i]!, (path = new Path2D()));
      path.moveTo(z[first]!, x[first]!);
      for (let j = first + 1; j < stop; j++) path.lineTo(z[j]!, x[j]!);
    } else if (style.lane === "tick" || (style.floorWidth > 0 && lengths[i] === 0)) {
      markers.push(i);
    }
  }
  const order = [...byKind.entries()].sort(([a], [b]) => kindStyle(a).floorWidth - kindStyle(b).floorWidth);
  const result = { centre, kinds: order.reverse(), markers, box: { uMin, uMax, vMin, vMax } };
  layerCache.set(branch, result);
  return result;
}

export interface FloorPlanHandle {
  fitAll(): void;
  fitBranch(branch: string): void;
  show(target: Target, zoom?: boolean): void;
}

interface Props {
  summary: LatticeSummary;
  branches: Map<string, BranchModel>;
  current: string | null;
  selection: Target | null;
  hover: Store<Target | null>;
  initialView?: View2D;
  scheme: string;
  onSelect(target: Target | null): void;
  onViewChange(view: View2D): void;
  handle?: Ref<FloorPlanHandle>;
}

export function FloorPlan(props: Props) {
  const { summary, branches, current, selection, hover, scheme, onSelect, onViewChange } = props;
  const view = useRef<View2D | null>(props.initialView ?? null);
  // Until the user pans or zooms, the view keeps fitting the whole machine as the pane resizes.
  const autoFit = useRef(!props.initialView);
  const fitted = useRef({ width: 0, height: 0 });
  const theme = useRef<Theme>(readTheme(null));
  const tooltip = useRef<HTMLDivElement | null>(null);
  const readout = useRef<HTMLDivElement | null>(null);
  const [labels, setLabels] = useState(true);
  const state = useRef({ branches, current, selection, labels });
  state.current = { branches, current, selection, labels };

  const summaryBox = useCallback(
    (only?: string): Box | null =>
      union(
        summary.branches
          .filter((b) => !only || b.id === only)
          .map((b) => ({ uMin: b.bounds.z_min, uMax: b.bounds.z_max, vMin: b.bounds.x_min, vMax: b.bounds.x_max })),
      ),
    [summary],
  );

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, { width, height }: { width: number; height: number }) => {
      const t = theme.current;
      ctx.fillStyle = t.background;
      ctx.fillRect(0, 0, width, height);
      if (!view.current || (autoFit.current && (fitted.current.width !== width || fitted.current.height !== height))) {
        const box = summaryBox();
        if (!box) return;
        view.current = fit(box, width, height);
        fitted.current = { width, height };
      }
      const v = view.current;
      const { branches, current, selection, labels } = state.current;
      drawGrid(ctx, v, width, height, t);
      const ratio = window.devicePixelRatio || 1;
      const ordered = [...branches.values()].sort((a, b) => Number(a.id === current) - Number(b.id === current));
      ctx.setTransform(ratio * v.k, 0, 0, -ratio * v.k, ratio * (width / 2 - v.k * v.cu), ratio * (height / 2 + v.k * v.cv));
      ctx.lineJoin = "round";
      for (const branch of ordered) {
        const layer = layers(branch);
        const isCurrent = branch.id === current;
        ctx.globalAlpha = isCurrent ? 1 : 0.55;
        ctx.lineCap = "round";
        ctx.strokeStyle = BRANCH_COLORS[branch.order % BRANCH_COLORS.length]!;
        ctx.lineWidth = (isCurrent ? 2 : 1.25) / v.k;
        ctx.stroke(layer.centre);
        ctx.lineCap = "butt";
        ctx.globalAlpha = isCurrent ? 0.85 : 0.5;
        for (const [kind, path] of layer.kinds) {
          ctx.strokeStyle = kindColor(kind, t.dark);
          ctx.lineWidth = bandWidth(kind, v, isCurrent) / v.k;
          ctx.stroke(path);
        }
      }
      ctx.globalAlpha = 1;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      for (const branch of ordered) drawMarkers(ctx, branch, v, width, height, branch.id === current, t);
      drawClosures(ctx, summary, branches, v, width, height, t);
      if (labels) drawLabels(ctx, ordered, current, v, width, height, t);
      const hovered = hover.get();
      if (hovered && !sameTarget(hovered, selection)) outline(ctx, branches, hovered, v, width, height, t, t.highlight, current);
      if (selection) outline(ctx, branches, selection, v, width, height, t, t.selection, current);
      drawScaleBar(ctx, v, width, height, t);
    },
    [summary, summaryBox, hover],
  );

  const { canvasRef, sizeRef, requestDraw } = useCanvas(draw);

  useEffect(() => {
    theme.current = readTheme(canvasRef.current);
    requestDraw();
  }, [scheme, canvasRef, requestDraw]);
  useEffect(requestDraw, [branches, current, selection, labels, requestDraw]);
  useEffect(() => hover.subscribe(requestDraw), [hover, requestDraw]);

  const setView = useCallback(
    (next: View2D, user = true) => {
      autoFit.current &&= !user;
      view.current = next;
      onViewChange(next);
      requestDraw();
    },
    [onViewChange, requestDraw],
  );

  const fitAll = useCallback(() => {
    const box = summaryBox();
    if (!box) return;
    const { width, height } = sizeRef.current;
    setView(fit(box, width, height), false);
    autoFit.current = true;
    fitted.current = { width, height };
  }, [summaryBox, setView, sizeRef]);

  useImperativeHandle(
    props.handle,
    () => ({
      fitAll,
      fitBranch(id: string) {
        const box = summaryBox(id);
        if (box) setView(fit(box, sizeRef.current.width, sizeRef.current.height));
      },
      show(target: Target, zoom = false) {
        const branch = state.current.branches.get(target.branch);
        if (!branch || !view.current) return;
        const box = elementBox(branch, target.index);
        const { width, height } = sizeRef.current;
        const [u, v] = [(box.uMin + box.uMax) / 2, (box.vMin + box.vMax) / 2];
        const [sx, sy] = toScreen(view.current, width, height, u, v);
        if (zoom) {
          const pad = Math.max(10, branch.lengths[target.index]! * 3);
          setView(fit({ uMin: u - pad, uMax: u + pad, vMin: v - pad, vMax: v + pad }, width, height));
        } else if (sx < 40 || sx > width - 40 || sy < 40 || sy > height - 40) {
          setView({ ...view.current, cu: u, cv: v });
        }
      },
    }),
    [summaryBox, setView, sizeRef, fitAll],
  );

  // Pointer interaction: drag to pan, wheel or pinch to zoom, click to select.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const pointers = new Map<number, { x: number; y: number }>();
    let dragged = false;
    let pinch = 0;
    const local = (event: { clientX: number; clientY: number }) => {
      const rect = canvas.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    };
    const pick = (x: number, y: number): Target | null => {
      const v = view.current;
      if (!v) return null;
      const { width, height } = sizeRef.current;
      const [u, w] = toWorld(v, width, height, x, y);
      const radius = 8 / v.k;
      let best: Target | null = null;
      let bestScore = Infinity;
      for (const branch of state.current.branches.values()) {
        const bonus = branch.id === state.current.current ? 0.2 * radius : 0;
        const found = branch.segments.nearest(u, w, radius, (point, d) => {
          const style = kindStyle(branch.kinds[branch.occurrenceOfPoint[point]!]!);
          if (style.priority === 0) return Infinity; // drifts are never picked
          return Math.max(0, d - style.floorWidth / 2) + (3 - style.priority) * radius - bonus;
        });
        if (!found) continue;
        const i = branch.occurrenceOfPoint[found.point]!;
        const style = kindStyle(branch.kinds[i]!);
        if (style.priority === 0) continue;
        const score = Math.max(0, found.distance - style.floorWidth / 2) + (3 - style.priority) * radius - bonus;
        if (score < bestScore) {
          bestScore = score;
          best = { branch: branch.id, index: i };
        }
      }
      return best;
    };
    const showTip = (x: number, y: number, target: Target | null) => {
      const tip = tooltip.current;
      if (!tip) return;
      const branch = target && state.current.branches.get(target.branch);
      if (!branch || !target) {
        tip.hidden = true;
        return;
      }
      const i = target.index;
      const strength = strengthText(branch, i, (value) => formatNumber(value, 5));
      tip.replaceChildren(
        line("strong", branch.names[i]!),
        line("span", `${branch.kinds[i]} · ${branch.name} · s = ${formatMetres(branch.s[i]!, 3)} m · L = ${formatNumber(branch.lengths[i]!, 5)} m`),
        ...(strength ? [line("span", strength)] : []),
        ...(branch.devices[i] ? [line("span", branch.devices[i]!)] : []),
        ...(branch.components[i] ? [line("span", `one magnet with other lines: ${branch.components[i]}`)] : []),
      );
      tip.hidden = false;
      const { width } = sizeRef.current;
      tip.style.transform = `translate(${Math.min(x + 14, width - tip.offsetWidth - 4)}px, ${y + 16}px)`;
    };
    const showReadout = (x: number, y: number) => {
      const v = view.current;
      if (!v || !readout.current) return;
      const [u, w] = toWorld(v, sizeRef.current.width, sizeRef.current.height, x, y);
      readout.current.textContent = `z ${u.toFixed(2)} m   x ${w.toFixed(2)} m`;
    };

    const down = (event: PointerEvent) => {
      canvas.setPointerCapture(event.pointerId);
      pointers.set(event.pointerId, local(event));
      dragged = false;
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        pinch = Math.hypot(a!.x - b!.x, a!.y - b!.y);
      }
    };
    const move = (event: PointerEvent) => {
      const p = local(event);
      const previous = pointers.get(event.pointerId);
      if (previous && view.current) {
        if (pointers.size === 2) {
          pointers.set(event.pointerId, p);
          const [a, b] = [...pointers.values()];
          const distance = Math.hypot(a!.x - b!.x, a!.y - b!.y);
          if (pinch > 0) {
            const { width, height } = sizeRef.current;
            setView(zoomAt(view.current, distance / pinch, (a!.x + b!.x) / 2, (a!.y + b!.y) / 2, width, height));
          }
          pinch = distance;
          dragged = true;
          return;
        }
        const dx = p.x - previous.x, dy = p.y - previous.y;
        if (dragged || Math.abs(dx) + Math.abs(dy) > 3) {
          dragged = true;
          pointers.set(event.pointerId, p);
          setView(pan(view.current, dx, dy));
        }
        if (tooltip.current) tooltip.current.hidden = true;
        return;
      }
      const target = pick(p.x, p.y);
      hover.set(target);
      showTip(p.x, p.y, target);
      showReadout(p.x, p.y);
    };
    const up = (event: PointerEvent) => {
      const p = local(event);
      pointers.delete(event.pointerId);
      if (pointers.size < 2) pinch = 0;
      if (!dragged && pointers.size === 0) onSelect(pick(p.x, p.y));
    };
    const leave = () => {
      hover.set(null);
      if (tooltip.current) tooltip.current.hidden = true;
    };
    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      if (!view.current) return;
      const p = local(event);
      const { width, height } = sizeRef.current;
      const scale = event.deltaMode === 1 ? 0.05 : 0.0015;
      setView(zoomAt(view.current, Math.exp(-event.deltaY * scale), p.x, p.y, width, height));
      if (tooltip.current) tooltip.current.hidden = true;
    };
    const key = (event: KeyboardEvent) => {
      const v = view.current;
      if (!v) return;
      const { width, height } = sizeRef.current;
      const steps: Record<string, () => View2D> = {
        "+": () => zoomAt(v, 1.25, width / 2, height / 2, width, height),
        "=": () => zoomAt(v, 1.25, width / 2, height / 2, width, height),
        "-": () => zoomAt(v, 0.8, width / 2, height / 2, width, height),
        ArrowLeft: () => pan(v, 60, 0),
        ArrowRight: () => pan(v, -60, 0),
        ArrowUp: () => pan(v, 0, 60),
        ArrowDown: () => pan(v, 0, -60),
      };
      const step = steps[event.key];
      if (step) {
        event.preventDefault();
        setView(step());
      } else if (event.key === "0") {
        fitAll();
      }
    };
    const doubleClick = fitAll;
    canvas.addEventListener("pointerdown", down);
    canvas.addEventListener("pointermove", move);
    canvas.addEventListener("pointerup", up);
    canvas.addEventListener("pointercancel", up);
    canvas.addEventListener("pointerleave", leave);
    canvas.addEventListener("wheel", wheel, { passive: false });
    canvas.addEventListener("keydown", key);
    canvas.addEventListener("dblclick", doubleClick);
    return () => {
      canvas.removeEventListener("pointerdown", down);
      canvas.removeEventListener("pointermove", move);
      canvas.removeEventListener("pointerup", up);
      canvas.removeEventListener("pointercancel", up);
      canvas.removeEventListener("pointerleave", leave);
      canvas.removeEventListener("wheel", wheel);
      canvas.removeEventListener("keydown", key);
      canvas.removeEventListener("dblclick", doubleClick);
    };
  }, [canvasRef, sizeRef, hover, onSelect, setView, fitAll]);

  const kindsShown = Object.keys(summary.kinds).filter((k) => kindStyle(k).floorWidth > 0);
  return (
    <div className="floor">
      <canvas ref={canvasRef} className="floor-canvas" tabIndex={0} aria-label="Floor plan" data-testid="floor-canvas" />
      <div ref={tooltip} className="tooltip" hidden />
      <div className="floor-controls">
        <button type="button" onClick={fitAll} title="Show every branch (0, or double-click)">
          Fit all
        </button>
        <button type="button" disabled={!current} onClick={() => {
          const box = current && summaryBox(current);
          if (box) setView(fit(box, sizeRef.current.width, sizeRef.current.height));
        }} title="Show the current branch">Fit branch</button>
        <label title="Element names when zoomed in">
          <input type="checkbox" checked={labels} onChange={(e) => setLabels(e.target.checked)} /> Labels
        </label>
      </div>
      <div className="legend">
        {kindsShown.map((kind) => (
          <span key={kind}><i style={{ background: swatch(kind) }} />{kind}</span>
        ))}
      </div>
      <div ref={readout} className="readout" />
    </div>
  );
}

function line(tag: "strong" | "span", text: string): HTMLElement {
  const element = document.createElement(tag);
  element.textContent = text;
  return element;
}

function elementBox(branch: BranchModel, i: number): Box {
  let uMin = Infinity, uMax = -Infinity, vMin = Infinity, vMax = -Infinity;
  for (let j = branch.offsets[i]!; j < branch.offsets[i + 1]!; j++) {
    uMin = Math.min(uMin, branch.z[j]!);
    uMax = Math.max(uMax, branch.z[j]!);
    vMin = Math.min(vMin, branch.x[j]!);
    vMax = Math.max(vMax, branch.x[j]!);
  }
  return { uMin, uMax, vMin, vMax };
}

function drawGrid(ctx: CanvasRenderingContext2D, v: View2D, width: number, height: number, t: Theme) {
  const step = niceStep(80 / v.k);
  const [u0, v1] = toWorld(v, width, height, 0, 0);
  const [u1, v0] = toWorld(v, width, height, width, height);
  ctx.strokeStyle = t.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let u = Math.ceil(u0 / step) * step; u <= u1; u += step) {
    const [sx] = toScreen(v, width, height, u, 0);
    ctx.moveTo(Math.round(sx) + 0.5, 0);
    ctx.lineTo(Math.round(sx) + 0.5, height);
  }
  for (let w = Math.ceil(v0 / step) * step; w <= v1; w += step) {
    const [, sy] = toScreen(v, width, height, 0, w);
    ctx.moveTo(0, Math.round(sy) + 0.5);
    ctx.lineTo(width, Math.round(sy) + 0.5);
  }
  ctx.stroke();
}

function drawScaleBar(ctx: CanvasRenderingContext2D, v: View2D, width: number, height: number, t: Theme) {
  const metres = niceStep(120 / v.k);
  const pixels = metres * v.k;
  const x = width - pixels - 16, y = height - 18;
  ctx.strokeStyle = t.text;
  ctx.fillStyle = t.text;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, y - 4);
  ctx.lineTo(x, y);
  ctx.lineTo(x + pixels, y);
  ctx.lineTo(x + pixels, y - 4);
  ctx.stroke();
  ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(metres >= 1 ? `${metres} m` : `${metres * 100} cm`, x + pixels / 2, y - 6);
  // Axes: z to the right, x up.
  const ax = 18, ay = height - 18;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(ax + 22, ay);
  ctx.moveTo(ax, ay);
  ctx.lineTo(ax, ay - 22);
  ctx.stroke();
  ctx.textAlign = "left";
  ctx.fillText("z", ax + 25, ay + 4);
  ctx.fillText("x", ax - 3, ay - 26);
}

/** Screen width of an element band: its physical width, but never thinner than a few pixels. */
function bandWidth(kind: string, v: View2D, isCurrent: boolean): number {
  return Math.max(kindStyle(kind).floorWidth * v.k, isCurrent ? 4 : 3);
}

/** Elements without a band (BPMs, markers) as small dots, as in the earlier viewer; forks as ticks across the beam. */
function drawMarkers(
  ctx: CanvasRenderingContext2D,
  branch: BranchModel,
  v: View2D,
  width: number,
  height: number,
  isCurrent: boolean,
  t: Theme,
) {
  if (v.k < 0.4) return;
  const { markers } = layers(branch);
  ctx.lineWidth = 1.5;
  for (const i of markers) {
    const kind = branch.kinds[i]!;
    if (kind === "Marker" && v.k < 20) continue;
    const j = branch.offsets[i]!;
    const [sx, sy] = toScreen(v, width, height, branch.z[j]!, branch.x[j]!);
    if (sx < -10 || sy < -10 || sx > width + 10 || sy > height + 10) continue;
    const color = kindColor(kind, t.dark);
    if (kind === "Fork") {
      const theta = branch.theta[i]!;
      const nx = Math.sin(theta), ny = Math.cos(theta); // across the beam, on screen
      ctx.globalAlpha = isCurrent ? 1 : 0.55;
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(sx - nx * 8, sy - ny * 8);
      ctx.lineTo(sx + nx * 8, sy + ny * 8);
      ctx.stroke();
    } else {
      ctx.globalAlpha = isCurrent ? 0.8 : 0.45;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(sx, sy, 2, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1;
}

function drawClosures(
  ctx: CanvasRenderingContext2D,
  summary: LatticeSummary,
  branches: Map<string, BranchModel>,
  v: View2D,
  width: number,
  height: number,
  t: Theme,
) {
  for (const closure of summary.closures.filter(isOpen)) {
    const fork = summary.forks.find((f) => f.id === closure.fork);
    const branch = fork && branches.get(fork.from_branch);
    const i = branch?.index.get(closure.fork);
    if (!branch || i === undefined) continue;
    const j = branch.offsets[i]!;
    const [sx, sy] = toScreen(v, width, height, branch.z[j]!, branch.x[j]!);
    ctx.strokeStyle = t.danger;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(sx, sy, 9, 0, 2 * Math.PI);
    ctx.stroke();
    if (v.k > 2) {
      ctx.fillStyle = t.danger;
      ctx.font = "11px system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(
        `closure ${formatNumber(closure.position_error * 1e3, 3)} mm, ${formatNumber(closure.angle_error * 1e3, 3)} mrad`,
        sx + 12,
        sy - 10,
      );
    }
  }
}

function drawLabels(
  ctx: CanvasRenderingContext2D,
  branches: BranchModel[],
  current: string | null,
  v: View2D,
  width: number,
  height: number,
  t: Theme,
) {
  if (v.k < 3) return;
  ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  const taken = new Set<number>();
  const cell = (x: number, y: number) => Math.floor(x / 90) * 4096 + Math.floor(y / 16);
  for (const branch of [...branches].reverse()) {
    ctx.fillStyle = branch.id === current ? t.text : t.muted;
    for (let i = 0; i < branch.n; i++) {
      const style = kindStyle(branch.kinds[i]!);
      if (style.priority < 2 || (branch.kinds[i] === "Instrument" && v.k < 12)) continue;
      const first = branch.offsets[i]!, last = branch.offsets[i + 1]! - 1;
      const [sx, sy] = toScreen(v, width, height, (branch.z[first]! + branch.z[last]!) / 2, (branch.x[first]! + branch.x[last]!) / 2);
      if (sx < 0 || sy < 0 || sx > width || sy > height) continue;
      const theta = branch.theta[i]!;
      const off = Math.max(style.floorWidth * v.k, 4) / 2 + 6;
      const lx = sx + Math.sin(theta) * off, ly = sy + Math.cos(theta) * off;
      const key = cell(lx, ly);
      if (taken.has(key) || taken.has(key + 1) || taken.has(key - 1)) continue;
      taken.add(key);
      ctx.fillText(branch.names[i]!, lx, ly);
    }
  }
  ctx.textBaseline = "alphabetic";
}

/** Hover or selection: the element redrawn at full strength with a thin outline of ``color`` around it. */
function outline(
  ctx: CanvasRenderingContext2D,
  branches: Map<string, BranchModel>,
  target: Target,
  v: View2D,
  width: number,
  height: number,
  t: Theme,
  color: string,
  current: string | null,
) {
  const branch = branches.get(target.branch);
  const kind = branch?.kinds[target.index];
  if (!branch || !kind || isDrift(kind)) return;
  const first = branch.offsets[target.index]!, stop = branch.offsets[target.index + 1]!;
  const points: [number, number][] = [];
  for (let j = first; j < stop; j++) points.push(toScreen(v, width, height, branch.z[j]!, branch.x[j]!));
  const span = Math.hypot(points[0]![0] - points.at(-1)![0], points[0]![1] - points.at(-1)![1]);
  const fill = kindColor(kind, t.dark);
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = color;
  if (kindStyle(kind).floorWidth === 0 || span < 3) {
    const [sx, sy] = points[0]!;
    ctx.fillStyle = fill;
    ctx.beginPath();
    ctx.arc(sx, sy, 4, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
    return;
  }
  const band = bandWidth(kind, v, branch.id === current);
  const path = new Path2D();
  points.forEach(([x, y], k) => (k ? path.lineTo(x, y) : path.moveTo(x, y)));
  ctx.lineCap = "butt";
  ctx.strokeStyle = fill;
  ctx.lineWidth = band;
  ctx.stroke(path);
  ctx.lineJoin = "miter";
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.stroke(bandOutline(points, band / 2 + 1, 1));
  ctx.lineJoin = "round";
}

/** Closed outline of a band of half width ``half`` along ``points``, ``extend`` pixels past each end. */
function bandOutline(points: [number, number][], half: number, extend: number): Path2D {
  const n = points.length;
  const left: [number, number][] = [], right: [number, number][] = [];
  for (let k = 0; k < n; k++) {
    const [ax, ay] = points[Math.max(k - 1, 0)]!, [bx, by] = points[Math.min(k + 1, n - 1)]!;
    const length = Math.hypot(bx - ax, by - ay) || 1;
    const tx = (bx - ax) / length, ty = (by - ay) / length;
    const shift = k === 0 ? -extend : k === n - 1 ? extend : 0;
    const [px, py] = points[k]!;
    const cx = px + tx * shift, cy = py + ty * shift;
    left.push([cx - ty * half, cy + tx * half]);
    right.push([cx + ty * half, cy - tx * half]);
  }
  const path = new Path2D();
  [...left, ...right.reverse()].forEach(([x, y], k) => (k ? path.lineTo(x, y) : path.moveTo(x, y)));
  path.closePath();
  return path;
}
