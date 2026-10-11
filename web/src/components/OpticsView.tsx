// Design optics of the current branch along s, under the branch's lattice
// strip: beta functions on the left axis, dispersion on the right, and a
// ring's tunes, chromaticities and momentum compaction. Strip and plot share
// the s range, scale and margins, so every element sits over its stretch of
// the optics; they also share the hover and selection of every other view.
//
// Drag in either to zoom to an s range, right click to undo, shift-drag to
// pan, double-click for the whole branch (sInteraction). An engine worker on
// the server computes the optics, once per revision, engine and fidelity;
// TrackPad starts Julia on its first use.

import { useCallback, useEffect, useImperativeHandle, useRef, useState, type ReactNode, type Ref } from "react";

import { api, type EngineInfo } from "../api/client";
import { occurrenceAt, type BranchModel } from "../model/branch";
import { formatMetres, formatNumber, formatTick } from "../model/format";
import { isDrift } from "../model/kinds";
import {
  CURVES,
  extent,
  isFieldFree,
  niceAxis,
  opticsModel,
  pointAt,
  segmentKind,
  segmentValue,
  valueAt,
  type Axis,
  type OpticsModel,
} from "../model/optics";
import { isRoute } from "../model/route";
import { sameTarget, useStoreSelect, type Store, type Target } from "../model/store";
import { updateHash } from "../model/urlState";
import { niceStep, S_MARGIN, sScale, type SRange, type ZoomHistory } from "../model/viewport";
import { attachSInteraction } from "./sInteraction";
import { readTheme, useCanvas, type Theme } from "./useCanvas";

const FIDELITIES = ["fast", "standard", "precise"];
const ENGINE_NAMES: Record<string, string> = { xsuite: "Xsuite", trackpad: "TrackPad" };
/** Plot area inside the canvas: the s axis's ticks, labels and title go below it, px. */
const TOP = 8;
const BOTTOM = 34;
const TICK = 4;
const HINT = "Drag to zoom to a range · right click: undo · shift-drag: pan · double-click: all of it";

export interface OpticsViewHandle {
  /** The model drawn now, for tests and notebooks driving the page. */
  model(): OpticsModel | null;
}

interface Props {
  digest: string;
  branch: BranchModel | null;
  selection: Target | null;
  hover: Store<Target | null>;
  range: Store<SRange | null>;
  /** The s range being selected with the mouse, shown by every view along s. */
  band: Store<SRange | null>;
  history: ZoomHistory;
  scheme: string;
  initialEngine?: string;
  initialFidelity?: string;
  /** The machine's operating modes, whose settings the optics may have instead of the lattice's. */
  modes?: string[];
  initialMode?: string;
  /** The lattice strip, drawn between the controls and the plot. */
  strip?: ReactNode;
  /** The panel's tabs, first in the controls. */
  tabs?: ReactNode;
  onSelect(target: Target | null): void;
  handle?: Ref<OpticsViewHandle>;
}

interface Status {
  key: string;
  model: OpticsModel | null;
  loading: boolean;
  error: string | null;
}

const engineName = (name: string) => ENGINE_NAMES[name] ?? name;

/** The branch's kinds by occurrence, when the optics are of the same occurrences; else none. */
function kindsOf(branch: BranchModel, model: OpticsModel): readonly string[] {
  return model.occurrences.length === branch.n && model.occurrences[0] === branch.ids[0] ? branch.kinds : [];
}

const wholeRange = (b: BranchModel): SRange => ({ s0: 0, s1: Math.max(b.length, 1e-3) });

export function summaryText(model: OpticsModel): string {
  const f = (v: number | null, digits: number) => (v === null ? "–" : formatNumber(v, digits));
  if (!model.available) return "";
  if (model.periodic)
    return `Q ${f(model.tuneX, 6)}, ${f(model.tuneY, 6)} · ξ ${f(model.chromaX, 4)}, ${f(model.chromaY, 4)} · αc ${f(model.momentumCompaction, 4)}`;
  const { s, values } = model.exits;
  const n = s.length - 1;
  const mu = (key: string) => values[key]![n]! - values[key]![0]!;
  return `from ${model.origin} · μ ${f(mu("mu_x"), 5)}, ${f(mu("mu_y"), 5)} (2π)`;
}

/** The visible s range as text; re-renders by itself, not the whole panel, as the range changes. */
function RangeReadout({ range, branch }: { range: Store<SRange | null>; branch: BranchModel | null }) {
  const text = useStoreSelect(range, (r) =>
    r && branch
      ? `s ${formatMetres(Math.max(r.s0, 0), 2)} – ${formatMetres(Math.min(r.s1, branch.length), 2)} of ${formatMetres(branch.length, 2)} m`
      : "",
  );
  return (
    <span className="range-readout" title={HINT}>
      {text}
    </span>
  );
}

export function OpticsView(props: Props) {
  const { digest, branch, selection, hover, range, band, history, scheme, onSelect } = props;
  const [engines, setEngines] = useState<EngineInfo[] | null>(null);
  const [engine, setEngine] = useState(props.initialEngine ?? "xsuite");
  const [fidelity, setFidelity] = useState(props.initialFidelity ?? "standard");
  const modes = props.modes ?? [];
  const [chosenMode, setMode] = useState(props.initialMode ?? "");
  const mode = modes.includes(chosenMode) ? chosenMode : "";
  const [shown, setShown] = useState<Record<string, boolean>>({ beta_x: true, beta_y: true, eta_x: true, eta_y: false });
  const [status, setStatus] = useState<Status>({ key: "", model: null, loading: false, error: null });
  const [elapsed, setElapsed] = useState(0);
  const theme = useRef<Theme>(readTheme(null));
  const tooltip = useRef<HTMLDivElement | null>(null);
  const cursor = useRef<number | null>(null);
  const state = useRef({ branch, selection, model: status.model, shown });
  state.current = { branch, selection, model: status.model, shown };

  useEffect(() => {
    api.engines().then(setEngines, () => setEngines([]));
  }, []);
  useEffect(() => updateHash({ engine, fidelity, mode: mode || undefined }), [engine, fidelity, mode]);

  const branchId = branch?.id ?? null;
  useEffect(() => {
    if (!branchId) return;
    let stop = false;
    const key = `${digest}/${branchId}/${engine}/${fidelity}/${mode}`;
    setStatus((s) => ({ key, model: s.model, loading: true, error: null }));
    setElapsed(0);
    const started = performance.now();
    const timer = window.setInterval(() => setElapsed(Math.round((performance.now() - started) / 1000)), 1000);
    api
      .optics(digest, branchId, engine, fidelity, mode)
      .then(
        (data) => {
          if (stop) return;
          try {
            setStatus({ key, model: opticsModel(data), loading: false, error: null });
          } catch (e) {
            setStatus({ key, model: null, loading: false, error: (e as Error).message });
          }
        },
        (e: Error) => !stop && setStatus({ key, model: null, loading: false, error: e.message }),
      )
      .finally(() => window.clearInterval(timer));
    return () => {
      stop = true;
      window.clearInterval(timer);
    };
  }, [digest, branchId, engine, fidelity, mode]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, { width, height }: { width: number; height: number }) => {
      const t = theme.current;
      ctx.fillStyle = t.background;
      ctx.fillRect(0, 0, width, height);
      const { branch, selection, model, shown } = state.current;
      const view = range.get();
      if (!branch || !view) return;
      const k = sScale(view, width);
      const toX = (s: number) => S_MARGIN + (s - view.s0) * k;
      const x0 = S_MARGIN, x1 = width - S_MARGIN, y0 = TOP, y1 = height - BOTTOM;
      if (x1 <= x0 || y1 <= y0) return;
      const sLeft = Math.max(view.s0, 0), sRight = Math.min(view.s1, branch.length);
      const ready = !!model && model.branch === branch.id && model.available;
      const kinds = ready ? kindsOf(branch, model) : [];
      const points = ready ? model.points : null;
      const axisOf = (side: "left" | "right"): Axis | null => {
        if (!points) return null;
        const keys = CURVES.filter((c) => c.axis === side && shown[c.key]).map((c) => c.key);
        const found = extent(points, keys, sLeft, sRight, kinds);
        return found ? niceAxis(found[0], found[1], Math.max(2, Math.floor((y1 - y0) / 36))) : null;
      };
      const axes = { left: axisOf("left"), right: axisOf("right") };
      const toY = (axis: Axis, v: number) => y1 - ((v - axis.min) / (axis.max - axis.min)) * (y1 - y0);
      const step = niceStep(90 / k);
      const sTicks: number[] = [];
      for (let s = Math.ceil(sLeft / step) * step; s <= sRight + 1e-9; s += step) sTicks.push(s);
      const valueTicks = (axis: Axis) => {
        const out: number[] = [];
        for (let v = axis.min; v <= axis.max + axis.step / 2; v += axis.step) out.push(Math.abs(v) < axis.step * 1e-9 ? 0 : v);
        return out;
      };

      // Inside the plot area: bands, grid, curves and cursor.
      ctx.save();
      ctx.beginPath();
      ctx.rect(x0, y0, x1 - x0, y1 - y0);
      ctx.clip();
      const shade = (s0: number, length: number, color: string, alpha: number) => {
        ctx.globalAlpha = alpha;
        ctx.fillStyle = color;
        ctx.fillRect(toX(s0) - (length ? 0 : 1), y0, Math.max(length * k, 2), y1 - y0);
        ctx.globalAlpha = 1;
      };
      const element = (target: Target | null, color: string, alpha: number) => {
        if (target && target.branch === branch.id)
          shade(branch.s[target.index]!, branch.lengths[target.index]!, color, alpha);
      };
      element(selection, t.selection, 0.1);
      const hovered = hover.get();
      if (!sameTarget(hovered, selection)) element(hovered, t.highlight, 0.18);
      const selecting = band.get();
      if (selecting) shade(selecting.s0, selecting.s1 - selecting.s0, t.selection, 0.12);

      ctx.strokeStyle = t.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (const s of sTicks) {
        const x = Math.round(toX(s)) + 0.5;
        ctx.moveTo(x, y0);
        ctx.lineTo(x, y1);
      }
      const gridAxis = axes.left ?? axes.right;
      if (gridAxis)
        for (const v of valueTicks(gridAxis)) {
          const y = Math.round(toY(gridAxis, v)) + 0.5;
          ctx.moveTo(x0, y);
          ctx.lineTo(x1, y);
        }
      ctx.stroke();
      if (axes.right && axes.right.min < 0 && axes.right.max > 0) {
        ctx.strokeStyle = t.axis;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        const y = Math.round(toY(axes.right, 0)) + 0.5;
        ctx.moveTo(x0, y);
        ctx.lineTo(x1, y);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Curves: straight between the engine's points, exact parabolas for beta in field-free elements.
      if (points) {
        const n = points.s.length - 1;
        const first = Math.max(pointAt(points, sLeft) - 1, 0), last = Math.min(pointAt(points, sRight) + 1, n);
        ctx.lineWidth = 1.6;
        ctx.lineJoin = "round";
        for (const curve of CURVES) {
          const axis = axes[curve.axis];
          const v = points.values[curve.key];
          if (!shown[curve.key] || !axis || !v) continue;
          ctx.strokeStyle = t.dark ? curve.darkColor : curve.color;
          ctx.setLineDash(curve.dash);
          ctx.beginPath();
          ctx.moveTo(toX(points.s[first]!), toY(axis, v[first]!));
          for (let p = first; p < last; p++) {
            const length = points.s[p + 1]! - points.s[p]!, px = length * k;
            if (px > 6 && curve.key.startsWith("beta") && isFieldFree(segmentKind(points, p, kinds))) {
              const steps = Math.min(48, Math.ceil(px / 4));
              for (let j = 1; j < steps; j++) {
                const at = (length * j) / steps;
                ctx.lineTo(toX(points.s[p]! + at), toY(axis, segmentValue(points, curve.key, p, at, kinds)));
              }
            }
            ctx.lineTo(toX(points.s[p + 1]!), toY(axis, v[p + 1]!));
          }
          ctx.stroke();
        }
        ctx.setLineDash([]);

        const s = cursor.current;
        if (s !== null && s >= sLeft && s <= sRight) {
          const x = Math.round(toX(s)) + 0.5;
          ctx.strokeStyle = t.axis;
          ctx.beginPath();
          ctx.moveTo(x, y0);
          ctx.lineTo(x, y1);
          ctx.stroke();
          for (const curve of CURVES) {
            const axis = axes[curve.axis];
            if (!shown[curve.key] || !axis) continue;
            ctx.fillStyle = t.dark ? curve.darkColor : curve.color;
            ctx.beginPath();
            ctx.arc(x, toY(axis, valueAt(points, curve.key, s, kinds)), 3, 0, 2 * Math.PI);
            ctx.fill();
          }
        }
      }
      ctx.restore();

      // The frame, the axes' ticks and labels outside it, and their titles.
      ctx.strokeStyle = t.axis;
      ctx.lineWidth = 1;
      ctx.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0 - 1, y1 - y0 - 1);
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillStyle = t.muted;
      ctx.beginPath();
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (const s of sTicks) {
        const x = Math.round(toX(s)) + 0.5;
        ctx.moveTo(x, y1);
        ctx.lineTo(x, y1 + TICK);
        ctx.fillText(formatTick(s, step), x, y1 + TICK + 2);
      }
      ctx.textBaseline = "middle";
      const valueAxis = (axis: Axis | null, side: "left" | "right") => {
        if (!axis) return;
        ctx.textAlign = side === "left" ? "right" : "left";
        const edge = side === "left" ? x0 : x1;
        const out = side === "left" ? -1 : 1;
        for (const v of valueTicks(axis)) {
          const y = Math.round(toY(axis, v)) + 0.5;
          ctx.moveTo(edge, y);
          ctx.lineTo(edge + out * TICK, y);
          ctx.fillText(formatTick(v, axis.step), edge + out * (TICK + 3), Math.min(Math.max(y, y0 + 5), y1 - 5));
        }
      };
      valueAxis(axes.left, "left");
      valueAxis(axes.right, "right");
      ctx.stroke();

      ctx.fillStyle = t.text;
      ctx.font = "12px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText("s (m)", (x0 + x1) / 2, height - 1);
      const title = (text: string, x: number, angle: number) => {
        ctx.save();
        ctx.translate(x, (y0 + y1) / 2);
        ctx.rotate(angle);
        ctx.textBaseline = "middle";
        ctx.fillText(text, 0, 0);
        ctx.restore();
      };
      if (axes.left) title("β (m)", 10, -Math.PI / 2);
      if (axes.right) title("η (m)", width - 10, Math.PI / 2);
      ctx.textBaseline = "alphabetic";
    },
    [hover, range, band],
  );

  const { canvasRef, sizeRef, requestDraw } = useCanvas(draw);

  useEffect(() => {
    theme.current = readTheme(canvasRef.current);
    requestDraw();
  }, [scheme, canvasRef, requestDraw]);
  useEffect(requestDraw, [branch, selection, status, shown, requestDraw]);
  useEffect(() => hover.subscribe(requestDraw), [hover, requestDraw]);
  useEffect(() => range.subscribe(requestDraw), [range, requestDraw]);
  useEffect(() => band.subscribe(requestDraw), [band, requestDraw]);

  useImperativeHandle(props.handle, () => ({ model: () => state.current.model }), []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    /** The element under s, except drifts, which are never picked. */
    const pick = (s: number): Target | null => {
      const b = state.current.branch;
      if (!b || s < 0 || s > b.length) return null;
      const i = occurrenceAt(b, s);
      return isDrift(b.kinds[i]!) ? null : { branch: b.id, index: i };
    };
    const inPlot = (x: number, y: number) =>
      x >= S_MARGIN && x <= sizeRef.current.width - S_MARGIN && y >= TOP && y <= sizeRef.current.height - BOTTOM;
    const hideTip = () => {
      if (tooltip.current) tooltip.current.hidden = true;
    };
    const showTip = (x: number, y: number, s: number) => {
      const tip = tooltip.current;
      const { branch: b, model } = state.current;
      if (!tip) return;
      if (!b || !model || model.branch !== b.id || !model.available || s < 0 || s > b.length) {
        tip.hidden = true;
        return;
      }
      const kinds = kindsOf(b, model);
      const value = (key: string) => valueAt(model.points, key, s, kinds);
      const i = occurrenceAt(b, s);
      const lines = [
        `s = ${formatMetres(s, 3)} m${isDrift(b.kinds[i]!) ? "" : ` · ${b.labels[i]}`}`,
        `βx ${formatNumber(value("beta_x"), 5)} m   βy ${formatNumber(value("beta_y"), 5)} m`,
        `αx ${formatNumber(value("alpha_x"), 4)}   αy ${formatNumber(value("alpha_y"), 4)}`,
        `ηx ${formatNumber(value("eta_x"), 4)} m   ηy ${formatNumber(value("eta_y"), 4)} m`,
        `μx ${formatNumber(value("mu_x"), 5)}   μy ${formatNumber(value("mu_y"), 5)} (2π)`,
      ];
      tip.textContent = "";
      lines.forEach((text, k) => {
        const element = document.createElement(k ? "span" : "strong");
        element.textContent = text;
        tip.append(element);
      });
      tip.hidden = false;
      tip.style.transform = `translate(${Math.min(x + 14, sizeRef.current.width - tip.offsetWidth - 4)}px, ${Math.max(0, Math.min(y + 10, sizeRef.current.height - tip.offsetHeight - 4))}px)`;
    };
    return attachSInteraction({
      canvas,
      range,
      band,
      history,
      limits: () => (state.current.branch ? wholeRange(state.current.branch) : null),
      hover(x, y, s) {
        if (!inPlot(x, y)) {
          cursor.current = null;
          hover.set(null);
          hideTip();
        } else {
          cursor.current = s;
          hover.set(pick(s));
          showTip(x, y, s);
        }
        requestDraw();
      },
      leave() {
        cursor.current = null;
        hover.set(null);
        hideTip();
        requestDraw();
      },
      click(x, y, s) {
        if (inPlot(x, y)) onSelect(pick(s));
      },
      busy: hideTip,
    });
  }, [canvasRef, sizeRef, hover, range, band, history, onSelect, requestDraw]);

  const model = status.model && branch && status.model.branch === branch.id ? status.model : null;
  const info = engines?.find((e) => e.name === engine);
  let message: string | null = null;
  if (status.loading)
    message =
      `Computing optics with ${engineName(engine)} (${fidelity})… ${elapsed} s` +
      (engine === "trackpad" && elapsed >= 3 ? " — TrackPad starts Julia on its first use, which takes a little while" : "");
  else if (status.error) message = `Cannot compute optics with ${engineName(engine)}: ${status.error}`;
  else if (model && !model.available) message = `No optics for this branch: ${model.origin}`;

  return (
    <div className="optics">
      <div className="optics-bar">
        {props.tabs}
        <span className="optics-legend" role="group" aria-label="Optics curves">
          {CURVES.map((c) => (
            <button
              key={c.key}
              type="button"
              aria-pressed={!!shown[c.key]}
              title={`${model?.columns.get(c.key)?.description ?? c.label} (${c.axis} axis)`}
              onClick={() => setShown((s) => ({ ...s, [c.key]: !s[c.key] }))}
            >
              <i
                style={{
                  borderTopColor: scheme === "dark" ? c.darkColor : c.color,
                  borderTopStyle: c.dash.length ? "dashed" : "solid",
                }}
              />
              {c.label}
            </button>
          ))}
        </span>
        <span
          className="optics-summary"
          data-testid="optics-summary"
          title={model && !status.loading ? `${summaryText(model)} (${model.origin})` : ""}
        >
          {model && !status.loading ? summaryText(model) : ""}
        </span>
        <RangeReadout range={range} branch={branch} />
        <button
          type="button"
          title={`Whole ${isRoute(branch) ? "route" : "branch"} (double-click). ${HINT}`}
          onClick={() => {
            const shownRange = range.get();
            if (!branch) return;
            if (shownRange) history.push(shownRange);
            range.set(wholeRange(branch));
          }}
        >
          Whole {isRoute(branch) ? "route" : "branch"}
        </button>
        <span className="optics-engine">
          <select aria-label="Optics engine" value={engine} onChange={(e) => setEngine(e.target.value)}>
            {(engines ?? [{ name: engine, available: true, fidelities: FIDELITIES, method: "", version: null }]).map((e) => (
              <option key={e.name} value={e.name} disabled={!e.available} title={e.method}>
                {engineName(e.name)}
                {e.available ? "" : " (not installed)"}
              </option>
            ))}
          </select>
          <select aria-label="Optics fidelity" value={fidelity} onChange={(e) => setFidelity(e.target.value)}>
            {(info?.fidelities ?? FIDELITIES).map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
          {modes.length ? (
            <select
              aria-label="Optics settings"
              value={mode}
              title="Whose settings the optics have: the lattice's (the design), or an operating mode's (settings/<mode>.yaml)"
              onChange={(e) => setMode(e.target.value)}
            >
              <option value="">design settings</option>
              {modes.map((m) => (
                <option key={m} value={m}>
                  mode {m}
                </option>
              ))}
            </select>
          ) : null}
          <span
            className="muted optics-method"
            title={`${info?.method ?? ""}${model?.version ? `, ${engineName(model.engine)} ${model.version}` : ""}${model?.sampled ? "; sampled inside magnets at the integrator steps" : "; values at element exits"}`}
          >
            {model?.method === "TPSA" ? "TPSA" : model?.method ? "finite differences" : ""}
          </span>
        </span>
      </div>
      {props.strip ? <div className="optics-strip">{props.strip}</div> : null}
      <div className="optics-plot">
        <canvas ref={canvasRef} className="optics-canvas" aria-label="Optics along s" data-testid="optics-canvas" />
        <div ref={tooltip} className="tooltip" hidden />
        {message ? <div className="optics-message">{message}</div> : null}
      </div>
    </div>
  );
}
