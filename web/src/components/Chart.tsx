// A small x-y chart on a canvas: lines and points, vertical and horizontal
// markers, a linear or log y axis, and a hover readout of the nearest point.
// The Check tab draws its spectra, drifts and per-BPM comparisons with it.

import { useCallback, useEffect, useRef } from "react";

import { formatNumber, formatTick } from "../model/format";
import { niceAxis, type Axis } from "../model/optics";
import { readTheme, useCanvas, type Theme } from "./useCanvas";

export interface Series {
  label: string;
  color: string;
  darkColor: string;
  x: readonly number[];
  y: readonly (number | null | undefined)[];
  /** line, points, or both (the default). */
  style?: "line" | "points" | "both";
  dash?: number[];
}

export interface Marker {
  /** A vertical line at x, or a horizontal one at y. */
  x?: number;
  y?: number;
  label?: string;
  color: string;
  darkColor: string;
  dash?: number[];
}

interface Props {
  title: string;
  xLabel: string;
  yLabel: string;
  series: Series[];
  markers?: Marker[];
  /** A log10 y axis over these decades. */
  log?: [number, number];
  xDomain?: [number, number];
  /** Whether the y axis reaches zero (default); not for tunes. */
  yZero?: boolean;
  /** Names of the first series' points (BPMs), for the readout. */
  names?: readonly string[];
  scheme: string;
  testId?: string;
}

const LEFT = 52;
const RIGHT = 12;
const TOP = 8;
const BOTTOM = 32;

export function Chart(props: Props) {
  const theme = useRef<Theme>(readTheme(null));
  const state = useRef(props);
  state.current = props;
  const cursor = useRef<number | null>(null);
  const axisRef = useRef<Axis | null>(null);
  const readout = useRef<HTMLDivElement | null>(null);

  const draw = useCallback((ctx: CanvasRenderingContext2D, { width, height }: { width: number; height: number }) => {
    const t = theme.current;
    const { series, markers = [], log, xDomain, xLabel, yLabel } = state.current;
    ctx.fillStyle = t.background;
    ctx.fillRect(0, 0, width, height);
    const x0 = LEFT, x1 = width - RIGHT, y0 = TOP, y1 = height - BOTTOM;
    if (x1 - x0 < 40 || y1 - y0 < 30) return;
    const xs = series.flatMap((s) => s.x).concat(markers.flatMap((m) => (m.x === undefined ? [] : [m.x])));
    const ys = series
      .flatMap((s) => s.y.filter((v): v is number => typeof v === "number" && Number.isFinite(v)))
      .concat(markers.flatMap((m) => (m.y === undefined ? [] : [m.y])));
    const xAxis: Axis = xDomain
      ? { min: xDomain[0], max: xDomain[1], step: niceAxis(xDomain[0], xDomain[1], 6, false).step }
      : xs.length
        ? niceAxis(Math.min(...xs), Math.max(...xs), Math.max(2, Math.floor((x1 - x0) / 80)), false)
        : { min: 0, max: 1, step: 0.2 };
    const yAxis: Axis = log
      ? { min: log[0], max: log[1], step: 1 }
      : niceAxis(
          ys.length ? Math.min(...ys) : -1,
          ys.length ? Math.max(...ys) : 1,
          Math.max(2, Math.floor((y1 - y0) / 30)),
          state.current.yZero ?? true,
        );
    const toX = (v: number) => x0 + ((v - xAxis.min) / (xAxis.max - xAxis.min)) * (x1 - x0);
    const toY = (v: number) => {
      const w = log ? Math.log10(Math.max(v, 10 ** (log[0] - 1))) : v;
      return y1 - ((w - yAxis.min) / (yAxis.max - yAxis.min)) * (y1 - y0);
    };
    ctx.font = "11px system-ui, sans-serif";
    // Grid and axes.
    ctx.strokeStyle = t.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    const xTicks: number[] = [];
    for (let v = Math.ceil(xAxis.min / xAxis.step - 1e-9) * xAxis.step; v <= xAxis.max + 1e-9; v += xAxis.step) xTicks.push(v);
    const yTicks: number[] = [];
    for (let v = yAxis.min; v <= yAxis.max + 1e-9; v += yAxis.step) yTicks.push(Math.abs(v) < yAxis.step * 1e-9 ? 0 : v);
    for (const v of xTicks) {
      const x = Math.round(toX(v)) + 0.5;
      ctx.moveTo(x, y0);
      ctx.lineTo(x, y1);
    }
    for (const v of yTicks) {
      const y = Math.round(log ? y1 - ((v - yAxis.min) / (yAxis.max - yAxis.min)) * (y1 - y0) : toY(v)) + 0.5;
      ctx.moveTo(x0, y);
      ctx.lineTo(x1, y);
    }
    ctx.stroke();
    ctx.save();
    ctx.beginPath();
    ctx.rect(x0, y0, x1 - x0, y1 - y0);
    ctx.clip();
    for (const m of markers) {
      ctx.strokeStyle = t.dark ? m.darkColor : m.color;
      ctx.setLineDash(m.dash ?? [4, 3]);
      ctx.beginPath();
      if (m.x !== undefined) {
        ctx.moveTo(Math.round(toX(m.x)) + 0.5, y0);
        ctx.lineTo(Math.round(toX(m.x)) + 0.5, y1);
      } else if (m.y !== undefined) {
        ctx.moveTo(x0, Math.round(toY(m.y)) + 0.5);
        ctx.lineTo(x1, Math.round(toY(m.y)) + 0.5);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      if (m.label) {
        ctx.fillStyle = t.dark ? m.darkColor : m.color;
        ctx.textBaseline = "top";
        ctx.textAlign = "left";
        if (m.x !== undefined) ctx.fillText(m.label, toX(m.x) + 3, y0 + 2);
        else if (m.y !== undefined) ctx.fillText(m.label, x0 + 4, toY(m.y) + 2);
      }
    }
    for (const s of series) {
      const color = t.dark ? s.darkColor : s.color;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash(s.dash ?? []);
      if (s.style !== "points") {
        ctx.beginPath();
        let drawing = false;
        s.x.forEach((xv, i) => {
          const yv = s.y[i];
          if (yv === null || yv === undefined || !Number.isFinite(yv) || (log && yv <= 0)) {
            drawing = false;
            return;
          }
          if (drawing) ctx.lineTo(toX(xv), toY(yv));
          else ctx.moveTo(toX(xv), toY(yv));
          drawing = true;
        });
        ctx.stroke();
      }
      ctx.setLineDash([]);
      if (s.style !== "line") {
        s.x.forEach((xv, i) => {
          const yv = s.y[i];
          if (yv === null || yv === undefined || !Number.isFinite(yv)) return;
          ctx.beginPath();
          ctx.arc(toX(xv), toY(yv), s.style === "points" ? 3 : 2.2, 0, 2 * Math.PI);
          ctx.fill();
        });
      }
    }
    const c = cursor.current;
    if (c !== null) {
      ctx.strokeStyle = t.axis;
      ctx.beginPath();
      ctx.moveTo(Math.round(toX(c)) + 0.5, y0);
      ctx.lineTo(Math.round(toX(c)) + 0.5, y1);
      ctx.stroke();
    }
    ctx.restore();
    ctx.strokeStyle = t.axis;
    ctx.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0 - 1, y1 - y0 - 1);
    ctx.fillStyle = t.muted;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const v of xTicks) ctx.fillText(formatTick(v, xAxis.step), toX(v), y1 + 4);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const v of yTicks) {
      const y = log ? y1 - ((v - yAxis.min) / (yAxis.max - yAxis.min)) * (y1 - y0) : toY(v);
      ctx.fillText(log ? `1e${v}` : formatTick(v, yAxis.step), x0 - 5, Math.min(Math.max(y, y0 + 5), y1 - 5));
    }
    ctx.fillStyle = t.text;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(xLabel, (x0 + x1) / 2, height - 1);
    ctx.save();
    ctx.translate(11, (y0 + y1) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = "middle";
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
    axisRef.current = xAxis;
  }, []);

  const { canvasRef, sizeRef, requestDraw } = useCanvas(draw);
  useEffect(() => {
    theme.current = readTheme(canvasRef.current);
    requestDraw();
  }, [props.scheme, canvasRef, requestDraw]);
  useEffect(requestDraw, [props.series, props.markers, props.log, props.xDomain, requestDraw]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const move = (event: MouseEvent) => {
      const axis = axisRef.current;
      const box = canvas.getBoundingClientRect();
      const px = event.clientX - box.left;
      const { width } = sizeRef.current;
      const tip = readout.current;
      const { series, names } = state.current;
      const first = series[0];
      if (!axis || !tip || !first || !first.x.length || px < LEFT || px > width - RIGHT) {
        cursor.current = null;
        if (tip) tip.hidden = true;
        requestDraw();
        return;
      }
      const v = axis.min + ((px - LEFT) / (width - LEFT - RIGHT)) * (axis.max - axis.min);
      let best = 0;
      first.x.forEach((xv, i) => {
        if (Math.abs(xv - v) < Math.abs(first.x[best]! - v)) best = i;
      });
      cursor.current = first.x[best]!;
      tip.textContent = [
        `${names?.[best] ? `${names[best]} · ` : ""}${formatNumber(first.x[best]!, 6)}`,
        ...series.map((s) => {
          const j = s === first ? best : s.x.indexOf(first.x[best]!);
          const yv = j >= 0 ? s.y[j] : undefined;
          return `${s.label}: ${yv === null || yv === undefined ? "—" : formatNumber(yv, 5)}`;
        }),
      ].join("  ");
      tip.hidden = false;
      requestDraw();
    };
    const leave = () => {
      cursor.current = null;
      if (readout.current) readout.current.hidden = true;
      requestDraw();
    };
    canvas.addEventListener("mousemove", move);
    canvas.addEventListener("mouseleave", leave);
    return () => {
      canvas.removeEventListener("mousemove", move);
      canvas.removeEventListener("mouseleave", leave);
    };
  }, [canvasRef, sizeRef, requestDraw]);

  return (
    <figure className="chart" data-testid={props.testId}>
      <figcaption>
        <span className="chart-title">{props.title}</span>
        {props.series.map((s) => (
          <span key={s.label} className="chart-key">
            <i style={{ background: s.color }} /> {s.label}
          </span>
        ))}
      </figcaption>
      <div className="chart-plot">
        <canvas ref={canvasRef} aria-label={props.title} />
        <div ref={readout} className="chart-readout" hidden />
      </div>
    </figure>
  );
}
