// The current branch laid out along s: the synoptic strip of a beamline, drawn
// directly above the optics plot and sharing its s scale and margins, so every
// element sits over its stretch of the optics.
//
// Dipoles straddle the axis; quadrupoles and other multipoles sit above it
// when focusing (in x) and below when defocusing; BPMs and markers are ticks.
// Forks out of and into the branch are drawn as flags that lead to the other
// branch. Only the occurrences inside the visible range are drawn, found by
// binary search on s. Mouse and wheel work as in the optics plot
// (sInteraction): drag to zoom to a range, right click to undo.

import { useCallback, useEffect, useImperativeHandle, useRef, type Ref } from "react";

import type { LatticeSummary } from "../api/client";
import { occurrenceAt, strengthText, type BranchModel } from "../model/branch";
import { formatMetres, formatNumber } from "../model/format";
import { kindColor, kindStyle } from "../model/kinds";
import { isRoute, routePasses } from "../model/route";
import { sameTarget, type Store, type Target } from "../model/store";
import { clampRange, niceStep, S_MARGIN as MARGIN, type SRange, type ZoomHistory } from "../model/viewport";
import { attachSInteraction } from "./sInteraction";
import { readTheme, useCanvas, type Theme } from "./useCanvas";

/** Height of the fork flags' row at the top, px. */
const FLAGS = 15;

/** Width the visible s range spans, px (the optics plot below uses the same, so s lines up). */
const span = (width: number) => Math.max(width - 2 * MARGIN, 1);

export interface LinearViewHandle {
  showAll(): void;
  show(index: number, zoom?: boolean): void;
}

interface Props {
  summary: LatticeSummary;
  branch: BranchModel | null;
  selection: Target | null;
  hover: Store<Target | null>;
  range: Store<SRange | null>;
  /** The s range being selected with the mouse, shown by every view along s. */
  band: Store<SRange | null>;
  history: ZoomHistory;
  scheme: string;
  onSelect(target: Target | null): void;
  onFollowFork(fork: { branch: string; occurrence: string | null; s: number }): void;
  handle?: Ref<LinearViewHandle>;
}

interface Flag {
  s: number;
  label: string;
  branch: string;
  occurrence: string | null;
  landing: number;
  /** Clickable extent of the drawn flag and its label, px; undefined when out of view. */
  hit?: [number, number];
}

export function LinearView(props: Props) {
  const { summary, branch, selection, hover, range, band, history, scheme, onSelect, onFollowFork } = props;
  const theme = useRef<Theme>(readTheme(null));
  const tooltip = useRef<HTMLDivElement | null>(null);
  const flags = useRef<Flag[]>([]);
  const state = useRef({ branch, selection });
  state.current = { branch, selection };

  const limits = useCallback((b: BranchModel): SRange => ({ s0: 0, s1: Math.max(b.length, 1e-3) }), []);

  useEffect(() => {
    if (!branch) {
      flags.current = [];
      return;
    }
    const names = new Map(summary.branches.map((b) => [b.id, b.name]));
    if (isRoute(branch)) {
      // A route's flags: every fork from the stretches it passes, and every landing on them but its own joins.
      flags.current = [
        ...summary.forks.flatMap((f) => {
          // every pass: a route going round a ring passes its forks on each turn
          return routePasses(branch, f.from_branch, f.s_from).map((s) => ({
            s,
            label: `→ ${names.get(f.to_branch)}`,
            branch: f.to_branch,
            occurrence: f.to_occurrence,
            landing: f.s_to,
          }));
        }),
        ...summary.forks
          .filter((f) => !branch.index.has(f.id))
          .flatMap((f) => {
            return routePasses(branch, f.to_branch, f.s_to).map((s) => ({
              s,
              label: `← ${names.get(f.from_branch)}`,
              branch: f.from_branch,
              occurrence: f.id,
              landing: f.s_from,
            }));
          }),
      ];
      return;
    }
    flags.current = [
      ...summary.forks
        .filter((f) => f.from_branch === branch.id)
        .map((f) => ({ s: f.s_from, label: `→ ${names.get(f.to_branch)}`, branch: f.to_branch, occurrence: f.to_occurrence, landing: f.s_to })),
      ...summary.forks
        .filter((f) => f.to_branch === branch.id)
        .map((f) => ({ s: f.s_to, label: `← ${names.get(f.from_branch)}`, branch: f.from_branch, occurrence: f.id, landing: f.s_from })),
    ];
  }, [summary, branch]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, { width, height }: { width: number; height: number }) => {
      const t = theme.current;
      ctx.fillStyle = t.background;
      ctx.fillRect(0, 0, width, height);
      const { branch, selection } = state.current;
      const shown = range.get();
      if (!branch || !shown) return;
      const { s0, s1 } = shown;
      const k = span(width) / (s1 - s0);
      const toX = (s: number) => MARGIN + (s - s0) * k;
      // Elements are drawn over the visible range only: the margins hold the optics plot's axes.
      const sLeft = Math.max(s0, 0), sRight = Math.min(s1, branch.length);
      const top = FLAGS, axis = top + (height - top) / 2;
      // Height of a dipole block; other kinds are fractions of it (kinds.ts), as in the earlier viewer.
      const dipole = Math.min(40, Math.max(12, (height - top) * 0.62));

      /** Where occurrence i is drawn, or null for kinds that are not (drifts). */
      const block = (i: number) => {
        const style = kindStyle(branch.kinds[i]!);
        if (style.lane === "none") return null;
        const x0 = toX(branch.s[i]!), w = branch.lengths[i]! * k, h = style.height * dipole;
        if (style.lane === "tick" || w === 0) return { x: Math.round(x0) - 0.75, y: axis - h / 2, w: 1.5, h, below: false };
        const polarity = style.lane === "polarity" ? branch.polarity[i]! : 0;
        const y = polarity > 0 ? axis - h : polarity < 0 ? axis : axis - h / 2;
        return { x: x0, y, w: Math.max(w, 1.2), h, below: polarity < 0 };
      };

      ctx.save();
      ctx.beginPath();
      ctx.rect(MARGIN, 0, width - 2 * MARGIN, height);
      ctx.clip();

      // The s grid of the optics plot below, continued through the strip.
      const step = niceStep(90 / k);
      ctx.strokeStyle = t.grid;
      ctx.font = "11px system-ui, sans-serif";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let s = Math.ceil(sLeft / step) * step; s <= sRight + 1e-9; s += step) {
        const x = Math.round(toX(s)) + 0.5;
        ctx.moveTo(x, top);
        ctx.lineTo(x, height);
      }
      ctx.stroke();
      const selecting = band.get();
      if (selecting) {
        ctx.fillStyle = t.selection;
        ctx.globalAlpha = 0.12;
        ctx.fillRect(toX(selecting.s0), 0, (selecting.s1 - selecting.s0) * k, height);
        ctx.globalAlpha = 1;
      }
      ctx.strokeStyle = t.axis;
      ctx.beginPath();
      ctx.moveTo(Math.max(toX(0), 0), axis + 0.5); // the beam line exists over the branch only
      ctx.lineTo(Math.min(toX(branch.length), width), axis + 0.5);
      ctx.stroke();

      // Elements in view
      const first = occurrenceAt(branch, sLeft);
      const last = occurrenceAt(branch, sRight);
      const labels: [number, number, string][] = [];
      ctx.globalAlpha = 0.85;
      for (let i = first; i <= last; i++) {
        const r = block(i);
        if (!r || (branch.kinds[i] === "Marker" && k < 20)) continue;
        ctx.fillStyle = kindColor(branch.kinds[i]!, t.dark);
        ctx.fillRect(r.x, r.y, r.w, r.h);
        if (r.w > 40) labels.push([r.x + r.w / 2, r.below ? r.y + r.h + 11 : r.y - 4, branch.labels[i]!]);
      }
      ctx.globalAlpha = 1;
      ctx.fillStyle = t.text;
      ctx.textAlign = "center";
      let lastRight = -Infinity;
      for (const [x, y, name] of labels) {
        const w = ctx.measureText(name).width;
        if (x - w / 2 < lastRight + 6) continue;
        ctx.fillText(name, x, y);
        lastRight = x + w / 2;
      }

      // Forks
      const forkColor = kindColor("Fork", t.dark);
      ctx.textAlign = "left";
      for (const flag of flags.current) {
        flag.hit = undefined;
        if (flag.s < sLeft || flag.s > sRight) continue;
        const x = Math.round(toX(flag.s)) + 0.5;
        const label = ctx.measureText(flag.label).width;
        const onRight = x + 3 + label <= width; // else the label goes left of its line
        flag.hit = onRight ? [x - 4, x + 3 + label] : [x - 3 - label, x + 4];
        ctx.strokeStyle = forkColor;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = forkColor;
        ctx.fillText(flag.label, onRight ? x + 3 : x - 3 - label, 11);
      }

      // Hover and selection: an outline around the element's own block.
      const mark = (target: Target | null, color: string, lineWidth: number) => {
        if (!target || target.branch !== branch.id) return;
        const r = block(target.index);
        if (!r || r.x + r.w < 0 || r.x > width) return;
        ctx.fillStyle = kindColor(branch.kinds[target.index]!, t.dark);
        ctx.fillRect(r.x, r.y, r.w, r.h);
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.strokeRect(r.x - 1.5, r.y - 1.5, r.w + 3, r.h + 3);
      };
      const hovered = hover.get();
      if (!sameTarget(hovered, selection)) mark(hovered, t.highlight, 2);
      mark(selection, t.selection, 1.5);
      ctx.restore();
    },
    [hover, range, band],
  );

  const { canvasRef, sizeRef, requestDraw } = useCanvas(draw);

  useEffect(() => {
    theme.current = readTheme(canvasRef.current);
    requestDraw();
  }, [scheme, canvasRef, requestDraw]);
  useEffect(requestDraw, [branch, selection, summary, requestDraw]);
  useEffect(() => hover.subscribe(requestDraw), [hover, requestDraw]);
  useEffect(() => range.subscribe(requestDraw), [range, requestDraw]);
  useEffect(() => band.subscribe(requestDraw), [band, requestDraw]);

  useImperativeHandle(
    props.handle,
    () => ({
      showAll() {
        const b = state.current.branch;
        const shown = range.get();
        if (!b) return;
        if (shown) history.push(shown);
        range.set(limits(b));
      },
      show(index: number, zoom = false) {
        const b = state.current.branch;
        const shown = range.get();
        if (!b || !shown) return;
        const s = b.s[index]!, end = s + b.lengths[index]!;
        const span = shown.s1 - shown.s0;
        if (zoom) {
          const width = Math.max(30, 6 * (end - s));
          range.set(clampRange({ s0: (s + end) / 2 - width / 2, s1: (s + end) / 2 + width / 2 }, limits(b)));
        } else if (end < shown.s0 || s > shown.s1) {
          range.set(clampRange({ s0: (s + end) / 2 - span / 2, s1: (s + end) / 2 + span / 2 }, limits(b)));
        }
      },
    }),
    [range, history, limits],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const pick = (x: number, y: number): Target | null => {
      const b = state.current.branch;
      const shown = range.get();
      if (!b || !shown || y < FLAGS) return null;
      const k = span(sizeRef.current.width) / (shown.s1 - shown.s0);
      const s = shown.s0 + (x - MARGIN) / k;
      const tolerance = 4 / k;
      let best: number | null = null, bestScore = Infinity;
      for (let i = occurrenceAt(b, s - tolerance); i < b.n && b.s[i]! <= s + tolerance; i++) {
        const style = kindStyle(b.kinds[i]!);
        if (style.lane === "none") continue; // drifts and other undrawn kinds are never picked
        const end = b.s[i]! + b.lengths[i]!;
        const d = s < b.s[i]! ? b.s[i]! - s : s > end ? s - end : 0;
        if (d > tolerance) continue;
        const score = d + (3 - style.priority) * tolerance;
        if (score < bestScore) {
          bestScore = score;
          best = i;
        }
      }
      return best === null ? null : { branch: b.id, index: best };
    };
    const flagAt = (x: number, y: number) =>
      y < FLAGS ? flags.current.find((f) => f.hit && x >= f.hit[0] && x <= f.hit[1]) : undefined;
    const showTip = (x: number, y: number, target: Target | null) => {
      const tip = tooltip.current;
      const b = state.current.branch;
      if (!tip) return;
      if (!b || !target) {
        tip.hidden = true;
        return;
      }
      const i = target.index;
      const strength = strengthText(b, i, (v) => formatNumber(v, 5));
      tip.textContent = "";
      const lines = [
        b.labels[i]!,
        ...(b.eicNames[i] ? [`lattice name ${b.names[i]}`] : []),
        `${b.kinds[i]} · s = ${formatMetres(b.s[i]!, 4)} … ${formatMetres(b.s[i]! + b.lengths[i]!, 4)} m`,
        ...(strength ? [strength] : []),
        ...(b.devices[i] ? [b.devices[i]!] : []),
        ...(b.components[i] ? [`one magnet with other lines: ${b.components[i]}`] : []),
      ];
      lines.forEach((text, k) => {
        const element = document.createElement(k ? "span" : "strong");
        element.textContent = text;
        tip.append(element);
      });
      tip.hidden = false;
      tip.style.transform = `translate(${Math.min(x + 14, sizeRef.current.width - tip.offsetWidth - 4)}px, ${y + 14}px)`;
    };
    const hideTip = () => {
      if (tooltip.current) tooltip.current.hidden = true;
    };
    return attachSInteraction({
      canvas,
      range,
      band,
      history,
      limits: () => (state.current.branch ? limits(state.current.branch) : null),
      hover(x, y) {
        canvas.style.cursor = flagAt(x, y) ? "pointer" : "";
        const target = pick(x, y);
        hover.set(target);
        showTip(x, y, target);
      },
      leave() {
        hover.set(null);
        hideTip();
      },
      click(x, y) {
        const flag = flagAt(x, y);
        if (flag) onFollowFork({ branch: flag.branch, occurrence: flag.occurrence, s: flag.landing });
        else onSelect(pick(x, y));
      },
      busy: hideTip,
    });
  }, [canvasRef, sizeRef, hover, range, band, history, limits, onSelect, onFollowFork]);

  return (
    <div className="linear">
      <canvas ref={canvasRef} className="linear-canvas" aria-label="Linear lattice view" data-testid="linear-canvas" />
      <div ref={tooltip} className="tooltip" hidden />
    </div>
  );
}
