// BPM orbits of the current ring or route, from a virtual machine's run (ADR
// 0007): x along s above, y below, N turns (a route's: shots) apart, the newest
// opaque and the older ones fading. A ring's run records the orbit turn after
// turn; a route's records each shot's trajectory, a single pass through its
// branches. While the run goes on the view follows its latest turns; paused,
// it steps back and forth through what the run has kept. It sits under the
// lattice strip and shares its s range, scale and margins, so every BPM sits
// under its element, and the hover and selection of every view.
//
// The data are BPM readings as a BPM system gives them (mm, with each BPM's
// offset and noise), read by the server from the run's folder; the page asks
// for the few orbits it shows, never for the turn-by-turn history. Every user
// sees the same runs: the server only reads them.

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { api, type CaptureEntry, type OrbitFrames, type RunEntry } from "../api/client";
import { va, type MachineEntry } from "../api/va";
import { occurrenceAt, type BranchModel } from "../model/branch";
import { formatMetres, formatNumber, formatTick } from "../model/format";
import { isDrift } from "../model/kinds";
import { CURVES, niceAxis, type Axis } from "../model/optics";
import { fade, nearestBpm, readingExtent, runsFor } from "../model/orbits";
import { isRoute } from "../model/route";
import { Cancelled, type Auth } from "./Auth";
import { sameTarget, type Store, type Target } from "../model/store";
import { niceStep, S_MARGIN, sScale, type SRange, type ZoomHistory } from "../model/viewport";
import { attachSInteraction } from "./sInteraction";
import { readTheme, useCanvas, type Theme } from "./useCanvas";

const POLL_MS = 1000;
const TOP = 8;
const GAP = 12;
const BOTTOM = 34;
const PLANES = [
  { key: "x" as const, title: "x (mm)", curve: CURVES[0]! },
  { key: "y" as const, title: "y (mm)", curve: CURVES[1]! },
];

interface Props {
  digest: string;
  latticeName: string;
  branch: BranchModel | null;
  runs: RunEntry[];
  /** Shown: only then does it follow the run. */
  active?: boolean;
  /** The run to show first, e.g. a virtual accelerator machine's (from the URL). */
  initialRun?: string;
  /** The run of the machine the viewer works with: shown unless the user picked another. */
  preferRun?: string;
  /** The virtual accelerator's machine the viewer works with, and how to show what it tracks. */
  machine?: MachineEntry | null;
  onShowMachine?(): void;
  /** Who is logged in, asked when they capture a machine's buffer. */
  auth?: Auth;
  selection: Target | null;
  hover: Store<Target | null>;
  range: Store<SRange | null>;
  band: Store<SRange | null>;
  history: ZoomHistory;
  scheme: string;
  /** The lattice strip, drawn between the controls and the plot. */
  strip?: ReactNode;
  /** The panel's tabs, first in the controls. */
  tabs?: ReactNode;
  onSelect(target: Target | null): void;
}

const wholeRange = (b: BranchModel): SRange => ({ s0: 0, s1: Math.max(b.length, 1e-3) });

const CAPTURE_NAMES: Record<string, string> = { kick: "kick", inject: "injection", "post-mortem": "post-mortem", request: "request" };

/** "kick at turn 12288 (4096 turns)": what a capture is, for the data chooser. */
const captureLabel = (c: CaptureEntry) =>
  `${CAPTURE_NAMES[c.kind] ?? c.kind} at turn ${c.trigger} (${c.turns} turns${c.partial ? ", partial" : ""})`;

/** Events marked in the controls: what changed the beam. */
const EVENT_NAMES: Record<string, string> = { kick: "kick", settings: "settings", inject: "injection", loss: "loss" };

/** " · kick at 250, 2298 · loss at 31, 33 (+4)": the latest three of each kind. */
function eventText(events: readonly { kind: string; turn: number }[]): string {
  const byKind = new Map<string, number[]>();
  events.forEach((e) => byKind.set(e.kind, [...(byKind.get(e.kind) ?? []), e.turn]));
  return [...byKind]
    .map(([kind, turns]) => ` · ${EVENT_NAMES[kind]} at ${turns.slice(-3).join(", ")}${turns.length > 3 ? ` (+${turns.length - 3})` : ""}`)
    .join("");
}

/** A positive whole number from an input, or the fallback. */
const whole = (text: string, fallback: number, max: number) => {
  const n = Math.round(Number(text));
  return Number.isFinite(n) && n >= 1 ? Math.min(n, max) : fallback;
};

export function BpmView(props: Props) {
  const { digest, latticeName, branch, runs, selection, hover, range, band, history, scheme, onSelect } = props;
  const [every, setEvery] = useState(5);
  const [count, setCount] = useState(5);
  const [live, setLive] = useState(true);
  const [end, setEnd] = useState<number | null>(null);
  const [chosen, setChosen] = useState<string | null>(props.initialRun ?? null);
  const [frames, setFrames] = useState<OrbitFrames | null>(null);
  const [error, setError] = useState<string | null>(null);
  /** Where the orbits come from: the BPMs' buffer (the latest turns), the slow orbit, or a capture. */
  const [source, setSource] = useState("buffer");
  const [captures, setCaptures] = useState<CaptureEntry[]>([]);
  const awaiting = useRef<number | null>(null);
  const theme = useRef<Theme>(readTheme(null));
  const tooltip = useRef<HTMLDivElement | null>(null);
  const cursor = useRef<number | null>(null);

  // The runs of this ring or route: of this revision first, then of the same lattice and names.
  const candidates = useMemo(
    () => (branch ? runsFor(runs, digest, latticeName, branch).map((c) => c.run) : []),
    [runs, branch, digest, latticeName],
  );
  const run =
    candidates.find((r) => r.id === chosen) ?? candidates.find((r) => r.id === props.preferRun) ?? candidates[0] ?? null;
  const runId = run?.id ?? null;
  const otherRevision = !!run && run.lattice_digest !== digest;
  const unit = frames?.unit ?? run?.unit ?? "turn";

  const state = useRef({ branch, selection, frames });
  state.current = { branch, selection, frames };

  const active = props.active ?? true;
  const ring = run?.kind === "ring";
  const fixed = source.startsWith("capture:");
  useEffect(() => {
    setSource("buffer");
    setCaptures([]);
    setLive(true);
    setEnd(null);
  }, [runId]);
  // A ring run's captures, as they come: a new one asked for here is shown once saved.
  const loadCaptures = useCallback(() => {
    if (runId && ring) api.captures(runId).then(setCaptures, () => undefined);
  }, [runId, ring]);
  useEffect(() => {
    if (!active) return;
    loadCaptures();
    const timer = window.setInterval(loadCaptures, 3 * POLL_MS);
    return () => window.clearInterval(timer);
  }, [active, loadCaptures]);
  useEffect(() => {
    if (awaiting.current === null || captures.length <= awaiting.current) return;
    awaiting.current = null;
    setSource(`capture:${captures[captures.length - 1]!.id}`);
    setLive(false);
    setEnd(null);
  }, [captures]);
  useEffect(() => {
    if (!runId) {
      setFrames(null);
      return;
    }
    if (!active) return;
    let stop = false;
    let timer: number | undefined;
    const controller = new AbortController();
    const load = () =>
      api.orbits(runId, every, count, live ? undefined : (end ?? undefined), controller.signal, source).then(
        (next) => {
          if (stop) return;
          setFrames((previous) =>
            previous &&
            previous.run === next.run &&
            previous.end_turn === next.end_turn &&
            previous.every === next.every &&
            previous.turns.join() === next.turns.join()
              ? previous
              : next,
          );
          setError(null);
          if (live && !fixed) timer = window.setTimeout(load, next.state === "running" ? POLL_MS : 4 * POLL_MS);
        },
        (e: Error) => {
          if (stop || e.name === "AbortError") return;
          setError(e.message);
          if (live && !fixed) timer = window.setTimeout(load, 4 * POLL_MS);
        },
      );
    load();
    return () => {
      stop = true;
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [runId, every, count, live, end, active]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, { width, height }: { width: number; height: number }) => {
      const t = theme.current;
      ctx.fillStyle = t.background;
      ctx.fillRect(0, 0, width, height);
      const { branch, selection, frames } = state.current;
      const view = range.get();
      if (!branch || !view) return;
      const k = sScale(view, width);
      const toX = (s: number) => S_MARGIN + (s - view.s0) * k;
      const x0 = S_MARGIN, x1 = width - S_MARGIN;
      const h = (height - TOP - BOTTOM - GAP) / 2;
      if (x1 <= x0 || h <= 10) return;
      const sLeft = Math.max(view.s0, 0), sRight = Math.min(view.s1, branch.length);
      const step = niceStep(90 / k);
      const sTicks: number[] = [];
      for (let s = Math.ceil(sLeft / step) * step; s <= sRight + 1e-9; s += step) sTicks.push(s);
      ctx.font = "11px system-ui, sans-serif";

      PLANES.forEach((plane, p) => {
        const y0 = TOP + p * (h + GAP), y1 = y0 + h;
        const found = frames ? readingExtent(frames, plane.key, sLeft, sRight) : null;
        const axis: Axis = niceAxis(found?.[0] ?? -1, found?.[1] ?? 1, Math.max(2, Math.floor(h / 32)));
        const toY = (v: number) => y1 - ((v - axis.min) / (axis.max - axis.min)) * (y1 - y0);
        const ticks: number[] = [];
        for (let v = axis.min; v <= axis.max + axis.step / 2; v += axis.step) ticks.push(Math.abs(v) < axis.step * 1e-9 ? 0 : v);

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
          if (target && target.branch === branch.id) shade(branch.s[target.index]!, branch.lengths[target.index]!, color, alpha);
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
        for (const v of ticks) {
          const y = Math.round(toY(v)) + 0.5;
          ctx.moveTo(x0, y);
          ctx.lineTo(x1, y);
        }
        ctx.stroke();
        ctx.strokeStyle = t.axis;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(x0, Math.round(toY(0)) + 0.5);
        ctx.lineTo(x1, Math.round(toY(0)) + 0.5);
        ctx.stroke();
        ctx.setLineDash([]);

        // Orbits, oldest first and faintest; the newest on top, with its BPMs marked.
        if (frames) {
          const color = t.dark ? plane.curve.darkColor : plane.curve.color;
          const n = frames.turns.length;
          frames[plane.key].forEach((orbit, f) => {
            const newest = f === n - 1;
            ctx.globalAlpha = fade(f, n);
            ctx.strokeStyle = color;
            ctx.fillStyle = color;
            ctx.lineWidth = newest ? 2 : 1.2;
            ctx.beginPath();
            let drawing = false;
            orbit.forEach((v, i) => {
              if (v === null || v === undefined) {
                drawing = false;
                return;
              }
              const x = toX(frames.s[i]!), y = toY(v);
              if (drawing) ctx.lineTo(x, y);
              else ctx.moveTo(x, y);
              drawing = true;
            });
            ctx.stroke();
            orbit.forEach((v, i) => {
              if (v === null || v === undefined) return;
              ctx.beginPath();
              ctx.arc(toX(frames.s[i]!), toY(v), newest ? 2.8 : 1.6, 0, 2 * Math.PI);
              ctx.fill();
            });
          });
          ctx.globalAlpha = 1;
        }
        const s = cursor.current;
        if (s !== null && s >= sLeft && s <= sRight) {
          ctx.strokeStyle = t.axis;
          ctx.beginPath();
          ctx.moveTo(Math.round(toX(s)) + 0.5, y0);
          ctx.lineTo(Math.round(toX(s)) + 0.5, y1);
          ctx.stroke();
        }
        ctx.restore();

        // Frame, value axis and title.
        ctx.strokeStyle = t.axis;
        ctx.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0 - 1, y1 - y0 - 1);
        ctx.fillStyle = t.muted;
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.beginPath();
        for (const v of ticks) {
          const y = Math.round(toY(v)) + 0.5;
          ctx.moveTo(x0, y);
          ctx.lineTo(x0 - 4, y);
          ctx.fillText(formatTick(v, axis.step), x0 - 7, Math.min(Math.max(y, y0 + 5), y1 - 5));
        }
        ctx.stroke();
        ctx.save();
        ctx.translate(10, (y0 + y1) / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = "center";
        ctx.fillStyle = t.dark ? plane.curve.darkColor : plane.curve.color;
        ctx.font = "12px system-ui, sans-serif";
        ctx.fillText(plane.title, 0, 0);
        ctx.restore();
        ctx.font = "11px system-ui, sans-serif";
      });

      // The s axis, under the lower plot.
      const y1 = height - BOTTOM;
      ctx.strokeStyle = t.axis;
      ctx.fillStyle = t.muted;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.beginPath();
      for (const s of sTicks) {
        const x = Math.round(toX(s)) + 0.5;
        ctx.moveTo(x, y1);
        ctx.lineTo(x, y1 + 4);
        ctx.fillText(formatTick(s, step), x, y1 + 6);
      }
      ctx.stroke();
      ctx.fillStyle = t.text;
      ctx.font = "12px system-ui, sans-serif";
      ctx.textBaseline = "bottom";
      ctx.fillText("s (m)", (x0 + x1) / 2, height - 1);
      ctx.textBaseline = "alphabetic";
    },
    [hover, range, band],
  );

  const { canvasRef, sizeRef, requestDraw } = useCanvas(draw);

  useEffect(() => {
    theme.current = readTheme(canvasRef.current);
    requestDraw();
  }, [scheme, canvasRef, requestDraw]);
  useEffect(requestDraw, [branch, selection, frames, requestDraw]);
  useEffect(() => hover.subscribe(requestDraw), [hover, requestDraw]);
  useEffect(() => range.subscribe(requestDraw), [range, requestDraw]);
  useEffect(() => band.subscribe(requestDraw), [band, requestDraw]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const inPlot = (x: number, y: number) =>
      x >= S_MARGIN && x <= sizeRef.current.width - S_MARGIN && y >= TOP && y <= sizeRef.current.height - BOTTOM;
    /** The BPM within a few pixels of s, as an occurrence of the branch where its name is known. */
    const bpmAt = (s: number) => {
      const { frames, branch } = state.current;
      const view = range.get();
      if (!frames || !branch || !view) return null;
      const i = nearestBpm(frames.s, s, 8 / sScale(view, sizeRef.current.width));
      if (i === null) return null;
      const name = frames.bpms[i]!;
      const index = branch.labels.indexOf(name);
      return { i, name, target: index >= 0 ? { branch: branch.id, index } : null };
    };
    const pick = (s: number): Target | null => {
      const near = bpmAt(s);
      if (near?.target) return near.target;
      const b = state.current.branch;
      if (!b || s < 0 || s > b.length) return null;
      const i = occurrenceAt(b, s);
      return isDrift(b.kinds[i]!) ? null : { branch: b.id, index: i };
    };
    const hideTip = () => {
      if (tooltip.current) tooltip.current.hidden = true;
    };
    const showTip = (x: number, y: number, s: number) => {
      const tip = tooltip.current;
      const { frames } = state.current;
      const near = bpmAt(s);
      if (!tip || !frames || !near) {
        hideTip();
        return;
      }
      const reading = (v: number | null | undefined) => (v === null || v === undefined ? "—" : formatNumber(v, 4));
      const lines = [
        `${near.name} · s = ${formatMetres(frames.s[near.i]!, 3)} m`,
        ...frames.turns
          .map((turn, f) => {
            const when = frames.times?.[f];
            const label = when === undefined ? `${frames.unit} ${turn}` : new Date(when * 1000).toLocaleTimeString();
            return `${label}: x ${reading(frames.x[f]![near.i])}  y ${reading(frames.y[f]![near.i])} mm`;
          })
          .reverse(),
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

  const newest = frames?.turns[frames.turns.length - 1];
  const step = (direction: 1 | -1) => {
    if (!frames || newest === undefined) return;
    setLive(false);
    const next = Math.min(Math.max(newest + 1 + direction * every, frames.first_turn + 1), frames.end_turn);
    setEnd(next);
  };
  const marked = (frames?.events ?? []).filter((e) => e.kind in EVENT_NAMES);
  /** The viewer's machine, when it tracks something else than what is shown and there is no run here. */
  const elsewhere = !run && props.machine && props.onShowMachine ? props.machine : null;
  const route = isRoute(branch);

  let message: string | null = null;
  if (!branch) message = null;
  else if (!branch.periodic && !route)
    message =
      "A line's BPMs read one pass at a time: its trajectories are shown for a route through it. " +
      "Name one in routes.yaml (a route may be a single line) and choose it above.";
  else if (!run)
    message = route
      ? `No run of route ${branch.name} on the server. Start one with \`eicvibe-virtual route MACHINE --route ` +
        `${branch.name} --out DIR\` and serve it with \`eicvibe-web … --runs DIR\`.`
      : "No virtual-ring run of this ring on the server. Start one with `eicvibe-virtual ring MACHINE --branch " +
        `${branch.name} --out DIR\` and serve it with \`eicvibe-web … --runs DIR\`.`;
  else if (error) message = `Cannot read the run's orbits: ${error}`;
  else if (!frames) message = "Reading the run…";

  return (
    <div className="bpm">
      <div className="optics-bar bpm-bar">
        {props.tabs}
        {candidates.length > 1 ? (
          <select aria-label="Run" value={runId ?? ""} onChange={(e) => setChosen(e.target.value)}>
            {candidates.map((r) => (
              <option key={r.id} value={r.id}>
                {r.machine ? `${r.machine}: ` : ""}
                {r.id} ({r.engine}, {r.state})
              </option>
            ))}
          </select>
        ) : run ? (
          <span className="muted" title={`${run.engine}, ${run.fidelity}; started ${run.start}`}>
            {run.id}
          </span>
        ) : null}
        {run ? (
          <span className={`bpm-state ${frames?.state ?? run.state}`} data-testid="bpm-state">
            {frames?.state ?? run.state}
          </span>
        ) : null}
        {otherRevision ? <span className="muted" title="The run tracked another revision of this lattice">other revision</span> : null}
        {ring ? (
          <select
            aria-label="BPM data"
            value={source}
            title="The BPMs' buffer holds the latest turns; the slow orbit averages each second; a capture is the buffer frozen by a kick, injection, beam loss or request"
            onChange={(e) => {
              setSource(e.target.value);
              setLive(!e.target.value.startsWith("capture:"));
              setEnd(null);
            }}
          >
            <option value="buffer">turn by turn: the latest {run!.buffer_turns || ""} turns</option>
            {run!.slow_period > 0 ? <option value="slow">slow orbit: every {run!.slow_period} s</option> : null}
            {captures.map((c) => (
              <option key={c.id} value={`capture:${c.id}`}>
                {captureLabel(c)}
              </option>
            ))}
          </select>
        ) : null}
        {ring && props.auth && props.machine && run!.machine === props.machine.id && run!.state === "running" ? (
          <button
            type="button"
            title="Freeze the BPMs' buffer now and save it, as a BPM system's acquisition on request"
            onClick={() => {
              awaiting.current = captures.length;
              props
                .auth!.withLogin(`To capture ${props.machine!.name}'s BPM buffer, log in.`, () => va.acquire(props.machine!.id))
                .then(loadCaptures, (e: Error) => {
                  awaiting.current = null;
                  if (!(e instanceof Cancelled)) setError(e.message);
                });
            }}
          >
            Acquire
          </button>
        ) : null}
        {run && props.machine && run.id === props.machine.run && props.machine.warnings.length ? (
          <span className="admin-error" title={`Machine ${props.machine.name}:\n${props.machine.warnings.join("\n")}`} data-testid="bpm-warning">
            ⚠ {props.machine.warnings.length} warning{props.machine.warnings.length === 1 ? "" : "s"}
          </span>
        ) : null}
        <label title={`${unit === "shot" ? "Shots" : unit === "sample" ? "Samples" : "Turns"} between the orbits shown`}>
          every{" "}
          <input
            type="number"
            min={1}
            value={every}
            aria-label="Turns between orbits"
            onChange={(e) => setEvery(whole(e.target.value, every, 1_000_000))}
          />{" "}
          {unit}s
        </label>
        <label title="Orbits shown; older ones fade">
          last{" "}
          <input
            type="number"
            min={1}
            max={50}
            value={count}
            aria-label="Orbits shown"
            onChange={(e) => setCount(whole(e.target.value, count, 50))}
          />
        </label>
        <span className="bpm-turns" data-testid="bpm-turns">
          {frames
            ? `${unit}s ${frames.turns[0]}–${newest} of ${fixed ? `${frames.first_turn}–${frames.end_turn}` : frames.end_turn}` +
              (frames.sample_period ? ` (${frames.sample_period} s apart)` : "") +
              eventText(marked)
            : ""}
        </span>
        <span className="bpm-controls">
          <button type="button" onClick={() => step(-1)} disabled={!frames} title={`${every} ${unit}s back`} aria-label="Back">
            ◀
          </button>
          <button
            type="button"
            onClick={() => {
              setLive((l) => !l);
              setEnd(newest !== undefined ? newest + 1 : null);
            }}
            disabled={!run || fixed}
            aria-pressed={live}
            title={live ? `Pause on these ${unit}s` : `Follow the run's latest ${unit}s`}
          >
            {live ? "Pause" : "Live"}
          </button>
          <button type="button" onClick={() => step(1)} disabled={!frames || live} title={`${every} ${unit}s on`} aria-label="Forward">
            ▶
          </button>
        </span>
      </div>
      {props.strip ? <div className="optics-strip">{props.strip}</div> : null}
      <div className="optics-plot">
        <canvas ref={canvasRef} className="optics-canvas" aria-label="BPM orbits along s" data-testid="bpm-canvas" />
        <div ref={tooltip} className="tooltip" hidden />
        {message ? (
          <div className="optics-message" data-testid="bpm-message">
            <div>
              {elsewhere ? (
                <p>
                  The machine {elsewhere.name} tracks {elsewhere.kind === "route" ? "the route" : "the ring"} {elsewhere.target}.{" "}
                  <button type="button" onClick={props.onShowMachine}>
                    Show {elsewhere.target}
                  </button>
                </p>
              ) : null}
              <p>{message}</p>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
