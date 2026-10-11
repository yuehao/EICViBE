// The Check tab: a ring run's tracking against the model of its lattice and
// settings (ADR 0007), the measurement made on a real ring after a pinger's
// kick. The tune from the spectrum of the BPMs' turn-by-turn readings, the
// phase advance between BPMs and beta from the tune line, the closed orbit from
// the mean readings, and how tune and amplitude drift as the bunch decoheres;
// beside each, what the engine's optics give. A tune that moves with the
// amplitude is not an error of the tracking: the detuning curve, single
// particles tracked by the same engine, shows where it should be.
//
// The server reads the run's history and computes the model in an engine
// worker; the page asks again when its choices change, or on Check again.

import { useEffect, useState, type ReactNode } from "react";

import { api, type CaptureEntry, type EngineInfo, type RunCheck, type RunEntry } from "../api/client";
import type { BranchModel } from "../model/branch";
import { agrees, decades, lineOf, measuredPoint, tuneDifference, tuneText } from "../model/check";
import { formatNumber } from "../model/format";
import { CURVES } from "../model/optics";
import { runsFor } from "../model/orbits";
import type { AiStatus } from "../api/ai";
import { AiAnswer, aiName, useAnswer } from "./Ai";
import { Chart, type Marker, type Series } from "./Chart";

interface Props {
  /** The runs to choose from: the viewer's ring's, or every ring's on the check page. */
  runs: RunEntry[];
  /** What to say when there is none; null says nothing (no ring chosen). */
  empty: string | null;
  /** Shown: only then does it ask the server. */
  active?: boolean;
  /** The run to check unless the user picks another: the viewer's machine's, or the page's link's. */
  preferRun?: string;
  scheme: string;
  tabs?: ReactNode;
  /** A link to the check page, for room to read it (in the viewer's panel). */
  pageLink?: boolean;
  /** The optional local model (ADR 0009), to explain the check in a few sentences. */
  ai?: AiStatus | null;
}

/** The ring runs the viewer's Check tab offers for its ring, and what to say when there are none. */
export function viewerRuns(
  runs: RunEntry[],
  digest: string,
  latticeName: string,
  branch: BranchModel | null,
): { runs: RunEntry[]; empty: string | null } {
  if (!branch) return { runs: [], empty: null };
  if (!branch.periodic)
    return { runs: [], empty: "The check compares a ring's turn-by-turn BPM data with its optics: choose a ring with a run." };
  return {
    runs: runsFor(runs, digest, latticeName, branch)
      .map((c) => c.run)
      .filter((r) => r.kind === "ring"),
    empty:
      "No run of this ring on the server. Start one (a machine on the administration page, or `eicvibe-virtual ring`), kick its beam, then check it here.",
  };
}

const X = CURVES[0]!;
const Y = CURVES[1]!;
const PLANES = [
  { key: "x" as const, curve: X },
  { key: "y" as const, curve: Y },
];
const GREY = { color: "#6b7280", darkColor: "#9ca3af" };
const colors = (c: { color: string; darkColor: string }) => ({ color: c.color, darkColor: c.darkColor });

const CAPTURE_NAMES: Record<string, string> = { kick: "kick", inject: "injection", "post-mortem": "post-mortem", request: "request" };

/** A positive whole number from an input, or undefined. */
const whole = (text: string) => {
  const n = Math.round(Number(text));
  return text.trim() !== "" && Number.isFinite(n) && n >= 0 ? n : undefined;
};

const series = (label: string, curve: { color: string; darkColor: string }, x: readonly number[], y: readonly (number | null)[], extra: Partial<Series> = {}): Series => ({
  label,
  color: curve.color,
  darkColor: curve.darkColor,
  x,
  y,
  ...extra,
});

export function CheckView(props: Props) {
  const { runs: candidates, scheme } = props;
  const active = props.active ?? true;
  const [chosen, setChosen] = useState<string | null>(null);
  const [captures, setCaptures] = useState<CaptureEntry[]>([]);
  const [source, setSource] = useState("kick");
  const [turnsText, setTurnsText] = useState("1024");
  const [firstText, setFirstText] = useState("");
  const [turns, setTurns] = useState(1024);
  const [first, setFirst] = useState<number | undefined>(undefined);
  const [engine, setEngine] = useState("");
  const [engines, setEngines] = useState<EngineInfo[]>([]);
  const [detuning, setDetuning] = useState(false);
  const [again, setAgain] = useState(0);
  const [check, setCheck] = useState<RunCheck | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const explanation = useAnswer();

  const run = candidates.find((r) => r.id === chosen) ?? candidates.find((r) => r.id === props.preferRun) ?? candidates[0] ?? null;
  const runId = run?.id ?? null;

  useEffect(() => {
    if (active) api.engines().then((all) => setEngines(all.filter((e) => e.available)), () => undefined);
  }, [active]);
  useEffect(() => {
    setCaptures([]);
    setCheck(null);
    setSource("kick");
    if (active && runId) api.captures(runId).then(setCaptures, () => undefined);
  }, [runId, active]);

  useEffect(() => {
    if (!active || !runId) return;
    const controller = new AbortController();
    explanation.clear(); // it explained other numbers (clear is stable)
    setBusy(true);
    setError(null);
    api.check(runId, { source, turns, first, engine: engine || undefined, detuning }, controller.signal).then(
      (next) => {
        setCheck(next);
        setBusy(false);
      },
      (e: Error) => {
        if (e.name === "AbortError") return;
        setError(e.message);
        setBusy(false);
      },
    );
    return () => controller.abort();
  }, [active, runId, source, turns, first, engine, detuning, again, explanation.clear]);

  const commit = () => {
    const n = whole(turnsText);
    setTurns(n === undefined ? 1024 : Math.min(Math.max(n, 64), 8192));
    setFirst(whole(firstText));
  };

  let message: string | null = null;
  if (!run) message = props.empty;
  else if (error) message = `Cannot check the run: ${error}`;
  else if (!check) message = "Checking: reading the BPMs and computing the model…";

  return (
    <div className="bpm check">
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
          <span className="muted">{run.id}</span>
        ) : null}
        {run ? (
          <select
            aria-label="Data checked"
            value={source}
            title="A kick's capture from the kick on is what a pinger measurement uses; the buffer holds the latest turns"
            onChange={(e) => {
              setSource(e.target.value);
              setFirstText("");
              setFirst(undefined);
            }}
          >
            <option value="kick">the latest kick, from the kick on</option>
            <option value="buffer">the latest turns</option>
            {captures.map((c) => (
              <option key={c.id} value={`capture:${c.id}`}>
                {CAPTURE_NAMES[c.kind] ?? c.kind} at turn {c.trigger}
              </option>
            ))}
          </select>
        ) : null}
        <label title="Turns analysed (64 to 8192): more resolve the tune better, but the bunch decoheres">
          <input
            type="number"
            min={64}
            max={8192}
            aria-label="Turns checked"
            value={turnsText}
            onChange={(e) => setTurnsText(e.target.value)}
            onBlur={commit}
            onKeyDown={(e) => e.key === "Enter" && commit()}
          />{" "}
          turns
        </label>
        <label title="First turn analysed; empty: the kick, or the latest turns">
          from{" "}
          <input
            type="number"
            min={0}
            aria-label="First turn checked"
            placeholder="kick"
            value={firstText}
            onChange={(e) => setFirstText(e.target.value)}
            onBlur={commit}
            onKeyDown={(e) => e.key === "Enter" && commit()}
          />
        </label>
        <label title="The engine whose optics are the model: the run's own, or another to cross-check">
          model{" "}
          <select aria-label="Model engine" value={engine} onChange={(e) => setEngine(e.target.value)}>
            <option value="">{run ? `${run.engine} (the run's)` : "the run's"}</option>
            {engines
              .filter((e) => e.name !== run?.engine)
              .map((e) => (
                <option key={e.name} value={e.name}>
                  {e.name}
                </option>
              ))}
          </select>
        </label>
        <label title="Track 8 single particles with the model's engine, kicked up to the measured amplitude: the tune each amplitude should have">
          <input type="checkbox" checked={detuning} onChange={(e) => setDetuning(e.target.checked)} /> tune vs amplitude
        </label>
        <button type="button" disabled={!run || busy} onClick={() => setAgain((n) => n + 1)} title="Check the run again (its buffer moves on)">
          {busy ? "Checking…" : "Check again"}
        </button>
        {props.ai && run ? (
          <button
            type="button"
            disabled={!check || busy || explanation.busy || !props.ai.available}
            title={
              props.ai.available
                ? `${aiName(props.ai)} reads these numbers and writes a few sentences (seconds; it can be wrong)`
                : `No model answers now: ${props.ai.reason}`
            }
            onClick={() =>
              explanation.start("explain-check", { run: run.id, source, turns, first: first ?? null, engine: engine || null, detuning })
            }
          >
            ✦ Explain
          </button>
        ) : null}
        {props.pageLink && run ? (
          <a className="button" href={`check.html#run=${encodeURIComponent(run.id)}`} target="_blank" rel="noreferrer" title="The check on a page of its own, with room for its charts">
            Full page ↗
          </a>
        ) : null}
      </div>
      <div className="check-body">
        {message ? (
          <div className="optics-message" data-testid="check-message">
            <p>{message}</p>
          </div>
        ) : check ? (
          <>
            {props.ai ? (
              <AiAnswer status={props.ai} {...explanation} onStop={explanation.stop} testId="check-explanation" />
            ) : null}
            <CheckResult check={check} scheme={scheme} />
          </>
        ) : null}
      </div>
    </div>
  );
}

function CheckResult({ check, scheme }: { check: RunCheck; scheme: string }) {
  const where =
    check.trigger !== null && check.trigger !== undefined
      ? `turns ${check.first_turn}–${check.first_turn + check.turns} of the capture of turn ${check.trigger}`
      : `turns ${check.first_turn}–${check.first_turn + check.turns}`;
  const spectrumScale = decades(PLANES.flatMap((p) => check[p.key].spectrum));
  const markers = (lines: (number | null)[], labels: string[]): Marker[] =>
    lines.flatMap((x, i) => (x === null ? [] : [{ x, label: labels[i], ...colors(i ? Y : X) }]));
  const modelLines = PLANES.map((p) => lineOf(check[p.key].model_tune));
  const qs = check.synchrotron_tune === null || check.synchrotron_tune === undefined ? null : lineOf(check.synchrotron_tune);
  const drift = check.drift_turns;
  const detuning = check.detuning;
  return (
    <>
      <section className="check-summary" data-testid="check-summary">
        <table className="kv check-table">
          <thead>
            <tr>
              <th />
              <th title="From the BPM spectra; the integer part is the model's">measured tune</th>
              <th title="The engine's optics of the run's lattice and settings">model</th>
              <th title="The run's own one-turn map when it started (what its bunch was matched to)">run's map</th>
              <th>difference</th>
              <th title="Median over the BPMs, right after the kick and over the turns checked">amplitude (mm)</th>
              <th title="Measured phase advance from the first BPM minus the model's">phase rms</th>
              <th title="From the oscillation amplitudes; BPM gain errors (1 % rms) are in it">β-beat rms</th>
            </tr>
          </thead>
          <tbody>
            {PLANES.map(({ key }) => {
              const p = check[key];
              const difference = tuneDifference(p);
              return (
                <tr key={key} data-testid={`check-${key}`}>
                  <th>{key}</th>
                  <td>{tuneText(p.measured_tune, p.measured_full)}</td>
                  <td>{p.model_tune === null || p.model_tune === undefined ? "—" : p.model_tune.toFixed(4)}</td>
                  <td>{tuneText(p.map_tune)}</td>
                  <td className={difference === null ? "" : agrees(difference) ? "check-good" : "check-off"}>
                    {difference === null ? "—" : `${difference >= 0 ? "+" : ""}${difference.toFixed(4)}`}
                  </td>
                  <td>
                    {formatNumber(p.initial_amplitude, 3)} → {formatNumber(p.amplitude, 3)}
                  </td>
                  <td>{p.phase_rms === null || p.phase_rms === undefined ? "—" : `${p.phase_rms.toFixed(2)}°`}</td>
                  <td>{p.beta_beating_rms === null || p.beta_beating_rms === undefined ? "—" : `${(100 * p.beta_beating_rms).toFixed(1)} %`}</td>
                </tr>
              );
            })}
            <tr>
              <th>s</th>
              <td>—</td>
              <td colSpan={2}>{qs === null ? "—" : `${qs.toFixed(4)} (run's map)`}</td>
              <td colSpan={4} className="muted">
                The synchrotron tune: a sideband of the betatron lines at ±Qs.
              </td>
            </tr>
          </tbody>
        </table>
        <p className="muted">
          {where}; {check.bpms.length} BPMs; model: {check.engine} ({check.fidelity}), settings {check.settings}.
        </p>
        {check.notes.map((note) => (
          <p key={note} className="check-note">
            {note}
          </p>
        ))}
      </section>
      <div className="check-charts">
        <Chart
          title="Spectrum of the readings"
          testId="check-spectrum"
          xLabel="frequency (turns⁻¹)"
          yLabel="line (mm)"
          xDomain={[0, 0.5]}
          log={spectrumScale}
          series={PLANES.map((p) => series(p.key, p.curve, check.frequencies, check[p.key].spectrum, { style: "line" }))}
          markers={[
            ...markers(modelLines, ["model x", "model y"]),
            ...(qs === null ? [] : [{ x: qs, label: "Qs", ...GREY }]),
          ]}
          scheme={scheme}
        />
        <Chart
          title={`Tune over the turns (${check.drift_window}-turn stretches)`}
          testId="check-drift"
          xLabel="turn"
          yLabel="line"
          yZero={false}
          xDomain={[check.first_turn, check.first_turn + check.turns]}
          series={PLANES.map((p) => series(p.key, p.curve, drift, check[p.key].drift_tune))}
          markers={PLANES.flatMap((p, i) =>
            modelLines[i] === null ? [] : [{ y: modelLines[i]!, label: `model ${p.key}`, ...colors(p.curve) }],
          )}
          scheme={scheme}
        />
        <Chart
          title="Oscillation amplitude over the turns"
          xLabel="turn"
          xDomain={[check.first_turn, check.first_turn + check.turns]}
          yLabel="amplitude (mm)"
          series={PLANES.map((p) => series(p.key, p.curve, drift, check[p.key].drift_amplitude))}
          scheme={scheme}
        />
        {detuning ? (
          <Chart
            title={`Tune against amplitude: single particles kicked in ${detuning.plane}, ${detuning.turns} turns`}
            testId="check-detuning"
            xLabel={`amplitude in ${detuning.plane} (mm, median over the BPMs)`}
            yLabel="line"
            yZero={false}
            series={[
              series("model x", X, detuning.amplitude, detuning.tune_x),
              series("model y", Y, detuning.amplitude, detuning.tune_y),
              ...PLANES.flatMap((p) => {
                const point = measuredPoint(check, p.key);
                return point ? [series(`measured ${p.key}`, GREY, [point.amplitude], [point.tune], { style: "points" })] : [];
              }),
            ]}
            scheme={scheme}
          />
        ) : null}
        <Chart
          title="Phase advance: measured minus model"
          testId="check-phase"
          xLabel="s (m)"
          yLabel="degrees"
          names={check.bpms}
          series={PLANES.filter((p) => check[p.key].phase_error.length).map((p) => series(p.key, p.curve, check.s, check[p.key].phase_error))}
          markers={[{ y: 0, ...GREY, dash: [] }]}
          scheme={scheme}
        />
        <Chart
          title="Beta from the amplitudes: relative to the model"
          testId="check-beta"
          xLabel="s (m)"
          yLabel="Δβ/β (%)"
          names={check.bpms}
          series={PLANES.filter((p) => check[p.key].beta_beating.length).map((p) =>
            series(p.key, p.curve, check.s, check[p.key].beta_beating.map((v) => (v === null ? null : 100 * v))),
          )}
          markers={[{ y: 0, ...GREY, dash: [] }]}
          scheme={scheme}
        />
        <Chart
          title="Orbit: mean readings (points, the BPMs' offsets in them) and the model's closed orbit (dashed)"
          testId="check-orbit"
          xLabel="s (m)"
          yLabel="mm"
          names={check.bpms}
          series={PLANES.flatMap((p) => [
            series(`${p.key} measured`, p.curve, check.s, check[p.key].orbit, { style: "points" }),
            series(`${p.key} model`, p.curve, check.s, check[p.key].model_orbit, { style: "line", dash: [5, 3] }),
          ])}
          scheme={scheme}
        />
      </div>
    </>
  );
}
