// A machine's errors (ADR 0011): what makes the virtual machine differ from its
// model, as a real one does. Static errors of magnets chosen by kind and name
// (offsets, roll, pitch and yaw about the centre, the main field's error), power
// supplies whose current fluctuates (noise and ripple; every magnet of a supply
// together), and the BPMs' own errors and noise. They are drawn from the seed
// when the machine starts, the same every start; "New draw" picks another seed,
// another machine with the same tolerances. Only whoever may configure the
// machine sees them (administrators for a shared machine, its owner for a
// sandbox). The magnets' and supplies' change only while it is stopped; the
// BPMs', which read the beam and leave it alone, also while it runs.
//
// Shown in the units magnet and BPM people use: µm, mrad, 10⁻⁴ of the field,
// ppm of the current; the server keeps SI (metres, radians, fractions; BPMs in mm).

import { useEffect, useState } from "react";

import { va, VaError, type ErrorSpec, type ErrorsEntry, type MagnetErrors, type MachineEntry, type SupplyNoise } from "../api/va";

type Kind = NonNullable<MagnetErrors["select"]>["kinds"][number];
const KINDS: Kind[] = ["Bend", "Quadrupole", "Sextupole", "Octupole", "Kicker", "Solenoid", "Multipole"];

interface MagnetDraft {
  kinds: string[];
  names: string;
  x: string;
  y: string;
  roll: string;
  pitch: string;
  yaw: string;
  field: string;
  cut: string;
}

interface SupplyDraft {
  kinds: string[];
  names: string;
  rms: string;
  bandwidth: string;
  ripple: string;
}

interface Draft {
  seed: string;
  magnets: MagnetDraft[];
  supplies: SupplyDraft[];
  bpms: { offset: string; gain: string; roll: string; tbt: string; orbit: string };
}

/** ``value`` in a display unit (``scale`` of them per SI unit), as typed text. */
const shown = (value: number | null | undefined, scale: number) => (value ? String(Number((value * scale).toPrecision(6))) : "");

/** Typed text in a display unit back to SI; blank is 0. Throws on anything else. */
function parsed(text: string, scale: number, what: string): number {
  if (text.trim() === "") return 0;
  const n = Number(text);
  if (!Number.isFinite(n) || n < 0) throw new Error(`${what}: "${text}" is not a number ≥ 0`);
  return n / scale;
}

const patterns = (text: string) =>
  text
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

function draftOf(spec: ErrorSpec): Draft {
  return {
    seed: spec.seed === null || spec.seed === undefined ? "" : String(spec.seed),
    magnets: (spec.magnets ?? []).map((m) => ({
      kinds: [...(m.select?.kinds ?? [])],
      names: (m.select?.names ?? []).join(", "),
      x: shown(m.x_offset, 1e6),
      y: shown(m.y_offset, 1e6),
      roll: shown(m.roll, 1e3),
      pitch: shown(m.pitch, 1e3),
      yaw: shown(m.yaw, 1e3),
      field: shown(m.field, 1e4),
      cut: String(m.cut ?? 2.5),
    })),
    supplies: (spec.supplies ?? []).map((s) => ({
      kinds: [...(s.select?.kinds ?? [])],
      names: (s.select?.names ?? []).join(", "),
      rms: shown(s.rms, 1e6),
      bandwidth: s.bandwidth ? String(s.bandwidth) : "",
      ripple: (s.ripple ?? []).map((r) => `${r.frequency}:${Number((r.amplitude * 1e6).toPrecision(6))}`).join(", "),
    })),
    bpms: {
      offset: shown(spec.bpms?.offset_rms, 1e3),
      gain: shown(spec.bpms?.gain_rms, 1e2),
      roll: shown(spec.bpms?.roll_rms, 1e3),
      tbt: shown(spec.bpms?.resolution, 1e3),
      orbit: shown(spec.bpms?.orbit_resolution, 1e3),
    },
  };
}

function specOf(draft: Draft, base: ErrorSpec): ErrorSpec {
  const seed = draft.seed.trim() === "" ? null : Number(draft.seed);
  if (seed !== null && !Number.isInteger(seed)) throw new Error(`seed: "${draft.seed}" is not a whole number`);
  const magnets: MagnetErrors[] = draft.magnets.map((m, k) => {
    const row = `magnets, row ${k + 1}`;
    const cut = parsed(m.cut || "2.5", 1, `${row}, cut`);
    if (cut <= 0) throw new Error(`${row}: the cut must be above 0 sigma`);
    return {
      select: { kinds: m.kinds as Kind[], names: patterns(m.names), branches: [] },
      x_offset: parsed(m.x, 1e6, `${row}, x offset`),
      y_offset: parsed(m.y, 1e6, `${row}, y offset`),
      roll: parsed(m.roll, 1e3, `${row}, roll`),
      pitch: parsed(m.pitch, 1e3, `${row}, pitch`),
      yaw: parsed(m.yaw, 1e3, `${row}, yaw`),
      field: parsed(m.field, 1e4, `${row}, field`),
      cut,
    };
  });
  const supplies: SupplyNoise[] = draft.supplies.map((s, k) => {
    const row = `supplies, row ${k + 1}`;
    const ripple = patterns(s.ripple).map((line) => {
      const [f, a] = line.split(":").map((v) => v.trim());
      const frequency = Number(f);
      if (!Number.isFinite(frequency) || frequency <= 0 || a === undefined) throw new Error(`${row}: ripple "${line}" is not Hz:ppm`);
      return { frequency, amplitude: parsed(a, 1e6, `${row}, ripple`) };
    });
    const bandwidth = s.bandwidth.trim() === "" ? null : parsed(s.bandwidth, 1, `${row}, bandwidth`) || null;
    return {
      select: { kinds: s.kinds as Kind[], names: patterns(s.names), branches: [] },
      rms: parsed(s.rms, 1e6, `${row}, noise`),
      bandwidth,
      ripple,
    };
  });
  return {
    seed,
    magnets,
    supplies,
    bpms: {
      seed: base.bpms?.seed ?? 7,
      offset_rms: parsed(draft.bpms.offset, 1e3, "BPM offsets"),
      gain_rms: parsed(draft.bpms.gain, 1e2, "BPM gains"),
      roll_rms: parsed(draft.bpms.roll, 1e3, "BPM rolls"),
      resolution: parsed(draft.bpms.tbt, 1e3, "BPM turn-by-turn noise"),
      orbit_resolution: parsed(draft.bpms.orbit, 1e3, "BPM closed-orbit noise"),
    },
  };
}

const NEW_MAGNETS: MagnetDraft = { kinds: ["Quadrupole"], names: "", x: "", y: "", roll: "", pitch: "", yaw: "", field: "", cut: "2.5" };
/** EIC tuning studies' tolerances for bends, quadrupoles and sextupoles: 0.2 mm offsets, 0.5 mrad roll. */
const TYPICAL: MagnetDraft = { ...NEW_MAGNETS, kinds: ["Bend", "Quadrupole", "Sextupole"], x: "200", y: "200", roll: "0.5" };
const NEW_SUPPLY: SupplyDraft = { kinds: ["Bend"], names: "", rms: "10", bandwidth: "", ripple: "" };

function Kinds({ value, onChange, label, disabled }: { value: string[]; onChange(kinds: string[]): void; label: string; disabled: boolean }) {
  return (
    <span className="error-kinds" role="group" aria-label={label}>
      {KINDS.map((k) => (
        <label key={k} title={k}>
          <input type="checkbox" checked={value.includes(k)} disabled={disabled} onChange={(e) => onChange(e.target.checked ? [...value, k] : value.filter((v) => v !== k))} />
          {k === "Quadrupole" ? "Quad" : k === "Sextupole" ? "Sext" : k === "Octupole" ? "Oct" : k}
        </label>
      ))}
    </span>
  );
}

/** How many magnets (or supplies) a row chooses on the machine: none is a mistake, said in red. */
function Chosen({ count, unit, row }: { count: number | undefined; unit: "magnet" | "supply"; row: number }) {
  if (count === undefined) return <span className="muted">…</span>;
  if (count === 0)
    return (
      <span className="admin-error" data-testid={`chooses-${unit}-${row}`} title="No magnet of this machine has these kinds and names: nothing would be drawn. Leave Names blank for all.">
        none
      </span>
    );
  return (
    <span className="muted" data-testid={`chooses-${unit}-${row}`}>
      {count} {unit === "supply" ? (count === 1 ? "supply" : "supplies") : count === 1 ? "magnet" : "magnets"}
    </span>
  );
}

interface Props {
  machine: MachineEntry;
  onError(e: unknown): void;
  onSaved?(): void;
}

export function ErrorsEditor({ machine, onError, onSaved }: Props) {
  const [base, setBase] = useState<ErrorSpec | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [reason, setReason] = useState("");
  const [note, setNote] = useState<string | null>(null);
  const [refused, setRefused] = useState<string | null>(null);
  /** What each row chooses on this machine, as the draft stands: magnets by row, supplies by row. */
  const [chosen, setChosen] = useState<ErrorsEntry["chosen"] | null>(null);

  useEffect(() => {
    setBase(null);
    setDraft(null);
    setNote(null);
    setRefused(null);
    setChosen(null);
    va.errors(machine.id).then(
      (entry) => {
        setBase(entry.errors);
        setDraft(draftOf(entry.errors));
        setChosen(entry.chosen);
      },
      (e: Error) => (e instanceof VaError && (e.status === 403 || e.status === 401) ? setRefused(e.message) : onError(e)),
    );
  }, [machine.id, onError]);
  // As the draft changes, what it would choose (a row choosing nothing shows at once).
  useEffect(() => {
    if (!draft || !base) return;
    let spec: ErrorSpec;
    try {
      spec = specOf(draft, base);
    } catch {
      return; // not a spec yet: the field being typed says why when saved
    }
    let stale = false;
    const timer = window.setTimeout(
      () =>
        va.checkErrors(machine.id, spec).then(
          (entry) => !stale && setChosen(entry.chosen),
          () => undefined,
        ),
      300,
    );
    return () => {
      stale = true;
      window.clearTimeout(timer);
    };
  }, [draft, base, machine.id]);

  if (refused) return <p className="muted">{`${machine.name}'s errors are for whoever may configure it: ${refused}`}</p>;
  if (!draft || !base) return <p className="muted">Reading {machine.name}'s errors…</p>;
  const stopped = machine.state === "stopped" || machine.state === "failed";
  const running = machine.state === "running";
  // The BPMs read the beam and leave it alone: their errors change while it runs too.
  const bpmsOpen = stopped || running;
  const set = (change: Partial<Draft>) => setDraft({ ...draft, ...change });
  const magnet = (k: number, change: Partial<MagnetDraft>) => set({ magnets: draft.magnets.map((m, i) => (i === k ? { ...m, ...change } : m)) });
  const supply = (k: number, change: Partial<SupplyDraft>) => set({ supplies: draft.supplies.map((s, i) => (i === k ? { ...s, ...change } : s)) });
  const number = (value: string, onChange: (v: string) => void, label: string, placeholder = "0", open = stopped) => (
    <input className="error-number" inputMode="decimal" aria-label={label} placeholder={placeholder} value={value} onChange={(e) => onChange(e.target.value)} disabled={!open} />
  );
  const bpm = (key: keyof Draft["bpms"], label: string) => number(draft.bpms[key], (v) => set({ bpms: { ...draft.bpms, [key]: v } }), label, "0", bpmsOpen);
  const save = () => {
    let spec: ErrorSpec;
    try {
      spec = specOf(draft, base);
    } catch (e) {
      return setNote((e as Error).message);
    }
    va.setErrors(machine.id, spec, reason).then(
      (saved) => {
        setBase(saved.errors);
        setDraft(draftOf(saved.errors));
        setChosen(saved.chosen);
        setReason("");
        setNote(
          running
            ? `Saved: ${machine.name}'s BPMs read with these errors from its next turn.`
            : `Saved: ${machine.name} draws these errors (seed ${saved.errors.seed}) when it starts.`,
        );
        onSaved?.();
      },
      (e: Error) => setNote(e.message),
    );
  };

  return (
    <div className="errors-editor" data-testid="errors-editor">
      {running ? (
        <p className="admin-error">
          {machine.name} is running: its BPMs' errors change from its next turn. Stop it to change its magnets' and supplies' errors or the seed: it draws them as it starts.
        </p>
      ) : !stopped ? (
        <p className="admin-error">
          {machine.name} is {machine.state}: its errors change once it runs or has stopped.
        </p>
      ) : null}
      <h3>Magnets</h3>
      <p className="muted">Each row's errors are drawn per magnet: Gaussian, rms, cut at the sigma given; rows add. Offsets and rotations are of the body's centre (yaw turns the exit to +x, pitch to −y); the field error scales the main field (a bend's dipole, a quadrupole's gradient, a corrector's kick).</p>
      <table className="errors-table">
        <thead>
          <tr>
            <th>Magnets</th>
            <th>Names</th>
            <th>x µm</th>
            <th>y µm</th>
            <th>roll mrad</th>
            <th>pitch mrad</th>
            <th>yaw mrad</th>
            <th>field 10⁻⁴</th>
            <th>cut σ</th>
            <th>chooses</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {draft.magnets.map((m, k) => (
            <tr key={k}>
              <td>
                <Kinds value={m.kinds} onChange={(kinds) => magnet(k, { kinds })} label={`Magnet kinds, row ${k + 1}`} disabled={!stopped} />
              </td>
              <td>
                <input aria-label={`Names, row ${k + 1}`} placeholder="blank: all; or qf*, b_*" title="Element names, shell patterns, comma separated; blank chooses every magnet of the kinds ticked" value={m.names} onChange={(e) => magnet(k, { names: e.target.value })} disabled={!stopped} />
              </td>
              <td>{number(m.x, (x) => magnet(k, { x }), `x offset µm, row ${k + 1}`)}</td>
              <td>{number(m.y, (y) => magnet(k, { y }), `y offset µm, row ${k + 1}`)}</td>
              <td>{number(m.roll, (roll) => magnet(k, { roll }), `roll mrad, row ${k + 1}`)}</td>
              <td>{number(m.pitch, (pitch) => magnet(k, { pitch }), `pitch mrad, row ${k + 1}`)}</td>
              <td>{number(m.yaw, (yaw) => magnet(k, { yaw }), `yaw mrad, row ${k + 1}`)}</td>
              <td>{number(m.field, (field) => magnet(k, { field }), `field error 1e-4, row ${k + 1}`)}</td>
              <td>{number(m.cut, (cut) => magnet(k, { cut }), `cut sigma, row ${k + 1}`, "2.5")}</td>
              <td>
                <Chosen count={chosen?.magnets[k]} unit="magnet" row={k + 1} />
              </td>
              <td>
                <button type="button" aria-label={`Remove magnets row ${k + 1}`} onClick={() => set({ magnets: draft.magnets.filter((_, i) => i !== k) })} disabled={!stopped}>
                  ✕
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="admin-row">
        <button type="button" onClick={() => set({ magnets: [...draft.magnets, { ...NEW_MAGNETS }] })} disabled={!stopped}>
          + Magnet errors
        </button>
        <button type="button" title="EIC tuning studies' tolerances: 0.2 mm offsets and 0.5 mrad rolls of bends, quadrupoles and sextupoles" onClick={() => set({ magnets: [...draft.magnets, { ...TYPICAL }] })} disabled={!stopped}>
          + Typical alignment
        </button>
      </div>

      <h3>Power supplies</h3>
      <p className="muted">The current of every supply driving the magnets chosen fluctuates about its setting, all its magnets together: noise (white, a new value every turn, or of a bandwidth) and ripple lines. A ring feels it turn by turn; a route, shot by shot.</p>
      <table className="errors-table">
        <thead>
          <tr>
            <th>Magnets</th>
            <th>Names</th>
            <th>noise ppm rms</th>
            <th>bandwidth Hz</th>
            <th>ripple Hz:ppm, …</th>
            <th>chooses</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {draft.supplies.map((s, k) => (
            <tr key={k}>
              <td>
                <Kinds value={s.kinds} onChange={(kinds) => supply(k, { kinds })} label={`Supply magnet kinds, row ${k + 1}`} disabled={!stopped} />
              </td>
              <td>
                <input aria-label={`Supply names, row ${k + 1}`} placeholder="blank: all" title="Element names, shell patterns, comma separated; blank chooses every magnet of the kinds ticked" value={s.names} onChange={(e) => supply(k, { names: e.target.value })} disabled={!stopped} />
              </td>
              <td>{number(s.rms, (rms) => supply(k, { rms }), `noise ppm, row ${k + 1}`)}</td>
              <td>{number(s.bandwidth, (bandwidth) => supply(k, { bandwidth }), `bandwidth Hz, row ${k + 1}`, "every turn")}</td>
              <td>
                <input aria-label={`Ripple, row ${k + 1}`} placeholder="60:5, 360:2" value={s.ripple} onChange={(e) => supply(k, { ripple: e.target.value })} disabled={!stopped} />
              </td>
              <td>
                <Chosen count={chosen?.supplies[k]} unit="supply" row={k + 1} />
              </td>
              <td>
                <button type="button" aria-label={`Remove supplies row ${k + 1}`} onClick={() => set({ supplies: draft.supplies.filter((_, i) => i !== k) })} disabled={!stopped}>
                  ✕
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="admin-row">
        <button type="button" onClick={() => set({ supplies: [...draft.supplies, { ...NEW_SUPPLY }] })} disabled={!stopped}>
          + Fluctuating supply
        </button>
      </div>

      <h3>BPMs</h3>
      <p className="muted">Each BPM reads gain × (roll of (beam − offset)) + noise, its offset, gain and roll drawn from the seed: changed here, each keeps its own, scaled to the new rms. They change while the machine runs, from its next turn.</p>
      <div className="admin-row errors-bpms">
        <label>offsets µm rms {bpm("offset", "BPM offsets µm")}</label>
        <label>gains % rms {bpm("gain", "BPM gains %")}</label>
        <label>rolls mrad rms {bpm("roll", "BPM rolls mrad")}</label>
        <label title="Each turn's reading, at full intensity; more as the beam thins">turn-by-turn noise µm {bpm("tbt", "BPM turn-by-turn noise µm")}</label>
        <label title="Each sample of the closed-orbit (slow) acquisition">closed-orbit noise µm {bpm("orbit", "BPM closed-orbit noise µm")}</label>
      </div>

      <div className="admin-row">
        <label title="Draws every error: the same seed, the same machine">
          seed <input className="error-number error-seed" aria-label="Seed" value={draft.seed} onChange={(e) => set({ seed: e.target.value })} disabled={!stopped} />
        </label>
        <button type="button" title="Another seed: another machine with the same tolerances" onClick={() => set({ seed: String(1 + Math.floor(Math.random() * 2 ** 31)) })} disabled={!stopped}>
          New draw
        </button>
        <input className="setpoint-reason" placeholder="Why (goes into the log, without the seed)" aria-label="Reason for the errors" value={reason} onChange={(e) => setReason(e.target.value)} disabled={!bpmsOpen} />
        <button type="button" onClick={save} disabled={!bpmsOpen}>
          Save errors
        </button>
      </div>
      {note ? (
        <p className={note.startsWith("Saved") ? "muted" : "admin-error"} data-testid="errors-note">
          {note}
        </p>
      ) : null}
    </div>
  );
}
