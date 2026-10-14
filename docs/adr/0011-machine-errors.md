# ADR 0011: A virtual machine's errors: its magnets, power supplies and BPMs

Status: accepted (2026-10-13). Scope: `eicvibe.virtual.errors`, the engines' `BodyShiftP` (pitch and yaw), `RingRun` and `RouteRun`, and the virtual accelerator's machines (`eicvibe.va`, the admin page, the viewer's sandboxes). Depends on ADR 0001 (PALS `BodyShiftP`), ADR 0006 (engines, ramps), ADR 0007 (virtual runs and BPMs) and ADR 0008 (machines, roles, the log).

## Context

Until now a virtual machine was its model: every magnet where the lattice puts it and as strong as its setting makes it, and only the BPMs had errors (offsets, gains, noise). A closed orbit was then zero until someone moved a corrector, a tune was the model's, and orbit correction, local bumps and tune knobs (ADR 0010) were tried on a machine with nothing to correct. Real machines are never their models: magnets are misaligned and their fields off by parts in 10⁴, power supplies ripple and drift, and the BPMs read through offsets, gains and rolls of their own. Physicists study tolerances by drawing such errors from their rms values and correcting the machine that results (the EIC's tuning studies use 0.2 mm offsets and 0.5 mrad rolls of dipoles, quadrupoles and sextupoles; its RCS studies add main-dipole field errors and power-supply jitter), and operators meet them every day.

## Decisions

1. **Errors are the machine's, never its settings.** A machine's settings keep the design and what people set; its errors act only where values enter the engine (`MachineErrors.engine_values`): a field error scales the setting of its magnet's main field, a body error adds to its `BodyShiftP`. So the settings and their readbacks, the operators' model (`eicvibe.operations`), `run.json`, the events and the log never show them, as on the real machine, where nobody reads a magnet's misalignment from the control system. What a run drew is kept in its `hidden.json`, beside the BPMs' hidden errors, for whoever benchmarks a correction against the truth.
2. **What errors, and how they are drawn** (`ErrorSpec`):
   - **Magnets** (`MagnetErrors`), chosen by kind, branch and name patterns: the offsets of the body's centre, its roll, pitch and yaw about the centre (PALS `BodyShiftP`, right-handed: a yaw turns the exit to +x, a pitch to −y, as Bmad's rotations are about the centre), and the relative error of its main field: a bend's dipole field, a quadrupole's gradient, a sextupole's and an octupole's, each plane of a corrector (its calibration), a solenoid's strength. Each is a Gaussian of the rms given, drawn per magnet and cut at 2.5 σ by default (draws beyond are drawn again), as alignment and magnetic measurement leave them. Rows add: a magnet chosen twice has both.
   - **Power supplies** (`SupplyNoise`): the current of every supply driving the magnets chosen fluctuates about its setting, and every magnet on one supply follows it together (a string of dipoles is one current; the registry's circuits say which magnets share one, and without a registry each magnet has its own). Noise of a relative rms, white (a new value every turn) or of a first-order bandwidth, and ripple lines (frequency, relative amplitude, a random phase each). A ring feels it turn by turn, at the ring's own revolution period; the current is sampled at least twenty times per period of its fastest part and held in between, so the engine is not asked to step every turn for a slow ripple. A route feels it shot by shot: each shot finds its supplies' currents where their noise is then (pulse-to-pulse jitter).
   - **BPMs** (`BpmModel`): each BPM's offset, gain and roll (the roll mixes the planes), all hidden; the turn-by-turn noise; and the noise of the closed-orbit (slow) acquisition, now its own. A real BPM system's slow orbit is a narrow-band average with its own noise (NSLS-II: about 1 µm turn by turn, 10 µm single bunch before its improved processing, 0.2 µm in fast acquisition, 0.3 µm long-term), not the turn-by-turn noise averaged over the turns the simulation happened to track in a second, which made a slow virtual ring's orbit noisier than a fast one's. The slow orbit now averages the readings without their turn-by-turn noise and adds `orbit_resolution` (0.3 µm by default) to every sample.
3. **The engines take pitch and yaw.** `BodyShiftP.x_rot` and `y_rot` join the parameters every engine changes in place (`SETTINGS`). Xsuite rotates the body exactly about its centre (`rot_shift_anchor` = half the length; its `rot_y_rad` is PALS's yaw and its `rot_x_rad` minus PALS's pitch, checked numerically); TrackPad shifts and turns the beam into the body's frame at its entrance and back at its exit, to first order in the angles (its T1 and T2, before and after the roll). The conformance suite (ADR 0006) holds both to the exact matrices of a quadrupole offset, pitched and yawed about its centre. A longitudinal offset (`z_offset`) is still refused.
4. **A machine draws its errors from its seed whenever it starts.** The same seed gives the same machine every start; another seed ("New draw") another machine with the same tolerances. A new machine gets a seed drawn when it is made; a sandbox, a copy of its machine, gets the same errors. A machine from before errors keeps the BPM errors it always had. Errors change only while the machine is stopped: real errors do not jump under a running beam, and a study changes the machine, not a running one. A change without a seed keeps the machine's.
5. **A choice of nothing is a mistake, and a machine its errors break is still run.** Every row says how many magnets (or supplies) it chooses on the machine, as it is edited, and a row that chooses none is refused when saved and warned of when a machine starts with it (a name such as "any" matches no magnet; blank names choose them all). A machine whose errors leave it no closed orbit or no stable optics cannot hold a beam, as a real one could not: the run still starts, the injector sends the bunch the design asks for (matched to the ring without its errors), the machine carries a warning, and the beam is lost. The EIN's RCS shows how near that is: quadrupole offsets of 0.01, 0.1 and 0.3 mm rms give a closed orbit of 0.16, 1.6 and 4.7 mm rms at its 192 BPMs; at 1 mm it has none (the sextupoles, fed down by the orbit, make the one-turn map unstable).
6. **Errors are for whoever may configure the machine.** On a shared machine, administrators see and change them; on a sandbox, its owner. Machine listings do not carry them. The log records what kinds of errors and how large, never the seed, which would let anyone draw the machine's errors again.
7. **Where they are chosen.** The admin page's Errors tab (and each machine's Errors button), the viewer's Errors window for one's own sandbox, `GET`/`PUT /api/v1/va/machines/{id}/errors` (the spec, and what each row chooses), `POST …/errors/check` (what a spec would choose; nothing saved) and a new machine's `errors`, the command line's `--errors FILE` (an `ErrorSpec` in YAML or JSON), and Python. The editor shows the units magnet and BPM people use: µm, mrad, 10⁻⁴ of the field, ppm of the current, Hz; "Typical alignment" adds the EIC tuning studies' 0.2 mm and 0.5 mrad for bends, quadrupoles and sextupoles.

## Validation (2026-10-13)

- A ring with misaligned and yawed quadrupoles reads, at every BPM, the closed orbit the same engine finds for the drawn errors (Xsuite and TrackPad, to 1 nm); its settings, events and `run.json` show nothing of them.
- All the demo ring's bends on one string, a slow ripple of 10⁻⁴: the orbit at every BPM moves by −D η, the dispersion times the current's change, to 5 % (both engines, no RF), as a field change of η bends the beam as a momentum change of −η would.
- A BPM's roll reads y in x by its sine; the slow orbit at rest scatters by `orbit_resolution` alone, however noisy the turns are.
- A route's dipole supplies with 10⁻³ of noise move its trajectory shot to shot; without, every shot is the same.
- The demo ring's quadrupoles 20 % off leave it no stable optics: the run starts, says so, and loses the bunch.

## Consequences

- Orbit correction, bumps and tune knobs (ADR 0010) now have a machine to correct, and the truth to judge them against.
- Fluctuating supplies make the engine change settings every few turns, or every turn for white noise: Xsuite then tracks a turn at a time between changes, which costs where tracking is fast (the demo ring) and little where a turn is slow (the RCS).
- Not yet: higher-order (multipole) field errors, girders that move magnets together, ground motion and slow drifts, BPMs that fail, errors that change while a machine runs.

## Using it

```yaml
# errors.yaml: eicvibe-virtual ring MACHINE --branch rcs --out runs/rcs1 --errors errors.yaml
seed: 42
magnets:
  - select: {kinds: [Bend, Quadrupole, Sextupole]}
    x_offset: 2.0e-4      # m rms
    y_offset: 2.0e-4
    roll: 5.0e-4          # rad rms
  - select: {kinds: [Quadrupole]}
    field: 5.0e-4         # relative rms
supplies:
  - select: {kinds: [Bend]}
    rms: 1.0e-5           # relative, white: every turn
    ripple: [{frequency: 360, amplitude: 2.0e-6}]
bpms: {offset_rms: 0.1, gain_rms: 0.01, roll_rms: 1.0e-3, resolution: 0.005, orbit_resolution: 3.0e-4}  # mm, rad
```

```python
from eicvibe.virtual import ErrorSpec, MachineErrors, RingRun

spec = ErrorSpec.model_validate(yaml_data)
errors = MachineErrors(spec, lattice, ["rcs"], registry)   # drawn once, from the seed
run = RingRun(lattice, "rcs", "runs/rcs1", registry=registry, errors=errors, bpm=spec.bpm_model()).start()
```

## Sources

- EIC optics tuning (CERN Indico 1186798, 2022): 0.2 mm and 0.5 mrad rms for dipoles, quadrupoles and sextupoles.
- Lovelace, Lin, Montag, Ranjbar, magnetic errors in the EIC RCS (IPAC'23 WEPL088); Bergan, RCS error tolerance studies (IPAC'26 WEP1329): main-dipole field errors, misalignments, power-supply jitter.
- Podobedov and Blaskiewicz, ESR dipole power-supply ripple, attenuated by the vacuum chamber's eddy currents.
- NSLS-II BPM system: Hu et al., PAC 2011 MOP198; IBIC 2016 TUCL02: resolution turn by turn and in fast and slow acquisition.
