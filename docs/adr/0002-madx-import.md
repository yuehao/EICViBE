# ADR 0002: MAD-X import through cpymad

Status: proposed (2026-10-06). Scope: `eicvibe.importers.madx`. Depends on ADR 0001.

## Decision

MAD-X itself runs the script, through cpymad (the optional dependency `eicvibe[madx]`). The importer then reads each requested sequence back in expanded form and converts it with `LatticeBuilder`. CALL, macros, WHILE, SEQEDIT, MATCH and deferred expressions therefore behave exactly as in MAD-X, and the imported strengths are the final ones.

```python
from eicvibe.importers import import_madx

result = import_madx("lattice/esr.madx", ["esr"], kind="periodic")
result.lattice      # validated Lattice
result.warnings     # what was dropped or approximated
```

- The script runs in its own directory, so relative CALLs resolve. A STOP ends it, and MAD-X stays usable for the import.
- `import_cpymad(madx, ...)` imports from a session the caller has already prepared.
- `sequences` defaults to MAD-X's current sequence. The first requested sequence becomes the root branch. `kind` is `periodic` by default and can be given per sequence.
- Each MAD-X element becomes one definition, and each sequence node becomes one occurrence. Element classes become tags (`qf: mq` gives `qf` the tag `mq`). Implicit drifts become definitions of their own, per sequence.
- The sequence's `$start` marker becomes the `BeginningEle`. The BEAM becomes `ReferenceP` with `pc_ref` only, since MAD-X masses differ slightly from `SPECIES`. An ion is matched to a species by charge and by mass within 0.1 %.
- Ids derive from the lattice name and the MAD-X names. Re-importing an edited script keeps the ids of unchanged elements, so `diff` shows only what changed, e.g. new matched strengths.
- After building, every occurrence's s and each branch length are compared with MAD-X; a mismatch is a warning.
- Each conversion that is not a plain copy produces an `ImportNote`. Conventions (sign flips, unit changes) are noted once per import. Anything dropped or approximated is a warning on its element. `strict=True` raises on elements that have no counterpart.

## Support matrix

| MAD-X | Imported as |
| --- | --- |
| DRIFT (and implicit drifts) | `Drift`; zero-length drifts are skipped |
| MARKER | `Marker` |
| SBEND | `Bend`: `angle_ref`, `e1`, `e2`, `edge*_int` = FINT·HGAP (FINTX < 0 means FINT), `h1`, `h2`, `tilt_ref` = TILT. K1, K1S, K2, K2S, K3, K3S and K0S become per-length multipoles. K0 is imported only if it differs from ANGLE/L |
| RBEND | As SBEND, with L converted from chord to arc length under `option, rbarc`, and E1, E2 + ANGLE/2 |
| QUADRUPOLE, SEXTUPOLE, OCTUPOLE | Same kind; KN and KNS are per-length strengths, and TILT is set on every order present |
| MULTIPOLE | `Multipole`: KNL, KSL become `KnNL`, `KsNL`; TILT is set on each order |
| KNL, KSL on thick elements | Integrated, unless the order also has a per-length strength; then they are spread over the length |
| SOLENOID | `Solenoid`: `Ksol` = KS |
| HKICKER, VKICKER, KICKER, TKICKER | `Kicker`: `Kn0L` = −HKICK (or −KICK), `Ks0L` = VKICK (or KICK); tags `HCOR`/`VCOR` |
| RFCAVITY, CRABCAVITY | `RFCavity`, `CrabCavity`: voltage = VOLT·10⁶, phase = 2π·LAG mod 2π with `zero_phase: BELOW_TRANSITION`, HARMON (else FREQ·10⁶) |
| MONITOR, HMONITOR, VMONITOR, INSTRUMENT | `Instrument` with its length |
| PLACEHOLDER, COLLIMATOR, RCOLLIMATOR, ECOLLIMATOR | `Drift` (`Marker` if zero length) with the aperture |
| XROTATION, YROTATION, SROTATION | `Patch` with `x_rot`, `y_rot`, `z_rot` = ANGLE |
| TRANSLATION | `Patch` with offsets DX, DY, DS |
| BEAMBEAM | `BeamBeam`: SIGX, SIGY, CHARGE, NPART |
| APERTYPE CIRCLE, ELLIPSE, RECTANGLE | `ApertureP` limits, shifted by APER_OFFSET |
| RECTELLIPSE, LHCSCREEN, RECTCIRCLE | Ellipse, rectangle, or `VERTICES` outline with elliptical arcs |
| APER_VX, APER_VY | `VERTICES`, reordered counterclockwise |
| Any other element (DIPEDGE, RFMULTIPOLE, ELSEPARATOR, MATRIX, …) | Warning; `Drift` or `Marker`, or an error with `strict` |

Signs were checked against MAD-X itself. XROTATION, YROTATION and SROTATION turn the frame as right-handed rotations (SURVEY). SROTATION(a) followed by a bend and SROTATION(−a) equals the bend with TILT = a. The RF phase map uses MAD-X's LAG convention: LAG = 0 is the zero crossing stable below transition, 0.5 the one stable above, and 0.25 is on crest.

## Bmad lattices, through Bmad's own writer (2026-10-10)

A Bmad design is imported as MAD-X written by Bmad itself: `eicvibe.importers.bmad.bmad_to_madx` runs `bmad_to_mad_sad_elegant -madx -noaperture` on a copy of the design's folder, so Bmad resolves its CALLs, overlays and superposition and nothing is written beside the design files. Bmad is the authority on its language, as MAD-X is on its own. The writer loses three things, which the conversion puts back, each with an `ImportNote`:

- **Elements superimposed on a `PIPE`.** Bmad makes each overlap a slave with two lords, the pipe and the element (`LTR_PIP01\LTR_QUA01`), and the writer names it after the pipe (`LTR_PIP01_L1001`). Such pipes are read as drifts. A pipe is a drift with an aperture, and apertures are not written, so nothing else changes; the conversion checks that every element keeps its length and position. Each element then keeps its own name.
- **Monitors and instruments**, which the writer writes as drifts (`write_lattice_mad_format.f90`, the drift case). Bmad's element list, read through Tao, names them, and they become MAD-X monitors and instruments again. A monitor that a zero-length element splits keeps its name on its first piece.
- **The sign of a patch's `x_pitch`.** The writer gives `YROTATION = -x_pitch`. MAD-X 5.09 turns the frame as Bmad does: `x_pitch = +a` and `YROTATION = +a` both turn it by `a` toward +x and give a beam on the axis px = −a (Tao and MAD-X TWISS and SURVEY on a drift, patch, drift line). The angle's sign is corrected. The writer's `XROTATION = -y_pitch` and `SROTATION = tilt` are right; so are its translations.

The Bmad programs come from `$EICVIBE_BMAD_BIN`, `$ACC_EXE`, the PATH or `~/packages/bmad-ecosystem/production/bin`. Checked on the EIN's LTR against Tao, element by element up to the RCS injection point: floor coordinates to 1e-13 m, optics to 5e-7 m, and the orbit at the injection point exactly (x = −30.900 mm, px = +4.391 mrad). Every s after a patch is 1.4e-4 m shorter than Bmad's, since Bmad gives a patch a path length (here 30.9 mm × sin 4.39 mrad) and MAD-X does not. The RTE's floor and orbit agree likewise (8e-13 m, 1e-5 m); with the writer's sign they were 0.8 m and 67 mm off.

## Not imported (warnings)

- Multipole orders above 10 (`MAX_MULTIPOLE_ORDER`). MAD-X allows up to 20. The RHIC proton lattice uses orders up to 10, in its interaction-region field-error multipoles.
- Thin solenoids (KSI), MULTIPOLE ANGLE (a thin bend of the reference orbit), CRABCAVITY TILT (the crabbing plane), BEAMBEAM offsets and shapes, APER_TILT, TRUE_RBEND, KTAP, and quadrupole fringe settings.
- Element attributes DX, DY, DS, DTHETA, DPHI and DPSI (except on TRANSLATION).
- EFCOMP and EALIGN errors are not imported and not detected: cpymad does not expose them per element, and ETABLE did not report them in tests with MAD-X 5.09.03.
- Initial optics of a line (BETA0, or the initial values given to TWISS) are commands, not sequence data. A `linac` branch gets a note to set `TwissP`.

## Checked on real lattices

Positions and lengths were compared element by element with MAD-X TWISS and SURVEY tables, after the import in the same session. They agree exactly. The EICViBE length equals the SURVEY length exactly and the declared sequence length to within 1e-10 m, which is float summation over thousands of elements. The TWISS summary LENGTH is rounded to 10 significant digits. `test_lengths_match_madx_twiss_and_survey` repeats this comparison on the test lattices, including an RBEND and rotations.

| Lattice | Occurrences | EICViBE length | MAD-X TWISS output (LENGTH) |
| --- | --- | --- | --- |
| EIC RCS (`RCSV3.madx`): three MATCH blocks with USE_MACRO, nested sequences, a deferred voltage set after matching | 5367 | 1421.257252858 m | 1421.257253 (also in the saved `RCSV3.twiss`, MAD-X 5.09.01) |
| RHIC 250 GeV protons; the import takes about 7.6 s, of which 5.3 s is MAD-X running the script | 7821 | 3833.845181459 m | 3833.845181 (also in the saved `test.twiss`, MAD-X 5.04.02) |
