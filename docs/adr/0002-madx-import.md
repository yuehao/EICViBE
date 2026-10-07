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

## Not imported (warnings)

- Multipole orders above 10 (`MAX_MULTIPOLE_ORDER`). MAD-X allows up to 20. The RHIC proton lattice uses orders up to 10, in its interaction-region field-error multipoles.
- Thin solenoids (KSI), MULTIPOLE ANGLE (a thin bend of the reference orbit), CRABCAVITY TILT (the crabbing plane), BEAMBEAM offsets and shapes, APER_TILT, TRUE_RBEND, KTAP, and quadrupole fringe settings.
- Element attributes DX, DY, DS, DTHETA, DPHI and DPSI (except on TRANSLATION).
- EFCOMP and EALIGN errors are not imported and not detected: cpymad does not expose them per element, and ETABLE did not report them in tests with MAD-X 5.09.03.
- Initial optics of a line (BETA0, or the initial values given to TWISS) are commands, not sequence data. A `linac` branch gets a note to set `TwissP`.

## Checked on real lattices

- EIC RCS (`RCSV3.madx`): three MATCH blocks with USE_MACRO, nested sequences, and a deferred voltage set after matching. Imported 5367 occurrences over 1421.257 m with no position warnings.
- RHIC 250 GeV protons: 7821 occurrences over 3833.845 m. The import takes about 7.6 s, of which 5.3 s is MAD-X running the script.
