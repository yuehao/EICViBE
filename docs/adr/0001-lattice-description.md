# ADR 0001: Lattice description

Status: proposed (2026-09-24). Scope: `eicvibe.domain.lattice`, the M0/M1 domain contract for machine content. Routes, cycles, beams and engine settings belong to scenarios and are out of scope.

## Decision

A `Lattice` is an immutable, fully validated description of one machine revision. A `LatticeBuilder` produces and edits lattices. The domain package imports only the standard library and pydantic; a test enforces this.

```python
b = LatticeBuilder("rcs")
b.define("QF", "Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": 1.2}, tags=["QUAD"])
b.define("D", "Drift", length=2.0)
b.add_branch("ring", "periodic", ["QF", "D"] * 40)
b.set_overrides(("ring", "QF_3"), BodyShiftP={"x_offset": 1e-4})   # one magnet's misalignment
lattice = b.build()                  # validated, immutable, lattice.digest = "sha256:…"
edited = LatticeBuilder.from_lattice(lattice)   # same ids; edit, then build() again
```

### Identity, names and position are separate (audit A09, backlog B08)

- Every element definition, branch, occurrence and shared component has an opaque `id` that is generated once and survives edits and renames. Ids are random by default, so two users editing the same base revision cannot allocate the same id. Importers use `derived_id`, which makes re-importing a file reproducible.
- Names are unique labels (definitions and branches globally, occurrences within their branch). A string that is the id of one entity and the name of another is rejected rather than guessed.
- Index and s position are derived per revision (`lattice.topology`) and never used as identity. `diff(old, new)` matches by id, so an insertion is one added occurrence and a rename is one `name` change.

### Families, placements and hardware

- An `ElementDefinition` is a family: kind, length, parameter groups and tags. Every occurrence that places it shares its values. This follows the PALS/Bmad/MAD-X split between element definitions and line references, and Xsuite's `Replica`.
- An `Occurrence` is one placement. Placement-specific values (trims, misalignments, apertures) are `ParameterOverride`s on the occurrence. The effective element is the definition plus its overrides, validated as a whole. Legacy copied a prototype into every placement (`QF_1…QF_n`), so a family change needed n edits and the family survived only in `inherit`.
- A `SharedComponent` declares occurrences in different branches to be one piece of hardware with distinct optical traversals (roles such as `injection`, `circulation`, `extraction`). It replaces `MultipassComponentSpec`; it plays the role of Bmad's multipass lord and PALS's multipass "physical element sets". No boundary markers are needed, because forks can land on either side of an occurrence.

### Forks land at an explicit side

`ForkLink` names a destination branch, an optional landing occurrence, and `to_side` (`entrance` by default, as legacy tracking assumed; or `exit`). PALS gives the destination element (default: the first) but not the side; a shared septum needs "continue after the circulating traversal". `connect_injection(via=…)` places the fork after the injection-side traversal and lands at the exit of the ring-side traversal, so the hardware is passed once. `connect_extraction(via=…)` places the fork before the ring-side traversal and lands at the entrance of the extraction-side traversal.

### Strict structure, separate plausibility

- Construction rejects anything ambiguous or unresolvable: unknown kinds or groups, unknown references, fork links that do not resolve, a BeginningEle that is not first, redundant pairs set together (`frequency`+`harmon`, `voltage`+`gradient`, `Ksol`+`Bsol`, `e1`+`e1_rect`, `etap_x`+`deta_x_ds`, per-length `Kn1`+integrated `Kn1L` of one multipole order, two energy or two time settings in `ReferenceChangeP`), `pc_ref`/`E_tot_ref` that disagree for a known species, NaN/Inf, and booleans or strings in numeric fields. All problems are reported together, each with a code.
- `diagnose()` returns warnings only, like Bmad's lattice sanity checks: RF phase that looks like degrees, per-length strength on a zero-length magnet, unknown species, cavity without frequency, an elliptical aperture missing some limits (PALS treats it as inactive), a `FloorP` that a fork overrides, and implausible magnitudes (`diagnostics.SANITY_THRESHOLDS`). Legacy raised on those magnitudes.
- Tunable settings (quadrupole and multipole strengths, solenoid strength, kicks, cavity voltage) are checked against per-element operating limits in a `ParameterLimits` table, kept outside the `Lattice` because limits belong to the hardware and change independently of revisions. Entries are keyed by definition id (a family on one supply) or occurrence id (a magnet with its own trim), and record their source (`placeholder`, `manual`, `epics`) and PV. For now every element gets kind placeholders equal to the legacy thresholds (`limits.PLACEHOLDER_LIMITS`).

### Canonical content and digest

`to_dict()` omits defaults and default-only groups. Definitions, branches and components are sorted by name; occurrences keep physical order; overrides and tags are sorted. `-0.0` becomes `0.0`. The digest is SHA-256 of sorted-key compact JSON, so equal content gives an equal digest, and adding a defaulted field later leaves existing digests unchanged. `format_version` is required; a golden test pins the encoding, and changing it requires a version bump and a migration.

### Parameter groups follow PALS

Groups follow the PALS parameter groups as of October 2026: a stored field has the PALS name, unit and meaning, so canonical content is a PALS subset. Where PALS offers several forms of one quantity, one is stored and the importer converts the others: bends keep `length` and `angle_ref` (not `g_ref`, `radius_ref`, `Bn0_ref`, `L_chord`, `L_rectangle`), apertures keep limits (not `x_center`/`x_width`), and multipoles keep normalized strengths up to order 10 (`MAX_MULTIPOLE_ORDER`); the fields `BnN`, `BsN`, `BnNL`, `BsNL` are kept only for a line whose reference momentum along it comes from tracking its beam, such as an injector from its gun (amended 2026-10-13, ADR 0012), one form per order. `ControlP` is an extension: `is_on` is the ungrouped PALS element parameter, and `scale` has no PALS counterpart. `MetaP.author` and `MetaP.date` are custom metadata keys, which PALS permits. Patches are rigid and have zero length (PALS `flexible: false`, `user_sets_length: true`). Not adopted yet: `ElectricMultipoleP`, `CoordinateSetP`, `TaylorP`, `ForkFromP` (a parser output), and the groups PALS marks as under construction. Element kinds use PALS names too, such as `Bend` rather than `SBend` or `RBend`; engine adapters translate them. The groups each kind takes (`elements.ALLOWED_GROUPS`) are the PALS kind lists, without groups not adopted here and without output groups, with `FloorP` only on `BeginningEle`. Kinds not adopted yet: `Fiducial` and `FloorShift` (they need `CoordinateSetP`), `UnionEle`, `Placeholder` (a lattice-expansion artifact), and the kinds PALS marks as under construction, except `EGun`: an electron gun, from its cathode, adopted with `RFP` for its voltage (without a frequency it is DC), as Bmad's PALS writer gives it (amended 2026-10-14, ADR 0012).

### Units are data

Every parameter field records its unit in the JSON schema (`parameter_info("MagneticMultipoleP", "Kn1").unit == "1/m^2"`), as pytac and LUME attach units to fields and variables. Web and notebook clients can label values from the generated schema instead of keeping their own tables. Conventions are in `conventions.py`: SI and eV, arc length for bends, normalized multipole strengths (per length or integrated), phases in rad, and positive bend angle toward −x.

## Changes from the legacy model

| Legacy | New |
| --- | --- |
| `SBend`, `RBend`, `Monitor` kinds | PALS `Bend` and `Instrument`; the old spellings get a hint, and a rectangular bend sets `e1` = `e2` = `angle_ref`/2 |
| Lowercase/legacy aliases (`kn1`, `angle`, `hkick`, `freq`, `dx`…) | Rejected; the importer maps them and reports the mapping |
| `MisalignP` | `BodyShiftP` |
| `ForkP` on the element, targets by name | `ForkLink` on the occurrence, targets by id, explicit side |
| `MarkerP`, `MonitorP` | Dropped (unused, or empty) |
| `ReferenceChangeP` | PALS `ReferenceChangeP`, on `Patch` |
| `MetaP.ID` | PALS `MetaP.ID`: an external identification, unrelated to stable ids |
| `BendP.g_ref`, `BendP.L_chord` | Derived from `angle_ref` and length, not stored |
| `BendP.hgap`, `edge1_int`/`edge2_int` without a unit | PALS `edge*_int` in m (MAD-X FINT·HGAP); `hgap` dropped |
| `KickerP` (`Kn0`, `Ks0`, `tilt`, `single_plane`) | `MagneticMultipoleP.Kn0L`, `Ks0L`, `tilt0`; plane restrictions become tags such as `HCOR` |
| `MagneticMultipoleP.tilt` for all orders | Per-order `tilt0`…`tilt10` |
| `PatchP.x_pitch`, `y_pitch`, `tilt`, `t_offset` | `x_rot`, `y_rot` = −`y_pitch` (right-handed), `z_rot`; `t_offset` becomes `ReferenceChangeP.dtime_ref` |
| `TrackingP` fields (`num_steps`, `method`, `track_start`, `track_end`) | One table of settings per program, e.g. `{"xsuite": {"num_steps": 10}}` |
| `ReferenceP.location` | Dropped: PALS reference parameters are at the upstream end |
| `ControlP.on` | `ControlP.is_on` |
| `RFP.zero_phase` (float) | The PALS enumeration (`ACCELERATING`, `BELOW_TRANSITION`, `ABOVE_TRANSITION`) |
| RF phase in rad/2π | rad, as PALS since October 2026 |
| `ApertureP` and `BeamBeamP` 0.0 meaning "unset" | `None` |
| `TwissP` in x/y (`beta_x`, `alpha_x`, …), `beta_*` default 1.0 | PALS normal modes (`beta_a`, `alpha_a`, …) plus coupling `cmat`, phases and `pz` derivatives; `beta_*` required, no silent default |
| Specialized validation failures downgraded to warnings | Structural errors raise; magnitudes warn |
| `connect(..., connection_type="expanded")`, `apply_misalignment`, change cache on Xsuite objects | Not ported; routes, error seeds and change sets are scenario/application concerns |

## Ideas from other packages

Adopted here: definition/placement separation (PALS, Bmad, MAD-X, Xsuite), tags as families (pytac, MML, OpenXAL node types), machine-readable units (pytac, LUME), multipass-style shared hardware (Bmad, PALS), sanity diagnostics (Bmad), stable node ids (OpenXAL), versioned JSON documents (Xsuite `to_dict`, PALS).

Proposed, not implemented:

1. **Knobs and deferred expressions** (MAD-X `:=`, Xsuite `line.vars`, Bmad overlays/groups, Beamlines.jl). This should be the M4 variable registry, bound to parameter paths (`definition or occurrence id`, `group.parameter`). Start with linear knobs, and keep one authority per parameter so that a knob and a direct value cannot both set it. Shared definitions already cover a power-supply family.
2. **`field_master`** (PALS). A ramp must declare whether fields or normalized strengths are held fixed as reference energy changes. This belongs with the RCS cycle port.
3. **Calibrations and data sources** (pytac `PolyUnitConv`/`PchipUnitConv` and live/sim sources; OpenXAL DESIGN/LIVE synchronization). These belong to the read-only control-binding layer, keyed by occurrence id with calibration revisions. Its first consumer replaces the placeholder limits: read each setting PV's drive limits (DRVL/DRVH, or pvAccess `control.limitLow/limitHigh`), convert from engineering units through the calibration and the reference rigidity, and write `source=epics` entries.
4. **Install at s** (MAD-X `SEQEDIT INSTALL`, Xsuite insert at s): split a drift occurrence. It needs a rule for which half keeps the id.
5. **PALS import/export with a support matrix**, and a legacy YAML importer that produces a report (aliases, sign conversions, dropped fields). The MAD-X importer (ADR 0002) follows this pattern.
6. **Superperiodicity** (pyAT `periodicity`), only once every adapter honours it; an ignored field would silently change circumference and harmonic number.

## Open questions for physics review

1. **Kicker sign.** The legacy MAD-X/Bmad importers store `hkick` in `KickerP.Kn0` unchanged, and the Xsuite adapter applies `Kn0` as `knl[0]`. MAD-X defines `HKICK = −KNL[0]`, so legacy horizontal corrector kicks appear sign-flipped in Xsuite. The new contract stores kicks as `MagneticMultipoleP.Kn0L` with the multipole sign; the importer must negate `hkick`. Please confirm, and check TrackPad's convention.
2. Resolved: zero-length multipoles use the PALS integrated strengths `KnNL`/`KsNL`; a per-length strength on a zero-length magnet warns.
3. Species are electron, positron, antiproton, proton, deuteron, helium-3, gold-197 and uranium-238, the ions fully stripped. Masses are CODATA 2022; gold and uranium are AME2020 atomic masses minus the electrons plus their binding energy (A·u would be 71 MeV too heavy for gold). The pc/E tolerance is 1e-9 relative plus 1 eV. Do the engines use the same masses?
4. Warning thresholds in `LIMITS` are the legacy values; some (a 10 m misalignment) are too loose to be useful.
5. Is `entrance` the right default landing side, and does TrackPad already land forks at the entrance?
6. What does `ControlP.scale` scale, exactly?
