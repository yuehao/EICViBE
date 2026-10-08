# ADR 0005: The machine as files people read and edit

Status: proposed (2026-10-07). Scope: `eicvibe.store`, `eicvibe.domain.machine.description`, the `eicvibe` command, and how the EIN and the viewer use them. Depends on ADR 0001 (lattice), ADR 0003 (registry, power circuits) and ADR 0004 (viewer).

## Decisions

1. **One folder per machine holds what people decide, split by subject.** The lattice, element names, power supplies (strings and transfer functions), other devices and operating settings each have their own file. Later, magnet measurements and ramp profiles get theirs. Physicists, engineers and operators edit their own files and review them in version control.
2. **These files are the source of truth.** Design files such as MAD-X are imported into them. Importing again compares the import with the files and shows the differences; the lattice files are replaced only when asked (`--write`). Naming, device and settings files are never rewritten by a tool after they are first written.
3. **YAML for structured data, CSV for long tables.** YAML allows comments and stays close to PALS. Measured excitation curves, and later measurement series and ramp tables, are CSV files that open in any spreadsheet. Round-trip YAML (`ruamel.yaml`) keeps comments where a tool updates a file.
4. **Files refer by name; ids keep identity.** Magnets and instruments are `branch/occurrence` (lattice names) and elements are referred to by name. Every lattice entry still carries its stable `id`, which devices and history attach to. An entry without an `id` gets one derived from its name. EIC names are never written as references: they follow from `naming.yaml`, so renaming an area renames everything consistently and changes no identity.
5. **The registry is assembled from the files, every time.** `MachineDescription` (`eicvibe.domain.machine.description`) is what the files hold: the naming rules, the supplies with their circuits, magnets and transfer functions, the BPMs, RF stations and tune monitors. `assemble_registry` expands it with the naming rules into the registry of ADR 0003. The rule-based generator (`describe_machine`) only writes the starting files. Device ids derive from supply keys and lattice ids, never from names.
6. **`build/` is generated and never edited.** `eicvibe build` writes the canonical lattice JSON, the registry and a manifest of the source files' checksums. Tools may read `build/`; people edit the source files.

## Layout

```text
machine.yaml              index: name, root, branches in beam order, provenance of the lattice
lattice/<branch>.yaml     a branch: its element definitions, then its line in beam order
lattice/shared.yaml       components several branches pass through (the BAR septum)
naming.yaml               EIC naming rules: areas, tokens, shared magnets, explicit names
devices/power.yaml        supplies; circuits (one magnet, or a string in series); transfer functions
devices/transfer/*.csv    measured transfer functions: current, field
devices/diagnostics.yaml  BPMs and tune monitors
devices/rf.yaml           RF stations and the ranges of their settings
settings/<mode>.yaml      operating settings, e.g. bar_mode.yaml: parameter values by occurrence
measurements/             reserved: the magnetic measurement database
ramps/                    reserved: synchrotron ramp profiles
build/                    generated: lattice.json, registry.json, manifest.json
```

Each file starts with a comment that explains its entries. A short entry is one line, so a branch's line or a list of individually powered supplies reads like a table; a string lists one magnet per line. For example, in `devices/power.yaml`:

```yaml
transfer:
  dipole-placeholder: {kind: polynomial, coefficients: [0.0, 0.001], field_unit: T, source: placeholder}
supplies:
  ltr/ltr_dip01: {sets: Kn0, transfer: dipole-placeholder, limits: [0.0, 876.0], magnets: [ltr/ltr_dip01]}
  rcs.b0: # 156 magnets in series
    sets: Kn0
    transfer: dipole-placeholder
    polarity: -1
    limits: [0.0, 109.0]
    magnets:
      - rcs/b0_1
      - {magnet: rcs/b0_17, transfer: b0-017}   # a measured curve for one magnet
```

A measured curve is `{kind: table, file: transfer/b0-017.csv, field_unit: T, source: measured}`. A service is `{kind: service, service: magnet-tf, magnet: B0-017, hysteresis: true, fallback: dipole-placeholder, field_unit: T}`.

## Workflow

- `python -m eicvibe.machines.ein EIN_FOLDER` imports the EIN design files into `EIN_FOLDER/machine`. The first run writes every file. Later runs print what the design files change and write the lattice files only with `--write`, keeping ids by name (`adopt_ids`).
- `eicvibe check FOLDER` loads the folder and reports problems: references that do not resolve, strings whose magnets need different currents, lattice diagnostics. `eicvibe build FOLDER` also writes `build/`.
- `eicvibe-web FOLDER` serves the machine folder. A saved edit shows on the next request. A file that does not load leaves the last good revision in place, and the server logs the file and line.

## Not yet

- Formats for `measurements/` (the magnetic measurement database: per-magnet measurement runs, harmonics, field maps) and `ramps/` (ramp profiles: time series of energy, RF and family currents). The plan is a YAML index with CSV series. They will be defined from real data, not ahead of it.
- Rewriting a lattice file loses comments inside it; the device, naming and settings files are not rewritten.
