"""The devices as files: naming, power supplies with transfer functions, diagnostics, RF.

- ``naming.yaml``: EIC naming rules (areas, tokens, shared magnets, explicit names).
- ``devices/power.yaml``: supplies, the magnets each drives (in series when
  several) and each magnet's transfer function, from a library of named
  transfer functions at the top of the file; measured curves are CSV tables
  in ``devices/transfer/``.
- ``devices/diagnostics.yaml``: BPMs and tune monitors.
- ``devices/rf.yaml``: RF stations and the ranges of their settings.

Magnets and instruments are referred to as ``branch/occurrence`` (lattice
names), which do not change when EIC names do.
"""

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from eicvibe.domain.lattice import Lattice
from eicvibe.domain.machine import (
    Bpm,
    CircuitMagnet,
    MachineDescription,
    Naming,
    NamingArea,
    PolynomialTransfer,
    RfStation,
    ServiceTransfer,
    SharedName,
    Supply,
    SupplyCircuit,
    TableTransfer,
    TuneMonitor,
)

from .yamlio import FileError, block, comment, flow, mapping, read_table, read_yaml, write_table, write_yaml

NAMING_FILE = "naming.yaml"
POWER_FILE = "devices/power.yaml"
DIAGNOSTICS_FILE = "devices/diagnostics.yaml"
RF_FILE = "devices/rf.yaml"
TRANSFER_DIR = "devices/transfer"

_NAMING_HEADER = """EIC names (EIC-SEG-RSI-009 Rev 02) of elements, devices and channels follow these rules.
areas: per branch, a prefix (the whole branch, positions in m from its start) or a
  list of sections {prefix, start, end, origin, direction, period}. Temporary
  until SEG assigns the prefixes; changing them renames, but keeps every identity.
tokens: abbreviations of controllers and signals.
shared: a magnet that several branches pass through is named once, at its home traversal.
names: explicit names that replace the rules', e.g. rcs/b0_1: rcs-dh111."""

_POWER_HEADER = """Power supplies and the magnets they drive (ADR 0003, eicvibe.domain.machine.power).
A supply has one circuit, or one per plane (a two-plane corrector: circuits h and v).
A circuit drives one magnet, or a string of magnets in series that share its current.
For each circuit: `sets` the lattice parameter (Kn1 means MagneticMultipoleP.Kn1),
`limits` of its setting (A), `polarity` (-1 when wired reversed), `transfer` the
transfer function from current to field, and `magnets` as branch/occurrence. A magnet
entry may give its own transfer, polarity or sets: {magnet: rcs/b0_17, transfer: b0-017};
a magnet that several branches pass through lists its traversals: [itl/iseptum, etl/eseptum].
A supply's EIC name is its first magnet's plus -ps unless `name` gives one.
Transfer functions, named under `transfer:` and used by name:
  polynomial: field = c0 + c1 I + ...   (a linear approximation has two coefficients)
  table: a measured curve, a CSV file of current and field (devices/transfer/)
  service: computed by a transfer-function service (nonlinear, hysteresis), with a
    fallback used where the service is not available.
field_unit is T, T/m, ... per length, or T m, T, ... with integrated: true."""

_DIAGNOSTICS_HEADER = """Beam instruments. A BPM reads x and y (:rdx, :rdy) in mm, unless an entry
{at: branch/occurrence, unit: ..., scale: ...} says otherwise (scale: channel units per m).
A tune monitor reads the tunes of a ring (branch)."""

_RF_HEADER = """RF stations: the ranges of their voltage (MV) and phase (deg) settings."""


class References:
    """Readable names of a lattice's occurrences, branches and components, and back."""

    def __init__(self, lattice: Lattice) -> None:
        self.lattice = lattice
        self.ref = {o.id: f"{b.name}/{o.name}" for b in lattice.branches for o in b.occurrences}
        self.id = {ref: occurrence for occurrence, ref in self.ref.items()}
        self.branch_name = {b.id: b.name for b in lattice.branches}
        self.branch_id = {b.name: b.id for b in lattice.branches}
        self.component_name = {c.id: c.name for c in lattice.shared_components}
        self.component_id = {c.name: c.id for c in lattice.shared_components}

    def occurrence(self, ref: Any, path: Path, where: str) -> str:
        found = self.id.get(str(ref))
        if found is None:
            raise FileError(path, f"{where}: no occurrence {ref!r} in the lattice (write branch/occurrence)")
        return found

    def branch(self, name: Any, path: Path, where: str) -> str:
        if name not in self.branch_id:
            raise FileError(path, f"{where}: no branch {name!r}")
        return self.branch_id[name]


# -- writing -------------------------------------------------------------------


def write_devices(description: MachineDescription, lattice: Lattice, folder: Path) -> list[Path]:
    """Write naming.yaml and the device files of ``description``; returns the paths written."""
    folder, refs = Path(folder), References(lattice)
    return [
        _write_naming(description.naming, lattice, refs, folder / NAMING_FILE),
        *_write_power(description.supplies, refs, folder),
        _write_diagnostics(description, refs, folder / DIAGNOSTICS_FILE),
        _write_rf(description.rf, refs, folder / RF_FILE),
    ]


def _write_naming(naming: Naming, lattice: Lattice, refs: References, path: Path) -> Path:
    areas = mapping()
    for branch_id, sections in naming.areas.items():
        length = lattice.layout(branch_id).length
        only = sections[0] if len(sections) == 1 else None
        if (
            only is not None
            and only.start == 0.0
            and only.end >= length
            and only.model_dump(exclude_defaults=True).keys() <= {"prefix", "start", "end"}
        ):
            areas[refs.branch_name[branch_id]] = only.prefix
        else:
            areas[refs.branch_name[branch_id]] = block(
                flow(a.model_dump(exclude_defaults=True)) for a in sections
            )
    tokens = naming.model_dump(exclude={"areas", "shared", "names"})
    data = mapping([("areas", areas), ("tokens", flow(tokens))])
    data["shared"] = mapping(
        (refs.component_name[s.component], flow({"home": s.home, "abbreviation": s.abbreviation}))
        for s in naming.shared
    )
    data["names"] = mapping((refs.ref[o], name) for o, name in naming.names.items())
    return write_yaml(path, data, _NAMING_HEADER)


_KIND_OF_UNIT = {
    ("T", False): "dipole",
    ("T/m", False): "quadrupole",
    ("T/m^2", False): "sextupole",
    ("T/m^3", False): "octupole",
    ("T m", True): "corrector",
    ("T", True): "thin-quadrupole",
    ("T/m", True): "thin-sextupole",
    ("T/m^2", True): "thin-octupole",
}


def _write_power(supplies, refs: References, folder: Path) -> list[Path]:
    path = folder / POWER_FILE
    library = mapping()
    named: dict[str, str] = {}
    written: list[Path] = []

    def name_of(transfer) -> str:
        key = transfer.model_dump_json()
        if key in named:
            return named[key]
        if isinstance(transfer, ServiceTransfer):
            base = f"service-{transfer.magnet}".lower().replace("/", "-")
        else:
            base = _KIND_OF_UNIT.get((transfer.field_unit, transfer.integrated), "magnet")
            base += f"-{transfer.source.value}"
        name, n = base, 2
        while name in library:
            name, n = f"{base}-{n}", n + 1
        named[key] = name
        library[name] = _transfer_spec(transfer, name, folder, written, name_of)
        return name

    entries = mapping()
    for supply in supplies:
        entry = mapping()
        if supply.name:
            entry["name"] = supply.name
        if len(supply.circuits) == 1 and supply.circuits[0].plane is None:
            _fill_circuit(entry, supply.circuits[0], refs, name_of)
        else:
            entry["circuits"] = mapping(
                (circuit.plane, flow(_fill_circuit(mapping(), circuit, refs, name_of)))
                for circuit in supply.circuits
            )
        magnets = sum(len(c.magnets) for c in supply.circuits)
        # A supply of one magnet reads best as one line; a string, one magnet per line.
        entries[supply.key] = flow(entry) if magnets == len(supply.circuits) == 1 else entry
        if len(supply.circuits) == 1 and magnets > 1:
            entries.yaml_add_eol_comment(f"{magnets} magnets in series", supply.key)
    data = mapping([("transfer", library), ("supplies", entries)])
    written.insert(0, write_yaml(path, data, _POWER_HEADER))
    return written


def _fill_circuit(entry, circuit: SupplyCircuit, refs: References, name_of) -> Any:
    magnets = circuit.magnets
    common = {
        "sets": _most(m.parameter for m in magnets),
        "transfer": _most(name_of(m.transfer) for m in magnets),
        "polarity": _most(m.polarity for m in magnets),
    }
    entry["sets"] = _short(common["sets"])
    entry["transfer"] = common["transfer"]
    if common["polarity"] != 1:
        entry["polarity"] = common["polarity"]
    entry["limits"] = flow(list(circuit.limits))
    if circuit.unit != "A":
        entry["unit"] = circuit.unit
    items = []
    for magnet in magnets:
        where = [refs.ref[o] for o in magnet.occurrences]
        own = {}
        if magnet.parameter != common["sets"]:
            own["sets"] = _short(magnet.parameter)
        if name_of(magnet.transfer) != common["transfer"]:
            own["transfer"] = name_of(magnet.transfer)
        if magnet.polarity != common["polarity"]:
            own["polarity"] = magnet.polarity
        at = where[0] if len(where) == 1 else flow(where)
        items.append(flow({"magnet": at, **own}) if own else at)
    entry["magnets"] = flow(items) if len(items) <= 3 else block(items)
    return entry


def _transfer_spec(transfer, name: str, folder: Path, written: list[Path], name_of) -> Any:
    spec: dict[str, Any] = {"kind": transfer.kind}
    if isinstance(transfer, PolynomialTransfer):
        spec["coefficients"] = list(transfer.coefficients)
    elif isinstance(transfer, TableTransfer):
        relative = f"transfer/{name}.csv"
        written.append(
            write_table(
                folder / TRANSFER_DIR / f"{name}.csv",
                [f"current [{transfer.engineering_unit}]", f"field [{transfer.field_unit}]"],
                zip(transfer.currents, transfer.fields, strict=True),
            )
        )
        spec["file"] = relative
    else:
        spec |= {"service": transfer.service, "magnet": transfer.magnet}
        if transfer.hysteresis:
            spec["hysteresis"] = True
        if transfer.fallback is not None:
            spec["fallback"] = name_of(transfer.fallback)
    spec["field_unit"] = transfer.field_unit
    if transfer.integrated:
        spec["integrated"] = True
    if transfer.engineering_unit != "A":
        spec["engineering_unit"] = transfer.engineering_unit
    spec["source"] = transfer.source.value
    return flow(spec)


def _write_diagnostics(description: MachineDescription, refs: References, path: Path) -> Path:
    bpms = []
    for bpm in description.bpms:
        if bpm.unit == "mm" and bpm.scale == 1e3:
            bpms.append(refs.ref[bpm.occurrence])
        else:
            bpms.append(flow({"at": refs.ref[bpm.occurrence], "unit": bpm.unit, "scale": bpm.scale}))
    data = mapping([("bpms", block(bpms))])
    data["tune_monitors"] = flow([refs.branch_name[t.branch] for t in description.tune_monitors])
    return write_yaml(path, data, _DIAGNOSTICS_HEADER)


def rename_references(folder: Path, renames: Mapping[str, str]) -> list[Path]:
    """Replace the references ``branch/old`` by ``branch/new`` in the naming and device files.

    For an import that renamed occurrences where they are: everything else in
    the files, comments included, stays as people wrote it. A supply keyed by
    a renamed magnet takes the new name as its key, and with it a new id; its
    EIC name and channels stay. Returns the files changed.
    """
    changed = []
    for path in [folder / NAMING_FILE, *sorted((folder / "devices").glob("*.yaml"))]:
        if not path.is_file():
            continue
        text = original = path.read_text()
        for old, new in renames.items():
            text = re.sub(rf"(?<![\w./-]){re.escape(old)}(?![\w.])", new, text)
        if text != original:
            path.write_text(text)
            changed.append(path)
    return changed


def set_supplies(
    folder: Path,
    transfers: Mapping[str, tuple[PolynomialTransfer, str]],
    circuits: Mapping[tuple[str, str | None], tuple[str, tuple[float, float]]],
) -> Path:
    """Give supplies' circuits transfer functions and limits, in the power file, in place.

    ``transfers`` are named transfer functions with a note each, added under ``transfer:`` (or
    replacing one of the same name); ``circuits`` maps a supply's key and circuit (its plane, or None
    for a supply of one circuit) to the transfer function's name and the limits it takes. A magnet
    entry with a transfer function of its own keeps it. Everything else in the file, comments
    included, stays as people wrote it. Returns the file.
    """
    path = Path(folder) / POWER_FILE
    data = read_yaml(path)
    if not isinstance(data, Mapping) or not isinstance(data.get("supplies"), Mapping):
        raise FileError(path, "no supplies to set")
    if data.get("transfer") is None:
        data["transfer"] = mapping()
    library = data["transfer"]
    for name, (transfer, note) in transfers.items():
        library[name] = _transfer_spec(transfer, name, Path(folder), [], lambda t: t)
        if note:
            comment(library, name, note)
    supplies = data["supplies"]
    for (key, plane), (name, limits) in circuits.items():
        if key not in supplies:
            raise FileError(path, f"no supply {key}")
        entry = supplies[key]
        if plane is not None:
            entry = (entry.get("circuits") or {}).get(plane)
            if entry is None:
                raise FileError(path, f"supply {key} has no circuit {plane}")
        if name not in library:
            raise FileError(path, f"supply {key}: no transfer function {name!r}")
        entry["transfer"] = name
        entry["limits"] = flow([float(v) for v in limits])
    return write_yaml(path, data)


def add_bpms(folder: Path, bpms: Sequence[str], branches: Sequence[str]) -> Path | None:
    """Add BPMs at ``bpms`` (``branch/occurrence``) to the diagnostics file, in its words.

    Each goes after the last BPM of its branch, or else before the first BPM
    of a branch that comes later in ``branches``; the rest of the file stays
    as it is. Returns the file, or None when there was nothing to add.
    """
    path = folder / DIAGNOSTICS_FILE
    lines = path.read_text().splitlines(keepends=True)
    try:
        start = next(i for i, line in enumerate(lines) if re.match(r"bpms:\s*$", line))
    except StopIteration:
        raise FileError(
            f"{path}: no block list of bpms to add {len(bpms)} BPMs to; add them by hand"
        ) from None
    end = start + 1
    while end < len(lines) and re.match(r"\s+- ", lines[end]):
        end += 1
    entries = lines[start + 1 : end]
    indent = re.match(r"(\s+)- ", entries[0]).group(1) if entries else "  "
    listed = {m.group(1) for line in entries if (m := re.search(r"([\w.]+/[\w.]+)", line))}
    order = {branch: i for i, branch in enumerate(branches)}

    def branch_of(line: str) -> int:
        m = re.search(r"([\w.]+)/", line)
        return order.get(m.group(1), len(order)) if m else len(order)

    added = 0
    for bpm in bpms:
        if bpm in listed:
            continue
        rank = order.get(bpm.split("/")[0], len(order))
        at = next((i for i, line in enumerate(entries) if branch_of(line) > rank), len(entries))
        entries.insert(at, f"{indent}- {bpm}\n")
        listed.add(bpm)
        added += 1
    if not added:
        return None
    path.write_text("".join([*lines[: start + 1], *entries, *lines[end:]]))
    return path


def _write_rf(stations, refs: References, path: Path) -> Path:
    cavities = mapping(
        (refs.ref[s.occurrence], flow({"voltage": list(s.voltage), "phase": list(s.phase)})) for s in stations
    )
    return write_yaml(path, mapping([("cavities", cavities)]), _RF_HEADER)


def _most(values):
    values = list(values)
    return max(dict.fromkeys(values), key=values.count)


def _short(parameter: str) -> str:
    return parameter.removeprefix("MagneticMultipoleP.")


# -- reading -------------------------------------------------------------------


def read_devices(folder: Path, lattice: Lattice) -> MachineDescription:
    """The machine description in ``folder``'s naming and device files (missing files: none of that)."""
    folder, refs = Path(folder), References(lattice)
    naming = _read_naming(folder / NAMING_FILE, lattice, refs)
    supplies = _read_power(folder, refs)
    bpms, tunes = _read_diagnostics(folder / DIAGNOSTICS_FILE, refs)
    rf = _read_rf(folder / RF_FILE, refs)
    try:
        return MachineDescription(
            lattice=lattice.name, naming=naming, supplies=supplies, bpms=bpms, rf=rf, tune_monitors=tunes
        )
    except ValidationError as error:
        raise FileError(folder, f"the device files do not describe valid devices:\n{error}") from None


def _read_naming(path: Path, lattice: Lattice, refs: References) -> Naming:
    if not path.exists():
        return Naming()
    data = _mapping(read_yaml(path) or {}, path)
    areas = {}
    for branch, value in _mapping(data.get("areas") or {}, path, "areas").items():
        branch_id = refs.branch(branch, path, "areas")
        if isinstance(value, str):
            sections = (NamingArea(prefix=value, start=0.0, end=lattice.layout(branch_id).length + 1.0),)
        else:
            sections = tuple(NamingArea(**_plain(s)) for s in value)
        areas[branch_id] = sections
    shared = tuple(
        SharedName(
            component=refs.component_id.get(name) or _fail(path, f"shared: no shared component {name!r}"),
            **_plain(entry),
        )
        for name, entry in _mapping(data.get("shared") or {}, path, "shared").items()
    )
    names = {
        refs.occurrence(ref, path, "names"): name
        for ref, name in _mapping(data.get("names") or {}, path, "names").items()
    }
    try:
        return Naming(areas=areas, shared=shared, names=names, **_plain(data.get("tokens") or {}))
    except ValidationError as error:
        raise FileError(path, str(error)) from None


def _read_power(folder: Path, refs: References) -> tuple[Supply, ...]:
    path = folder / POWER_FILE
    if not path.exists():
        return ()
    data = _mapping(read_yaml(path) or {}, path)
    specs = _mapping(data.get("transfer") or {}, path, "transfer")
    cache: dict[str, Any] = {}

    def transfer(value, where: str):
        if isinstance(value, str):
            if value not in specs:
                raise FileError(path, f"{where}: no transfer function {value!r} under transfer:")
            if value not in cache:
                cache[value] = _transfer(_plain(specs[value]), value, path, folder, transfer)
            return cache[value]
        return _transfer(_plain(value), where, path, folder, transfer)

    supplies = []
    for key, entry in _mapping(data.get("supplies") or {}, path, "supplies").items():
        entry = _mapping(entry, path, f"supply {key}")
        if "circuits" in entry:
            circuits = [
                _circuit(
                    _mapping(c, path, f"supply {key} circuit {plane}"),
                    str(plane),
                    path,
                    f"supply {key} circuit {plane}",
                    refs,
                    transfer,
                )
                for plane, c in _mapping(entry["circuits"], path, f"supply {key} circuits").items()
            ]
        else:
            circuits = [_circuit(entry, None, path, f"supply {key}", refs, transfer)]
        try:
            supplies.append(Supply(key=str(key), name=entry.get("name"), circuits=tuple(circuits)))
        except ValidationError as error:
            raise FileError(path, f"supply {key}: {error}") from None
    return tuple(supplies)


def _circuit(entry, plane, path: Path, where: str, refs: References, transfer) -> SupplyCircuit:
    for needed in ("sets", "limits", "magnets"):
        if needed not in entry:
            raise FileError(path, f"{where}: needs {needed}")
    default_transfer = entry.get("transfer")
    magnets = []
    for number, item in enumerate(entry["magnets"], start=1):
        here = f"{where}, magnet {number}"
        own = dict(_plain(item)) if isinstance(item, Mapping) else {"magnet": _plain(item)}
        at = own["magnet"] if "magnet" in own else _fail(path, f"{here}: needs magnet")
        occurrences = tuple(refs.occurrence(r, path, here) for r in (at if isinstance(at, list) else [at]))
        chosen = own.get("transfer", default_transfer)
        if chosen is None:
            raise FileError(path, f"{here}: no transfer function")
        try:
            magnets.append(
                CircuitMagnet(
                    occurrences=occurrences,
                    parameter=_long(own.get("sets", entry["sets"])),
                    transfer=transfer(chosen, here),
                    polarity=own.get("polarity", entry.get("polarity", 1)),
                )
            )
        except ValidationError as error:
            raise FileError(path, f"{here}: {error}") from None
    try:
        return SupplyCircuit(
            plane=plane, limits=tuple(entry["limits"]), unit=entry.get("unit", "A"), magnets=tuple(magnets)
        )
    except ValidationError as error:
        raise FileError(path, f"{where}: {error}") from None


def _transfer(spec: dict, where: str, path: Path, folder: Path, resolve):
    kind = spec.get("kind")
    try:
        if kind == "table":
            if "file" not in spec:
                raise FileError(path, f"transfer {where}: a table needs file:")
            table = folder / "devices" / spec.pop("file")
            _, rows = read_table(table)
            if any(len(row) != 2 for row in rows):
                raise FileError(table, "a transfer table has two columns, current and field")
            return TableTransfer(currents=tuple(r[0] for r in rows), fields=tuple(r[1] for r in rows), **spec)
        if kind == "service":
            if isinstance(spec.get("fallback"), str):
                spec["fallback"] = resolve(spec["fallback"], f"transfer {where} fallback")
            return ServiceTransfer(**spec)
        if kind == "polynomial":
            return PolynomialTransfer(**spec)
    except ValidationError as error:
        raise FileError(path, f"transfer {where}: {error}") from None
    raise FileError(path, f"transfer {where}: kind is polynomial, table or service, not {kind!r}")


def _read_diagnostics(path: Path, refs: References) -> tuple[tuple[Bpm, ...], tuple[TuneMonitor, ...]]:
    if not path.exists():
        return (), ()
    data = _mapping(read_yaml(path) or {}, path)
    bpms = []
    for number, item in enumerate(data.get("bpms") or [], start=1):
        own = dict(_plain(item)) if isinstance(item, Mapping) else {"at": item}
        at = own.pop("at", None) or _fail(path, f"bpm {number}: needs at")
        bpms.append(Bpm(occurrence=refs.occurrence(at, path, f"bpm {number}"), **own))
    tunes = [
        TuneMonitor(branch=refs.branch(b, path, "tune_monitors")) for b in data.get("tune_monitors") or []
    ]
    return tuple(bpms), tuple(tunes)


def _read_rf(path: Path, refs: References) -> tuple[RfStation, ...]:
    if not path.exists():
        return ()
    data = _mapping(read_yaml(path) or {}, path)
    stations = []
    for ref, entry in _mapping(data.get("cavities") or {}, path, "cavities").items():
        entry = _plain(_mapping(entry, path, f"cavity {ref}"))
        try:
            stations.append(RfStation(occurrence=refs.occurrence(ref, path, f"cavity {ref}"), **entry))
        except ValidationError as error:
            raise FileError(path, f"cavity {ref}: {error}") from None
    return tuple(stations)


def _long(parameter: str) -> str:
    return parameter if "." in parameter else f"MagneticMultipoleP.{parameter}"


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def _mapping(value: Any, path: Path, where: str = "the file") -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FileError(path, f"{where} must be a mapping")
    return value


def _fail(path: Path, message: str):
    raise FileError(path, message)
