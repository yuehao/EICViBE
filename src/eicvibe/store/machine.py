"""A machine folder: the files people edit, and what is built from them (ADR 0005).

::

    machine.yaml              index: name, root and branches in beam order, provenance
    lattice/<branch>.yaml     each branch: element definitions and its line
    lattice/shared.yaml       components several branches pass through
    naming.yaml               EIC naming rules
    devices/power.yaml        supplies, strings and transfer functions
    devices/transfer/*.csv    measured transfer functions
    devices/diagnostics.yaml  BPMs and tune monitors
    devices/rf.yaml           RF stations
  routes.yaml               the beam's paths through several branches, by branch name
    settings/*.yaml           operating settings, e.g. a mode's magnet strengths
    measurements/             reserved: the magnetic measurement database
    ramps/                    reserved: synchrotron ramp profiles
    build/                    written by ``eicvibe build``; never edited

:func:`load_machine` reads the lattice and assembles the registry from these
files, every time, so an edit counts as soon as it is saved.
"""

import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from eicvibe.domain.lattice import ElementKind, Lattice, Route, route_through
from eicvibe.domain.machine import MachineDescription, MachineRegistry, assemble_registry

from .device_files import References, read_devices, write_devices
from .lattice_files import (
    _BRANCH_HEADER,
    LATTICE_DIR,
    OLD_BRANCH_HEADERS,
    SHARED_FILE,
    branch_order,
    read_lattice,
    write_lattice,
)
from .yamlio import FileError, block, flow, mapping, read_yaml, write_yaml

MACHINE_FILE = "machine.yaml"
SETTINGS_DIR = "settings"
BUILD_DIR = "build"
FORMAT = "eicvibe-machine 1"

_MACHINE_HEADER = """The {name} machine for EICViBE (ADR 0005). People edit these files; `eicvibe check`
validates them and `eicvibe build` writes build/ (never edit build/).
  lattice/<branch>.yaml     each branch: element definitions and its line, in beam order
  lattice/shared.yaml       components several branches pass through (one magnet, several beams)
  naming.yaml               EIC naming rules
  devices/power.yaml        power supplies, strings and transfer functions (devices/transfer/*.csv)
  devices/diagnostics.yaml  BPMs and tune monitors
  devices/rf.yaml           RF stations
  settings/*.yaml           operating settings, e.g. a mode's magnet strengths
  measurements/, ramps/     reserved: magnetic measurements, ramp profiles"""

_SETTINGS_HEADER = """{description}
Values replace the lattice's: occurrence (branch/name) -> parameter -> value, in lattice units."""


@dataclass(frozen=True)
class Setting:
    """One value of an operating mode: a lattice parameter of an occurrence."""

    occurrence: str
    parameter: str
    value: float


@dataclass(frozen=True)
class Machine:
    folder: Path
    lattice: Lattice
    description: MachineDescription
    registry: MachineRegistry
    source: dict[str, Any]
    """Provenance of the lattice, as machine.yaml records it."""
    routes: tuple[Route, ...] = ()
    """The beam's paths through several branches (routes.yaml)."""


ROUTES_FILE = "routes.yaml"
ROUTES_HEADER = """\
Routes: the beam's paths through several branches, by branch name in beam order.
A route enters a branch where the previous branch's fork lands and leaves it at its fork
to the next branch, going round a ring when that fork lies behind; a route that ends in a
ring it entered part-way goes round once.
"""


def read_routes(folder: Path, lattice: Lattice) -> tuple[Route, ...]:
    """The routes in a machine folder's ``routes.yaml``; none without the file."""
    return read_routes_file(Path(folder) / ROUTES_FILE, lattice)


def read_routes_file(path: Path, lattice: Lattice) -> tuple[Route, ...]:
    """The routes in a routes file (``routes: {name: [branch, ...]}``); none without the file."""
    path = Path(path)
    if not path.is_file():
        return ()
    data = read_yaml(path) or {}
    entries = data.get("routes") if isinstance(data, dict) else None
    if not isinstance(entries, dict):
        raise FileError(path, "needs a mapping 'routes' of route names to lists of branch names")
    routes = []
    for name, branches in entries.items():
        if not isinstance(branches, list) or not branches:
            raise FileError(path, f"route {name}: give its branches as a list of names")
        try:
            routes.append(route_through(lattice, str(name), [str(b) for b in branches]))
        except (KeyError, ValueError) as error:
            raise FileError(path, f"route {name}: {error}") from None
    return tuple(routes)


def write_routes(folder: Path, routes: dict[str, list[str]]) -> Path | None:
    """Write ``routes.yaml`` if there is none: people edit it, so a tool never rewrites it."""
    path = Path(folder) / ROUTES_FILE
    if path.exists():
        return None
    return write_routes_file(path, routes)


def write_routes_file(path: Path, routes: dict[str, list[str]]) -> Path:
    """Write a routes file, ``routes: {name: [branch, ...]}``."""
    body = mapping([("routes", mapping((name, flow(list(branches))) for name, branches in routes.items()))])
    return write_yaml(path, body, ROUTES_HEADER)


def is_machine(folder: Path) -> bool:
    return (Path(folder) / MACHINE_FILE).is_file()


def read_index(folder: Path) -> dict[str, Any]:
    path = Path(folder) / MACHINE_FILE
    data = read_yaml(path)
    if not isinstance(data, dict) or "name" not in data or "branches" not in data:
        raise FileError(path, "needs name and branches")
    if data.get("format", FORMAT) != FORMAT:
        raise FileError(path, f"format {data.get('format')!r} is not {FORMAT!r}")
    return data


def load_lattice(folder: Path) -> Lattice:
    index = read_index(folder)
    branches = [str(b) for b in index["branches"]]
    return read_lattice(Path(folder), str(index["name"]), branches, index.get("root", branches[0]))


def load_machine(folder: Path) -> Machine:
    """The lattice, device description and registry in ``folder``."""
    folder = Path(folder)
    index = read_index(folder)
    lattice = load_lattice(folder)
    description = read_devices(folder, lattice)
    try:
        registry = assemble_registry(lattice, description)
    except (ValueError, KeyError) as error:
        raise FileError(folder, f"cannot assemble the registry: {error}") from None
    routes = read_routes(folder, lattice)
    return Machine(folder, lattice, description, registry, dict(index.get("source") or {}), routes)


def write_machine(
    folder: Path,
    lattice: Lattice,
    description: MachineDescription | None = None,
    *,
    source: dict[str, Any] | None = None,
) -> list[Path]:
    """Write a new machine folder: machine.yaml, the lattice files and (if given) the device files."""
    folder = Path(folder)
    order = branch_order(lattice)
    written = [write_index(folder, lattice, source, order)]
    names = eic_names(assemble_registry(lattice, description)) if description is not None else None
    written += write_lattice(lattice, folder, order, names)
    if description is not None:
        written += write_devices(description, lattice, folder)
    for reserved in ("measurements", "ramps"):
        (folder / reserved).mkdir(parents=True, exist_ok=True)
    return written


def write_index(
    folder: Path, lattice: Lattice, source: dict[str, Any] | None, order: list[str] | None = None
) -> Path:
    order = order or branch_order(lattice)
    data = mapping(
        [
            ("format", FORMAT),
            ("name", lattice.name),
            ("root", lattice.branch(lattice.root_branch).name if lattice.root_branch else order[0]),
            ("branches", flow(order)),
        ]
    )
    if source:
        data["source"] = _yaml_tree(source)
    return write_yaml(folder / MACHINE_FILE, data, _MACHINE_HEADER.format(name=lattice.name))


def write_settings(
    folder: Path, name: str, settings: list[Setting], lattice: Lattice, description: str
) -> Path:
    refs = References(lattice)
    values = mapping()
    for setting in settings:
        values.setdefault(refs.ref[setting.occurrence], {})[setting.parameter] = setting.value
    data = mapping([("settings", mapping((ref, flow(v)) for ref, v in values.items()))])
    return write_yaml(
        Path(folder) / SETTINGS_DIR / f"{name}.yaml", data, _SETTINGS_HEADER.format(description=description)
    )


def read_settings(folder: Path, name: str, lattice: Lattice) -> list[Setting]:
    path = Path(folder) / SETTINGS_DIR / f"{name}.yaml"
    data = read_yaml(path) or {}
    refs = References(lattice)
    return [
        Setting(refs.occurrence(ref, path, "settings"), str(parameter), float(value))
        for ref, values in (data.get("settings") or {}).items()
        for parameter, value in values.items()
    ]


def source_files(folder: Path) -> list[Path]:
    """Every file people edit in a machine folder (not build/), in a stable order."""
    folder = Path(folder)
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and BUILD_DIR not in p.relative_to(folder).parts[:1]
    )


def check_machine(folder: Path) -> list[str]:
    """Problems with a machine folder; empty when it loads and every reference resolves."""
    machine = load_machine(folder)
    problems = [f"lattice: {d}" for d in machine.lattice.diagnose()]
    problems += [f"registry: {p}" for p in machine.registry.check_against(machine.lattice)]
    problems += [f"power: {p}" for p in machine.registry.power_problems(machine.lattice)]
    stale = [c for c in update_names(folder, machine, write=False) if c.startswith("eic ")]
    if stale:
        problems.append(
            f"names: {len(stale)} EIC names in the lattice files differ from naming.yaml's rules "
            f"(`eicvibe build` updates them), e.g. {stale[0]}"
        )
    return problems


def eic_names(registry: MachineRegistry) -> dict[str, str]:
    """EIC names of the lattice's occurrences, by occurrence id (a shared magnet's at each traversal)."""
    return {occurrence: slot.name for slot in registry.slots for occurrence in slot.occurrences}


def update_names(folder: Path, machine: Machine | None = None, *, write: bool = True) -> list[str]:
    """Bring the lattice files' ``eic`` names to naming.yaml's rules, and drop ids only occurrences need.

    Edits the files in place, keeping their comments and layout; returns what changed (or, without
    ``write``, what would): ``eic <branch>/<name>: old -> new`` and ``ids <file>: removed n``.
    """
    folder = Path(folder)
    machine = machine or load_machine(folder)
    names = eic_names(machine.registry)
    directory = folder / LATTICE_DIR
    changes: list[str] = []
    for branch in read_index(folder)["branches"]:
        path = directory / f"{branch}.yaml"
        data = read_yaml(path)
        edited = removed = 0
        if "id" in data:
            del data["id"]
            removed += 1
        for entry in (data.get("elements") or {}).values():
            if isinstance(entry, dict) and "id" in entry:
                del entry["id"]
                removed += 1
        if _old_header(data, str(branch), machine.lattice.name):
            data.ca.comment = None
            data.yaml_set_start_comment(_BRANCH_HEADER.format(branch=branch, lattice=machine.lattice.name))
            edited += 1
        drifts = 0
        for entry in data.get("line") or []:
            placed = machine.lattice.placement(str(entry["name"]), branch=str(branch))
            if placed.kind is ElementKind.DRIFT and "id" in entry:
                del entry["id"]
                drifts += 1
            wanted = names.get(placed.id)
            have = entry.get("eic")
            if have == wanted:
                continue
            changes.append(f"eic {branch}/{entry['name']}: {have or '(none)'} -> {wanted or '(none)'}")
            edited += 1
            if wanted is None:
                del entry["eic"]
            elif "eic" in entry:
                entry["eic"] = wanted
            else:
                entry.insert(1, "eic", wanted)
        if removed:
            changes.append(
                f"ids {path.relative_to(folder)}: removed {removed} (definitions and branches go by name)"
            )
        if drifts:
            changes.append(f"ids {path.relative_to(folder)}: removed {drifts} of drifts (they go by name)")
        if write and (edited or removed or drifts):
            write_yaml(path, data)
    shared = directory / SHARED_FILE
    if shared.exists():
        data = read_yaml(shared) or {}
        removed = 0
        for entry in data.values():
            if isinstance(entry, dict) and "id" in entry:
                del entry["id"]
                removed += 1
        if removed:
            changes.append(f"ids {shared.relative_to(folder)}: removed {removed} (components go by name)")
            if write:
                write_yaml(shared, data)
    return changes


def _old_header(data: Any, branch: str, lattice: str) -> bool:
    """Whether a branch file still starts with a header the tools wrote before, unedited."""
    comment = getattr(getattr(data, "ca", None), "comment", None)
    tokens = comment[1] if comment and len(comment) > 1 and comment[1] else []
    text = "".join(t.value for t in tokens)
    return any(
        text == "".join(f"# {line}\n" for line in old.format(branch=branch, lattice=lattice).splitlines())
        for old in OLD_BRANCH_HEADERS
    )


def build_machine(folder: Path) -> dict[str, Path]:
    """Write build/lattice.json, build/registry.json and build/manifest.json from the source files.

    First brings the lattice files' EIC names up to date (:func:`update_names`).
    """
    folder = Path(folder)
    update_names(folder)
    machine = load_machine(folder)
    out = folder / BUILD_DIR
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "lattice": out / "lattice.json",
        "registry": out / "registry.json",
        "manifest": out / "manifest.json",
    }
    paths["lattice"].write_text(machine.lattice.to_json(indent=1) + "\n")
    registry = machine.registry.model_dump_json(indent=1)
    paths["registry"].write_text(registry + "\n")
    manifest = {
        "machine": machine.lattice.name,
        "lattice_digest": machine.lattice.digest,
        "registry_sha256": hashlib.sha256(registry.encode()).hexdigest(),
        "eicvibe_version": version("eicvibe"),
        "files": {
            str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_files(folder)
        },
        "source": machine.source,
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=1) + "\n")
    return paths


def _yaml_tree(value: Any) -> Any:
    if isinstance(value, dict):
        small = all(not isinstance(v, dict | list) for v in value.values())
        return flow(value) if small else mapping((k, _yaml_tree(v)) for k, v in value.items())
    if isinstance(value, list):
        return block(_yaml_tree(v) for v in value)
    return value
