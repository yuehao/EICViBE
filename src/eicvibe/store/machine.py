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

from eicvibe.domain.lattice import Lattice
from eicvibe.domain.machine import MachineDescription, MachineRegistry, assemble_registry

from .device_files import References, read_devices, write_devices
from .lattice_files import branch_order, read_lattice, write_lattice
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
    return Machine(folder, lattice, description, registry, dict(index.get("source") or {}))


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
    written += write_lattice(lattice, folder, order)
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
    return problems


def build_machine(folder: Path) -> dict[str, Path]:
    """Write build/lattice.json, build/registry.json and build/manifest.json from the source files."""
    folder = Path(folder)
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
