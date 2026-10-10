"""The lattice as files: ``lattice/<branch>.yaml`` for each branch and ``lattice/shared.yaml``.

A branch file has the branch's element definitions and then its line, the
occurrences in beam order, one per line. Everything refers by name: an
occurrence names its element, a fork names the branch and the occurrence it
leads to, a shared component names its traversals as ``branch/occurrence``.
A definition placed in several branches is written in the first branch's
file and named from the others. Definitions, branches, shared components and
drifts are identified by their names; every other occurrence (a magnet,
monitor, cavity or marker in the beam line) keeps a stable ``id``, its identity
for devices, settings and history through renames, and a new one may leave it
out to get an id derived from its name. An occurrence's ``eic`` is its EIC name (EIC-SEG-RSI-009) from
the naming rules of ``naming.yaml``, written and kept current by the tools.
Values are in the units and conventions of the lattice model (SI, angles in
rad, normalized strengths; ADR 0001).
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from eicvibe.domain.lattice import FORMAT_VERSION, ElementKind, Lattice, derived_id

from .yamlio import FileError, block, flow, mapping, read_yaml, write_yaml

LATTICE_DIR = "lattice"
SHARED_FILE = "shared.yaml"

#: Flow-style element entries longer than this are written one key per line.
_ONE_LINE = 150

_BRANCH_HEADER = """Branch {branch} of {lattice}: its element definitions, then its line in beam order.
Units SI, angles in rad, strengths normalized (PALS conventions, ADR 0001).
Elements are referred to by name. `eic` is the element's EIC name, from the rules in
naming.yaml; `eicvibe build` keeps it current (to choose a name, add it to naming.yaml's
names). Keep each occurrence's `id` (drifts have none: they go by name): it is its
identity for devices, settings and history. A new occurrence may leave it out; it then
gets an id derived from its name."""

#: Headers the tools wrote before, replaced by the current one where a file still has them.
OLD_BRANCH_HEADERS = (
    """Branch {branch} of {lattice}: its element definitions, then its line in beam order.
Units SI, angles in rad, strengths normalized (PALS conventions, ADR 0001).
Elements are referred to by name. Keep each `id`: it is the entry's identity for
devices and history. A new element or occurrence may leave it out; it then gets
an id derived from its name.""",
    """Branch {branch} of {lattice}: its element definitions, then its line in beam order.
Units SI, angles in rad, strengths normalized (PALS conventions, ADR 0001).
Elements are referred to by name. `eic` is the element's EIC name, from the rules in
naming.yaml; `eicvibe build` keeps it current (to choose a name, add it to naming.yaml's
names). Keep each occurrence's `id`: it is its identity for devices, settings and
history. A new occurrence may leave it out; it then gets an id derived from its name.""",
)

_SHARED_HEADER = """Physical components that several branches pass through, such as a septum: one
piece of hardware, a traversal (branch/occurrence) for each beam that passes it."""


def branch_order(lattice: Lattice) -> list[str]:
    """Branch names in beam order: the root, then along forks, then any others."""
    order: list[str] = []

    def visit(branch_id: str) -> None:
        if branch_id in order:
            return
        order.append(branch_id)
        for edge in sorted(lattice.forks_from(branch_id), key=lambda e: e.s_from):
            visit(edge.to_branch)

    if lattice.root_branch:
        visit(lattice.root_branch)
    for branch in lattice.branches:
        visit(branch.id)
    return [lattice.branch(b).name for b in order]


def name_ids(lattice: Lattice) -> Lattice:
    """The lattice as its files identify it: definitions, branches, shared components and drifts by name.

    Other occurrences keep their ids. Reading back what :func:`write_lattice` wrote gives this lattice.
    """
    data = lattice.to_dict()
    elements = {e["id"]: derived_id("ele", lattice.name, e["name"]) for e in data.get("elements", [])}
    branches = {b["id"]: derived_id("br", lattice.name, b["name"]) for b in data.get("branches", [])}
    drifts = {
        o.id: derived_id("occ", lattice.name, b.name, o.name)
        for b in lattice.branches
        for o in b.occurrences
        if lattice.element(o.element).kind is ElementKind.DRIFT
    }
    for element in data.get("elements", []):
        element["id"] = elements[element["id"]]
    for branch in data.get("branches", []):
        branch["id"] = branches[branch["id"]]
        for occurrence in branch.get("occurrences", []):
            occurrence["id"] = drifts.get(occurrence["id"], occurrence["id"])
            occurrence["element"] = elements[occurrence["element"]]
            if "fork" in occurrence:
                fork = occurrence["fork"]
                fork["to_branch"] = branches[fork["to_branch"]]
                if fork.get("to_occurrence"):
                    fork["to_occurrence"] = drifts.get(fork["to_occurrence"], fork["to_occurrence"])
    if data.get("root_branch"):
        data["root_branch"] = branches[data["root_branch"]]
    for component in data.get("shared_components", []):
        component["id"] = derived_id("cmp", lattice.name, component["name"])
        for traversal in component.get("traversals", []):
            traversal["occurrence"] = drifts.get(traversal["occurrence"], traversal["occurrence"])
    return Lattice.from_dict(data)


def write_lattice(
    lattice: Lattice, folder: Path, order: list[str] | None = None, eic: Mapping[str, str] | None = None
) -> list[Path]:
    """Write the branch files and the shared-component file; returns the paths written.

    ``eic`` gives occurrences' EIC names, by occurrence id (a registry's slots).
    """
    eic = eic or {}
    order = order or branch_order(lattice)
    directory = Path(folder) / LATTICE_DIR
    owner: dict[str, str] = {}  # definition id -> branch whose file holds it
    for name in order:
        for occurrence in lattice.branch(name).occurrences:
            owner.setdefault(occurrence.element, name)
    for element in lattice.elements:
        owner.setdefault(element.id, order[0])
    names = {e.id: e.name for e in lattice.elements}
    branch_names = {b.id: b.name for b in lattice.branches}
    occurrence_names = {o.id: o.name for b in lattice.branches for o in b.occurrences}
    written = []
    for name in order:
        branch = lattice.branch(name)
        placed_here = list(dict.fromkeys(o.element for o in branch.occurrences))
        unplaced = [e.id for e in lattice.elements if owner[e.id] == name and e.id not in placed_here]
        elements = mapping()
        for element_id in placed_here + unplaced:
            if owner[element_id] == name:
                elements[names[element_id]] = _element_entry(lattice.element(element_id))
        line = block(
            _occurrence_entry(
                o,
                names,
                branch_names,
                occurrence_names,
                eic.get(o.id),
                lattice.element(o.element).kind is not ElementKind.DRIFT,
            )
            for o in branch.occurrences
        )
        data = mapping([("branch", branch.name), ("kind", branch.kind.value)])
        data["elements"] = elements
        data["line"] = line
        header = _BRANCH_HEADER.format(branch=branch.name, lattice=lattice.name)
        written.append(write_yaml(directory / f"{name}.yaml", data, header))
    shared = mapping()
    for component in lattice.shared_components:
        traversals = {}
        for t in component.traversals:
            branch_name = branch_names[lattice.placement(t.occurrence).branch_id]
            traversals[t.role] = f"{branch_name}/{occurrence_names[t.occurrence]}"
        shared[component.name] = flow({"traversals": traversals})
    written.append(write_yaml(directory / SHARED_FILE, shared, _SHARED_HEADER))
    return written


def _element_entry(element) -> Any:
    data = element.model_dump(mode="json", exclude_defaults=True, exclude={"id", "name"})
    data = {"kind": element.kind.value, **{k: v for k, v in data.items() if k != "kind"}}
    one_line = flow(data)
    if len(str(dict(data))) <= _ONE_LINE:
        return one_line
    entry = mapping((key, flow(value)) for key, value in data.items())
    return entry


def _occurrence_entry(
    occurrence, names, branch_names, occurrence_names, eic: str | None = None, with_id: bool = True
) -> Any:
    data: dict[str, Any] = {"name": occurrence.name}
    if eic:
        data["eic"] = eic
    data["element"] = names[occurrence.element]
    if occurrence.tags:
        data["tags"] = list(occurrence.tags)
    if occurrence.overrides:
        overrides: dict[str, dict[str, Any]] = {}
        for override in occurrence.overrides:
            overrides.setdefault(override.group, {})[override.parameter] = override.value
        data["overrides"] = overrides
    if occurrence.fork is not None:
        fork = occurrence.fork.model_dump(mode="json", exclude_defaults=True)
        link: dict[str, Any] = {"to": branch_names[occurrence.fork.to_branch]}
        if occurrence.fork.to_occurrence is not None:
            link["at"] = occurrence_names[occurrence.fork.to_occurrence]
        for key, renamed in (
            ("to_side", "side"),
            ("direction", "direction"),
            ("propagate_reference", "propagate_reference"),
        ):
            if key in fork:
                link[renamed] = fork[key]
        data["fork"] = link
    if with_id:  # drifts go by name: nothing attaches to them
        data["id"] = occurrence.id
    return flow(data)


# -- reading -------------------------------------------------------------------


def read_lattice(folder: Path, name: str, branches: list[str], root: str | None) -> Lattice:
    """The lattice in ``folder``'s lattice files, for the branches listed in machine.yaml."""
    directory = Path(folder) / LATTICE_DIR
    files = {b: _mapping(read_yaml(directory / f"{b}.yaml"), directory / f"{b}.yaml") for b in branches}
    definitions: dict[str, dict[str, Any]] = {}
    for branch, data in files.items():
        path = directory / f"{branch}.yaml"
        for element_name, entry in _mapping(data.get("elements") or {}, path, "elements").items():
            if element_name in definitions:
                raise FileError(path, f"element {element_name!r} is also defined in another branch file")
            entry = dict(_mapping(entry, path, f"element {element_name}"))
            definitions[element_name] = {
                **entry,
                "name": element_name,
                "id": entry.get("id") or derived_id("ele", name, element_name),
            }
    branch_ids = {b: files[b].get("id") or derived_id("br", name, b) for b in branches}
    occurrence_ids: dict[str, dict[str, str]] = {}
    lines: dict[str, list] = {}
    for branch, data in files.items():
        path = directory / f"{branch}.yaml"
        if data.get("branch", branch) != branch:
            raise FileError(
                path, f"says it is branch {data.get('branch')!r}, but machine.yaml lists it as {branch!r}"
            )
        line = data.get("line") or []
        if not isinstance(line, list):
            raise FileError(path, "line must be a list of occurrences")
        lines[branch] = line
        ids = occurrence_ids[branch] = {}
        for number, entry in enumerate(line, start=1):
            entry = _mapping(entry, path, f"line item {number}")
            if "name" not in entry or "element" not in entry:
                raise FileError(path, f"line item {number} needs a name and an element")
            ids[entry["name"]] = entry.get("id") or derived_id("occ", name, branch, entry["name"])

    branches_out = []
    for branch in branches:
        path = directory / f"{branch}.yaml"
        occurrences = []
        for number, entry in enumerate(lines[branch], start=1):
            where = f"line item {number} ({entry['name']})"
            element = definitions.get(entry["element"])
            if element is None:
                raise FileError(path, f"{where}: no element {entry['element']!r} in the lattice files")
            occurrence: dict[str, Any] = {
                "id": occurrence_ids[branch][entry["name"]],
                "name": entry["name"],
                "element": element["id"],
            }
            if entry.get("tags"):
                occurrence["tags"] = list(entry["tags"])
            if entry.get("overrides"):
                occurrence["overrides"] = [
                    {"group": group, "parameter": parameter, "value": value}
                    for group, values in _mapping(entry["overrides"], path, f"{where} overrides").items()
                    for parameter, value in _mapping(values, path, f"{where} overrides").items()
                ]
            if entry.get("fork"):
                occurrence["fork"] = _fork(entry["fork"], path, where, branch_ids, occurrence_ids)
            occurrences.append(occurrence)
        data = files[branch]
        branches_out.append(
            {
                "id": branch_ids[branch],
                "name": branch,
                "kind": data.get("kind", "linac"),
                "occurrences": occurrences,
            }
        )

    shared_path = directory / SHARED_FILE
    components = []
    if shared_path.exists():
        for component_name, entry in _mapping(read_yaml(shared_path) or {}, shared_path).items():
            entry = _mapping(entry, shared_path, component_name)
            traversals = [
                {"role": role, "occurrence": _reference(ref, shared_path, component_name, occurrence_ids)}
                for role, ref in _mapping(entry.get("traversals") or {}, shared_path, component_name).items()
            ]
            components.append(
                {
                    "id": entry.get("id") or derived_id("cmp", name, component_name),
                    "name": component_name,
                    "traversals": traversals,
                }
            )
    data = {
        "format_version": FORMAT_VERSION,
        "name": name,
        "elements": [_definition(d) for d in definitions.values()],
        "branches": branches_out,
        "root_branch": branch_ids[root] if root else None,
        "shared_components": components,
    }
    try:
        return Lattice.from_dict(data)
    except ValidationError as error:
        raise FileError(directory, f"the lattice files do not describe a valid lattice:\n{error}") from None


def _definition(entry: dict[str, Any]) -> dict[str, Any]:
    return {key: (list(value) if key == "tags" else _plain(value)) for key, value in entry.items()}


def _plain(value: Any) -> Any:
    """ruamel's commented containers as plain dicts and lists."""
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def _fork(entry, path, where, branch_ids, occurrence_ids) -> dict[str, Any]:
    entry = _mapping(entry, path, f"{where} fork")
    target = entry.get("to")
    if target not in branch_ids:
        raise FileError(path, f"{where}: fork to unknown branch {target!r}")
    link: dict[str, Any] = {"to_branch": branch_ids[target]}
    if entry.get("at") is not None:
        landing = occurrence_ids[target].get(entry["at"])
        if landing is None:
            raise FileError(path, f"{where}: branch {target!r} has no occurrence {entry['at']!r}")
        link["to_occurrence"] = landing
    for renamed, key in (
        ("side", "to_side"),
        ("direction", "direction"),
        ("propagate_reference", "propagate_reference"),
    ):
        if renamed in entry:
            link[key] = entry[renamed]
    return link


def _reference(ref: Any, path: Path, where: str, occurrence_ids: dict[str, dict[str, str]]) -> str:
    """The occurrence id of ``branch/occurrence``."""
    branch, _, occurrence = str(ref).partition("/")
    found = occurrence_ids.get(branch, {}).get(occurrence)
    if found is None:
        raise FileError(path, f"{where}: no occurrence {ref!r} (write branch/occurrence)")
    return found


def _mapping(value: Any, path: Path, where: str = "the file") -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FileError(path, f"{where} must be a mapping of names to entries")
    return value
