"""Differences between two lattice revisions, matched by stable id.

Because entities are matched by id, a rename is one ``name`` change, and an
insertion is one added occurrence rather than a shift of everything after
it. Indices and s positions are derived and are not compared.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Any

from .elements import ElementDefinition
from .parameters import GROUP_TYPES, TrackingP


class ChangeKind(StrEnum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


@dataclass(frozen=True, slots=True)
class Change:
    """One difference. ``field`` is a dotted path such as ``MagneticMultipoleP.Kn1``."""

    kind: ChangeKind
    entity: str
    """``lattice``, ``element``, ``branch``, ``occurrence`` or ``shared_component``."""
    id: str
    field: str
    old: Any
    new: Any

    def __str__(self) -> str:
        target = f"{self.entity} {self.id}" if self.id else self.entity
        if self.kind is ChangeKind.MODIFIED:
            return f"{target}: {self.field} {self.old!r} -> {self.new!r}"
        return f"{target} {self.kind.value}: {self.old if self.kind is ChangeKind.REMOVED else self.new}"


def diff(old, new) -> tuple[Change, ...]:
    """Changes that turn lattice ``old`` into lattice ``new``."""
    changes: list[Change] = []
    changes += _compare("lattice", {"": _lattice_fields(old)}, {"": _lattice_fields(new)}, {}, {})

    changes += _compare(
        "element",
        {e.id: _element_fields(e) for e in old.elements},
        {e.id: _element_fields(e) for e in new.elements},
        {e.id: e.name for e in old.elements},
        {e.id: e.name for e in new.elements},
    )

    changes += _compare(
        "branch",
        {b.id: {"name": b.name, "kind": b.kind} for b in old.branches},
        {b.id: {"name": b.name, "kind": b.kind} for b in new.branches},
        {b.id: b.name for b in old.branches},
        {b.id: b.name for b in new.branches},
    )
    changes += _order_changes(old, new)

    old_occ = {o.id: (b, o) for b in old.branches for o in b.occurrences}
    new_occ = {o.id: (b, o) for b in new.branches for o in b.occurrences}
    changes += _compare(
        "occurrence",
        {i: _occurrence_fields(b, o) for i, (b, o) in old_occ.items()},
        {i: _occurrence_fields(b, o) for i, (b, o) in new_occ.items()},
        {i: f"{b.name}/{o.name}" for i, (b, o) in old_occ.items()},
        {i: f"{b.name}/{o.name}" for i, (b, o) in new_occ.items()},
    )

    changes += _compare(
        "shared_component",
        {c.id: _component_fields(c) for c in old.shared_components},
        {c.id: _component_fields(c) for c in new.shared_components},
        {c.id: c.name for c in old.shared_components},
        {c.id: c.name for c in new.shared_components},
    )
    return tuple(changes)


def _compare(
    entity: str,
    old: Mapping[str, dict[str, Any]],
    new: Mapping[str, dict[str, Any]],
    old_labels: Mapping[str, str],
    new_labels: Mapping[str, str],
) -> list[Change]:
    changes = [
        Change(ChangeKind.REMOVED, entity, i, "", old_labels.get(i), None)
        for i in sorted(old.keys() - new.keys())
    ]
    changes += [
        Change(ChangeKind.ADDED, entity, i, "", None, new_labels.get(i))
        for i in sorted(new.keys() - old.keys())
    ]
    for identifier in sorted(old.keys() & new.keys()):
        before, after = old[identifier], new[identifier]
        for key in sorted(before.keys() | after.keys()):
            a, b = _plain(before.get(key)), _plain(after.get(key))
            if a != b:
                changes.append(Change(ChangeKind.MODIFIED, entity, identifier, key, a, b))
    return changes


def _order_changes(old, new) -> list[Change]:
    """A ``branch.order`` change when occurrences present in both revisions are reordered."""
    changes = []
    new_branches = {b.id: b for b in new.branches}
    for branch in old.branches:
        other = new_branches.get(branch.id)
        if other is None:
            continue
        kept = {o.id for o in other.occurrences} & {o.id for o in branch.occurrences}
        before = [o.id for o in branch.occurrences if o.id in kept]
        after = [o.id for o in other.occurrences if o.id in kept]
        if before != after:
            changes.append(Change(ChangeKind.MODIFIED, "branch", branch.id, "order", before, after))
    return changes


def _lattice_fields(lattice) -> dict[str, Any]:
    return {"name": lattice.name, "root_branch": lattice.root_branch}


def _element_fields(e: ElementDefinition) -> dict[str, Any]:
    # Every group is flattened; an absent group contributes its defaults, so
    # adding a group of defaults is not a change.
    fields: dict[str, Any] = {"name": e.name, "kind": e.kind, "length": e.length, "tags": e.tags}
    for group_name, model in GROUP_TYPES.items():
        group = getattr(e, group_name)
        for name, info in model.model_fields.items():
            if group is not None:
                value = getattr(group, name)
            else:
                value = None if info.is_required() else info.default
            fields[f"{group_name}.{name}"] = value
        if isinstance(group, TrackingP):
            for program, table in group.programs().items():
                for name, value in table.items():
                    fields[f"{group_name}.{program}.{name}"] = value
    return fields


def _occurrence_fields(branch, occurrence) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "name": occurrence.name,
        "branch": branch.id,
        "element": occurrence.element,
        "tags": occurrence.tags,
    }
    for override in occurrence.overrides:
        fields[f"overrides.{override.group}.{override.parameter}"] = override.value
    if occurrence.fork is not None:
        for name, value in occurrence.fork.model_dump().items():
            fields[f"fork.{name}"] = value
    return fields


def _component_fields(component) -> dict[str, Any]:
    fields: dict[str, Any] = {"name": component.name}
    for traversal in component.traversals:
        fields[f"traversals.{traversal.role}"] = traversal.occurrence
    return fields


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return list(value)
    return value
