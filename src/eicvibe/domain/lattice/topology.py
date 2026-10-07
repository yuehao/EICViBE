"""Derived, read-only index of a validated lattice.

:func:`index_lattice` checks every cross reference in a lattice and, if they
all resolve, returns a :class:`Topology`: s positions, effective elements,
the fork graph and name lookups. A ``Lattice`` builds it once during
validation; nothing here is part of the serialized description.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from pydantic import ValidationError

from .elements import CONTROL_KINDS, ElementDefinition, ElementKind

if TYPE_CHECKING:
    from .model import Branch, ForkDirection, ForkSide, Lattice, Occurrence, SharedComponent

#: Relative and absolute (m) tolerance for lengths of traversals of one component.
SHARED_LENGTH_REL_TOL = 1e-2
SHARED_LENGTH_ABS_TOL = 1e-6


@dataclass(frozen=True, slots=True)
class PlacedElement:
    """An occurrence at its position in one revision.

    ``element`` is the effective element (definition plus the occurrence's
    overrides); ``definition`` is the shared family definition. ``index`` and
    the s values change when the branch is edited; ``occurrence.id`` does not.
    """

    occurrence: Occurrence
    branch_id: str
    index: int
    s_enter: float
    s_exit: float
    definition: ElementDefinition
    element: ElementDefinition

    @property
    def id(self) -> str:
        return self.occurrence.id

    @property
    def name(self) -> str:
        return self.occurrence.name

    @property
    def kind(self) -> ElementKind:
        return self.element.kind

    @property
    def length(self) -> float:
        return self.element.length

    @property
    def tags(self) -> frozenset[str]:
        """Tags of the definition and of the occurrence."""
        return frozenset(self.definition.tags) | frozenset(self.occurrence.tags)


@dataclass(frozen=True, slots=True)
class BranchLayout:
    """A branch with every occurrence placed along s."""

    branch: Branch
    placements: tuple[PlacedElement, ...]

    @property
    def id(self) -> str:
        return self.branch.id

    @property
    def periodic(self) -> bool:
        return self.branch.periodic

    @property
    def length(self) -> float:
        return self.placements[-1].s_exit if self.placements else 0.0


@dataclass(frozen=True, slots=True)
class ForkEdge:
    """A directed connection from a fork occurrence to a landing point.

    ``id`` is the fork occurrence id. The beam leaves ``from_branch`` at
    ``s_from`` and continues in ``to_branch`` at ``s_to``: the ``to_side`` of
    ``to_occurrence``, or the branch entrance when ``to_occurrence`` is None.
    """

    id: str
    from_branch: str
    from_index: int
    s_from: float
    to_branch: str
    to_occurrence: str | None
    to_side: ForkSide
    s_to: float
    direction: ForkDirection
    propagate_reference: bool


@dataclass(frozen=True, slots=True, eq=False)
class Topology:
    """Lookups and derived geometry of one lattice revision. Keys are ids."""

    elements: Mapping[str, ElementDefinition]
    element_names: Mapping[str, str]
    layouts: Mapping[str, BranchLayout]
    branch_names: Mapping[str, str]
    placements: Mapping[str, PlacedElement]
    forks: Mapping[str, ForkEdge]
    outgoing: Mapping[str, tuple[ForkEdge, ...]]
    incoming: Mapping[str, tuple[ForkEdge, ...]]
    components: Mapping[str, SharedComponent]
    component_names: Mapping[str, str]
    component_of: Mapping[str, str]


class _Problems(list[str]):
    def add(self, code: str, message: str) -> None:
        self.append(f"[{code}] {message}")


def index_lattice(lattice: Lattice) -> Topology:
    """Validate the cross references of ``lattice`` and index it.

    Raises ``ValueError`` listing every problem found, not only the first.
    """
    problems = _Problems()

    seen_ids: dict[str, str] = {}

    def claim(identifier: str, what: str) -> None:
        if identifier in seen_ids:
            problems.add("duplicate-id", f"id {identifier!r} is used by {seen_ids[identifier]} and {what}")
        else:
            seen_ids[identifier] = what

    def names_of(items, what: str) -> dict[str, str]:
        by_name: dict[str, str] = {}
        for item in items:
            if item.name in by_name:
                problems.add("duplicate-name", f"{what} name {item.name!r} is not unique")
            by_name[item.name] = item.id
        return by_name

    elements = {e.id: e for e in lattice.elements}
    for e in lattice.elements:
        claim(e.id, f"element {e.name!r}")
    element_names = names_of(lattice.elements, "element")

    branches = {b.id: b for b in lattice.branches}
    for b in lattice.branches:
        claim(b.id, f"branch {b.name!r}")
    branch_names = names_of(lattice.branches, "branch")

    if lattice.root_branch is not None and lattice.root_branch not in branches:
        problems.add("unknown-branch", f"root branch {lattice.root_branch!r} does not exist")

    # Place every occurrence and resolve its effective element.
    layouts: dict[str, BranchLayout] = {}
    placements: dict[str, PlacedElement] = {}
    for branch in lattice.branches:
        names_of(branch.occurrences, f"occurrence in branch {branch.name!r} with")
        placed: list[PlacedElement] = []
        s = 0.0
        for index, occurrence in enumerate(branch.occurrences):
            where = f"occurrence {branch.name}/{occurrence.name}"
            claim(occurrence.id, where)
            definition = elements.get(occurrence.element)
            if definition is None:
                problems.add("unknown-element", f"{where} places unknown element {occurrence.element!r}")
                continue
            try:
                element = definition.with_overrides(occurrence.overrides)
            except (ValidationError, KeyError) as error:
                problems.add("invalid-override", f"{where}: {_first_line(error)}")
                continue
            if element.kind is ElementKind.BEGINNING_ELE and index != 0:
                problems.add("beginning-position", f"{where}: BeginningEle must be the first occurrence")
            if (element.kind is ElementKind.FORK) != (occurrence.fork is not None):
                problems.add(
                    "fork-link",
                    f"{where}: Fork occurrences need a fork link and other kinds must not have one",
                )
            item = PlacedElement(occurrence, branch.id, index, s, s + element.length, definition, element)
            placed.append(item)
            placements[occurrence.id] = item
            s = item.s_exit
        layouts[branch.id] = BranchLayout(branch, tuple(placed))

    forks = _index_forks(lattice, layouts, placements, problems)

    components = {c.id: c for c in lattice.shared_components}
    for c in lattice.shared_components:
        claim(c.id, f"shared component {c.name!r}")
    component_names = names_of(lattice.shared_components, "shared component")
    component_of = _index_components(lattice, placements, problems)

    if problems:
        raise ValueError("lattice is inconsistent:\n  " + "\n  ".join(problems))

    outgoing: dict[str, list[ForkEdge]] = {b: [] for b in branches}
    incoming: dict[str, list[ForkEdge]] = {b: [] for b in branches}
    for edge in forks.values():
        outgoing[edge.from_branch].append(edge)
        incoming[edge.to_branch].append(edge)

    return Topology(
        elements=MappingProxyType(elements),
        element_names=MappingProxyType(element_names),
        layouts=MappingProxyType(layouts),
        branch_names=MappingProxyType(branch_names),
        placements=MappingProxyType(placements),
        forks=MappingProxyType(forks),
        outgoing=MappingProxyType({b: tuple(e) for b, e in outgoing.items()}),
        incoming=MappingProxyType({b: tuple(e) for b, e in incoming.items()}),
        components=MappingProxyType(components),
        component_names=MappingProxyType(component_names),
        component_of=MappingProxyType(component_of),
    )


def _index_forks(
    lattice: Lattice,
    layouts: dict[str, BranchLayout],
    placements: dict[str, PlacedElement],
    problems: _Problems,
) -> dict[str, ForkEdge]:
    from .model import ForkSide

    forks: dict[str, ForkEdge] = {}
    for layout in layouts.values():
        for source in layout.placements:
            link = source.occurrence.fork
            if link is None:
                continue
            where = f"fork {layout.branch.name}/{source.name}"
            target_layout = layouts.get(link.to_branch)
            if target_layout is None:
                problems.add("fork-target", f"{where} targets unknown branch {link.to_branch!r}")
                continue
            if link.to_occurrence is None:
                if not target_layout.placements:
                    problems.add("fork-target", f"{where} targets empty branch {target_layout.branch.name!r}")
                    continue
                s_to = 0.0
            else:
                target = placements.get(link.to_occurrence)
                if target is None or target.branch_id != link.to_branch:
                    problems.add(
                        "fork-target",
                        f"{where} lands on {link.to_occurrence!r}, which is not an occurrence "
                        f"of branch {target_layout.branch.name!r}",
                    )
                    continue
                if target.kind is ElementKind.FORK:
                    problems.add("fork-target", f"{where} lands on another fork, {target.name!r}")
                    continue
                s_to = target.s_enter if link.to_side is ForkSide.ENTRANCE else target.s_exit
            forks[source.id] = ForkEdge(
                id=source.id,
                from_branch=layout.id,
                from_index=source.index,
                s_from=source.s_enter,
                to_branch=link.to_branch,
                to_occurrence=link.to_occurrence,
                to_side=link.to_side,
                s_to=s_to,
                direction=link.direction,
                propagate_reference=link.propagate_reference,
            )
    return forks


def _index_components(
    lattice: Lattice, placements: dict[str, PlacedElement], problems: _Problems
) -> dict[str, str]:
    component_of: dict[str, str] = {}
    for component in lattice.shared_components:
        where = f"shared component {component.name!r}"
        lengths: list[tuple[str, float]] = []
        for traversal in component.traversals:
            placement = placements.get(traversal.occurrence)
            if placement is None:
                problems.add(
                    "component-traversal",
                    f"{where} role {traversal.role!r} uses unknown occurrence {traversal.occurrence!r}",
                )
                continue
            if traversal.occurrence in component_of:
                problems.add(
                    "component-traversal",
                    f"{where}: occurrence {placement.name!r} already belongs to another component",
                )
            component_of[traversal.occurrence] = component.id
            if placement.kind in CONTROL_KINDS:
                problems.add(
                    "component-traversal",
                    f"{where} role {traversal.role!r}: a {placement.kind.value} is not physical hardware",
                )
            lengths.append((traversal.role, placement.length))
        for role, length in lengths[1:]:
            reference_role, reference = lengths[0]
            if not math.isclose(
                length, reference, rel_tol=SHARED_LENGTH_REL_TOL, abs_tol=SHARED_LENGTH_ABS_TOL
            ):
                problems.add(
                    "component-length",
                    f"{where}: role {role!r} is {length} m but {reference_role!r} is {reference} m",
                )
    return component_of


def _first_line(error: Exception) -> str:
    if isinstance(error, ValidationError):
        detail = error.errors()[0]
        location = ".".join(str(part) for part in detail["loc"])
        return f"{location}: {detail['msg']}" if location else detail["msg"]
    return str(error).strip("'\"")
