"""The immutable lattice description.

A :class:`Lattice` is the machine content of one revision: element
definitions, branches of placed occurrences, fork links between branches and
shared physical components. It holds no routes, cycles, beams or engine
settings; those belong to a scenario.

Identity and position are separate. Every definition, branch, occurrence and
component has an opaque ``id`` that survives edits and renames. Names are
unique labels for lookup. Indices and s positions are derived per revision
(see :mod:`~eicvibe.domain.lattice.topology`).

A ``Lattice`` validates completely when constructed. Edit one with
:class:`~eicvibe.domain.lattice.builder.LatticeBuilder`.
"""

import itertools
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from typing import Any, Literal, Self

from pydantic import Field, ValidationError, field_validator, model_validator

from ._base import DomainModel, Id, Label, Real, Tags, resolve_ref
from .canonical import canonical_json, content_digest
from .elements import ElementDefinition, ElementKind
from .parameters import ParticleP, ReferenceP, TwissP, parameter_adapter
from .topology import BranchLayout, ForkEdge, PlacedElement, Topology, index_lattice

FORMAT_VERSION = 1


class BranchKind(StrEnum):
    """Boundary condition of a branch's optics."""

    LINAC = "linac"
    """Open line: optics propagate from initial values at the entrance."""
    PERIODIC = "periodic"
    """Closed ring: optics are the periodic solution of one turn."""


class ForkSide(StrEnum):
    ENTRANCE = "entrance"
    EXIT = "exit"


class ForkDirection(StrEnum):
    FORWARD = "forward"
    REVERSE = "reverse"


class ParameterOverride(DomainModel):
    """One placement-specific parameter value, e.g. a trim or misalignment.

    ``value`` is checked against the field's type and bounds and stored in
    normalized form (``1`` becomes ``1.0`` for a real-valued field).
    """

    group: str
    parameter: str
    value: Real | int | bool | str | None

    @model_validator(mode="before")
    @classmethod
    def _normalize_value(cls, data: Any) -> Any:
        if not isinstance(data, Mapping) or "value" not in data:
            return data
        group, parameter = data.get("group"), data.get("parameter")
        if not isinstance(group, str) or not isinstance(parameter, str):
            return data
        try:
            value = parameter_adapter(group, parameter).validate_python(data["value"])
        except KeyError as error:
            raise ValueError(error.args[0]) from None
        except ValidationError as error:
            raise ValueError(f"{group}.{parameter}: {error.errors()[0]['msg']}") from None
        return {**data, "value": value}


class ForkLink(DomainModel):
    """Where the beam continues after a fork occurrence."""

    to_branch: Id
    to_occurrence: Id | None = Field(
        None, description="Landing occurrence; None lands at the branch entrance"
    )
    to_side: ForkSide = Field(
        ForkSide.ENTRANCE, description="Land before (entrance) or after (exit) to_occurrence"
    )
    direction: ForkDirection = ForkDirection.FORWARD
    propagate_reference: bool = Field(
        False, description="Carry the reference particle into the destination branch"
    )

    @model_validator(mode="after")
    def _side_needs_occurrence(self) -> Self:
        if self.to_occurrence is None and self.to_side is not ForkSide.ENTRANCE:
            raise ValueError("a fork landing at a branch entrance cannot use to_side='exit'")
        return self


class Occurrence(DomainModel):
    """One placement of an element definition in a branch.

    An occurrence is one physical placement (one magnet), unless a
    :class:`SharedComponent` declares it a traversal of hardware shared with
    other branches.
    """

    id: Id
    name: Label
    element: Id = Field(description="Id of the placed element definition")
    tags: Tags = Field((), description="Labels of this placement, e.g. 'sector3'")
    overrides: tuple[ParameterOverride, ...] = ()
    fork: ForkLink | None = Field(None, description="Required on, and only on, Fork occurrences")

    @field_validator("overrides")
    @classmethod
    def _canonical_overrides(cls, overrides: tuple[ParameterOverride, ...]) -> tuple[ParameterOverride, ...]:
        ordered = tuple(sorted(overrides, key=lambda o: (o.group, o.parameter)))
        for first, second in itertools.pairwise(ordered):
            if (first.group, first.parameter) == (second.group, second.parameter):
                raise ValueError(f"{first.group}.{first.parameter} is overridden twice")
        return ordered

    def override_map(self) -> dict[str, dict[str, Any]]:
        """Overrides grouped as ``{group: {parameter: value}}``."""
        grouped: dict[str, dict[str, Any]] = {}
        for override in self.overrides:
            grouped.setdefault(override.group, {})[override.parameter] = override.value
        return grouped


class Branch(DomainModel):
    """An ordered sequence of occurrences with its own s coordinate."""

    id: Id
    name: Label
    kind: BranchKind = BranchKind.LINAC
    occurrences: tuple[Occurrence, ...] = ()

    @property
    def periodic(self) -> bool:
        return self.kind is BranchKind.PERIODIC


class Traversal(DomainModel):
    """One optical representation of a shared component, e.g. its injection side."""

    role: Label
    occurrence: Id


class SharedComponent(DomainModel):
    """Physical hardware traversed by occurrences in several branches.

    Example: one septum with ``injection``, ``circulation`` and ``extraction``
    traversals, each with its own local optics. The hardware is counted once;
    each traversal keeps its own element and geometry.
    """

    id: Id
    name: Label
    traversals: tuple[Traversal, ...] = Field(min_length=2)

    @field_validator("traversals")
    @classmethod
    def _unique_traversals(cls, traversals: tuple[Traversal, ...]) -> tuple[Traversal, ...]:
        roles = [t.role for t in traversals]
        occurrences = [t.occurrence for t in traversals]
        if len(set(roles)) != len(roles):
            raise ValueError("traversal roles must be unique")
        if len(set(occurrences)) != len(occurrences):
            raise ValueError("an occurrence can be only one traversal of a component")
        return tuple(sorted(traversals, key=lambda t: t.role))

    def traversal(self, role: str) -> Traversal:
        for t in self.traversals:
            if t.role == role:
                return t
        raise KeyError(f"shared component {self.name!r} has no role {role!r}")


@dataclass(frozen=True, slots=True)
class InitialConditions:
    """Entrance conditions carried by a branch's ``BeginningEle``."""

    reference: ReferenceP | None
    twiss: TwissP | None
    particle: ParticleP | None


class Lattice(DomainModel):
    """Immutable machine description of one revision.

    Definitions, branches and components are kept sorted by name so that
    equal content compares and serializes identically. Occurrences keep
    their physical order.
    """

    format_version: Literal[1]
    name: Label
    elements: tuple[ElementDefinition, ...] = ()
    branches: tuple[Branch, ...] = ()
    root_branch: Id | None = Field(None, description="Default branch for tools that need one")
    shared_components: tuple[SharedComponent, ...] = ()

    @field_validator("elements", "branches", "shared_components")
    @classmethod
    def _sorted_by_name(cls, items: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(sorted(items, key=lambda item: item.name))

    @model_validator(mode="after")
    def _index(self) -> Self:
        # Seed the cached_property below; the index is built once per revision.
        vars(self)["topology"] = index_lattice(self)
        return self

    @cached_property
    def topology(self) -> Topology:
        """Derived index: placements, s positions, forks and lookups."""
        return index_lattice(self)

    @cached_property
    def digest(self) -> str:
        """``sha256:<hex>`` of the canonical serialization; the revision content hash."""
        return content_digest(self.to_dict())

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Canonical JSON-compatible form. Defaults and empty groups are omitted."""
        return self.model_dump(mode="json", exclude_defaults=True)

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return canonical_json(self.to_dict())
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, text: str | bytes) -> Self:
        return cls.model_validate_json(text)

    # -- lookups -------------------------------------------------------------

    def element(self, ref: str) -> ElementDefinition:
        """Element definition by id or name."""
        t = self.topology
        return t.elements[resolve_ref(ref, t.elements, t.element_names, "element")]

    def branch(self, ref: str) -> Branch:
        """Branch by id or name."""
        return self.layout(ref).branch

    def layout(self, ref: str) -> BranchLayout:
        """Placed occurrences of a branch, by branch id or name."""
        t = self.topology
        return t.layouts[resolve_ref(ref, t.layouts, t.branch_names, "branch")]

    def placement(self, ref: str, branch: str | None = None) -> PlacedElement:
        """An occurrence with its position and effective element.

        ``ref`` is an occurrence id, or an occurrence name within ``branch``.
        """
        placements = self.topology.placements
        if branch is None:
            if ref not in placements:
                raise KeyError(f"no occurrence with id {ref!r}; pass branch= to look up by name")
            return placements[ref]
        layout = self.layout(branch)
        by_id = {p.id: p for p in layout.placements}
        by_name = {p.name: p.id for p in layout.placements}
        return by_id[resolve_ref(ref, by_id, by_name, f"occurrence in branch {layout.branch.name!r}")]

    def shared_component(self, ref: str) -> SharedComponent:
        t = self.topology
        return t.components[resolve_ref(ref, t.components, t.component_names, "shared component")]

    def component_of(self, occurrence_id: str) -> SharedComponent | None:
        """The shared component an occurrence is a traversal of, if any."""
        component_id = self.topology.component_of.get(occurrence_id)
        return None if component_id is None else self.topology.components[component_id]

    @property
    def forks(self) -> tuple[ForkEdge, ...]:
        return tuple(self.topology.forks.values())

    def forks_from(self, branch: str) -> tuple[ForkEdge, ...]:
        return self.topology.outgoing[self.layout(branch).id]

    def forks_into(self, branch: str) -> tuple[ForkEdge, ...]:
        return self.topology.incoming[self.layout(branch).id]

    def initial_conditions(self, branch: str) -> InitialConditions | None:
        """Conditions on the branch's leading ``BeginningEle``, or None without one."""
        placements = self.layout(branch).placements
        if not placements or placements[0].kind is not ElementKind.BEGINNING_ELE:
            return None
        element = placements[0].element
        return InitialConditions(element.ReferenceP, element.TwissP, element.ParticleP)

    def select(
        self,
        branch: str,
        *,
        kind: ElementKind | str | Iterable[ElementKind | str] | None = None,
        element: str | None = None,
        tag: str | None = None,
        name: str | None = None,
        s_range: tuple[float, float] | None = None,
    ) -> tuple[PlacedElement, ...]:
        """Placements in a branch matching every given criterion.

        ``element`` selects a family by definition id or name; ``tag`` matches
        definition or occurrence tags. ``name`` is a regular expression
        searched in occurrence names. ``s_range`` keeps placements whose
        [s_enter, s_exit] intersects the closed range.
        """
        selected = self.layout(branch).placements
        if kind is not None:
            kinds = {ElementKind(kind)} if isinstance(kind, str) else {ElementKind(k) for k in kind}
            selected = tuple(p for p in selected if p.kind in kinds)
        if element is not None:
            definition_id = self.element(element).id
            selected = tuple(p for p in selected if p.definition.id == definition_id)
        if tag is not None:
            selected = tuple(p for p in selected if tag in p.tags)
        if name is not None:
            pattern = re.compile(name)
            selected = tuple(p for p in selected if pattern.search(p.name))
        if s_range is not None:
            start, end = s_range
            if start > end:
                raise ValueError("s_range start must not exceed its end")
            selected = tuple(p for p in selected if p.s_exit >= start and p.s_enter <= end)
        return selected

    def diagnose(self, limits=None):
        """Plausibility and limit warnings; see :func:`~eicvibe.domain.lattice.diagnostics.diagnose`."""
        from .diagnostics import diagnose

        return diagnose(self, limits)

    def __repr__(self) -> str:
        occurrences = sum(len(b.occurrences) for b in self.branches)
        return (
            f"Lattice(name={self.name!r}, elements={len(self.elements)}, "
            f"branches={len(self.branches)}, occurrences={occurrences}, digest={self.digest[:19]}…)"
        )

    __str__ = __repr__
