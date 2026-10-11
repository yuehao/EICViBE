"""Mutable authoring API that produces immutable :class:`Lattice` revisions.

Typical use::

    b = LatticeBuilder("fodo")
    b.define("QF", "Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": 1.2})
    b.define("QD", "Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": -1.2})
    b.define("D", "Drift", length=2.0)
    b.add_branch("ring", "periodic", ["QF", "D", "QD", "D"])
    lattice = b.build()

To edit a published revision, start from ``LatticeBuilder.from_lattice``:
every existing id is kept, so annotations and results attached to unchanged
occurrences stay attached.

References: element definitions, branches and shared components are named by
id or name. Occurrences are named by id, or by ``(branch, name)``; methods
that already take a branch accept an occurrence id or name within it.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Self

from ._base import resolve_ref
from .elements import ElementDefinition, ElementKind
from .ids import BRANCH, COMPONENT, ELEMENT, OCCURRENCE, IdFactory, random_id
from .model import (
    FORMAT_VERSION,
    Branch,
    BranchKind,
    ForkDirection,
    ForkLink,
    ForkSide,
    Lattice,
    Occurrence,
    ParameterOverride,
    SharedComponent,
    Traversal,
)
from .parameters import ParameterGroup

type OccurrenceRef = str | tuple[str, str]
type GroupValues = ParameterGroup | Mapping[str, Any] | None

_FORK_ELEMENT_NAME = "fork"


@dataclass
class _BranchDraft:
    id: str
    name: str
    kind: BranchKind
    occurrences: list[Occurrence] = field(default_factory=list)
    names: dict[str, str] = field(default_factory=dict)  # occurrence name -> id
    next_suffix: dict[str, int] = field(default_factory=dict)

    def index(self, occurrence_id: str) -> int:
        for index, occurrence in enumerate(self.occurrences):
            if occurrence.id == occurrence_id:
                return index
        raise KeyError(occurrence_id)


@dataclass
class _ComponentDraft:
    id: str
    name: str
    traversals: dict[str, str] = field(default_factory=dict)  # role -> occurrence id


class LatticeBuilder:
    """Accumulates edits and validates them into a :class:`Lattice`.

    Local rules (a definition's groups, an occurrence's overrides) are
    checked immediately; cross references are checked again by
    :meth:`build`.
    """

    def __init__(self, name: str, *, ids: IdFactory | None = None) -> None:
        self.name = name
        self.root_branch: str | None = None
        self._new = ids or random_id
        self._taken: set[str] = set()
        self._elements: dict[str, ElementDefinition] = {}
        self._element_names: dict[str, str] = {}
        self._branches: dict[str, _BranchDraft] = {}
        self._branch_names: dict[str, str] = {}
        self._where: dict[str, str] = {}  # occurrence id -> branch id
        self._components: dict[str, _ComponentDraft] = {}
        self._component_names: dict[str, str] = {}

    @classmethod
    def from_lattice(cls, lattice: Lattice, *, ids: IdFactory | None = None) -> Self:
        """A builder holding ``lattice``, keeping all of its ids."""
        builder = cls(lattice.name, ids=ids)
        builder._include(lattice, prefix="", fresh_ids=False, adopt_root=True)
        return builder

    # -- element definitions -----------------------------------------------

    def define(
        self,
        name: str,
        kind: ElementKind | str,
        *,
        length: float = 0.0,
        tags: Iterable[str] = (),
        **groups: GroupValues,
    ) -> str:
        """Add an element definition; returns its id."""
        self._check_new_name(name, self._element_names, "element")
        definition = ElementDefinition.model_validate(
            {
                "id": self._claim(ELEMENT),
                "name": name,
                "kind": kind,
                "length": length,
                "tags": tuple(tags),
                **groups,
            }
        )
        self._elements[definition.id] = definition
        self._element_names[name] = definition.id
        return definition.id

    def update_element(
        self,
        ref: str,
        *,
        name: str | None = None,
        length: float | None = None,
        tags: Iterable[str] | None = None,
        **groups: GroupValues,
    ) -> None:
        """Change a definition, and with it every occurrence that places it.

        Group mappings update the named fields; a group instance replaces the
        group; ``None`` removes it.
        """
        old = self._elements[self._element_id(ref)]
        updated = old.with_groups(**groups)
        changes: dict[str, Any] = {}
        if name is not None and name != old.name:
            self._check_new_name(name, self._element_names, "element")
            changes["name"] = name
        if length is not None:
            changes["length"] = length
        if tags is not None:
            changes["tags"] = tuple(tags)
        if changes:
            updated = ElementDefinition.model_validate({**updated.model_dump(), **changes})
        for draft in self._branches.values():
            for occurrence in draft.occurrences:
                if occurrence.element == old.id:
                    _effective(updated, occurrence.overrides, f"{draft.name}/{occurrence.name}")
        self._elements[old.id] = updated
        if updated.name != old.name:
            del self._element_names[old.name]
            self._element_names[updated.name] = old.id

    def remove_element(self, ref: str) -> None:
        """Remove a definition that no occurrence places."""
        element_id = self._element_id(ref)
        placed = [
            f"{d.name}/{o.name}"
            for d in self._branches.values()
            for o in d.occurrences
            if o.element == element_id
        ]
        if placed:
            raise ValueError(f"element {ref!r} is still placed at {', '.join(placed[:5])}")
        del self._element_names[self._elements.pop(element_id).name]

    # -- branches ------------------------------------------------------------

    def add_branch(
        self, name: str, kind: BranchKind | str = BranchKind.LINAC, elements: Iterable[str] = ()
    ) -> str:
        """Add a branch, optionally placing ``elements`` in order; returns its id.

        The first branch added becomes the root branch.
        """
        self._check_new_name(name, self._branch_names, "branch")
        draft = _BranchDraft(self._claim(BRANCH), name, BranchKind(kind))
        self._branches[draft.id] = draft
        self._branch_names[name] = draft.id
        if self.root_branch is None:
            self.root_branch = draft.id
        self.extend(draft.id, elements)
        return draft.id

    def update_branch(
        self, ref: str, *, name: str | None = None, kind: BranchKind | str | None = None
    ) -> None:
        draft = self._branch(ref)
        if name is not None and name != draft.name:
            self._check_new_name(name, self._branch_names, "branch")
            del self._branch_names[draft.name]
            self._branch_names[name] = draft.id
            draft.name = name
        if kind is not None:
            draft.kind = BranchKind(kind)

    def set_root(self, ref: str | None) -> None:
        self.root_branch = None if ref is None else self._branch(ref).id

    def remove_branch(self, ref: str) -> None:
        """Remove a branch that is not the root and that nothing else refers to."""
        draft = self._branch(ref)
        if draft.id == self.root_branch:
            raise ValueError(f"branch {draft.name!r} is the root branch; set another root first")
        landing = [
            f"{d.name}/{o.name}"
            for d in self._branches.values()
            if d is not draft
            for o in d.occurrences
            if o.fork is not None and o.fork.to_branch == draft.id
        ]
        if landing:
            raise ValueError(f"branch {draft.name!r} is the target of forks {', '.join(landing)}")
        for occurrence in draft.occurrences:
            self._check_unshared(occurrence.id, f"{draft.name}/{occurrence.name}")
        for occurrence in draft.occurrences:
            del self._where[occurrence.id]
        del self._branches[draft.id]
        del self._branch_names[draft.name]

    # -- occurrences ---------------------------------------------------------

    def append(
        self,
        branch: str,
        element: str,
        *,
        name: str | None = None,
        tags: Iterable[str] = (),
        **overrides: Mapping[str, Any],
    ) -> str:
        """Place a definition at the end of a branch; returns the occurrence id.

        ``overrides`` are placement-specific values by group, e.g.
        ``BodyShiftP={"x_offset": 1e-4}``. The default name is
        ``<element>_<k>`` with the smallest free ``k`` in the branch.
        """
        draft = self._branch(branch)
        return self._place(draft, len(draft.occurrences), element, name, tags, overrides)

    def extend(self, branch: str, elements: Iterable[str]) -> list[str]:
        """Append several definitions; returns their occurrence ids."""
        draft = self._branch(branch)
        return [self._place(draft, len(draft.occurrences), e, None, (), {}) for e in elements]

    def insert(
        self,
        branch: str,
        element: str,
        *,
        before: str | None = None,
        after: str | None = None,
        name: str | None = None,
        tags: Iterable[str] = (),
        **overrides: Mapping[str, Any],
    ) -> str:
        """Place a definition before or after an occurrence of the branch."""
        draft = self._branch(branch)
        return self._place(draft, self._position(draft, before, after), element, name, tags, overrides)

    def update_occurrence(
        self,
        ref: OccurrenceRef,
        *,
        name: str | None = None,
        element: str | None = None,
        tags: Iterable[str] | None = None,
    ) -> None:
        """Rename, retag, or re-point an occurrence to another definition."""
        draft, index = self._locate(ref)
        old = draft.occurrences[index]
        changes: dict[str, Any] = {}
        if name is not None and name != old.name:
            self._check_new_name(name, draft.names, f"occurrence in branch {draft.name!r}")
            changes["name"] = name
        if element is not None:
            element_id = self._element_id(element)
            _effective(self._elements[element_id], old.overrides, f"{draft.name}/{old.name}")
            changes["element"] = element_id
        if tags is not None:
            changes["tags"] = tuple(tags)
        new = old.model_validate({**old.model_dump(), **changes})
        self._replace(draft, index, new)

    def set_overrides(self, ref: OccurrenceRef, **groups: Mapping[str, Any]) -> None:
        """Set placement-specific values, keeping other overrides."""
        draft, index = self._locate(ref)
        old = draft.occurrences[index]
        merged = old.override_map()
        for group, values in groups.items():
            merged.setdefault(group, {}).update(values)
        self._replace(draft, index, self._with_overrides(draft, old, merged))

    def clear_overrides(self, ref: OccurrenceRef, *targets: str) -> None:
        """Remove overrides: ``"BodyShiftP"`` clears a group, ``"BodyShiftP.x_offset"`` one value.

        With no targets, every override is removed.
        """
        draft, index = self._locate(ref)
        old = draft.occurrences[index]
        merged = old.override_map()
        if not targets:
            merged = {}
        for target in targets:
            group, _, parameter = target.partition(".")
            if parameter:
                merged.get(group, {}).pop(parameter, None)
            else:
                merged.pop(group, None)
        self._replace(draft, index, self._with_overrides(draft, old, merged))

    def remove(self, ref: OccurrenceRef) -> None:
        """Remove an occurrence that no fork lands on and no shared component uses."""
        draft, index = self._locate(ref)
        occurrence = draft.occurrences[index]
        label = f"{draft.name}/{occurrence.name}"
        landing = [
            f"{d.name}/{o.name}"
            for d in self._branches.values()
            for o in d.occurrences
            if o.fork is not None and o.fork.to_occurrence == occurrence.id
        ]
        if landing:
            raise ValueError(f"forks {', '.join(landing)} land on {label}")
        self._check_unshared(occurrence.id, label)
        del draft.occurrences[index]
        del draft.names[occurrence.name]
        draft.next_suffix.clear()  # the freed name may be the smallest free one
        del self._where[occurrence.id]

    # -- forks and shared components ---------------------------------------

    def add_fork(
        self,
        branch: str,
        to_branch: str,
        *,
        to: str | None = None,
        to_side: ForkSide | str = ForkSide.ENTRANCE,
        before: str | None = None,
        after: str | None = None,
        name: str | None = None,
        element: str | None = None,
        direction: ForkDirection | str = ForkDirection.FORWARD,
        propagate_reference: bool = False,
    ) -> str:
        """Place a fork in ``branch`` leading to ``to_branch``; returns its occurrence id.

        The beam lands at the ``to_side`` of occurrence ``to`` (a name or id in
        ``to_branch``), or at the branch entrance when ``to`` is None. The fork
        goes before/after the given occurrence, or at the end of the branch.
        """
        draft = self._branch(branch)
        target = self._branch(to_branch)
        link = ForkLink(
            to_branch=target.id,
            to_occurrence=None if to is None else target.occurrences[self._index_in(target, to)].id,
            to_side=ForkSide(to_side),
            direction=ForkDirection(direction),
            propagate_reference=propagate_reference,
        )
        position = self._position(draft, before, after)
        fork_element = self._element_id(element) if element is not None else self._fork_element()
        base = name or f"fork_{draft.name}_to_{target.name}"
        return self._place(
            draft, position, fork_element, self._free_name(draft, base, exact_first=True), (), {}, fork=link
        )

    def connect_injection(
        self,
        line: str,
        ring: str,
        *,
        at: str | None = None,
        via: tuple[str, str] | None = None,
        component: str | None = None,
        name: str | None = None,
    ) -> str:
        """Join the end of an injection line to a ring; returns the fork occurrence id.

        ``at``: the ring occurrence where injected beam lands (its entrance);
        the fork goes at the end of the line.

        ``via=(line_occurrence, ring_occurrence)``: the injection-side and
        circulating-side traversals of one physical component, e.g. a septum.
        They are registered as roles ``injection`` and ``circulation`` of the
        shared component ``component`` (default: the ring occurrence name).
        The fork goes right after the line traversal and lands at the exit
        of the ring traversal, so the hardware is passed exactly once.
        """
        line_draft, ring_draft = self._branch(line), self._branch(ring)
        if (at is None) == (via is None):
            raise ValueError("pass exactly one of at= or via=")
        if via is None:
            return self.add_fork(line_draft.id, ring_draft.id, to=at, name=name)
        line_occ = line_draft.occurrences[self._index_in(line_draft, via[0])]
        ring_occ = ring_draft.occurrences[self._index_in(ring_draft, via[1])]
        self.share(component or ring_occ.name, {"injection": line_occ.id, "circulation": ring_occ.id})
        return self.add_fork(
            line_draft.id, ring_draft.id, to=ring_occ.id, to_side=ForkSide.EXIT, after=line_occ.id, name=name
        )

    def connect_extraction(
        self,
        ring: str,
        line: str,
        *,
        at: str | None = None,
        via: tuple[str, str] | None = None,
        component: str | None = None,
        name: str | None = None,
    ) -> str:
        """Branch an extraction line off a ring; returns the fork occurrence id.

        ``at``: the ring occurrence after which the fork goes; the beam lands
        at the line entrance.

        ``via=(ring_occurrence, line_occurrence)``: circulating-side and
        extraction-side traversals of one physical component, registered as
        roles ``circulation`` and ``extraction``. The fork goes right before
        the ring traversal and lands at the entrance of the line traversal.
        """
        ring_draft, line_draft = self._branch(ring), self._branch(line)
        if (at is None) == (via is None):
            raise ValueError("pass exactly one of at= or via=")
        if via is None:
            return self.add_fork(ring_draft.id, line_draft.id, after=at, name=name)
        ring_occ = ring_draft.occurrences[self._index_in(ring_draft, via[0])]
        line_occ = line_draft.occurrences[self._index_in(line_draft, via[1])]
        self.share(component or ring_occ.name, {"circulation": ring_occ.id, "extraction": line_occ.id})
        return self.add_fork(ring_draft.id, line_draft.id, to=line_occ.id, before=ring_occ.id, name=name)

    def share(self, name: str, traversals: Mapping[str, OccurrenceRef]) -> str:
        """Declare occurrences to be traversals of one physical component.

        Adds roles to the component ``name``, creating it if needed; a role
        already bound to a different occurrence is an error. Returns its id.
        """
        resolved = {}
        for role, ref in traversals.items():
            draft, index = self._locate(ref)
            resolved[role] = draft.occurrences[index].id
        component_id = self._component_names.get(name)
        draft = self._components.get(component_id) if component_id else None
        for role, occurrence_id in resolved.items():
            if draft is not None and draft.traversals.get(role, occurrence_id) != occurrence_id:
                raise ValueError(
                    f"shared component {name!r} already binds role {role!r} to another occurrence"
                )
            owner = next(
                (c for c in self._components.values() if occurrence_id in c.traversals.values()), None
            )
            rebinding_same_role = (
                owner is draft and owner is not None and owner.traversals.get(role) == occurrence_id
            )
            if owner is not None and not rebinding_same_role:
                raise ValueError(f"occurrence {occurrence_id!r} is already a traversal of {owner.name!r}")
        if draft is None:
            draft = _ComponentDraft(self._claim(COMPONENT), name)
            self._components[draft.id] = draft
            self._component_names[name] = draft.id
        draft.traversals.update(resolved)
        return draft.id

    # -- composition -----------------------------------------------------------

    def include(self, other: Lattice, *, prefix: str = "", new_ids: bool = False) -> dict[str, str]:
        """Copy another lattice's definitions, branches and components into this one.

        ``prefix`` is prepended to copied definition, branch and component
        names. Ids are kept unless ``new_ids`` is set; a clash is an error.
        Definitions are never merged by name: two lattices each defining
        ``D1`` stay two families. Returns the mapping from old to new ids.
        """
        return self._include(other, prefix=prefix, fresh_ids=new_ids, adopt_root=self.root_branch is None)

    def build(self) -> Lattice:
        """Validate everything and return the immutable lattice."""
        return Lattice(
            format_version=FORMAT_VERSION,
            name=self.name,
            elements=tuple(self._elements.values()),
            branches=tuple(
                Branch(id=d.id, name=d.name, kind=d.kind, occurrences=tuple(d.occurrences))
                for d in self._branches.values()
            ),
            root_branch=self.root_branch,
            shared_components=tuple(
                SharedComponent(
                    id=c.id,
                    name=c.name,
                    traversals=tuple(Traversal(role=r, occurrence=o) for r, o in c.traversals.items()),
                )
                for c in self._components.values()
            ),
        )

    def __repr__(self) -> str:
        count = sum(len(d.occurrences) for d in self._branches.values())
        return (
            f"LatticeBuilder(name={self.name!r}, elements={len(self._elements)}, "
            f"branches={len(self._branches)}, occurrences={count})"
        )

    # -- internals -------------------------------------------------------------

    def _claim(self, prefix: str, candidate: str | None = None) -> str:
        if candidate is not None:
            if candidate in self._taken:
                raise ValueError(f"id {candidate!r} is already in use")
            self._taken.add(candidate)
            return candidate
        for _ in range(100):
            candidate = self._new(prefix)
            if candidate not in self._taken:
                self._taken.add(candidate)
                return candidate
        raise RuntimeError("id factory keeps returning ids that are already in use")

    @staticmethod
    def _check_new_name(name: str, existing: Mapping[str, str], what: str) -> None:
        if name in existing:
            raise ValueError(f"{what} name {name!r} is already in use")

    def _element_id(self, ref: str) -> str:
        return resolve_ref(ref, self._elements, self._element_names, "element")

    def _branch(self, ref: str) -> _BranchDraft:
        return self._branches[resolve_ref(ref, self._branches, self._branch_names, "branch")]

    def _index_in(self, draft: _BranchDraft, ref: str) -> int:
        by_id = {o.id: o for o in draft.occurrences}
        occurrence_id = resolve_ref(ref, by_id, draft.names, f"occurrence in branch {draft.name!r}")
        return draft.index(occurrence_id)

    def _locate(self, ref: OccurrenceRef) -> tuple[_BranchDraft, int]:
        if isinstance(ref, tuple):
            draft = self._branch(ref[0])
            return draft, self._index_in(draft, ref[1])
        if ref not in self._where:
            raise KeyError(f"no occurrence with id {ref!r}; use (branch, name) to refer by name")
        draft = self._branches[self._where[ref]]
        return draft, draft.index(ref)

    def _position(self, draft: _BranchDraft, before: str | None, after: str | None) -> int:
        if before is not None and after is not None:
            raise ValueError("pass at most one of before= and after=")
        if before is not None:
            return self._index_in(draft, before)
        if after is not None:
            return self._index_in(draft, after) + 1
        return len(draft.occurrences)

    def _free_name(self, draft: _BranchDraft, base: str, *, exact_first: bool = False) -> str:
        if exact_first and base not in draft.names:
            return base
        k = draft.next_suffix.get(base, 1)
        while f"{base}_{k}" in draft.names:
            k += 1
        draft.next_suffix[base] = k + 1
        return f"{base}_{k}"

    def _place(
        self,
        draft: _BranchDraft,
        position: int,
        element: str,
        name: str | None,
        tags: Iterable[str],
        overrides: Mapping[str, Mapping[str, Any]],
        *,
        fork: ForkLink | None = None,
    ) -> str:
        definition = self._elements[self._element_id(element)]
        if name is None:
            name = self._free_name(draft, definition.name)
        else:
            self._check_new_name(name, draft.names, f"occurrence in branch {draft.name!r}")
        flat = _flatten(overrides)
        _effective(definition, flat, f"{draft.name}/{name}")
        occurrence = Occurrence(
            id=self._claim(OCCURRENCE),
            name=name,
            element=definition.id,
            tags=tuple(tags),
            overrides=flat,
            fork=fork,
        )
        draft.occurrences.insert(position, occurrence)
        draft.names[name] = occurrence.id
        self._where[occurrence.id] = draft.id
        return occurrence.id

    def _replace(self, draft: _BranchDraft, index: int, new: Occurrence) -> None:
        old = draft.occurrences[index]
        draft.occurrences[index] = new
        if new.name != old.name:
            del draft.names[old.name]
            draft.names[new.name] = new.id
            draft.next_suffix.clear()

    def _with_overrides(
        self, draft: _BranchDraft, old: Occurrence, grouped: Mapping[str, Mapping[str, Any]]
    ) -> Occurrence:
        flat = _flatten(grouped)
        _effective(self._elements[old.element], flat, f"{draft.name}/{old.name}")
        return old.model_validate({**old.model_dump(), "overrides": flat})

    def _check_unshared(self, occurrence_id: str, label: str) -> None:
        for component in self._components.values():
            if occurrence_id in component.traversals.values():
                raise ValueError(f"{label} is a traversal of shared component {component.name!r}")

    def _fork_element(self) -> str:
        existing = self._element_names.get(_FORK_ELEMENT_NAME)
        if existing is None:
            return self.define(_FORK_ELEMENT_NAME, ElementKind.FORK)
        if self._elements[existing].kind is not ElementKind.FORK:
            raise ValueError(f"element {_FORK_ELEMENT_NAME!r} is not a Fork; pass element= to add_fork")
        return existing

    def _include(self, other: Lattice, *, prefix: str, fresh_ids: bool, adopt_root: bool) -> dict[str, str]:
        clashes = [
            f"{what} {prefix + item.name!r}"
            for what, items, names in (
                ("element", other.elements, self._element_names),
                ("branch", other.branches, self._branch_names),
                ("shared component", other.shared_components, self._component_names),
            )
            for item in items
            if prefix + item.name in names
        ]
        if clashes:
            raise ValueError(f"name clash: {', '.join(clashes[:10])}; pass prefix= to rename")

        ids = [e.id for e in other.elements] + [b.id for b in other.branches]
        ids += [o.id for b in other.branches for o in b.occurrences] + [c.id for c in other.shared_components]
        if not fresh_ids and (clashing := [i for i in ids if i in self._taken]):
            raise ValueError(f"id clash: {', '.join(clashing[:10])}; pass new_ids=True")

        mapping: dict[str, str] = {}

        def claim(old: str, prefix_: str) -> str:
            mapping[old] = self._claim(prefix_) if fresh_ids else self._claim(prefix_, old)
            return mapping[old]

        for e in other.elements:
            claim(e.id, ELEMENT)
        for b in other.branches:
            claim(b.id, BRANCH)
            for o in b.occurrences:
                claim(o.id, OCCURRENCE)
        for c in other.shared_components:
            claim(c.id, COMPONENT)

        for e in other.elements:
            copy = ElementDefinition.model_validate(
                {**e.model_dump(), "id": mapping[e.id], "name": prefix + e.name}
            )
            self._elements[copy.id] = copy
            self._element_names[copy.name] = copy.id
        for b in other.branches:
            draft = _BranchDraft(mapping[b.id], prefix + b.name, b.kind)
            for o in b.occurrences:
                data = {**o.model_dump(), "id": mapping[o.id], "element": mapping[o.element]}
                if o.fork is not None:
                    data["fork"] = {
                        **o.fork.model_dump(),
                        "to_branch": mapping[o.fork.to_branch],
                        "to_occurrence": mapping.get(o.fork.to_occurrence) if o.fork.to_occurrence else None,
                    }
                occurrence = Occurrence.model_validate(data)
                draft.occurrences.append(occurrence)
                draft.names[occurrence.name] = occurrence.id
                self._where[occurrence.id] = draft.id
            self._branches[draft.id] = draft
            self._branch_names[draft.name] = draft.id
        for c in other.shared_components:
            draft = _ComponentDraft(
                mapping[c.id], prefix + c.name, {t.role: mapping[t.occurrence] for t in c.traversals}
            )
            self._components[draft.id] = draft
            self._component_names[draft.name] = draft.id
        if adopt_root and other.root_branch is not None:
            self.root_branch = mapping[other.root_branch]
        return mapping


def _flatten(grouped: Mapping[str, Mapping[str, Any]]) -> tuple[ParameterOverride, ...]:
    return tuple(
        ParameterOverride(group=group, parameter=parameter, value=value)
        for group, values in grouped.items()
        for parameter, value in values.items()
    )


def _effective(
    definition: ElementDefinition, overrides: tuple[ParameterOverride, ...], label: str
) -> ElementDefinition:
    try:
        return definition.with_overrides(overrides)
    except (ValueError, KeyError) as error:
        raise ValueError(f"{label}: overrides do not apply to {definition.name!r}: {error}") from error


def design_values(
    lattice: Lattice, values: Mapping[tuple[str, str], Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    """``values`` by (occurrence id, ``Group.name``) as values of the definitions they place:
    definition id -> group -> name -> value.

    A definition placed more than once is refused, as its value would change every occurrence
    of it (a setting can change one alone), and so is a parameter its occurrence overrides.
    """
    uses: dict[str, int] = {}
    for branch in lattice.branches:
        for occurrence in branch.occurrences:
            uses[occurrence.element] = uses.get(occurrence.element, 0) + 1
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for (occurrence, parameter), value in values.items():
        placed = lattice.placement(occurrence)
        definition = placed.definition
        where = f"{lattice.branch(placed.branch_id).name}/{placed.name}"
        group, name = parameter.split(".", 1)
        if any((o.group, o.parameter) == (group, name) for o in placed.occurrence.overrides):
            raise ValueError(f"{where}: the occurrence overrides {parameter}; edit it there")
        if uses[definition.id] > 1:
            raise ValueError(
                f"{where}: its definition {definition.name} is placed {uses[definition.id]} times; "
                "setting it would change them all"
            )
        out.setdefault(definition.id, {}).setdefault(group, {})[name] = value
    return out


def with_design(lattice: Lattice, values: Mapping[tuple[str, str], Any]) -> Lattice:
    """``lattice`` with ``values`` (occurrence id, ``Group.name``) as its design: in the definitions
    their occurrences place (:func:`design_values`), every id kept."""
    builder = LatticeBuilder.from_lattice(lattice)
    for definition, groups in design_values(lattice, values).items():
        builder.update_element(definition, **groups)
    return builder.build()
