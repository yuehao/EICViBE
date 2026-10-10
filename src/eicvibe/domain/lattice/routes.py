"""Routes: the beam's paths through several branches, joined at forks.

A route is a sequence of segments, each a stretch of one branch from an
occurrence's entrance to an occurrence's exit. Consecutive segments join at a
fork (the first ends at the fork, the next starts where the fork lands) or
continue round a ring (the first ends at the ring's end, the next starts at
its start). People name a route by its branches in beam order (the machine
folder's ``routes.yaml``, ADR 0005); :func:`route_through` works out the
segments from the forks:

- a route enters a branch where the previous branch's fork lands (an exit
  landing starts at the occurrence after the landing one) and leaves it at
  the first fork to the next branch at or after that point;
- in a ring, a fork behind the entry point is reached by going round: the
  route runs to the ring's end and on from its start;
- a route that ends in a ring it entered part-way goes round once, back to
  where it came in; a route that ends in a line runs to its end.

The EIN's BAR route, for example, is ``ltr, itl, bar, etl, ltr, rcs``: the LTR
to its fork to the ITL; the ITL; the BAR from the septum round to its fork to
the ETL; the ETL; the LTR again, from where the ETL lands to its fork to the
RCS; and once round the RCS.
"""

import math
from collections.abc import Sequence

from pydantic import Field

from ._base import DomainModel
from .model import ForkDirection, ForkSide, Lattice, PlacedElement


class Segment(DomainModel):
    """Part of one branch, from ``start``'s entrance to ``end``'s exit (occurrence ids)."""

    branch: str
    start: str
    end: str


class Route(DomainModel):
    name: str = Field(min_length=1)
    branches: tuple[str, ...] = Field(description="The branch names it was given, in beam order")
    segments: tuple[Segment, ...]


def route_through(lattice: Lattice, name: str, branches: Sequence[str]) -> Route:
    """The route through ``branches`` (names or ids, in beam order), joined at their forks."""
    if not branches:
        raise ValueError(f"route {name}: no branches")
    segments: list[Segment] = []
    entry = 0
    for k, branch in enumerate(branches):
        layout = lattice.layout(branch)
        ids = [p.id for p in layout.placements]

        def part(a: int, b: int, layout=layout, ids=ids) -> None:
            segments.append(Segment(branch=layout.id, start=ids[a], end=ids[b]))

        if k == len(branches) - 1:
            part(entry, len(ids) - 1)
            if layout.periodic and entry > 0:
                part(0, entry - 1)  # once round, back to where the route came in
            break
        target = lattice.layout(branches[k + 1]).id
        forks = sorted(
            (ids.index(e.id), e)
            for e in lattice.forks_from(layout.id)
            if e.to_branch == target and e.direction is ForkDirection.FORWARD
        )
        if not forks:
            raise ValueError(f"route {name}: no fork from {layout.branch.name} to {branches[k + 1]}")
        ahead = [f for f in forks if f[0] >= entry]
        if ahead:
            at, edge = ahead[0]
            part(entry, at)
        elif layout.periodic:
            at, edge = forks[0]
            part(entry, len(ids) - 1)
            part(0, at)
        else:
            raise ValueError(
                f"route {name}: {layout.branch.name}'s fork to {branches[k + 1]} lies behind the entry"
            )
        landing = lattice.layout(target)
        landed = [p.id for p in landing.placements]
        if edge.to_occurrence is None:
            entry = 0
        else:
            entry = landed.index(edge.to_occurrence) + (1 if edge.to_side is ForkSide.EXIT else 0)
            if entry >= len(landed):
                raise ValueError(f"route {name}: {edge.id} lands at the end of {landing.branch.name}")
    route = Route(name=name, branches=tuple(branches), segments=tuple(segments))
    seen: set[str] = set()
    for placed in route_placements(lattice, route):
        if placed[1].id in seen:
            raise ValueError(f"route {name} passes {placed[1].name} twice; a route passes each element once")
        seen.add(placed[1].id)
    return route


def route_placements(lattice: Lattice, route: Route) -> list[tuple[int, PlacedElement, float]]:
    """Every occurrence of a route in beam order: ``(segment, placement, route s at its entrance)``."""
    out = []
    s = 0.0
    for k, segment in enumerate(route.segments):
        placements = lattice.layout(segment.branch).placements
        ids = [p.id for p in placements]
        start = placements[ids.index(segment.start)].s_enter
        for placed in placements[ids.index(segment.start) : ids.index(segment.end) + 1]:
            out.append((k, placed, s + placed.s_enter - start))
        s += placements[ids.index(segment.end)].s_exit - start
    return out


def route_length(lattice: Lattice, route: Route) -> float:
    total = 0.0
    for segment in route.segments:
        placements = {p.id: p for p in lattice.layout(segment.branch).placements}
        total += placements[segment.end].s_exit - placements[segment.start].s_enter
    return total


def route_problems(lattice: Lattice, route: Route) -> list[str]:
    """What keeps a route from being one pass at one energy: reference momenta that change at a join."""
    problems = []
    for before, after in zip(route.segments, route.segments[1:], strict=False):
        if before.branch == after.branch:
            continue
        a, b = (lattice.initial_conditions(s.branch) for s in (before, after))
        pa = a.reference.pc_ref if a and a.reference else None
        pb = b.reference.pc_ref if b and b.reference else None
        if pa and pb and not math.isclose(pa, pb, rel_tol=1e-6):
            names = [lattice.branch(s.branch).name for s in (before, after)]
            problems.append(
                f"the reference momentum changes from {names[0]} ({pa / 1e9:.4g} GeV/c) to {names[1]} "
                f"({pb / 1e9:.4g} GeV/c): an acceleration, not one pass"
            )
    return problems
