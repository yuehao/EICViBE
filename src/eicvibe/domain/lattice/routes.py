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

A ring the beam is *stored* in before it leaves (``stored``, by branch name)
gives the beam its own periodic optics where it leaves: an accumulator ring
such as the EIN's BAR damps whatever came in to its equilibrium. Without it a
route is one pass, and carries the optics it came in with.

A route may go round a ring a number of full ``turns`` before it goes on (from
where it came in, back to it, then on to its fork), so that a synchrotron's
circulating beam is shown once round before it is extracted; only those rings'
elements are passed more than once. A ring the beam is *accelerated* in
(``accelerated``) is left at the next branch's reference momentum: its magnets
ramp with the beam, so the optics go on unchanged, and the beam's angles and
momentum spread shrink as the momentum grows (adiabatic damping).

The EIN's BAR route, for example, is ``ltr, itl, bar, etl, ltr, rcs, rte``: the
LTR to its fork to the ITL; the ITL; the BAR from the septum round to its fork
to the ETL; the ETL; the LTR again, from where the ETL lands to its fork to the
RCS; once round the RCS (one turn) and on to its extraction; and the RTE.
"""

import math
from collections.abc import Mapping, Sequence

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
    stored: tuple[str, ...] = Field(
        (),
        description="Ids of the rings the beam is stored in before it leaves them: it leaves on their optics",
    )
    turns: tuple[tuple[str, int], ...] = Field(
        (),
        description="Full turns the beam goes round a ring (by id) before it goes on to its fork",
    )
    accelerated: tuple[str, ...] = Field(
        (),
        description="Ids of rings the beam is accelerated in: it leaves them at the next branch's momentum",
    )

    def turns_in(self, branch: str) -> int:
        """Full turns round ``branch`` (an id) before going on."""
        return dict(self.turns).get(branch, 0)


def route_through(
    lattice: Lattice,
    name: str,
    branches: Sequence[str],
    stored: Sequence[str] = (),
    turns: Mapping[str, int] | None = None,
    accelerated: Sequence[str] = (),
) -> Route:
    """The route through ``branches`` (names or ids, in beam order), joined at their forks.

    ``stored`` names the rings the beam is stored in before it leaves them; ``turns`` the full
    turns it goes round a ring (by name or id) before going on; ``accelerated`` the rings it is
    accelerated in, which it leaves at the next branch's momentum.
    """
    if not branches:
        raise ValueError(f"route {name}: no branches")
    on_route = {lattice.layout(b).id for b in branches}

    def rings(given: Sequence[str], does: str, done: str) -> list[str]:
        """``given`` as ring ids on the route: ``does`` what only a ring does, ``done`` in it."""
        out = []
        for ring in given:
            layout = lattice.layout(ring)
            if not layout.periodic:
                raise ValueError(f"route {name}: {layout.branch.name} is a line; only a ring {does}")
            if layout.id not in on_route:
                raise ValueError(f"route {name}: it does not pass {layout.branch.name}, which it {done}")
            out.append(layout.id)
        return out

    kept = rings(stored, "stores a beam", "stores the beam in")
    speeding = rings(accelerated, "accelerates a beam", "accelerates the beam in")
    last = lattice.layout(branches[-1]).id
    if last in speeding:
        raise ValueError(
            f"route {name}: it ends in {lattice.branch(last).name}, so leaves no ring accelerated"
        )
    turning: dict[str, int] = {}
    for ring, count in (turns or {}).items():
        if count < 0 or count != int(count):
            raise ValueError(f"route {name}: {count} turns round {ring}: a whole number, at least 0")
        if count:
            turning[rings([ring], "goes round", "goes round")[0]] = int(count)
    segments: list[Segment] = []
    entry = 0
    for k, branch in enumerate(branches):
        layout = lattice.layout(branch)
        ids = [p.id for p in layout.placements]

        def part(a: int, b: int, layout=layout, ids=ids) -> None:
            segments.append(Segment(branch=layout.id, start=ids[a], end=ids[b]))

        def round_(times: int, entry=entry, ids=ids) -> None:
            """Full turns from where the route came in, back to it."""
            for _ in range(times):
                part(entry, len(ids) - 1)
                if entry > 0:
                    part(0, entry - 1)

        if k == len(branches) - 1:
            if layout.periodic:
                round_(max(1, turning.get(layout.id, 0)))  # round, back to where the route came in
            else:
                part(entry, len(ids) - 1)
            break
        round_(turning.get(layout.id, 0))
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
    route = Route(
        name=name,
        branches=tuple(branches),
        segments=tuple(segments),
        stored=tuple(kept),
        turns=tuple(turning.items()),
        accelerated=tuple(speeding),
    )
    seen: set[str] = set()
    for _, placed, _ in route_placements(lattice, route):
        if placed.id in seen and placed.branch_id not in turning:
            raise ValueError(
                f"route {name} passes {placed.name} twice; only a ring it goes round (turns) is passed again"
            )
        seen.add(placed.id)
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
    """What keeps a route from being one pass at one energy: reference momenta that change at a join,
    other than where it leaves a ring it is accelerated in."""
    problems = []
    for before, after in zip(route.segments, route.segments[1:], strict=False):
        if before.branch == after.branch or before.branch in route.accelerated:
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
