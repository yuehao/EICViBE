"""Routes: the beam's paths through several branches, joined at their forks."""

import pytest

from eicvibe.api.demo import demo_lattice
from eicvibe.domain.lattice import (
    LatticeBuilder,
    SequentialIds,
    route_length,
    route_placements,
    route_problems,
    route_through,
)


def named(lattice, route):
    return [
        (lattice.branch(s.branch).name, lattice.placement(s.start).name, lattice.placement(s.end).name)
        for s in route.segments
    ]


def test_a_route_follows_the_forks_from_branch_to_branch():
    lattice = demo_lattice()
    route = route_through(lattice, "transfer", ["injection", "ring", "extraction"])
    assert named(lattice, route) == [
        ("injection", "LINE_START_1", "fork_injection_to_ring"),
        ("ring", "SEPTUM_1", "fork_ring_to_extraction"),
        ("extraction", "X_START_1", "BPM_1"),
    ]
    placements = route_placements(lattice, route)
    s = [at for _, _, at in placements]
    assert s == sorted(s) and len({p.id for _, p, _ in placements}) == len(placements)
    ring = lattice.layout("ring")
    fork = next(p for p in ring.placements if p.name == "fork_ring_to_extraction")
    injection, extraction = lattice.layout("injection").length, lattice.layout("extraction").length
    expected = injection + fork.s_exit - ring.placements[1].s_enter + extraction
    assert route_length(lattice, route) == pytest.approx(expected)


def test_a_route_that_ends_in_a_ring_goes_once_round():
    lattice = demo_lattice()
    route = route_through(lattice, "fill", ["injection", "ring"])
    assert named(lattice, route)[1:] == [
        ("ring", "SEPTUM_1", "D_66"),
        ("ring", "RING_START_1", "RING_START_1"),
    ]
    expected = lattice.layout("injection").length + lattice.layout("ring").length
    assert route_length(lattice, route) == pytest.approx(expected)


def wrapped():
    """A ring entered half way round, left by a fork near its start, and a line at another energy."""
    b = LatticeBuilder("wrap", ids=SequentialIds())
    reference = {"species_ref": "electron", "pc_ref": 1e9}
    b.define("START", "BeginningEle", ReferenceP=reference)
    b.define("D", "Drift", length=1.0)
    b.define("M", "Marker")
    b.add_branch("ring", "periodic", ["START", "D", "M", "D", "D", "M", "D", "D"])
    b.add_branch("line", "linac", ["START", "D", "D"])
    b.define("HOT", "BeginningEle", ReferenceP=reference | {"pc_ref": 2e9})
    b.add_branch("hot", "linac", ["HOT", "D"])
    b.add_branch("out", "linac", ["START", "D"])
    b.add_fork("line", "ring", to="M_2", to_side="exit", after="D_2")
    b.add_fork("ring", "out", after="M_1")
    b.add_fork("ring", "hot", after="M_2")
    return b.build()


def test_a_fork_behind_the_entry_is_reached_by_going_round():
    lattice = wrapped()
    route = route_through(lattice, "round", ["line", "ring", "out"])
    # an exit landing enters at the occurrence after it (the fork marker placed there); the fork to
    # "out" lies behind, so the route goes on to the ring's end and round from its start
    assert named(lattice, route)[1:3] == [
        ("ring", "fork_ring_to_hot", "D_5"),
        ("ring", "START_1", "fork_ring_to_out"),
    ]
    assert route_problems(lattice, route) == []


def test_a_route_that_cannot_be_one_pass_says_why():
    lattice = wrapped()
    with pytest.raises(ValueError, match="no fork from out"):
        route_through(lattice, "nowhere", ["out", "ring"])
    hot = route_through(lattice, "hot", ["line", "ring", "hot"])
    assert "reference momentum changes" in route_problems(lattice, hot)[0]


def test_a_route_goes_round_a_ring_its_turns_and_may_leave_it_accelerated():
    """A synchrotron's beam, shown once round before it is extracted to a line at a higher momentum."""
    lattice = wrapped()
    once = route_through(lattice, "fast", ["line", "ring", "hot"])
    assert named(lattice, once)[1:] == [
        ("ring", "fork_ring_to_hot", "fork_ring_to_hot"),
        ("hot", "HOT_1", "D_1"),
    ]
    round_ = route_through(lattice, "fast", ["line", "ring", "hot"], turns={"ring": 1}, accelerated=["ring"])
    assert named(lattice, round_)[1:] == [
        ("ring", "fork_ring_to_hot", "D_5"),  # once round from where it came in
        ("ring", "START_1", "M_2"),
        ("ring", "fork_ring_to_hot", "fork_ring_to_hot"),  # and on to the fork
        ("hot", "HOT_1", "D_1"),
    ]
    ring = lattice.layout("ring").length
    assert route_length(lattice, round_) == pytest.approx(route_length(lattice, once) + ring)
    assert round_.turns_in(lattice.layout("ring").id) == 1
    assert route_problems(lattice, once) and route_problems(lattice, round_) == []  # accelerated in the ring
    # Only a ring it goes round is passed again; turns and acceleration are a ring's.
    passes = [p.id for _, p, _ in route_placements(lattice, round_)]
    assert len(passes) > len(set(passes))
    with pytest.raises(ValueError, match="only a ring goes round"):
        route_through(lattice, "x", ["line", "ring", "hot"], turns={"line": 1})
    with pytest.raises(ValueError, match="ends in ring"):
        route_through(lattice, "x", ["line", "ring"], accelerated=["ring"])
    with pytest.raises(ValueError, match="whole number"):
        route_through(lattice, "x", ["line", "ring", "hot"], turns={"ring": -1})
