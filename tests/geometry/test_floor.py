import math

import pytest

from eicvibe.domain.lattice import ForkSide, LatticeBuilder, SequentialIds
from eicvibe.geometry import Frame, element_path, face_rotation, floor_geometry


def ring(cells: int = 8) -> LatticeBuilder:
    b = LatticeBuilder("ring", ids=SequentialIds())
    b.define("B", "Bend", length=2.0, BendP={"angle_ref": 2 * math.pi / cells})
    b.define("Q", "Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": 0.3})
    b.define("D", "Drift", length=1.0)
    b.add_branch("ring", "periodic", ["Q", "D", "B", "D", "B", "D"] * (cells // 2))
    return b


def test_a_ring_closes_and_bends_toward_minus_x():
    lattice = ring().build()
    frames = floor_geometry(lattice).branches[lattice.layout("ring").id].frames
    end = frames[-1].exit
    assert end.r == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert math.isclose(math.cos(end.angles[0]), 1.0, abs_tol=1e-12)  # one full turn
    first_bend = frames[2].exit
    assert first_bend.r[0] < 0 and first_bend.angles[0] < 0  # clockwise seen from above


def test_frame_angles_round_trip():
    frame = Frame.from_angles((1.0, 2.0, 3.0), theta=0.3, phi=-0.2, psi=0.1)
    assert frame.angles == pytest.approx((0.3, -0.2, 0.1))
    assert frame.compose(frame.inverse()).r == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)


def test_forks_place_branches_and_report_closures():
    b = LatticeBuilder("forks", ids=SequentialIds())
    b.define("BEG", "BeginningEle", FloorP={"x": 1.0, "z": 2.0, "theta": 0.5})
    b.define("D", "Drift", length=3.0)
    b.define("M", "Marker")
    b.define("B", "Bend", length=1.0, BendP={"angle_ref": 0.2})
    b.add_branch("main", "linac", ["BEG", "D", "M", "D"])
    b.add_branch("spur", "linac", ["D", "B", "D"])
    b.add_fork("main", "spur", after="M_1")
    b.add_fork("spur", "main", to="M_1", to_side="exit")  # back to where it left, after a bend
    lattice = b.build()
    geometry = floor_geometry(lattice)
    main = geometry.branches[lattice.layout("main").id]
    spur = geometry.branches[lattice.layout("spur").id]
    assert main.frames[0].entrance.r == pytest.approx((1.0, 0.0, 2.0))
    fork = lattice.placement("fork_main_to_spur", branch="main")
    assert spur.placed_by == fork.id
    assert spur.frames[0].entrance.r == pytest.approx(main.frames[fork.index].entrance.r)
    [closure] = geometry.closures
    assert closure.position_error > 1.0 and closure.angle_error == pytest.approx(0.2)


def test_patches_shift_and_turn_the_frame():
    b = LatticeBuilder("patch", ids=SequentialIds())
    b.define("P", "Patch", PatchP={"x_offset": 0.1, "y_rot": 0.05})
    b.define("D", "Drift", length=1.0)
    b.add_branch("line", "linac", ["P", "D"])
    lattice = b.build()
    end = floor_geometry(lattice).branches[lattice.layout("line").id].frames[-1].exit
    assert end.r == pytest.approx((0.1 + math.sin(0.05), 0.0, math.cos(0.05)))


def test_positions_agree_with_madx_survey():
    madx_module = pytest.importorskip("cpymad.madx")
    from eicvibe.importers import import_cpymad
    from tests.importers.test_madx import ELEMENTS

    with madx_module.Madx(stdout=False) as madx:
        madx.input(ELEMENTS)
        lattice = import_cpymad(madx, ["line"], kind="linac").lattice
        survey = madx.survey(sequence="line")
        rows = zip(survey.name, survey.keyword, survey.l, survey.x, survey.y, survey.z, strict=True)
        expected = [
            (x, y, z)
            for name, kind, length, x, y, z in rows
            if "$" not in name and not (kind == "drift" and length == 0)
        ]
    frames = floor_geometry(lattice).branches[lattice.layout("line").id].frames[1:]
    assert [c for f in frames for c in f.exit.r] == pytest.approx([c for r in expected for c in r], abs=1e-12)
    assert expected[-1][1] < -0.1  # the tilted bend and rotations leave the horizontal plane


def test_bend_paths_follow_the_arc():
    b = LatticeBuilder("arc", ids=SequentialIds())
    b.define("B", "Bend", length=1.0, BendP={"angle_ref": 0.1})
    b.define("Q", "Quadrupole", length=0.5)
    b.add_branch("line", "linac", ["B", "Q"])
    lattice = b.build()
    [bend, quad] = lattice.layout("line").placements
    frames = floor_geometry(lattice).branches[lattice.layout("line").id].frames
    path = element_path(frames[0].entrance, bend.element)
    assert len(path) == 11 and path[-1] == pytest.approx(frames[0].exit.r)
    rho = 1.0 / 0.1
    assert all(math.dist(p, (-rho, 0.0, 0.0)) == pytest.approx(rho) for p in path)
    assert element_path(frames[1].entrance, quad.element) == (frames[1].entrance.r, frames[1].exit.r)


def test_a_line_forking_into_the_root_ring_is_placed_backwards():
    b = ring()
    b.define("L", "Drift", length=5.0)
    b.add_branch("line", "linac", ["L", "B", "L"])
    b.add_fork("line", "ring", to="Q_2")
    lattice = b.build()
    geometry = floor_geometry(lattice)
    line = geometry.branches[lattice.layout("line").id]
    ring_frames = geometry.branches[lattice.layout("ring").id].frames
    landing = ring_frames[lattice.placement("Q_2", branch="ring").index].entrance
    assert line.placed_by == lattice.placement("fork_line_to_ring", branch="line").id
    assert line.frames[-1].exit.r == pytest.approx(landing.r, abs=1e-12)
    assert geometry.closures == ()


def test_a_rectangular_bend_has_parallel_faces():
    b = LatticeBuilder("rect", ids=SequentialIds())
    b.define("R", "Bend", length=1.0, BendP={"angle_ref": 0.2, "e1": 0.1, "e2": 0.1, "tilt_ref": 0.3})
    b.add_branch("line", "linac", ["R"])
    lattice = b.build()
    [placed] = lattice.layout("line").placements
    [frames] = floor_geometry(lattice).branches[lattice.layout("line").id].frames
    entrance = frames.entrance.then((0.0, 0.0, 0.0), face_rotation(placed.element, ForkSide.ENTRANCE))
    exit_ = frames.exit.then((0.0, 0.0, 0.0), face_rotation(placed.element, ForkSide.EXIT))
    assert [c for row in exit_.w for c in row] == pytest.approx(
        [c for row in entrance.w for c in row], abs=1e-12
    )


def septum_lattice(shared: bool):
    """A ring whose straight kicker is also a rectangular septum bend for injected and extracted beam."""
    a, edge = -0.12, 0.00175
    b = LatticeBuilder("septum", ids=SequentialIds())
    b.define("D", "Drift", length=2.0)
    b.define("B", "Bend", length=2.0, BendP={"angle_ref": math.pi / 2})
    b.define("K", "Kicker", length=0.4)
    b.define("SI", "Bend", length=0.401, BendP={"angle_ref": a, "e1": a - edge, "e2": edge})
    b.define("SX", "Bend", length=0.401, BendP={"angle_ref": a, "e1": edge, "e2": a - edge})
    b.add_branch("ring", "periodic", ["K", "D", "B", "D", "B", "D", "B", "D", "B", "D"])
    b.add_branch("inj", "linac", ["D", "SI"])
    b.add_branch("ext", "linac", ["SX", "D"])
    if shared:
        b.connect_injection("inj", "ring", via=("SI_1", "K_1"), component="septum")
        b.connect_extraction("ring", "ext", via=("K_1", "SX_1"), component="septum")
    else:
        b.add_fork("inj", "ring", to="K_1", to_side="exit")
        b.add_fork("ring", "ext", before="K_1", to="SX_1")
    return b.build()


@pytest.mark.parametrize(("shared", "expected"), [(True, 1.75e-3), (False, 0.0)])
def test_forks_through_one_magnet_match_its_faces_not_the_orbits(shared, expected):
    lattice = septum_lattice(shared)
    geometry = floor_geometry(lattice)

    def frames(name, branch):
        placed = lattice.placement(name, branch=branch)
        return geometry.branches[placed.branch_id].frames[placed.index]

    kicker, injected, extracted = frames("K_1", "ring"), frames("SI_1", "inj"), frames("SX_1", "ext")
    assert kicker.exit.r == pytest.approx(injected.exit.r, abs=1e-12)
    assert kicker.exit.angles[0] - injected.exit.angles[0] == pytest.approx(expected, abs=1e-12)
    assert extracted.entrance.angles[0] - kicker.entrance.angles[0] == pytest.approx(expected, abs=1e-12)
