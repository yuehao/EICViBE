"""Floor coordinates: where every element of a lattice sits in the global frame.

A frame is a position ``r`` and an orientation ``w`` whose columns are the
local x, y and z axes in global coordinates (PALS, "Coordinate Systems").
Walking a branch, each element moves the frame by a local displacement ``L``
and rotation ``S``: ``r += w L`` and ``w = w S``.

- A straight element of length l: ``L = (0, 0, l)``.
- A bend of arc length l, angle a and reference tilt t: with rho = l / a,
  ``L = R_z(t) (rho (cos a - 1), 0, rho sin a)`` and
  ``S = R_z(t) R_y(-a) R_z(-t)``, so a positive angle bends toward -x.
- A patch: ``L`` is its offsets and ``S = R_y(y_rot) R_x(x_rot) R_z(z_rot)``.

The root branch starts at its BeginningEle's FloorP (W = R_y(theta) R_x(phi)
R_z(psi)), or at the origin. Forks then place the other branches so that a
fork's frame coincides with the frame of the point it lands on, as PALS does
for a fork into a new branch. This works in both directions: a fork places
the branch it leads to, and a branch that forks into a placed one (an
injection line into a root ring, say) is placed so that its fork lands
there. A fork between two branches that are both placed already (a line
returning to the line it left) moves neither; the mismatch between the two
frames is reported as a :class:`Closure`. Branches no fork connects start at
their own FloorP or at the origin.

A fork that joins two traversals of one shared component at the same face
(as ``connect_injection`` and ``connect_extraction`` place them) matches
the magnet's pole faces rather than the two reference orbits: the beams
take different paths through one rigid magnet. A bend's entrance face is
turned by -e1 from its orbit and its exit face by +e2 (in its bending
plane), so a rectangular bend, e1 + e2 = angle, has parallel faces; the
faces of any other element are square to its orbit. The EIN septum is
such a magnet: a bend of -0.12 rad with parallel faces for the injected
and extracted beams, and a straight, pulsed-off kicker for the stored
beam, so the injected beam leaves it 1.75 mrad (e2) off the stored orbit.
"""

import math
from collections import deque
from dataclasses import dataclass

from eicvibe.domain.lattice import ElementDefinition, ElementKind, ForkSide, Lattice

type Vector = tuple[float, float, float]
type Matrix = tuple[Vector, Vector, Vector]
"""Rows of a 3x3 matrix."""

IDENTITY: Matrix = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _mul(a: Matrix, b: Matrix) -> Matrix:
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))  # type: ignore[return-value]


def _apply(a: Matrix, v: Vector) -> Vector:
    return tuple(sum(a[i][k] * v[k] for k in range(3)) for i in range(3))  # type: ignore[return-value]


def _transpose(a: Matrix) -> Matrix:
    return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def rotation_x(angle: float) -> Matrix:
    c, s = math.cos(angle), math.sin(angle)
    return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))


def rotation_y(angle: float) -> Matrix:
    c, s = math.cos(angle), math.sin(angle)
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))


def rotation_z(angle: float) -> Matrix:
    c, s = math.cos(angle), math.sin(angle)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class Frame:
    """A position and orientation in global coordinates."""

    r: Vector = (0.0, 0.0, 0.0)
    w: Matrix = IDENTITY

    def then(self, displacement: Vector, rotation: Matrix = IDENTITY) -> Frame:
        """The frame after moving by ``displacement`` and turning by ``rotation``, both in local axes."""
        moved = _apply(self.w, displacement)
        return Frame(
            (self.r[0] + moved[0], self.r[1] + moved[1], self.r[2] + moved[2]), _mul(self.w, rotation)
        )

    def compose(self, local: Frame) -> Frame:
        """``local``, a frame relative to this one, in global coordinates."""
        return self.then(local.r, local.w)

    def inverse(self) -> Frame:
        wt = _transpose(self.w)
        back = _apply(wt, self.r)
        return Frame((-back[0], -back[1], -back[2]), wt)

    @property
    def angles(self) -> tuple[float, float, float]:
        """(theta, phi, psi) with ``w = R_y(theta) R_x(phi) R_z(psi)``."""
        w = self.w
        phi = math.asin(max(-1.0, min(1.0, -w[1][2])))
        theta = math.atan2(w[0][2], w[2][2])
        psi = math.atan2(w[1][0], w[1][1])
        return theta, phi, psi

    @classmethod
    def from_angles(cls, r: Vector, theta: float = 0.0, phi: float = 0.0, psi: float = 0.0) -> Frame:
        return cls(r, _mul(_mul(rotation_y(theta), rotation_x(phi)), rotation_z(psi)))


def _bend_motion(length: float, angle: float, tilt: float) -> tuple[Vector, Matrix]:
    rho = length / angle
    in_plane = (rho * (math.cos(angle) - 1.0), 0.0, rho * math.sin(angle))
    tilt_matrix = rotation_z(tilt)
    rotation = _mul(_mul(tilt_matrix, rotation_y(-angle)), rotation_z(-tilt))
    return _apply(tilt_matrix, in_plane), rotation


def _bends(element: ElementDefinition) -> bool:
    return element.kind is ElementKind.BEND and element.BendP is not None and element.BendP.angle_ref != 0.0


def element_motion(element: ElementDefinition) -> tuple[Vector, Matrix]:
    """Local displacement and rotation across one element, entrance to exit."""
    length = element.length
    if _bends(element):
        return _bend_motion(length, element.BendP.angle_ref, element.BendP.tilt_ref)
    if element.kind is ElementKind.PATCH and element.PatchP is not None:
        p = element.PatchP
        rotation = _mul(_mul(rotation_y(p.y_rot), rotation_x(p.x_rot)), rotation_z(p.z_rot))
        return (p.x_offset, p.y_offset, p.z_offset), rotation
    return (0.0, 0.0, length), IDENTITY


def element_path(entrance: Frame, element: ElementDefinition, max_angle: float = 0.01) -> tuple[Vector, ...]:
    """Points on the reference orbit through ``element``, from its entrance to its exit.

    A bend's arc is sampled so that the orbit turns by at most ``max_angle``
    between points; every other element is its two end points (a patch jumps).
    """
    if not _bends(element) or element.length == 0.0:
        return entrance.r, entrance.then(*element_motion(element)).r
    angle, tilt = element.BendP.angle_ref, element.BendP.tilt_ref
    steps = max(1, math.ceil(abs(angle) / max_angle))
    points = [entrance.r]
    for step in range(1, steps + 1):
        fraction = step / steps
        points.append(entrance.then(*_bend_motion(element.length * fraction, angle * fraction, tilt)).r)
    return tuple(points)


@dataclass(frozen=True, slots=True)
class PlacedFrames:
    """Global entrance and exit frames of one occurrence."""

    occurrence: str
    entrance: Frame
    exit: Frame


@dataclass(frozen=True, slots=True)
class Closure:
    """A fork into an already placed branch, and how far its landing point is from where the fork is."""

    fork: str
    """Occurrence id of the fork."""
    position_error: float
    """Distance, m."""
    angle_error: float
    """Angle between the two orientations, rad."""


@dataclass(frozen=True, slots=True)
class BranchGeometry:
    branch: str
    """Branch id."""
    placed_by: str | None
    """Occurrence id of the fork that placed this branch (leading to it or from it); None for a start."""
    frames: tuple[PlacedFrames, ...]


@dataclass(frozen=True, slots=True)
class FloorGeometry:
    branches: dict[str, BranchGeometry]
    """By branch id."""
    closures: tuple[Closure, ...]


def _local_frames(lattice: Lattice, branch: str) -> list[tuple[str, Frame, Frame]]:
    """(occurrence id, entrance, exit) relative to the branch's start."""
    frame, frames = Frame(), []
    for placed in lattice.layout(branch).placements:
        exit_frame = frame.then(*element_motion(placed.element))
        frames.append((placed.id, frame, exit_frame))
        frame = exit_frame
    return frames


def _start(lattice: Lattice, branch: str) -> Frame:
    placements = lattice.layout(branch).placements
    floor = (
        placements[0].element.FloorP
        if placements and placements[0].kind is ElementKind.BEGINNING_ELE
        else None
    )
    if floor is None:
        return Frame()
    return Frame.from_angles((floor.x, floor.y, floor.z), floor.theta, floor.phi, floor.psi)


def face_rotation(element: ElementDefinition, side: ForkSide) -> Matrix:
    """Orientation of an element's pole face at ``side`` relative to its reference orbit there.

    The face's frame is the orbit frame turned by this rotation: -e1 at a
    bend's entrance and +e2 at its exit, about the axis normal to its bending
    plane. Faces of other elements are square to the orbit.
    """
    bend = element.BendP if element.kind is ElementKind.BEND else None
    angle = 0.0 if bend is None else (-bend.e1 if side is ForkSide.ENTRANCE else bend.e2)
    if angle == 0.0:
        return IDENTITY
    tilt = bend.tilt_ref
    return _mul(_mul(rotation_z(tilt), rotation_y(angle)), rotation_z(-tilt))


def _shared_faces(lattice: Lattice) -> dict[str, tuple[Matrix, Matrix]]:
    """Forks that join two traversals of one shared component at the same face.

    By fork id: the face rotations of the traversal next to the fork and of
    the traversal it lands on. The fork must sit right after (landing on an
    exit) or right before (landing on an entrance) a traversal of the
    component it lands on; elements that do not move the frame may lie
    between.
    """
    found = {}
    for edge in lattice.forks:
        if edge.to_occurrence is None:
            continue
        component = lattice.component_of(edge.to_occurrence)
        if component is None:
            continue
        traversals = {t.occurrence for t in component.traversals}
        placements = lattice.layout(edge.from_branch).placements
        step = -1 if edge.to_side is ForkSide.EXIT else 1
        k = edge.from_index + step
        while 0 <= k < len(placements) and placements[k].id not in traversals:
            if element_motion(placements[k].element) != ((0.0, 0.0, 0.0), IDENTITY):
                break
            k += step
        if 0 <= k < len(placements) and placements[k].id in traversals:
            landing = lattice.placement(edge.to_occurrence).element
            found[edge.id] = (
                face_rotation(placements[k].element, edge.to_side),
                face_rotation(landing, edge.to_side),
            )
    return found


def _angle_between(a: Matrix, b: Matrix) -> float:
    relative = _mul(_transpose(a), b)
    trace = relative[0][0] + relative[1][1] + relative[2][2]
    return math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))


def floor_geometry(lattice: Lattice) -> FloorGeometry:
    """Global frames of every occurrence of every branch, placing branches through forks."""
    local = {b.id: _local_frames(lattice, b.id) for b in lattice.branches}
    index = {b: {occ: (entrance, exit_) for occ, entrance, exit_ in frames} for b, frames in local.items()}
    placement: dict[str, tuple[Frame, str | None]] = {}
    closures: list[Closure] = []
    checked: set[str] = set()
    faces = _shared_faces(lattice)

    def landing(edge) -> Frame:
        """Frame of the landing point relative to the start of the branch the fork leads to."""
        if edge.to_occurrence is None:
            return Frame()
        entrance, exit_ = index[edge.to_branch][edge.to_occurrence]
        return exit_ if edge.to_side is ForkSide.EXIT else entrance

    def place(root: str) -> None:
        placement[root] = (_start(lattice, root), None)
        queue = deque([root])
        while queue:
            branch = queue.popleft()
            for edge in (*lattice.forks_from(branch), *lattice.forks_into(branch)):
                if edge.id in checked:
                    continue
                checked.add(edge.id)
                fork_local, land = index[edge.from_branch][edge.id][0], landing(edge)
                if edge.id in faces:  # one magnet: match its faces, not the orbits
                    fork_face, landing_face = faces[edge.id]
                    fork_local, land = (
                        fork_local.then((0.0, 0.0, 0.0), fork_face),
                        land.then((0.0, 0.0, 0.0), landing_face),
                    )
                if edge.from_branch in placement and edge.to_branch in placement:
                    fork_frame = placement[edge.from_branch][0].compose(fork_local)
                    there = placement[edge.to_branch][0].compose(land)
                    gap = math.dist(there.r, fork_frame.r)
                    closures.append(Closure(edge.id, gap, _angle_between(there.w, fork_frame.w)))
                elif edge.from_branch in placement:
                    fork_frame = placement[edge.from_branch][0].compose(fork_local)
                    placement[edge.to_branch] = (fork_frame.compose(land.inverse()), edge.id)
                    queue.append(edge.to_branch)
                else:
                    there = placement[edge.to_branch][0].compose(land)
                    placement[edge.from_branch] = (there.compose(fork_local.inverse()), edge.id)
                    queue.append(edge.from_branch)

    if lattice.root_branch:
        place(lattice.root_branch)
    for branch in lattice.branches:
        if branch.id not in placement:
            place(branch.id)

    branches = {}
    for branch_id, frames in local.items():
        origin, placed_by = placement[branch_id]
        branches[branch_id] = BranchGeometry(
            branch_id,
            placed_by,
            tuple(
                PlacedFrames(occ, origin.compose(entrance), origin.compose(exit_))
                for occ, entrance, exit_ in frames
            ),
        )
    return FloorGeometry(branches, tuple(closures))
