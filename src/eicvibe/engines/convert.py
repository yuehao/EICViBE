"""Reading an occurrence's parameters the way every engine needs them (ADR 0006).

Engines differ in how they build elements, not in what the lattice means:
integrated strengths, a bend's field and edges, an element's roll and
displacement, the reference particle and an RF frequency given as a
harmonic. Each helper refuses what it cannot represent with
:class:`~.contracts.UnsupportedRequest` naming the engine.
"""

import math
from typing import Any, NamedTuple

from eicvibe.domain.lattice import SPECIES, Lattice

from .contracts import UnsupportedRequest

C_LIGHT = 299_792_458.0
ORDERS = range(11)

type Groups = dict[str, dict[str, Any]]
type Frame = tuple[tuple[float, float, float], list[tuple[str, float]]]


def integrated(multipoles: dict[str, Any], prefix: str, order: int, length: float) -> float:
    """``KnL`` (or ``KsL``) of one order, from the per-length and integrated values together."""
    return multipoles.get(f"{prefix}{order}", 0.0) * length + multipoles.get(f"{prefix}{order}L", 0.0)


def per_length(multipoles: dict[str, Any], prefix: str, order: int, length: float) -> float:
    return integrated(multipoles, prefix, order, length) / length


def highest_order(multipoles: dict[str, Any], length: float) -> int:
    """The highest order with a normal or skew strength; 0 when there is none."""
    present = [
        n
        for n in ORDERS
        if integrated(multipoles, "Kn", n, length) or integrated(multipoles, "Ks", n, length)
    ]
    return max(present or [0])


def common_tilt(engine: str, multipoles: dict[str, Any], order: int) -> float:
    """The roll of an element's multipoles, which must all share the main order's."""
    main = multipoles.get(f"tilt{order}", 0.0)
    others = {multipoles.get(f"tilt{n}") for n in ORDERS if n != order and multipoles.get(f"tilt{n}")}
    if others - {main}:
        raise UnsupportedRequest(engine, ["multipoles of one element with different tilts"])
    return main


class Displacement(NamedTuple):
    """An element body's BodyShiftP: offsets of its centre (m) and right-handed rotations about it (rad).

    ``pitch`` is PALS ``x_rot`` (a positive one turns the exit end towards -y), ``yaw`` is ``y_rot``
    (towards +x), ``roll`` is ``z_rot``; all are about the body's centre, as Bmad's are.
    """

    x: float
    y: float
    roll: float
    pitch: float
    yaw: float


def displacement(engine: str, groups: Groups) -> Displacement:
    """An element's BodyShiftP as a :class:`Displacement`; a longitudinal offset is refused."""
    shift = groups.get("BodyShiftP", {})
    if shift.get("z_offset"):
        raise UnsupportedRequest(engine, ["BodyShiftP.z_offset is not converted yet"])
    return Displacement(
        shift.get("x_offset", 0.0),
        shift.get("y_offset", 0.0),
        shift.get("z_rot", 0.0),
        shift.get("x_rot", 0.0),
        shift.get("y_rot", 0.0),
    )


def bend_edges(engine: str, bend: dict[str, Any]) -> tuple[float, float]:
    """Pole-face angles relative to the sector faces, from ``e1``/``e2`` or ``e1_rect``/``e2_rect``."""
    for name in ("h1", "h2"):
        if bend.get(name):
            raise UnsupportedRequest(engine, [f"BendP.{name} (pole face curvature) is not converted yet"])
    angle = bend.get("angle_ref", 0.0)
    edges = []
    for sector, rect in (("e1", "e1_rect"), ("e2", "e2_rect")):
        if bend.get(sector) and bend.get(rect):
            raise UnsupportedRequest(engine, [f"BendP gives both {sector} and {rect}"])
        edges.append(bend.get(sector) or (bend.get(rect) + angle / 2 if bend.get(rect) else 0.0))
    return edges[0], edges[1]


def bend_field(bend: dict[str, Any], multipoles: dict[str, Any], length: float) -> float:
    """A bend's dipole field as ``Kn0`` (1/m): given, or the reference curvature if ``Kn0_from_g_ref``."""
    if multipoles.get("Kn0") or multipoles.get("Kn0L"):
        return per_length(multipoles, "Kn", 0, length)
    return bend.get("angle_ref", 0.0) / length if bend.get("Kn0_from_g_ref", True) else 0.0


def reference(engine: str, lattice: Lattice, branch: str) -> dict[str, float]:
    """The branch's reference particle: ``pc`` and ``mass`` (eV), ``charge`` (e) and ``beta0``."""
    conditions = lattice.initial_conditions(branch)
    ref = conditions.reference if conditions else None
    if ref is None or ref.pc_ref is None or ref.species_ref not in SPECIES:
        raise UnsupportedRequest(engine, [f"branch {lattice.branch(branch).name} has no reference particle"])
    species = SPECIES[ref.species_ref]
    return {
        "pc": ref.pc_ref,
        "mass": species.mass,
        "charge": species.charge,
        "beta0": ref.pc_ref / math.hypot(ref.pc_ref, species.mass),
    }


#: phi0 of each ``RFP.zero_phase``: the energy gain goes as sin(phase + phi0) (MAD-X's 2 pi LAG is phi0 = 0).
ZERO_PHASE = {"BELOW_TRANSITION": 0.0, "ACCELERATING": math.pi / 2, "ABOVE_TRANSITION": math.pi}


def rf_phase(engine: str, name: str, rf: dict[str, Any]) -> float:
    """The RF phase as MAD-X and Xsuite take it: an energy gain of |q| V sin(phase - omega tau)."""
    zero = rf.get("zero_phase") or "ACCELERATING"
    if zero not in ZERO_PHASE:
        raise UnsupportedRequest(engine, [f"{name}: RF zero_phase {zero}"])
    return rf.get("phase", 0.0) + ZERO_PHASE[zero]


def rf_frequency(engine: str, name: str, rf: dict[str, Any], revolution: float) -> float:
    """An RF frequency in Hz, given or from the harmonic number and the revolution frequency."""
    frequency = rf.get("frequency") or 0.0
    if not frequency and rf.get("harmon"):
        frequency = rf["harmon"] * revolution
    return frequency


def patch_frame(engine: str, patch: dict[str, Any]) -> Frame:
    """A patch as an offset of the frame, then its rotations in the order R_y, R_x, R_z.

    PALS offsets and right-handed rotations equal MAD-X's TRANSLATION and
    X/Y/SROTATION values; the order is the floor geometry's.
    """
    if patch.get("t_offset"):
        raise UnsupportedRequest(engine, ["a patch's t_offset is not converted yet"])
    offset = (patch.get("x_offset", 0.0), patch.get("y_offset", 0.0), patch.get("z_offset", 0.0))
    rotations = [(key, patch[key]) for key in ("y_rot", "x_rot", "z_rot") if patch.get(key)]
    return offset, rotations
