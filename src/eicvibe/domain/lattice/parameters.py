"""Typed parameter groups attached to element definitions.

Groups, field names, units and meanings follow the PALS standard
(https://pals-project.readthedocs.io/en/latest/lattice-element-parameter-groups.html,
as of October 2026): a field stored here means what the PALS field of the
same name means, so canonical content is a PALS subset. Where PALS offers
several ways to state one quantity (``g_ref`` or ``angle_ref``,
``x_center``/``x_width`` or ``x_min``/``x_max``, a field ``Bn1`` or a
normalized ``Kn1``), one is stored. An importer converts the others, and
maps legacy aliases such as ``kn1``, ``angle`` or ``hkick``, and records the
mapping in its report. ``ControlP`` is an EICViBE extension; its ``is_on`` is
the ungrouped PALS element parameter of that name.

Only hard invariants are enforced at construction: values that are
physically meaningless (negative beta, x_min >= x_max) or ambiguous (both
``frequency`` and ``harmon``). Suspicious magnitudes are warnings from
:func:`eicvibe.domain.lattice.diagnostics.diagnose`, never errors.

A field left at its default is equivalent to an absent field, and a group
whose fields are all defaults is equivalent to an absent group. Canonical
serialization omits both.
"""

import itertools
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from typing import Annotated, Any, ClassVar, Self

from pydantic import ConfigDict, Field, TypeAdapter, create_model, field_validator, model_validator
from pydantic_core import PydanticUndefined

from ._base import DomainModel, NonNegativeReal, PositiveInt, PositiveReal, Real
from .conventions import SPECIES, energies_agree


def _field(default: Any = PydanticUndefined, unit: str | None = None, description: str | None = None) -> Any:
    """A field whose unit is machine-readable (``unit`` in the JSON schema)."""
    return Field(default, description=description, json_schema_extra={"unit": unit} if unit else None)


class ParameterGroup(DomainModel):
    """Base class of all parameter groups."""

    group_name: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.group_name = cls.__name__

    def is_default(self) -> bool:
        """True when every field has its default value (the group carries no information)."""
        fields = type(self).model_fields
        return all(
            not field.is_required() and getattr(self, name) == field.default for name, field in fields.items()
        )


#: Highest multipole order stored (0 is the dipole, 10 the 22-pole). PALS allows any order.
MAX_MULTIPOLE_ORDER = 10

_POLES = ("dipole", "quadrupole", "sextupole", "octupole", "decapole", "dodecapole")


def _multipole_fields() -> dict[str, Any]:
    """Fields of MagneticMultipoleP for every order N: normalized strengths KnN, KsN, KnNL, KsNL;
    fields BnN, BsN, BnNL, BsNL; and tiltN."""
    orders = range(MAX_MULTIPOLE_ORDER + 1)
    pole = [_POLES[n] if n < len(_POLES) else f"{2 * n + 2}-pole" for n in orders]
    fields: dict[str, Any] = {}
    for part, side in (("n", "normal"), ("s", "skew")):
        for n in orders:
            unit = "1/m" if n == 0 else f"1/m^{n + 1}"
            fields[f"K{part}{n}"] = (Real, _field(0.0, unit, f"{side.capitalize()} {pole[n]} strength"))
    for part, side, kick in (("n", "normal", "-x"), ("s", "skew", "+y")):
        for n in orders:
            unit = {0: "rad", 1: "1/m"}.get(n, f"1/m^{n}")
            note = f"; > 0 kicks toward {kick}" if n == 0 else ""
            fields[f"K{part}{n}L"] = (Real, _field(0.0, unit, f"Integrated {side} {pole[n]} strength{note}"))
    for part, side in (("n", "normal"), ("s", "skew")):
        for n in orders:
            unit = "T" if n == 0 else "T/m" if n == 1 else f"T/m^{n}"
            fields[f"B{part}{n}"] = (Real, _field(0.0, unit, f"{side.capitalize()} {pole[n]} field"))
            unit = "T m" if n == 0 else "T" if n == 1 else "T/m" if n == 2 else f"T/m^{n - 1}"
            fields[f"B{part}{n}L"] = (Real, _field(0.0, unit, f"Integrated {side} {pole[n]} field"))
    for n in orders:
        fields[f"tilt{n}"] = (Real, _field(0.0, "rad", f"Rotation of the {pole[n]} about s"))
    return fields


#: The forms a multipole order's strength takes: normalized per length and integrated, and the
#: field per length and integrated. One order uses one form, for its normal and skew parts alike.
STRENGTH_FORMS = (("K", ""), ("K", "L"), ("B", ""), ("B", "L"))


def strength_parameter(group: MagneticMultipoleP | None, order: int, normal: bool = True) -> str:
    """The parameter that holds a multipole order's strength: the form the element gives (normalized
    or field, per length or integrated), else the normalized per-length one."""
    side = "n" if normal else "s"
    if group is not None:
        for letter, suffix in STRENGTH_FORMS:
            if getattr(group, f"{letter}n{order}{suffix}") or getattr(group, f"{letter}s{order}{suffix}"):
                return f"{letter}{side}{order}{suffix}"
    return f"K{side}{order}"


def has_fields(group: MagneticMultipoleP | None) -> bool:
    """Whether any order is given as a field (BnN, BnNL, ...): normalizing it needs the reference momentum."""
    return group is not None and any(
        getattr(group, f"B{side}{n}{suffix}")
        for n in range(MAX_MULTIPOLE_ORDER + 1)
        for side in ("n", "s")
        for suffix in ("", "L")
    )


# The fields are generated so that MAX_MULTIPOLE_ORDER is the only place the order limit is set.
_MagneticMultipoleFields = create_model(
    "_MagneticMultipoleFields", __base__=ParameterGroup, __module__=__name__, **_multipole_fields()
)


class MagneticMultipoleP(_MagneticMultipoleFields):
    """Magnetic multipoles of orders 0 to ``MAX_MULTIPOLE_ORDER`` (see ``conventions``).

    Each order N is given in one form: normalized per unit length (``KnN``,
    ``KsN``) or integrated over the element (``KnNL``, ``KsNL``, which are
    ``length`` times ``KnN``, ``KsN``); or as a field, per length (``BnN``,
    ``BsN``) or integrated (``BnNL``, ``BsNL``). Normalized strengths are the
    rule (ADR 0001); fields are for lines whose reference momentum along them
    is not known, such as an injector accelerating from the gun (ADR 0012),
    where a strength can only be normalized by tracking the beam. Zero-length
    elements need an integrated form; a kicker's kick angles are ``Kn0L`` and
    ``Ks0L``. ``tiltN`` rotates order N about s.
    """

    @model_validator(mode="after")
    def _one_form_per_order(self) -> Self:
        for n in range(MAX_MULTIPOLE_ORDER + 1):
            given = [
                f"{letter}n{n}{suffix}/{letter}s{n}{suffix}"
                for letter, suffix in STRENGTH_FORMS
                if getattr(self, f"{letter}n{n}{suffix}") or getattr(self, f"{letter}s{n}{suffix}")
            ]
            if len(given) > 1:
                raise ValueError(f"order {n} is given in one form only; got {' and '.join(given)}")
        return self


class BendRefGeometry(StrEnum):
    """Reference coordinates of a bend for multipoles and rectangular edge angles."""

    ARC = "ARC"
    CHORD = "CHORD"
    ENTRANCE_COORDS = "ENTRANCE_COORDS"
    EXIT_COORDS = "EXIT_COORDS"


class MultipoleGeometry(StrEnum):
    """Coordinates in which a bend's multipoles are given."""

    FOLLOWS_REF_GEOMETRY = "FOLLOWS_REF_GEOMETRY"
    VERTICALLY_PURE = "VERTICALLY_PURE"
    HORIZONTALLY_PURE = "HORIZONTALLY_PURE"
    CHORD = "CHORD"
    ENTRANCE_COORDS = "ENTRANCE_COORDS"
    EXIT_COORDS = "EXIT_COORDS"


class BendP(ParameterGroup):
    """Reference geometry of a bend. The arc length is the element ``length``.

    PALS accepts any two of an angle, a curvature (``g_ref``, ``radius_ref``,
    ``Bn0_ref``) and a length (``length``, ``L_chord``, ``L_rectangle``). Here
    a bend always has ``length`` and ``angle_ref``; the others follow from
    them, and storing them allowed contradictory values.

    The bending field is ``MagneticMultipoleP.Kn0`` (or ``Kn0L``). While both
    are zero and ``Kn0_from_g_ref`` is true, the field is the reference
    strength ``angle_ref / length``; a bend with no field sets
    ``Kn0_from_g_ref`` false.

    While ``e1`` and ``e1_rect`` are both zero, the entrance face is a sector
    face (e1 = 0) for ``ref_geometry`` ARC and a rectangular face
    (e1_rect = 0) otherwise; the exit face likewise. On an ARC reference, a
    rectangular bend (MAD-X RBEND) sets e1 = e2 = angle_ref / 2.
    """

    angle_ref: Real = _field(unit="rad", description="Reference bending angle; > 0 bends toward -x")
    e1: Real = _field(0.0, "rad", "Entrance face angle, sector frame")
    e2: Real = _field(0.0, "rad", "Exit face angle, sector frame")
    e1_rect: Real = _field(0.0, "rad", "Entrance face angle, rectangular frame of ref_geometry")
    e2_rect: Real = _field(0.0, "rad", "Exit face angle, rectangular frame of ref_geometry")
    edge1_int: Real = _field(0.0, "m", "Entrance fringe field integral (MAD-X FINT*HGAP)")
    edge2_int: Real = _field(0.0, "m", "Exit fringe field integral (MAD-X FINT*HGAP)")
    h1: Real = _field(0.0, "1/m", "Entrance pole face curvature; > 0 is convex")
    h2: Real = _field(0.0, "1/m", "Exit pole face curvature; > 0 is convex")
    Kn0_from_g_ref: bool = Field(True, description="Zero Kn0 and Kn0L mean the reference field")
    multipole_geometry: MultipoleGeometry = MultipoleGeometry.FOLLOWS_REF_GEOMETRY
    ref_geometry: BendRefGeometry = BendRefGeometry.ARC
    tilt_ref: Real = _field(0.0, "rad", "Rotation of the bend plane about s")

    @model_validator(mode="after")
    def _one_edge_frame(self) -> Self:
        for sector, rect in (("e1", "e1_rect"), ("e2", "e2_rect")):
            if getattr(self, sector) != 0.0 and getattr(self, rect) != 0.0:
                raise ValueError(f"{sector} and {rect} describe the same edge in different frames; set one")
        return self


class SolenoidP(ParameterGroup):
    Ksol: Real = _field(0.0, "1/m", "Normalized solenoid strength B_s/(B rho)")
    Bsol: Real = _field(0.0, "T", "Solenoid field")

    @model_validator(mode="after")
    def _one_strength(self) -> Self:
        if self.Ksol != 0.0 and self.Bsol != 0.0:
            raise ValueError("set either Ksol or Bsol, not both")
        return self


class CavityType(StrEnum):
    STANDING_WAVE = "STANDING_WAVE"
    TRAVELING_WAVE = "TRAVELING_WAVE"


class ZeroPhase(StrEnum):
    """What ``RFP.phase`` = 0 refers to."""

    ACCELERATING = "ACCELERATING"
    """Maximum acceleration."""
    BELOW_TRANSITION = "BELOW_TRANSITION"
    """Stable zero crossing for particles below transition."""
    ABOVE_TRANSITION = "ABOVE_TRANSITION"
    """Stable zero crossing for particles above transition."""


class RFP(ParameterGroup):
    """RF cavity settings.

    At time t the RF phase is 2 pi frequency t + ``phase`` + phi0, where
    ``zero_phase`` fixes phi0. The reference energy grows by ``dE_ref``
    through the element.
    """

    frequency: PositiveReal | None = _field(None, "Hz", "RF frequency")
    harmon: PositiveInt | None = Field(None, description="Harmonic number")
    voltage: NonNegativeReal = _field(0.0, "V", "Peak voltage")
    gradient: NonNegativeReal = _field(0.0, "V/m", "Accelerating gradient")
    phase: Real = _field(0.0, "rad", "RF phase")
    multipass_phase: Real = _field(0.0, "rad", "Phase added on multipass (shared-component) traversals")
    cavity_type: CavityType = CavityType.STANDING_WAVE
    num_cells: PositiveInt | None = Field(None, description="Number of cavity cells")
    zero_phase: ZeroPhase = ZeroPhase.ACCELERATING
    L_active: PositiveReal | None = _field(None, "m", "Length with RF field; None means the element length")
    dE_ref: Real = _field(0.0, "eV", "Reference energy change through the element")

    @model_validator(mode="after")
    def _unambiguous(self) -> Self:
        if self.frequency is not None and self.harmon is not None:
            raise ValueError("set either frequency or harmon, not both")
        if self.voltage != 0.0 and self.gradient != 0.0:
            raise ValueError("set either voltage or gradient, not both")
        return self


class ApertureShape(StrEnum):
    ELLIPTICAL = "ELLIPTICAL"
    RECTANGULAR = "RECTANGULAR"
    VERTICES = "VERTICES"
    CUSTOM_SHAPE = "CUSTOM_SHAPE"
    """Defined outside PALS; an adapter must know the shape."""


class ApertureLocation(StrEnum):
    ENTRANCE_END = "ENTRANCE_END"
    CENTER = "CENTER"
    EXIT_END = "EXIT_END"
    BOTH_ENDS = "BOTH_ENDS"
    NOWHERE = "NOWHERE"
    EVERYWHERE = "EVERYWHERE"


class ApertureVertex(DomainModel):
    """One vertex of a VERTICES aperture and the path to it from the previous one."""

    point: tuple[Real, Real] = _field(unit="m", description="(x, y) of the vertex")
    radius: tuple[PositiveReal, PositiveReal] | None = _field(
        None, "m", "Half widths of the elliptical arc to this vertex; None is a straight line"
    )
    convex: bool = Field(True, description="Take the arc whose ellipse center is toward the aperture center")
    tilt: Real = _field(0.0, "rad", "Counterclockwise rotation of the arc's ellipse")

    @field_validator("radius", mode="before")
    @classmethod
    def _circle(cls, value: Any) -> Any:
        # PALS allows a single radius for a circular arc.
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return (value[0], value[0])
        return value


class ApertureVertices(DomainModel):
    """Outline of a VERTICES aperture.

    Seen from the center, the vertices go once counterclockwise around it,
    each less than pi beyond the previous one, so that every ray from the
    center crosses the outline once.
    """

    center: tuple[Real, Real] = _field((0.0, 0.0), "m", "(x, y) of the center point")
    absolute_vertices: bool = Field(False, description="Vertex points are absolute, not relative to center")
    list: tuple[ApertureVertex, ...] = Field(min_length=3, description="Vertices in order")

    @model_validator(mode="after")
    def _star_shaped(self) -> Self:
        cx, cy = self.center if self.absolute_vertices else (0.0, 0.0)
        angles = []
        for vertex in self.list:
            x, y = vertex.point[0] - cx, vertex.point[1] - cy
            if x == 0.0 and y == 0.0:
                raise ValueError(f"vertex {vertex.point} is at the center")
            angles.append(math.atan2(y, x))
        steps = [(b - a) % math.tau for a, b in itertools.pairwise([*angles, angles[0]])]
        if not all(0.0 < step < math.pi for step in steps) or not math.isclose(sum(steps), math.tau):
            raise ValueError("vertices must go once counterclockwise around the center, less than pi apart")
        return self


class ApertureP(ParameterGroup):
    """Aperture limits. An unset limit is unlimited (legacy used 0.0 for that).

    An ELLIPTICAL aperture needs all four limits, or it is inactive.
    ``vertices`` is required only when ``shape`` is VERTICES, and may stay set
    while another shape is selected.
    """

    x_min: Real | None = _field(None, "m", "Lower x limit")
    x_max: Real | None = _field(None, "m", "Upper x limit")
    y_min: Real | None = _field(None, "m", "Lower y limit")
    y_max: Real | None = _field(None, "m", "Upper y limit")
    shape: ApertureShape = ApertureShape.ELLIPTICAL
    location: ApertureLocation = ApertureLocation.ENTRANCE_END
    vertices: ApertureVertices | None = None
    material: str = Field("", description="Material, preferably a chemical formula such as 'Cu'")
    thickness: NonNegativeReal = _field(0.0, "m", "Wall thickness")
    aperture_shifts_with_body: bool = Field(False, description="BodyShiftP misalignments move the aperture")
    aperture_active: bool = True

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        for axis in ("x", "y"):
            low, high = getattr(self, f"{axis}_min"), getattr(self, f"{axis}_max")
            if low is not None and high is not None and not low < high:
                raise ValueError(f"{axis}_min ({low}) must be less than {axis}_max ({high})")
        if self.shape is ApertureShape.VERTICES and self.vertices is None:
            raise ValueError("a VERTICES aperture needs vertices")
        return self


class ControlP(ParameterGroup):
    """Power state and scaling (an EICViBE extension)."""

    is_on: bool = Field(True, description="Element fields are on; off, the element is a drift")
    scale: PositiveReal = Field(1.0, description="Scale factor on element strengths")


class MetaP(ParameterGroup):
    """Descriptive metadata.

    ``ID`` is an external identification such as an inventory number, unrelated
    to the stable ids of the lattice. ``author`` and ``date`` are custom keys,
    which PALS allows in this group.
    """

    alias: str = ""
    ID: str = ""
    label: str = ""
    description: Annotated[str, Field(max_length=1000)] = ""
    location: str = Field("", description="Where the element is, e.g. 'East transfer line'")
    history: tuple[str, ...] = Field((), description="History entries, oldest first")
    author: str = ""
    date: str = ""


class BodyShiftP(ParameterGroup):
    """Misalignment of the element body. Replaces the legacy ``MisalignP``."""

    x_offset: Real = _field(0.0, "m", "Horizontal displacement")
    y_offset: Real = _field(0.0, "m", "Vertical displacement")
    z_offset: Real = _field(0.0, "m", "Longitudinal displacement")
    x_rot: Real = _field(0.0, "rad", "Rotation about x (pitch)")
    y_rot: Real = _field(0.0, "rad", "Rotation about y (yaw)")
    z_rot: Real = _field(0.0, "rad", "Rotation about s (roll)")


_SETTING_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SETTINGS_TABLE = TypeAdapter(dict[str, bool | int | Real | str])


class TrackingP(ParameterGroup):
    """Program-specific tracking settings, one table per program.

    Example: ``TrackingP={"xsuite": {"num_steps": 10}}``. Values are scalars
    that only the named program's adapter reads. A mapping given to
    ``with_groups`` replaces whole program tables. Tracking settings belong
    to definitions and cannot be overridden per occurrence.
    """

    # Tables are validated by _tables rather than a typed __pydantic_extra__,
    # which pydantic keeps in __dict__, where DomainModel pickling drops it.
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "additionalProperties": {
                "type": "object",
                "additionalProperties": {"type": ["boolean", "integer", "number", "string"]},
            }
        },
    )

    @model_validator(mode="before")
    @classmethod
    def _tables(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        tables = {}
        for program, table in data.items():
            for name in (program, *(table if isinstance(table, Mapping) else ())):
                if not isinstance(name, str) or not _SETTING_NAME.fullmatch(name):
                    raise ValueError(f"{name!r} is not a valid program or setting name")
            if table != {}:
                tables[program] = _SETTINGS_TABLE.validate_python(table)
        return tables

    def programs(self) -> dict[str, dict[str, bool | int | float | str]]:
        """Settings by program name."""
        return dict(self.model_extra or {})

    def is_default(self) -> bool:
        return not self.model_extra


class BeamBeamP(ParameterGroup):
    """Opposing beam at the interaction point. Unset optics turn off the hourglass effect."""

    sigma_x: PositiveReal | None = _field(None, "m", "Horizontal rms size")
    sigma_y: PositiveReal | None = _field(None, "m", "Vertical rms size")
    sigma_z: PositiveReal | None = _field(None, "m", "Bunch length")
    alpha_x: Real | None = Field(None, description="Horizontal alpha at the IP")
    beta_x: PositiveReal | None = _field(None, "m", "Horizontal beta at the IP")
    alpha_y: Real | None = Field(None, description="Vertical alpha at the IP")
    beta_y: PositiveReal | None = _field(None, "m", "Vertical beta at the IP")
    charge: Real | None = _field(None, "e", "Opposing particle charge")
    energy: PositiveReal | None = _field(None, "eV", "Opposing beam energy")
    N_particle: NonNegativeReal | None = Field(None, description="Particles per opposing bunch")


def _check_rest_energy(species_ref: str | None, E_tot_ref: float | None) -> None:
    species = SPECIES.get(species_ref) if species_ref else None
    if species is not None and E_tot_ref is not None and E_tot_ref < species.mass:
        raise ValueError(f"E_tot_ref {E_tot_ref} eV is below the {species.name} rest energy")


class ReferenceP(ParameterGroup):
    """Reference particle at a branch entrance (on its ``BeginningEle``)."""

    species_ref: str | None = Field(None, description="Particle species, e.g. 'electron' or 'gold197'")
    pc_ref: PositiveReal | None = _field(None, "eV", "Reference momentum times c")
    E_tot_ref: PositiveReal | None = _field(None, "eV", "Reference total energy")
    time_ref: Real = _field(0.0, "s", "Reference time")

    @model_validator(mode="after")
    def _consistent_energy(self) -> Self:
        species = SPECIES.get(self.species_ref) if self.species_ref else None
        if species is None:
            return self  # unknown species: diagnose() warns that it cannot check
        _check_rest_energy(self.species_ref, self.E_tot_ref)
        if (
            self.pc_ref is not None
            and self.E_tot_ref is not None
            and not energies_agree(self.pc_ref, self.E_tot_ref, species.mass)
        ):
            raise ValueError(
                f"pc_ref {self.pc_ref} eV and E_tot_ref {self.E_tot_ref} eV disagree for {species.name}"
            )
        return self


class ReferenceChangeP(ParameterGroup):
    """Reference adjustments at the exit of a ``Patch``.

    Unset values leave the reference unchanged. Set at most one of
    ``dE_ref``, ``dpc_ref``, ``E_tot_ref`` and ``pc_ref``, and at most one of
    ``dtime_ref`` and ``time_ref``.
    """

    dtime_ref: Real | None = _field(None, "s", "Reference time change")
    dE_ref: Real | None = _field(None, "eV", "Reference energy change")
    dpc_ref: Real | None = _field(None, "eV", "Reference momentum change, times c")
    time_ref: Real | None = _field(None, "s", "New reference time")
    E_tot_ref: PositiveReal | None = _field(None, "eV", "New reference total energy")
    pc_ref: PositiveReal | None = _field(None, "eV", "New reference momentum times c")
    species_ref: str | None = Field(None, description="New reference species")

    @model_validator(mode="after")
    def _one_change_each(self) -> Self:
        for names in (("dE_ref", "dpc_ref", "E_tot_ref", "pc_ref"), ("dtime_ref", "time_ref")):
            given = [name for name in names if getattr(self, name) is not None]
            if len(given) > 1:
                raise ValueError(f"set at most one of {', '.join(names)}; got {', '.join(given)}")
        _check_rest_energy(self.species_ref, self.E_tot_ref)
        return self


class FloorP(ParameterGroup):
    """Placement of a branch entrance in global (floor) coordinates, on its ``BeginningEle``.

    The orientation is W = R_y(theta) R_x(phi) R_z(psi). PALS ignores this
    group on a branch that a fork enters at its beginning element; the fork
    supplies the placement.
    """

    x: Real = _field(0.0, "m", "Global x")
    y: Real = _field(0.0, "m", "Global y")
    z: Real = _field(0.0, "m", "Global z")
    theta: Real = _field(0.0, "rad", "Rotation about the global y axis")
    phi: Real = _field(0.0, "rad", "Rotation about x")
    psi: Real = _field(0.0, "rad", "Rotation about z")


class TwissP(ParameterGroup):
    """Initial Twiss, coupling and dispersion at a branch entrance (PALS ``TwissP``).

    Beta, alpha and phase belong to the normal modes a and b of Sagan and
    Rubin (PRSTAB 2, 074001, 1999); with a zero coupling matrix ``cmat`` they
    are the x and y Twiss parameters. Dispersions are in x and y. ``pz`` is the
    dimensionless momentum coordinate of ``ParticleP``, so a ``*_dpz`` field
    has the unit of the quantity it differentiates. The beta functions have no
    default.
    """

    alpha_a: Real = Field(0.0, description='"a" mode alpha')
    alpha_b: Real = Field(0.0, description='"b" mode alpha')
    beta_a: PositiveReal = _field(unit="m", description='"a" mode beta')
    beta_b: PositiveReal = _field(unit="m", description='"b" mode beta')
    cmat11: Real = Field(0.0, description="Coupling matrix component C[1,1]")
    cmat12: Real = _field(0.0, "m", "Coupling matrix component C[1,2]")
    cmat21: Real = _field(0.0, "1/m", "Coupling matrix component C[2,1]")
    cmat22: Real = Field(0.0, description="Coupling matrix component C[2,2]")
    eta_x: Real = _field(0.0, "m", "x dispersion dx/dpz")
    eta_y: Real = _field(0.0, "m", "y dispersion dy/dpz")
    etap_x: Real = Field(0.0, description="x momentum dispersion dpx/dpz")
    etap_y: Real = Field(0.0, description="y momentum dispersion dpy/dpz")
    deta_x_ds: Real = Field(0.0, description="x dispersion derivative d(eta_x)/ds")
    deta_y_ds: Real = Field(0.0, description="y dispersion derivative d(eta_y)/ds")
    phi_a: Real = _field(0.0, "rad", '"a" mode phase')
    phi_b: Real = _field(0.0, "rad", '"b" mode phase')
    dbeta_a_dpz: Real = _field(0.0, "m", "beta_a dependence on pz")
    dalpha_a_dpz: Real = Field(0.0, description="alpha_a dependence on pz")
    dbeta_b_dpz: Real = _field(0.0, "m", "beta_b dependence on pz")
    dalpha_b_dpz: Real = Field(0.0, description="alpha_b dependence on pz")
    deta_x_dpz: Real = _field(0.0, "m", "eta_x dependence on pz")
    detap_x_dpz: Real = Field(0.0, description="etap_x dependence on pz")
    deta_y_dpz: Real = _field(0.0, "m", "eta_y dependence on pz")
    detap_y_dpz: Real = Field(0.0, description="etap_y dependence on pz")

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        # gamma^2 + det(C) = 1 for the normal-mode decomposition, so gamma is real only if det(C) <= 1.
        det = self.cmat11 * self.cmat22 - self.cmat12 * self.cmat21
        if det > 1.0:
            raise ValueError(f"coupling matrix determinant {det} exceeds 1, so gamma would be imaginary")
        # etap and deta_ds are related through the orbit in ParticleP; setting both can contradict it.
        for axis in ("x", "y"):
            if getattr(self, f"etap_{axis}") != 0.0 and getattr(self, f"deta_{axis}_ds") != 0.0:
                raise ValueError(f"set either etap_{axis} or deta_{axis}_ds, not both")
        return self


class ParticleP(ParameterGroup):
    """Initial phase-space coordinates and optional Gaussian second moments."""

    x: Real = 0.0
    px: Real = 0.0
    y: Real = 0.0
    py: Real = 0.0
    z: Real = 0.0
    pz: Real = 0.0
    spin_x: Real = 0.0
    spin_y: Real = 0.0
    spin_z: Real = 0.0
    sigma_xx: NonNegativeReal | None = None
    sigma_xpx: Real | None = None
    sigma_xy: Real | None = None
    sigma_xpy: Real | None = None
    sigma_xz: Real | None = None
    sigma_xpz: Real | None = None
    sigma_pxpx: NonNegativeReal | None = None
    sigma_pxy: Real | None = None
    sigma_pxpy: Real | None = None
    sigma_pxz: Real | None = None
    sigma_pxpz: Real | None = None
    sigma_yy: NonNegativeReal | None = None
    sigma_ypy: Real | None = None
    sigma_yz: Real | None = None
    sigma_ypz: Real | None = None
    sigma_pypy: NonNegativeReal | None = None
    sigma_pyz: Real | None = None
    sigma_pypz: Real | None = None
    sigma_zz: NonNegativeReal | None = None
    sigma_zpz: Real | None = None
    sigma_pzpz: NonNegativeReal | None = None


class PatchP(ParameterGroup):
    """Exit frame of a ``Patch`` relative to its entrance frame.

    The exit frame is the entrance frame shifted by the offsets and rotated
    by S = R_y(y_rot) R_x(x_rot) R_z(z_rot), right-handed rotations as in
    PALS; a bend of angle a turns the frame by y_rot = -a. Patches here are
    rigid and have zero length (PALS ``flexible: false``,
    ``user_sets_length: true``). Reference time and energy changes are in
    ``ReferenceChangeP``.
    """

    x_offset: Real = _field(0.0, "m", "Translation along x")
    y_offset: Real = _field(0.0, "m", "Translation along y")
    z_offset: Real = _field(0.0, "m", "Translation along z")
    x_rot: Real = _field(0.0, "rad", "Rotation about x")
    y_rot: Real = _field(0.0, "rad", "Rotation about y; a bend of angle a is y_rot = -a")
    z_rot: Real = _field(0.0, "rad", "Rotation about z")


#: All parameter group types by name.
GROUP_TYPES: dict[str, type[ParameterGroup]] = {
    group.group_name: group
    for group in (
        MagneticMultipoleP,
        BendP,
        SolenoidP,
        RFP,
        ApertureP,
        ControlP,
        MetaP,
        BodyShiftP,
        TrackingP,
        BeamBeamP,
        ReferenceP,
        ReferenceChangeP,
        FloorP,
        TwissP,
        ParticleP,
        PatchP,
    )
}


@dataclass(frozen=True, slots=True)
class ParameterInfo:
    """Machine-readable description of one parameter, for clients and registries."""

    group: str
    name: str
    unit: str | None
    description: str | None
    required: bool
    default: Any

    @property
    def path(self) -> str:
        return f"{self.group}.{self.name}"


def parameter_info(group: str, parameter: str) -> ParameterInfo:
    try:
        field = GROUP_TYPES[group].model_fields[parameter]
    except KeyError:
        raise KeyError(f"no parameter {group}.{parameter}") from None
    extra = field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}
    unit = extra.get("unit")
    return ParameterInfo(
        group=group,
        name=parameter,
        unit=unit if isinstance(unit, str) else None,
        description=field.description,
        required=field.is_required(),
        default=None if field.is_required() else field.default,
    )


def parameters_of(group: str) -> tuple[ParameterInfo, ...]:
    """Every parameter of a group, in declaration order."""
    return tuple(parameter_info(group, name) for name in GROUP_TYPES[group].model_fields)


@cache
def parameter_adapter(group: str, parameter: str) -> TypeAdapter[Any]:
    """Validator for one field of one group, with that field's type and bounds.

    Cross-field rules (e.g. ``x_min < x_max``) are not applied here; they
    run when the value is merged into a whole group.
    """
    try:
        field = GROUP_TYPES[group].model_fields[parameter]
    except KeyError:
        if group not in GROUP_TYPES:
            raise KeyError(f"unknown parameter group {group!r}") from None
        if group == TrackingP.group_name:
            raise KeyError("TrackingP settings belong to definitions and cannot be overridden") from None
        raise KeyError(f"{group} has no parameter {parameter!r}") from None
    return TypeAdapter(Annotated[field.annotation, *field.metadata])
