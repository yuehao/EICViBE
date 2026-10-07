"""Plausibility checks that warn but never reject.

A lattice that constructs is structurally sound. :func:`diagnose` looks for
values that are legal but probably mistakes: a degree-valued RF phase, a
per-length strength on a zero-length magnet, a magnitude far outside
practice. Importers report these next to their own findings.

Tunable settings (quadrupole strengths, kicks, cavity voltages) are checked
against each element's operating limits from
:class:`~eicvibe.domain.lattice.limits.ParameterLimits`. Everything else is
checked against the global :data:`SANITY_THRESHOLDS`. The legacy models
raised on several of these magnitudes; here they are warnings.
"""

import math
from dataclasses import dataclass

from .conventions import SPECIES
from .elements import ElementDefinition, ElementKind
from .limits import ParameterLimits, placeholder_limits, tunable_parameters
from .parameters import MAX_MULTIPOLE_ORDER, ApertureShape, parameter_info

#: Thresholds for values that are not tunable settings, ported from the legacy
#: validators. The multipole entries apply where the order is not tunable,
#: e.g. a gradient on a bend.
SANITY_THRESHOLDS: dict[str, float] = {
    "length": 10_000.0,  # m
    "MagneticMultipoleP.Kn1": 1e3,  # 1/m^2
    "MagneticMultipoleP.Kn2": 1e4,  # 1/m^3
    "MagneticMultipoleP.Kn3": 1e5,  # 1/m^4
    "angle": 2 * math.pi,  # rad; tilts, bend angles, rotations
    "edge": math.pi,  # rad
    "RFP.frequency.min": 1e6,  # Hz
    "RFP.frequency.max": 1e12,  # Hz
    "RFP.phase": 2 * math.pi,  # rad; larger values look like degrees
    "ApertureP.limit": 1.0,  # m
    "ControlP.scale": 100.0,
    "BodyShiftP.offset": 10.0,  # m
    "BeamBeamP.sigma_xy": 0.01,  # m
    "BeamBeamP.sigma_z": 1.0,  # m
}

_THICK_STRENGTH_KINDS = {
    ElementKind.QUADRUPOLE: ("MagneticMultipoleP", ("Kn1", "Ks1")),
    ElementKind.SEXTUPOLE: ("MagneticMultipoleP", ("Kn2", "Ks2")),
    ElementKind.OCTUPOLE: ("MagneticMultipoleP", ("Kn3", "Ks3")),
    ElementKind.KICKER: ("MagneticMultipoleP", ("Kn0", "Ks0")),
    ElementKind.MULTIPOLE: (
        "MagneticMultipoleP",
        tuple(f"K{part}{n}" for n in range(MAX_MULTIPOLE_ORDER + 1) for part in "ns"),
    ),
    ElementKind.SOLENOID: ("SolenoidP", ("Ksol", "Bsol")),
}


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """One warning about a definition, occurrence or branch."""

    code: str
    message: str
    subject: str
    """Id of the definition, occurrence or branch concerned."""

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


def diagnose(lattice, limits: ParameterLimits | None = None) -> tuple[Diagnostic, ...]:
    """Warnings for every definition, every overridden occurrence, and every branch.

    ``limits`` defaults to :func:`~eicvibe.domain.lattice.limits.placeholder_limits`.
    Definitions are checked once, not once per placement. An occurrence is
    checked separately when its overrides change its effective element or
    when it has limits of its own.
    """
    limits = placeholder_limits(lattice) if limits is None else limits
    own_limits = limits.targets()
    found: list[Diagnostic] = []
    for definition in lattice.elements:
        where = f"element {definition.name!r}"
        found += _check_element(definition, definition.id, where)
        found += _check_limits(definition, limits, (definition.id,), definition.id, where)
    for layout in lattice.topology.layouts.values():
        if not layout.placements:
            found.append(Diagnostic("empty-branch", f"branch {layout.branch.name!r} is empty", layout.id))
        elif (first := layout.placements[0]).element.FloorP is not None and any(
            edge.to_occurrence in (None, first.id) for edge in lattice.forks_into(layout.id)
        ):
            message = f"branch {layout.branch.name!r}: a fork enters at its start, so its FloorP is ignored"
            found.append(Diagnostic("floor-ignored", message, first.id))
        for placement in layout.placements:
            where = f"occurrence {layout.branch.name}/{placement.name}"
            if placement.occurrence.overrides:
                found += _check_element(placement.element, placement.id, where)
            if placement.occurrence.overrides or placement.id in own_limits:
                targets = (placement.id, placement.definition.id)
                found += _check_limits(placement.element, limits, targets, placement.id, where)
    return tuple(found)


def _check_limits(
    e: ElementDefinition, limits: ParameterLimits, targets: tuple[str, ...], subject: str, where: str
) -> list[Diagnostic]:
    found = []
    for path in tunable_parameters(e.kind):
        group, _, name = path.partition(".")
        value = e.parameter(group, name)
        limit = limits.limit_for(e.kind, path, targets)
        if value is None or limit is None or limit.contains(value):
            continue
        unit = parameter_info(group, name).unit or ""
        message = f"{where}: {path} = {value:g} {unit} is outside its limit {limit}".replace("  ", " ")
        found.append(Diagnostic("limit-exceeded", message, subject))
    return found


def _check_element(e: ElementDefinition, subject: str, where: str) -> list[Diagnostic]:
    found: list[Diagnostic] = []

    def warn(code: str, message: str) -> None:
        found.append(Diagnostic(code, f"{where}: {message}", subject))

    def over(value: float | None, threshold: str) -> bool:
        return value is not None and abs(value) > SANITY_THRESHOLDS[threshold]

    tunable = set(tunable_parameters(e.kind))

    if e.length > SANITY_THRESHOLDS["length"]:
        warn("length-large", f"length {e.length} m is implausibly long")

    if (spec := _THICK_STRENGTH_KINDS.get(e.kind)) and e.length == 0.0:
        group, names = spec
        if any(e.parameter(group, n) != 0.0 for n in names):
            warn(
                "thin-thick-strength",
                f"zero-length {e.kind.value} has a per-length strength, so no effect; use KnNL/KsNL",
            )

    if (m := e.MagneticMultipoleP) is not None:
        for name in ("Kn1", "Kn2", "Kn3"):
            skew = "Ks" + name[2:]
            for n in (name, skew):
                if f"MagneticMultipoleP.{n}" not in tunable and over(
                    getattr(m, n), f"MagneticMultipoleP.{name}"
                ):
                    warn("strength-large", f"MagneticMultipoleP.{n} = {getattr(m, n)} is implausibly large")
        for n in range(MAX_MULTIPOLE_ORDER + 1):
            if over(tilt := getattr(m, f"tilt{n}"), "angle"):
                warn("angle-large", f"MagneticMultipoleP.tilt{n} = {tilt} rad exceeds 2 pi")

    if (b := e.BendP) is not None:
        if over(b.angle_ref, "angle"):
            warn("angle-large", f"BendP.angle_ref = {b.angle_ref} rad exceeds 2 pi")
        for n in ("e1", "e2", "e1_rect", "e2_rect"):
            if over(getattr(b, n), "edge"):
                warn("angle-large", f"BendP.{n} = {getattr(b, n)} rad exceeds pi")

    if (rf := e.RFP) is not None:
        if rf.frequency is not None and not (
            SANITY_THRESHOLDS["RFP.frequency.min"] <= rf.frequency <= SANITY_THRESHOLDS["RFP.frequency.max"]
        ):
            warn("rf-frequency-range", f"RFP.frequency = {rf.frequency} Hz is outside 1 MHz - 1 THz")
        for n in ("phase", "multipass_phase"):
            if over(getattr(rf, n), "RFP.phase"):
                message = f"RFP.{n} = {getattr(rf, n)} exceeds 2 pi; phases are in rad, not degrees"
                warn("rf-phase-units", message)
        if (rf.voltage or rf.gradient) and rf.frequency is None and rf.harmon is None:
            warn("rf-frequency-missing", "an active cavity needs frequency or harmon")

    if (a := e.ApertureP) is not None:
        for n in ("x_min", "x_max", "y_min", "y_max"):
            if over(getattr(a, n), "ApertureP.limit"):
                warn("aperture-large", f"ApertureP.{n} = {getattr(a, n)} m exceeds 1 m")
        limits = (a.x_min, a.x_max, a.y_min, a.y_max)
        partial = None in limits and any(v is not None for v in limits)
        if a.shape is ApertureShape.ELLIPTICAL and a.aperture_active and partial:
            warn("aperture-inactive", "an ELLIPTICAL aperture needs all four limits; this one is inactive")

    if (c := e.ControlP) is not None and c.scale > SANITY_THRESHOLDS["ControlP.scale"]:
        warn("scale-large", f"ControlP.scale = {c.scale} exceeds 100")

    if (s := e.BodyShiftP) is not None:
        for n in ("x_offset", "y_offset", "z_offset"):
            if over(getattr(s, n), "BodyShiftP.offset"):
                warn("misalignment-large", f"BodyShiftP.{n} = {getattr(s, n)} m exceeds 10 m")
        for n in ("x_rot", "y_rot", "z_rot"):
            if over(getattr(s, n), "angle"):
                warn("angle-large", f"BodyShiftP.{n} = {getattr(s, n)} rad exceeds 2 pi")

    if (bb := e.BeamBeamP) is not None:
        for n in ("sigma_x", "sigma_y"):
            if over(getattr(bb, n), "BeamBeamP.sigma_xy"):
                warn("beam-size-large", f"BeamBeamP.{n} = {getattr(bb, n)} m exceeds 10 mm")
        if over(bb.sigma_z, "BeamBeamP.sigma_z"):
            warn("beam-size-large", f"BeamBeamP.sigma_z = {bb.sigma_z} m exceeds 1 m")

    if (r := e.ReferenceP) is not None:
        if r.species_ref is None:
            if r.pc_ref is not None or r.E_tot_ref is not None:
                warn("species-missing", "reference energy without a species")
        elif r.species_ref not in SPECIES:
            warn(
                "species-unknown",
                f"species {r.species_ref!r} is not in the species table; its energy was not checked",
            )

    return found
