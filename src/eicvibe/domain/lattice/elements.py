"""Element kinds and element definitions.

An :class:`ElementDefinition` is a reusable, parameter-bearing element: a
magnet family, a drift length, a monitor type. Branches place definitions as
:class:`~eicvibe.domain.lattice.model.Occurrence` objects; placement-specific
values (a trim, a misalignment) are occurrence overrides, not new definitions.
"""

from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Self

from pydantic import Field, field_validator, model_validator

from . import parameters as p
from ._base import DomainModel, Id, Label, NonNegativeReal, Tags
from .parameters import GROUP_TYPES, ParameterGroup

if TYPE_CHECKING:
    from .model import ParameterOverride


class ElementKind(StrEnum):
    """PALS element kinds supported by the description."""

    DRIFT = "Drift"
    BEND = "Bend"
    MULTIPOLE = "Multipole"
    QUADRUPOLE = "Quadrupole"
    SEXTUPOLE = "Sextupole"
    OCTUPOLE = "Octupole"
    SOLENOID = "Solenoid"
    KICKER = "Kicker"
    RFCAVITY = "RFCavity"
    CRABCAVITY = "CrabCavity"
    EGUN = "EGun"
    INSTRUMENT = "Instrument"
    MARKER = "Marker"
    BEAMBEAM = "BeamBeam"
    BEGINNING_ELE = "BeginningEle"
    FORK = "Fork"
    PATCH = "Patch"
    REFERENCE_CHANGE = "ReferenceChange"


#: Kind names of the legacy model and of engines such as MAD-X, with what to use instead.
#: Adapters translate PALS kinds back to engine kinds.
LEGACY_KIND_NAMES = {
    "SBend": "'Bend'",
    "RBend": "'Bend'; a rectangular bend sets e1 = e2 = angle_ref/2",
    "Monitor": "'Instrument'",
}

_BODY = ("ApertureP", "BodyShiftP", "MetaP", "TrackingP")
_MAGNET = (*_BODY, "MagneticMultipoleP")

#: Parameter groups each kind may carry: the PALS kind lists without the groups
#: not adopted here (ElectricMultipoleP, ForkP, ForkFromP, CoordinateSetP) and
#: without output groups, with FloorP only on BeginningEle, and with the
#: ControlP extension on every kind.
ALLOWED_GROUPS: dict[ElementKind, frozenset[str]] = {
    kind: frozenset({"ControlP", *groups})
    for kind, groups in {
        ElementKind.DRIFT: _BODY,
        ElementKind.BEND: (*_MAGNET, "BendP"),
        ElementKind.MULTIPOLE: _MAGNET,
        ElementKind.QUADRUPOLE: _MAGNET,
        ElementKind.SEXTUPOLE: _MAGNET,
        ElementKind.OCTUPOLE: _MAGNET,
        ElementKind.SOLENOID: (*_MAGNET, "SolenoidP"),
        ElementKind.KICKER: _MAGNET,
        ElementKind.RFCAVITY: (*_MAGNET, "RFP", "SolenoidP"),
        ElementKind.CRABCAVITY: (*_MAGNET, "RFP"),  # PALS omits RFP from the list but uses it in its example
        # PALS's EGun is under construction: RFP gives its voltage, as Bmad's PALS writer does (ADR 0012).
        ElementKind.EGUN: (*_MAGNET, "RFP"),
        ElementKind.INSTRUMENT: _MAGNET,
        ElementKind.MARKER: ("ApertureP", "BodyShiftP", "MetaP"),
        ElementKind.BEAMBEAM: (*_BODY, "BeamBeamP"),
        ElementKind.BEGINNING_ELE: (*_BODY, "FloorP", "ParticleP", "ReferenceP", "TwissP"),
        ElementKind.FORK: _BODY,
        # The PALS list omits ReferenceChangeP, but the PatchP text uses it for time and energy changes.
        ElementKind.PATCH: ("ApertureP", "MetaP", "TrackingP", "PatchP", "ReferenceChangeP"),
        ElementKind.REFERENCE_CHANGE: ("MetaP", "TrackingP", "ReferenceChangeP"),
    }.items()
}

#: Kinds that mark or route rather than act on the beam. They have zero length
#: and cannot be a traversal of a shared physical component.
CONTROL_KINDS = frozenset(
    {
        ElementKind.MARKER,
        ElementKind.BEGINNING_ELE,
        ElementKind.FORK,
        ElementKind.PATCH,
        ElementKind.REFERENCE_CHANGE,
    }
)
#: PALS also requires zero length for BeamBeam.
ZERO_LENGTH_KINDS = CONTROL_KINDS | {ElementKind.BEAMBEAM}


class ElementDefinition(DomainModel):
    """A reusable element: kind, length and parameter groups.

    Each group is an optional field named after its type. Groups not allowed
    for the kind are rejected; groups holding only defaults are dropped.
    """

    id: Id
    name: Label
    kind: ElementKind
    length: NonNegativeReal = Field(
        0.0, description="Path length; arc length for bends", json_schema_extra={"unit": "m"}
    )
    tags: Tags = Field((), description="Labels shared by every placement, e.g. 'HCOR'")

    MagneticMultipoleP: p.MagneticMultipoleP | None = None
    BendP: p.BendP | None = None
    SolenoidP: p.SolenoidP | None = None
    RFP: p.RFP | None = None
    BeamBeamP: p.BeamBeamP | None = None
    PatchP: p.PatchP | None = None
    ReferenceP: p.ReferenceP | None = None
    ReferenceChangeP: p.ReferenceChangeP | None = None
    FloorP: p.FloorP | None = None
    TwissP: p.TwissP | None = None
    ParticleP: p.ParticleP | None = None
    ApertureP: p.ApertureP | None = None
    ControlP: p.ControlP | None = None
    MetaP: p.MetaP | None = None
    BodyShiftP: p.BodyShiftP | None = None
    TrackingP: p.TrackingP | None = None

    @field_validator("kind", mode="before")
    @classmethod
    def _explain_legacy_kind(cls, value: Any) -> Any:
        if isinstance(value, str) and value in LEGACY_KIND_NAMES:
            raise ValueError(f"{value!r} is not a PALS element kind; use {LEGACY_KIND_NAMES[value]}")
        return value

    @field_validator(*GROUP_TYPES, mode="after")
    @classmethod
    def _drop_default_group(cls, group: ParameterGroup | None) -> ParameterGroup | None:
        return None if group is not None and group.is_default() else group

    @model_validator(mode="after")
    def _kind_rules(self) -> Self:
        allowed = ALLOWED_GROUPS[self.kind]
        present = [name for name in GROUP_TYPES if getattr(self, name) is not None]
        rejected = [name for name in present if name not in allowed]
        if rejected:
            raise ValueError(
                f"{self.kind.value} does not accept {', '.join(rejected)}; "
                f"allowed: {', '.join(sorted(allowed))}"
            )
        if self.kind in ZERO_LENGTH_KINDS and self.length != 0.0:
            raise ValueError(f"{self.kind.value} must have zero length")
        if self.kind is ElementKind.DRIFT and self.length == 0.0:
            raise ValueError("Drift must have positive length")
        return self

    def groups(self) -> dict[str, ParameterGroup]:
        """The groups that are set, by group name."""
        return {name: g for name in GROUP_TYPES if (g := getattr(self, name)) is not None}

    def parameter(self, group: str, name: str) -> Any:
        """Value of ``group.name``, or the field default when the group is unset.

        Returns ``None`` for a required field of an unset group (e.g. an absent
        ``TwissP.beta_a``). Raises ``KeyError`` if this kind cannot carry the
        group or the group has no such field.
        """
        if group not in ALLOWED_GROUPS[self.kind]:
            raise KeyError(f"{self.kind.value} cannot carry {group}")
        model = GROUP_TYPES[group]
        if name not in model.model_fields:
            raise KeyError(f"{group} has no parameter {name!r}")
        value = getattr(self, group)
        if value is not None:
            return getattr(value, name)
        field = model.model_fields[name]
        return None if field.is_required() else field.default

    def with_groups(self, **changes: ParameterGroup | Mapping[str, Any] | None) -> Self:
        """A validated copy with groups merged in.

        A mapping updates the named fields of the existing group (or starts a
        new one); a group instance replaces it; ``None`` removes it.
        """
        data = self.model_dump()
        for name, change in changes.items():
            if name not in GROUP_TYPES:
                raise KeyError(f"unknown parameter group {name!r}")
            if change is None or isinstance(change, ParameterGroup):
                data[name] = change
            else:
                data[name] = {**(data[name] or {}), **change}
        return type(self).model_validate(data)

    def with_overrides(self, overrides: Iterable[ParameterOverride]) -> Self:
        """The effective element for an occurrence carrying ``overrides``."""
        changes: dict[str, dict[str, Any]] = {}
        for override in overrides:
            changes.setdefault(override.group, {})[override.parameter] = override.value
        return self.with_groups(**changes) if changes else self

    @property
    def is_control(self) -> bool:
        """True for markers, forks, patches, reference changes and beginning elements."""
        return self.kind in CONTROL_KINDS
