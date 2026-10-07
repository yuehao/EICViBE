"""Operating limits of tunable element parameters.

Limits belong to the hardware (a power supply, a cavity), not to a lattice
revision, so they live in a :class:`ParameterLimits` table next to the
lattice rather than inside it. Each entry is keyed by the stable id of an
element definition (a family on one supply) or of an occurrence (one magnet
with its own trim supply) and a parameter path such as
``MagneticMultipoleP.Kn1``.

Today every limit is a placeholder: :func:`placeholder_limits` gives each
element the values of :data:`PLACEHOLDER_LIMITS` for its kind. Later a
control-system provider will replace them entry by entry with
``source=EPICS``, reading the drive limits of each setting PV (DRVL/DRVH, or
``control.limitLow``/``limitHigh`` over pvAccess). EPICS limits are in
engineering units (e.g. A); that provider must convert them to the
parameter's units through the magnet calibration and, for normalized
strengths, the reference rigidity. A normalized limit therefore holds at one
beam energy only.
"""

from collections.abc import Iterable, Sequence
from enum import StrEnum
from functools import cached_property
from typing import Self

from pydantic import Field, field_validator, model_validator

from ._base import DomainModel, Id, Real
from .elements import ElementKind
from .parameters import parameter_info


class LimitSource(StrEnum):
    PLACEHOLDER = "placeholder"
    """Generic value for the element kind; not a property of the real device."""
    MANUAL = "manual"
    """Entered by a person, e.g. from a magnet specification."""
    EPICS = "epics"
    """Read from the control system."""


class Limit(DomainModel):
    """Allowed range of one setting, in the parameter's units. ``None`` is unbounded."""

    minimum: Real | None = None
    maximum: Real | None = None
    source: LimitSource = LimitSource.PLACEHOLDER
    pv: str | None = Field(default=None, description="Setting PV the limit comes from, or will come from")

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError(f"minimum {self.minimum} exceeds maximum {self.maximum}")
        return self

    def contains(self, value: float) -> bool:
        return (self.minimum is None or value >= self.minimum) and (
            self.maximum is None or value <= self.maximum
        )

    def __str__(self) -> str:
        low = "-inf" if self.minimum is None else f"{self.minimum:g}"
        high = "inf" if self.maximum is None else f"{self.maximum:g}"
        return f"[{low}, {high}] ({self.source.value})"


def _symmetric(bound: float) -> Limit:
    return Limit(minimum=-bound, maximum=bound)


def _multipole(order: int, bound: float) -> dict[str, Limit]:
    """Normal and skew strengths of one order, per length and integrated, all within ``bound``."""
    names = (f"Kn{order}", f"Ks{order}", f"Kn{order}L", f"Ks{order}L")
    return {f"MagneticMultipoleP.{name}": _symmetric(bound) for name in names}


#: Tunable parameters of each element kind, with placeholder limits (the
#: legacy plausibility thresholds, reused for the integrated strengths).
#: Kinds not listed have no tunable setting here; a bend's field follows the
#: reference energy.
PLACEHOLDER_LIMITS: dict[ElementKind, dict[str, Limit]] = {
    ElementKind.QUADRUPOLE: _multipole(1, 1e3),
    ElementKind.SEXTUPOLE: _multipole(2, 1e4),
    ElementKind.OCTUPOLE: _multipole(3, 1e5),
    ElementKind.SOLENOID: {"SolenoidP.Ksol": _symmetric(1e2)},
    ElementKind.KICKER: _multipole(0, 0.1),
    ElementKind.RFCAVITY: {"RFP.voltage": Limit(minimum=0.0, maximum=1e9)},
    ElementKind.CRABCAVITY: {"RFP.voltage": Limit(minimum=0.0, maximum=1e9)},
}


def tunable_parameters(kind: ElementKind) -> tuple[str, ...]:
    """Parameter paths that have operating limits for this kind."""
    return tuple(PLACEHOLDER_LIMITS.get(kind, {}))


class LimitEntry(DomainModel):
    """The limit of one parameter of one element definition or occurrence."""

    target: Id = Field(description="Element definition id or occurrence id")
    parameter: str = Field(description="Parameter path, e.g. 'MagneticMultipoleP.Kn1'")
    limit: Limit

    @field_validator("parameter")
    @classmethod
    def _known_parameter(cls, parameter: str) -> str:
        group, _, name = parameter.partition(".")
        try:
            parameter_info(group, name)
        except KeyError as error:
            raise ValueError(error.args[0]) from None
        return parameter


class ParameterLimits(DomainModel):
    """Immutable table of limits; derive changed tables with :meth:`updated`."""

    entries: tuple[LimitEntry, ...] = ()

    @field_validator("entries")
    @classmethod
    def _unique(cls, entries: tuple[LimitEntry, ...]) -> tuple[LimitEntry, ...]:
        ordered = tuple(sorted(entries, key=lambda e: (e.target, e.parameter)))
        keys = [(e.target, e.parameter) for e in ordered]
        if len(set(keys)) != len(keys):
            raise ValueError("each (target, parameter) may have only one limit")
        return ordered

    @cached_property
    def lookup(self) -> dict[tuple[str, str], Limit]:
        return {(e.target, e.parameter): e.limit for e in self.entries}

    def get(self, target: str, parameter: str) -> Limit | None:
        """The entry for exactly this target, or None."""
        return self.lookup.get((target, parameter))

    def limit_for(self, kind: ElementKind, parameter: str, targets: Sequence[str]) -> Limit | None:
        """First entry among ``targets`` (most specific first), else the kind placeholder."""
        for target in targets:
            if (limit := self.lookup.get((target, parameter))) is not None:
                return limit
        return PLACEHOLDER_LIMITS.get(kind, {}).get(parameter)

    def targets(self) -> frozenset[str]:
        return frozenset(e.target for e in self.entries)

    def updated(self, entries: Iterable[LimitEntry]) -> Self:
        """A copy in which ``entries`` add to or replace existing entries."""
        merged = {(e.target, e.parameter): e for e in self.entries}
        merged.update(((e.target, e.parameter), e) for e in entries)
        return type(self)(entries=tuple(merged.values()))

    def with_limit(
        self,
        target: str,
        parameter: str,
        minimum: float | None = None,
        maximum: float | None = None,
        *,
        source: LimitSource = LimitSource.MANUAL,
        pv: str | None = None,
    ) -> Self:
        """A copy with one limit set."""
        limit = Limit(minimum=minimum, maximum=maximum, source=source, pv=pv)
        return self.updated([LimitEntry(target=target, parameter=parameter, limit=limit)])


def placeholder_limits(lattice) -> ParameterLimits:
    """One placeholder entry per tunable parameter of every element definition."""
    return ParameterLimits(
        entries=tuple(
            LimitEntry(target=definition.id, parameter=parameter, limit=limit)
            for definition in lattice.elements
            for parameter, limit in PLACEHOLDER_LIMITS.get(definition.kind, {}).items()
        )
    )
