"""The operations, declared once for every place that runs them (ADR 0017).

Each declaration names an operation and says what it does; gives its parameters, a pydantic
model with which the virtual accelerator validates a request and whose JSON Schema (with the
``x-form`` hints of :func:`form`) the viewer builds its form from and an agent's tools will
describe; says what verifies it once applied (the orbit at the BPMs, or the tune monitor) and on
which machines it runs; and calls the function of :mod:`eicvibe.operations` that computes it. A
new operation is added here, beside its function: the API, the viewer's form and the
verification follow from the declaration.

What a proposal predicts is what verifies it: an operation verified by the orbit predicts the
BPMs (``bpms``) and either the orbit ``before`` and ``after`` at them or its ``change`` there
(mm); one verified by the tunes predicts the ``tunes`` before and after.

This module imports no engine, so that the API can be made from it: an operation's function is
imported when the operation runs.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .model import MachineModel
    from .proposal import Proposal


def form(**hints: Any) -> dict[str, Any]:
    """How the viewer's form shows a parameter, as its JSON Schema's ``x-form``: ``prefix`` and ``unit``
    (text before and after its input), ``labels`` (an enumeration's values' names), ``placeholder``,
    ``when`` (the other parameters' values it is shown for) and ``advanced`` (left out of the form;
    the API and tools take it)."""
    return {"x-form": hints}


class Measured(Protocol):
    """What the machine measured, for an operation that starts from it."""

    def orbit(self, plane: str) -> Mapping[str, float]:
        """The BPMs' latest slow orbit in ``plane``: m, by BPM name."""
        ...


@dataclass(frozen=True)
class Operation:
    """An operation as everything that runs it sees it."""

    name: str
    label: str
    summary: str
    parameters: type[BaseModel]
    propose: Callable[[MachineModel, Any, Measured], Proposal]
    """Its proposal on a model of the machine as set, from its parameters and what the machine measured."""
    verified_by: Literal["orbit", "tunes"]
    machines: frozenset[str] = frozenset({"ring"})
    """The kinds of machine it runs on: ring, route."""


PLANES = form(labels={"x": "horizontal", "y": "vertical"})


class OrbitCorrection(BaseModel):
    """The orbit corrected in one plane, from the BPMs' latest slow orbit or from the model's own."""

    plane: Literal["x", "y"] = Field("x", title="Plane", json_schema_extra=PLANES)
    source: Literal["measured", "model"] = Field(
        "measured",
        title="Orbit from",
        description="The BPMs' latest slow orbit, or the model's own",
        json_schema_extra=form(labels={"measured": "from the BPMs", "model": "from the model"}),
    )
    method: Literal["svd", "micado"] = Field(
        "svd", title="Method", json_schema_extra=form(labels={"svd": "SVD, damped", "micado": "MICADO"})
    )
    damping: float = Field(
        0.01,
        gt=0,
        title="Damping",
        description="SVD: Tikhonov damping, of the largest singular value",
        json_schema_extra=form(advanced=True),
    )
    count: int = Field(
        3,
        ge=1,
        le=50,
        title="Correctors",
        description="MICADO: correctors to use",
        json_schema_extra=form(unit="correctors", when={"method": "micado"}),
    )
    correctors: list[str] | None = Field(
        None,
        title="Corrector channels",
        description="Only these corrector channels",
        json_schema_extra=form(advanced=True),
    )


class LocalBump(BaseModel):
    """A closed bump of the orbit at a point: an offset, and an angle if given."""

    plane: Literal["x", "y"] = Field("x", title="Plane", json_schema_extra=PLANES)
    at: str = Field(
        title="Bump at",
        description="A BPM's name (or an element's)",
        json_schema_extra=form(prefix="at", placeholder="a BPM's name"),
    )
    amplitude_mm: float = Field(
        title="Amplitude",
        description="The offset there, mm",
        examples=[1.0],
        json_schema_extra=form(unit="mm"),
    )
    angle_mrad: float | None = Field(
        None,
        title="Angle",
        description="The angle there, mrad: leave it out for an offset alone (three correctors); an angle "
        "takes four",
        json_schema_extra=form(unit="mrad", placeholder="angle"),
    )
    correctors: list[str] | None = Field(
        None,
        title="Corrector channels",
        description="Only these corrector channels",
        json_schema_extra=form(advanced=True),
    )


class TuneMove(BaseModel):
    """A change of the tunes, with the least beta-beating and dispersion change."""

    dqx: float = Field(
        0.0,
        title="Change of Qx",
        description="The change of the horizontal tune",
        examples=[0.01],
        json_schema_extra=form(prefix="ΔQx"),
    )
    dqy: float = Field(
        0.0,
        title="Change of Qy",
        description="The change of the vertical tune",
        json_schema_extra=form(prefix="ΔQy"),
    )
    quadrupoles: list[str] | None = Field(
        None,
        title="Quadrupole channels",
        description="Only these quadrupole channels",
        json_schema_extra=form(advanced=True),
    )


def _orbit_correction(model: MachineModel, p: OrbitCorrection, measured: Measured) -> Proposal:
    from .orbit import correct_orbit

    orbit = dict(measured.orbit(p.plane)) if p.source == "measured" else None
    return correct_orbit(
        model, p.plane, orbit, method=p.method, damping=p.damping, count=p.count, correctors=p.correctors
    )


def _local_bump(model: MachineModel, p: LocalBump, measured: Measured) -> Proposal:
    from .bump import local_bump

    angle = None if p.angle_mrad is None else p.angle_mrad * 1e-3
    return local_bump(model, p.plane, p.at, p.amplitude_mm * 1e-3, angle, correctors=p.correctors)


def _tune_move(model: MachineModel, p: TuneMove, measured: Measured) -> Proposal:
    from .tune import move_tunes

    return move_tunes(model, p.dqx, p.dqy, quadrupoles=p.quadrupoles)


#: The operations, in the order the viewer offers them.
OPERATIONS: dict[str, Operation] = {
    o.name: o
    for o in (
        Operation(
            "orbit_correction",
            "Orbit correction",
            "SVD (damped) or MICADO on the model's orbit response matrix, from the BPMs' latest slow orbit",
            OrbitCorrection,
            _orbit_correction,
            "orbit",
        ),
        Operation(
            "local_bump",
            "Local bump",
            "A closed bump: three correctors for an offset, four for an offset and an angle",
            LocalBump,
            _local_bump,
            "orbit",
        ),
        Operation(
            "tune_move",
            "Tune move",
            "The quadrupole knob that moves the tunes with the least beta-beating and dispersion change",
            TuneMove,
            _tune_move,
            "tunes",
        ),
    )
}


def operation(name: str | None) -> Operation:
    """The operation of this name."""
    found = OPERATIONS.get(name or "")
    if found is None:
        raise ValueError(f"no operation {name!r}; operations: {', '.join(OPERATIONS)}")
    return found
