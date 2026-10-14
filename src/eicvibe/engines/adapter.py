"""What every engine adapter is: the contract's checks in one place, the physics the engine's own.

:class:`~.model.Engine` and :class:`~.model.Session` describe the contract;
an adapter (Xsuite, TrackPad, any engine to come) implements it by subclassing
:class:`EngineAdapter` and :class:`EngineSession`. Their public methods check a
request the same way for every engine, against the engine's declared
capabilities and the branch, before calling the engine's own computation (the
underscored methods), so that engines refuse the same requests for the same
reasons and a new engine cannot skip a check. What the engine computes is held
to the same standard by the conformance tests (``tests/engines/
test_conformance.py``), which every registered engine passes (ADR 0006).
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

import numpy as np

from eicvibe.domain.lattice import Lattice

from .contracts import (
    BeamTrack,
    Capabilities,
    Computation,
    EngineUnavailable,
    OpticsRequest,
    OpticsResult,
    Options,
    OrbitRequest,
    OrbitResult,
    Ramp,
    TimeScale,
    TrackRequest,
    TrackResult,
    UnsupportedRequest,
)
from .model import SETTINGS, Values, require, span

#: Settings changed in place, by occurrence: occurrence -> parameter path -> value.
Changes = dict[str, dict[str, float]]


class EngineSession(ABC):
    """One branch in an engine, with the contract's checks before every computation.

    A subclass sets :attr:`engine` and implements :meth:`_update`, :meth:`_optics`,
    :meth:`_orbit`, :meth:`_track`, :meth:`_track_beam` and :meth:`close`; it calls
    :meth:`checked` on any other values it applies (a tracking call's).
    """

    engine: ClassVar[str]

    def __init__(
        self, capabilities: Capabilities, lattice: Lattice, branch: str, options: Options, values: Values
    ) -> None:
        self.capabilities = capabilities
        self.lattice = lattice
        self.layout = lattice.layout(branch)
        self.branch = self.layout.id
        self.options = options
        self.values: dict[tuple[str, str], float] = dict(values)
        self._occurrence_ids = [p.id for p in self.layout.placements]
        self._in_branch = frozenset(self._occurrence_ids)

    # -- settings --------------------------------------------------------------------------------

    def update(self, values: Values) -> None:
        """Change settings in place: parameters of :data:`~.model.SETTINGS`, on this branch's occurrences."""
        self._update(self.checked(values))

    def checked(self, values: Values) -> Changes:
        """``values`` by occurrence, every one a setting that changes in place, on this branch."""
        changes: Changes = {}
        for (occurrence, path), value in values.items():
            if path not in SETTINGS:
                raise UnsupportedRequest(self.engine, [f"{path} cannot change in place; open a new session"])
            if occurrence not in self._in_branch:
                raise UnsupportedRequest(self.engine, [f"occurrence {occurrence} is not in this branch"])
            changes.setdefault(occurrence, {})[path] = value
        return changes

    def checked_ramps(self, ramps: Sequence[Ramp]) -> None:
        """Every ramp moves settings of this branch, from a start to an end of the same settings."""
        for ramp in ramps:
            if set(ramp.start) != set(ramp.end):
                raise UnsupportedRequest(self.engine, ["a ramp starts and ends with the same settings"])
            self.checked(ramp.start)

    # -- computations ----------------------------------------------------------------------------

    def optics(self, request: OpticsRequest) -> OpticsResult:
        """The branch's optics: a ring's periodic solution, or a line (or part) from initial conditions."""
        self._require(request.options, Computation.OPTICS)
        occurrences, partial = self._span(request.start, request.end)
        if (partial or not self.layout.periodic) and request.initial is None:
            raise UnsupportedRequest(self.engine, ["a line's optics need initial conditions"])
        return self._optics(request, occurrences, partial)

    def orbit(self, request: OrbitRequest) -> OrbitResult:
        """A ring's closed orbit, or the trajectory through a line (or part) from its initial coordinates."""
        self._require(request.options, Computation.ORBIT)
        occurrences, partial = self._span(request.start, request.end)
        return self._orbit(request, occurrences, partial)

    def track(self, request: TrackRequest) -> TrackResult:
        """Particles tracked for some turns of a ring, or once through a line, recorded where asked."""
        if request.turns > 1 and not self.layout.periodic:
            raise UnsupportedRequest(self.engine, ["a line is passed once; turns > 1 needs a ring"])
        if request.turns > 1:
            time_scale = TimeScale.TURNS
        else:
            time_scale = TimeScale.EQUILIBRIUM if self.layout.periodic else TimeScale.SINGLE_PASS
        self._require(request.options, Computation.TRACKING, time_scale)
        self._observed(request.observe, self._in_branch)
        return self._track(request)

    def track_beam(
        self,
        coordinates,
        turns: int,
        observe: Sequence[str],
        options: Options,
        start: str | None = None,
        end: str | None = None,
        ramps: Sequence[Ramp] = (),
    ) -> BeamTrack:
        """Track a bunch (rows of x, px, y, py, zeta, delta) for ``turns`` turns, recording centroids.

        With ``start`` or ``end``, or on a line, once through that part, as a line. With ``ramps``
        (a ring's turns only), settings move turn by turn while it tracks; the session is left at
        their values after the last turn.
        """
        once = start is not None or end is not None or not self.layout.periodic
        if once and turns != 1:
            raise UnsupportedRequest(self.engine, ["a line, or part of a branch, is passed once: turns = 1"])
        if once and ramps:
            raise UnsupportedRequest(self.engine, ["settings ramp while a ring turns, not in one pass"])
        self._require(options, Computation.TRACKING, TimeScale.SINGLE_PASS if once else TimeScale.TURNS)
        occurrences, _ = self._span(start, end)
        self._observed(observe, frozenset(occurrences))
        self.checked_ramps(ramps)
        coordinates = np.asarray(coordinates, dtype=float).reshape(-1, 6)
        return self._track_beam(coordinates, turns, list(observe), options, occurrences, once, tuple(ramps))

    # -- the engine's own ------------------------------------------------------------------------

    @abstractmethod
    def _update(self, changes: Changes) -> None:
        """Apply checked settings in place; refuse (UnsupportedRequest) a change of an element's structure."""

    @abstractmethod
    def _optics(self, request: OpticsRequest, occurrences: list[str], partial: bool) -> OpticsResult:
        """``occurrences``: the part asked for; ``partial``: a part of the branch, as a line."""

    @abstractmethod
    def _orbit(self, request: OrbitRequest, occurrences: list[str], partial: bool) -> OrbitResult: ...

    @abstractmethod
    def _track(self, request: TrackRequest) -> TrackResult: ...

    @abstractmethod
    def _track_beam(
        self,
        coordinates: np.ndarray,
        turns: int,
        observe: list[str],
        options: Options,
        occurrences: list[str],
        once: bool,
        ramps: tuple[Ramp, ...],
    ) -> BeamTrack:
        """``once``: a line, or part of a branch, passed once; ``ramps`` only on a ring's turns."""

    @abstractmethod
    def close(self) -> None: ...

    # -- helpers ---------------------------------------------------------------------------------

    def _require(
        self, options: Options, computation: Computation, time_scale: TimeScale | None = None
    ) -> None:
        require(self.capabilities, self.lattice, self.branch, options, computation, time_scale)

    def _span(self, start: str | None, end: str | None) -> tuple[list[str], bool]:
        """The occurrences from ``start`` to ``end`` (the branch by default), and whether that is a part."""
        return span(self._occurrence_ids, start, end, self.engine), start is not None or end is not None

    def _observed(self, observe: Sequence[str], where: frozenset[str]) -> None:
        if missing := [o for o in observe if o not in where]:
            raise UnsupportedRequest(
                self.engine, [f"observation points outside the part tracked: {missing[:3]}"]
            )


class EngineAdapter(ABC):
    """An engine: whether it is installed, what it can do, and sessions on a branch.

    A subclass sets :attr:`name` and :attr:`unavailable` and implements :meth:`available`,
    :meth:`capabilities` and :meth:`_open`; :meth:`open` refuses, the same way for every engine,
    when the engine is missing or cannot do what the branch needs.
    """

    name: ClassVar[str]
    unavailable: ClassVar[str] = ""
    """What to say when it is not installed: how to install it."""

    @abstractmethod
    def available(self) -> bool:
        """Whether the engine's software is installed (a cheap check; starting it may still fail)."""

    @abstractmethod
    def capabilities(self) -> Capabilities: ...

    def open(
        self, lattice: Lattice, branch: str, options: Options | None = None, values: Values | None = None
    ) -> EngineSession:
        """A session on ``branch`` with ``values`` (any occurrences'; those of other branches are ignored)."""
        options = options or Options()
        if not self.available():
            raise EngineUnavailable(self.unavailable or f"{self.name} is not installed")
        capabilities = self.capabilities()
        require(capabilities, lattice, branch, options)
        return self._open(capabilities, lattice, branch, options, dict(values or {}))

    @abstractmethod
    def _open(
        self, capabilities: Capabilities, lattice: Lattice, branch: str, options: Options, values: Values
    ) -> EngineSession: ...
