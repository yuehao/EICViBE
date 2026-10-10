"""Physics engines behind one contract: several engines, fidelities, effects and time scales (ADR 0006).

The first engines are Xsuite (:mod:`.xsuite`) and TrackPad (:mod:`.trackpad`).
Each declares its :class:`~.contracts.Capabilities`; a request that needs
what an engine lacks is refused with the reason. Engines load lazily, so
importing this package needs neither Xsuite nor Julia.
"""

from importlib import import_module

from .contracts import (
    Capabilities,
    Computation,
    Effect,
    EngineUnavailable,
    Fidelity,
    OpticsRequest,
    OpticsResult,
    Options,
    OrbitRequest,
    OrbitResult,
    ParticleLost,
    Ramp,
    Support,
    TimeScale,
    TrackRequest,
    TrackResult,
    TwissInit,
    UnsupportedRequest,
)
from .model import (
    Engine,
    Program,
    Session,
    machine_optics,
    quasi_static,
    require,
    route_optics,
    route_start,
)

#: Engine name -> "module:class", loaded on first use.
ENGINES = {
    "xsuite": "eicvibe.engines.xsuite:XsuiteEngine",
    "trackpad": "eicvibe.engines.trackpad:TrackPadEngine",
}


def engine(name: str) -> Engine:
    """The engine called ``name``; :class:`EngineUnavailable` if it cannot load here."""
    if name not in ENGINES:
        raise KeyError(f"no engine {name!r}; there are {', '.join(ENGINES)}")
    module, _, cls = ENGINES[name].partition(":")
    try:
        return getattr(import_module(module), cls)()
    except ImportError as error:
        raise EngineUnavailable(f"engine {name} cannot load: {error}") from None


def select(
    lattice, branch: str, options: Options | None = None, computation=Computation.OPTICS, prefer=None
) -> Engine:
    """The first engine (in ``prefer`` order, default :data:`ENGINES` order) that can do this request."""
    options = options or Options()
    reasons = []
    for name in prefer or ENGINES:
        try:
            candidate = engine(name)
            if not candidate.available():
                raise EngineUnavailable(f"engine {name} is not installed")
            require(candidate.capabilities(), lattice, branch, options, computation)
            return candidate
        except (EngineUnavailable, UnsupportedRequest) as error:
            reasons.append(str(error))
    raise UnsupportedRequest("every engine", reasons)


__all__ = [
    "ENGINES",
    "Capabilities",
    "Computation",
    "Effect",
    "Engine",
    "EngineUnavailable",
    "Fidelity",
    "OpticsRequest",
    "OpticsResult",
    "Options",
    "OrbitRequest",
    "OrbitResult",
    "ParticleLost",
    "Ramp",
    "Program",
    "Session",
    "Support",
    "TimeScale",
    "TrackRequest",
    "TrackResult",
    "TwissInit",
    "UnsupportedRequest",
    "engine",
    "machine_optics",
    "route_optics",
    "route_start",
    "quasi_static",
    "require",
    "select",
]
