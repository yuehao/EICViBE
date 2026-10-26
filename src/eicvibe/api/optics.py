"""Optics for the viewer: the design optics of a revision, computed by an engine worker.

Engines that can sample the optics inside elements without changing their
results do so (TrackPad at its integrator steps), so the plot follows beta and
dispersion through long magnets.

The optics of a revision follow from its lattice alone, so like every other
response they are computed once per (revision, engine, fidelity) and shared
by every client. The whole machine is computed at once, because a line's
optics start from the branch it forks from; a request for one branch waits
for that computation, which other requests for the same revision share. Each
engine runs in a worker process of its own (:mod:`eicvibe.engines.worker`):
TrackPad's Julia runtime starts once, on the first request that needs it.
"""

import asyncio
import hashlib
import threading
from collections import OrderedDict
from concurrent.futures import Future

from eicvibe.domain.lattice import Lattice
from eicvibe.engines import ENGINES, engine
from eicvibe.engines.contracts import EngineUnavailable, Fidelity, MachineOptics, Options, UnsupportedRequest
from eicvibe.engines.worker import EngineWorker

from . import schemas
from .catalog import Payload, Revision

#: Significant digits sent; 1e-9 relative is far below any model's accuracy.
DIGITS = 9

METHODS = {
    "xsuite": "Xsuite Twiss: tracking with finite differences",
    "trackpad": "TrackPad Twiss: TPSA maps (exact derivatives)",
}

COLUMNS = (
    schemas.Column(key="beta_x", label="βx", unit="m", description="Horizontal beta function"),
    schemas.Column(key="beta_y", label="βy", unit="m", description="Vertical beta function"),
    schemas.Column(key="alpha_x", label="αx", unit=None, description="Horizontal alpha"),
    schemas.Column(key="alpha_y", label="αy", unit=None, description="Vertical alpha"),
    schemas.Column(key="eta_x", label="ηx", unit="m", description="Horizontal dispersion, dx/d(delta)"),
    schemas.Column(key="eta_y", label="ηy", unit="m", description="Vertical dispersion"),
    schemas.Column(
        key="etap_x", label="ηx'", unit=None, description="Derivative of the horizontal dispersion"
    ),
    schemas.Column(key="etap_y", label="ηy'", unit=None, description="Derivative of the vertical dispersion"),
    schemas.Column(
        key="mu_x", label="μx", unit="2π", description="Horizontal phase advance from the start, turns"
    ),
    schemas.Column(
        key="mu_y", label="μy", unit="2π", description="Vertical phase advance from the start, turns"
    ),
    schemas.Column(key="x", label="x", unit="m", description="Horizontal orbit (closed orbit of a ring)"),
    schemas.Column(key="y", label="y", unit="m", description="Vertical orbit"),
)


#: Columns of a profile: the optics, without the orbit.
PROFILE_COLUMNS = tuple(c.key for c in COLUMNS if c.key not in ("x", "y"))


class UnknownEngine(KeyError):
    pass


def _round(value: float) -> float:
    return float(f"{value:.{DIGITS}g}")


def _key(
    revision: Revision, name: str, fidelity: Fidelity, route: str | None, mode: str | None
) -> tuple[str, str, str, str]:
    """A computation's cache key: by what it computes, not by names, so that a route or a mode
    edited in its file (the lattice unchanged) is computed again."""
    values = revision.modes.get(mode, {}) if mode else {}
    found = revision.views.routes.get(route) if route else None
    what = (found.model_dump_json() if found else route or "") + repr(sorted(values.items()))
    return revision.digest, name, fidelity.value, hashlib.sha256(what.encode()).hexdigest()


def branch_optics(
    lattice: Lattice, machine: MachineOptics, branch: str, mode: str | None = None
) -> schemas.BranchOptics:
    """One branch of a machine's optics as the viewer reads it (from the server, or a notebook's kernel)."""
    if branch.startswith("route:"):
        branch_id, periodic = branch, False
    else:
        layout = lattice.layout(branch)
        branch_id, periodic = layout.id, layout.periodic
    result = machine.results.get(branch_id)
    common = {
        "branch": branch_id,
        "engine": machine.engine,
        "version": machine.version,
        "fidelity": machine.fidelity.value,
        "periodic": periodic,
        "mode": mode,
        "origin": machine.notes.get(branch_id, "not computed"),
        "columns": list(COLUMNS),
    }
    if result is None:
        return schemas.BranchOptics(
            profile=None,
            **common,
            method=None,
            available=False,
            occurrences=[],
            s=[],
            values={},
            tune_x=None,
            tune_y=None,
            chroma_x=None,
            chroma_y=None,
            momentum_compaction=None,
        )
    start = result.start
    values = {
        c.key: [_round(getattr(start, c.key))] + [_round(v) for v in getattr(result, c.key)] for c in COLUMNS
    }
    profile = None
    if result.profile is not None:
        p = result.profile
        profile = schemas.OpticsProfile(
            s=[_round(v) for v in p.s],
            index=list(p.index),
            values={k: [_round(v) for v in getattr(p, k)] for k in PROFILE_COLUMNS},
        )
    return schemas.BranchOptics(
        **common,
        profile=profile,
        method=result.method,
        available=True,
        occurrences=list(result.occurrences),
        s=[0.0] + [_round(v) for v in result.s],
        values=values,
        tune_x=result.tune_x,
        tune_y=result.tune_y,
        chroma_x=result.chroma_x,
        chroma_y=result.chroma_y,
        momentum_compaction=result.momentum_compaction,
    )


def engine_infos() -> list[schemas.EngineInfo]:
    """The engines that compute optics: whether installed here, their versions and fidelities."""
    out = []
    for name in ENGINES:
        candidate = engine(name)
        capabilities = candidate.capabilities()
        out.append(
            schemas.EngineInfo(
                name=name,
                available=candidate.available(),
                version=capabilities.version,
                fidelities=[f.value for f in Fidelity if f in capabilities.fidelities],
                method=METHODS.get(name, ""),
            )
        )
    return out


class OpticsService:
    """Machine optics by (revision, engine, fidelity), computed by engine workers and kept for reuse."""

    def __init__(self, *, keep: int = 12) -> None:
        self._keep = keep
        self._lock = threading.Lock()
        self._workers: dict[str, EngineWorker] = {}
        self._machines: OrderedDict[
            tuple[str, str, str, str], tuple[Future[MachineOptics], dict[str, Payload]]
        ] = OrderedDict()

    def engines(self) -> list[schemas.EngineInfo]:
        return engine_infos()

    def worker(self, name: str) -> EngineWorker:
        """The worker process of engine ``name``, started on first use and shared."""
        if name not in ENGINES:
            raise UnknownEngine(name)
        with self._lock:
            return self._workers.setdefault(name, EngineWorker(name))

    def machine(
        self,
        revision: Revision,
        name: str,
        fidelity: Fidelity,
        route: str | None = None,
        mode: str | None = None,
    ) -> tuple[Future[MachineOptics], dict]:
        """The machine's or a route's optics of a revision, in an operating mode's settings or the
        lattice's, computed once; a failure is tried again."""
        if mode is not None and mode not in revision.modes:
            raise KeyError(f"no mode {mode}")
        values = revision.modes[mode] if mode else {}
        worker = self.worker(name)
        if fidelity not in engine(name).capabilities().fidelities:
            raise UnsupportedRequest(name, [f"no {fidelity.value} fidelity"])
        key = _key(revision, name, fidelity, route, mode)
        with self._lock:
            found = self._machines.get(key)
            if found is None or (found[0].done() and found[0].exception() is not None):
                if not engine(name).available():
                    raise EngineUnavailable(f"engine {name} is not installed on the server")
                if route is None:
                    computing = worker.machine_optics(
                        revision.lattice, Options(fidelity=fidelity), inside=True, values=values
                    )
                else:
                    computing = worker.route_optics(
                        revision.lattice,
                        revision.views.routes[route],
                        Options(fidelity=fidelity),
                        values=values,
                        inside=True,
                    )
                found = (computing, {})
                self._machines[key] = found
            self._machines.move_to_end(key)
            while len(self._machines) > self._keep:
                self._machines.popitem(last=False)
            return found

    def computed(
        self, revision: Revision, name: str, fidelity: Fidelity, branch: str, mode: str | None = None
    ) -> schemas.BranchOptics | None:
        """A branch's optics if they have been computed already, else None; never starts a computation."""
        route = branch.removeprefix("route:") if branch.startswith("route:") else None
        with self._lock:
            found = self._machines.get(_key(revision, name, fidelity, route, mode))
        if found is None or not found[0].done() or found[0].exception() is not None:
            return None
        branch_id = branch if route is not None else revision.lattice.layout(branch).id
        return branch_optics(revision.lattice, found[0].result(), branch_id, mode)

    async def branch(
        self, revision: Revision, name: str, fidelity: Fidelity, branch: str, mode: str | None = None
    ) -> Payload:
        """One branch's optics as a response, waiting for the machine computation without blocking others."""
        route = branch.removeprefix("route:") if branch.startswith("route:") else None
        future, payloads = self.machine(revision, name, fidelity, route, mode)
        machine = await asyncio.wrap_future(future)
        branch_id = branch if route is not None else revision.lattice.layout(branch).id
        payload = payloads.get(branch_id)
        if payload is None:
            optics = branch_optics(revision.lattice, machine, branch_id, mode)
            payload = payloads[branch_id] = Payload.of(optics)
        return payload

    def close(self) -> None:
        with self._lock:
            for worker in self._workers.values():
                worker.close()
            self._workers.clear()
