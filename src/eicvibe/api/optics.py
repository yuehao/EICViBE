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
import threading
from collections import OrderedDict
from concurrent.futures import Future

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


def branch_optics(revision: Revision, machine: MachineOptics, branch: str) -> schemas.BranchOptics:
    """One branch of a machine's optics as the viewer reads it."""
    lattice = revision.lattice
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


class OpticsService:
    """Machine optics by (revision, engine, fidelity), computed by engine workers and kept for reuse."""

    def __init__(self, *, keep: int = 12) -> None:
        self._keep = keep
        self._lock = threading.Lock()
        self._workers: dict[str, EngineWorker] = {}
        self._machines: OrderedDict[
            tuple[str, str, str], tuple[Future[MachineOptics], dict[str, Payload]]
        ] = OrderedDict()

    def engines(self) -> list[schemas.EngineInfo]:
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

    def worker(self, name: str) -> EngineWorker:
        """The worker process of engine ``name``, started on first use and shared."""
        if name not in ENGINES:
            raise UnknownEngine(name)
        with self._lock:
            return self._workers.setdefault(name, EngineWorker(name))

    def machine(
        self, revision: Revision, name: str, fidelity: Fidelity, route: str | None = None
    ) -> tuple[Future[MachineOptics], dict]:
        """The machine's or a route's optics of a revision, computed once; a failure is tried again."""
        worker = self.worker(name)
        if fidelity not in engine(name).capabilities().fidelities:
            raise UnsupportedRequest(name, [f"no {fidelity.value} fidelity"])
        key = (revision.digest, name, fidelity.value, route or "")
        with self._lock:
            found = self._machines.get(key)
            if found is None or (found[0].done() and found[0].exception() is not None):
                if not engine(name).available():
                    raise EngineUnavailable(f"engine {name} is not installed on the server")
                if route is None:
                    computing = worker.machine_optics(
                        revision.lattice, Options(fidelity=fidelity), inside=True
                    )
                else:
                    computing = worker.route_optics(
                        revision.lattice,
                        revision.views.routes[route],
                        Options(fidelity=fidelity),
                        inside=True,
                    )
                found = (computing, {})
                self._machines[key] = found
            self._machines.move_to_end(key)
            while len(self._machines) > self._keep:
                self._machines.popitem(last=False)
            return found

    async def branch(self, revision: Revision, name: str, fidelity: Fidelity, branch: str) -> Payload:
        """One branch's optics as a response, waiting for the machine computation without blocking others."""
        route = branch.removeprefix("route:") if branch.startswith("route:") else None
        future, payloads = self.machine(revision, name, fidelity, route)
        machine = await asyncio.wrap_future(future)
        branch_id = branch if route is not None else revision.lattice.layout(branch).id
        payload = payloads.get(branch_id)
        if payload is None:
            payload = payloads[branch_id] = Payload.of(branch_optics(revision, machine, branch_id))
        return payload

    def close(self) -> None:
        with self._lock:
            for worker in self._workers.values():
                worker.close()
            self._workers.clear()
