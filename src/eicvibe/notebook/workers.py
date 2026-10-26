"""The kernel's engines, each in a worker process of its own, as the server's are (ADR 0016).

A notebook's kernel runs threads of its own (its channels, its widgets' messages), and Julia's
runtime, started in it by TrackPad, left Python's GIL held by one of its tasks: the kernel hung,
with every Python thread waiting for the GIL (seen in JupyterLab). So the engines the notebook's
helpers and widgets use run where the server's do, in
:class:`~eicvibe.engines.worker.EngineWorker` processes: one per engine, started on first use and
kept for the kernel's life (they end with it). The kernel stays free while they compute, and an
engine that crashes takes its worker, not the kernel.
"""

import threading
from collections.abc import Mapping
from concurrent.futures import Future

from eicvibe.domain.lattice import Lattice, Route
from eicvibe.engines import ENGINES, EngineUnavailable, engine
from eicvibe.engines.contracts import MachineOptics, Options
from eicvibe.engines.model import initial_conditions
from eicvibe.engines.worker import EngineWorker, RemoteSession

_WORKERS: dict[str, EngineWorker] = {}
_LOCK = threading.Lock()


def worker(name: str) -> EngineWorker:
    """The kernel's worker of engine ``name``, started on first use."""
    if name not in ENGINES:
        raise KeyError(f"no engine {name!r}; there are {', '.join(ENGINES)}")
    with _LOCK:
        found = _WORKERS.get(name)
        if found is None:
            if not engine(name).available():
                raise EngineUnavailable(f"engine {name} is not installed in this kernel's environment")
            found = _WORKERS[name] = EngineWorker(name)
        return found


def optics(
    lattice: Lattice,
    branch: str,
    name: str,
    options: Options,
    values: Mapping[tuple[str, str], float],
    routes: Mapping[str, Route],
    *,
    inside: bool = True,
) -> Future[MachineOptics]:
    """A branch's optics, or a route's (``route:<name>``), as the viewer computes them, in a worker:
    a ring's alone, a line with the branches its fork comes from, a route along it."""
    chosen = worker(name)
    if branch.startswith("route:"):
        route = routes.get(wanted := branch.removeprefix("route:"))
        if route is None:
            raise KeyError(f"no route {wanted}: a machine's routes.yaml holds its routes")
        return chosen.route_optics(lattice, route, options, inside=inside, values=dict(values))
    layout = lattice.layout(branch)
    alone = layout.periodic or initial_conditions(lattice, layout.id) is not None
    return chosen.machine_optics(
        lattice, options, inside=inside, values=dict(values), branches=[layout.id] if alone else None
    )


def session(
    lattice: Lattice, branch: str, name: str, options: Options, values: Mapping[tuple[str, str], float]
) -> RemoteSession:
    """An engine's session on a branch, open in its worker until closed."""
    return worker(name).open_session(lattice, branch, options, dict(values))
