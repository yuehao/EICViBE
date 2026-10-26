"""An engine in a process of its own (ADR 0003, ADR 0006).

A server keeps one worker per engine. The worker loads the engine once and
keeps it loaded between requests: Xsuite's kernels, and for TrackPad the Julia
runtime with its compiled code, which takes tens of seconds to start. Julia,
reached through juliacall, wants the main thread of its process, and a crash
inside an engine ends the worker, not the server; the next request starts a
new one.

Requests and results cross the process boundary as the contract's plain data
(machine optics as JSON; sessions' results pickled), so nothing
engine-specific leaves the worker. A :class:`RemoteSession` keeps a session open
in the worker, for work that goes on, such as a ring tracked turn after turn.

A worker's process ends when the process that started it does, however that
ends: a notebook's kernel restarted, a server stopped by force. Left alone it
would wait for calls that never come, holding its engine (Julia's runtime is a
gigabyte) and the output it inherited.
"""

import multiprocessing
import multiprocessing.connection
import os
import threading
import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import TYPE_CHECKING, Any

from eicvibe.domain.lattice import Lattice, Route

from .contracts import (
    BeamTrack,
    EngineUnavailable,
    MachineOptics,
    OpticsRequest,
    OpticsResult,
    Options,
    OrbitRequest,
    OrbitResult,
    ParticleLost,
    TrackRequest,
    TrackResult,
    TwissInit,
    UnsupportedRequest,
)

if TYPE_CHECKING:
    from .remote import EngineHost, RemoteEngine


class EngineWorker:
    """One engine in its own process, here or on an engine host (:mod:`.remote`, ADR 0015); requests run
    one at a time, in order."""

    def __init__(self, name: str, host: EngineHost | None = None) -> None:
        self.name = name
        self.host = host
        """Where the engine runs: None here, in a process of its own."""
        self._pool: ProcessPoolExecutor | RemoteEngine | None = None
        self._generation = 0
        """How many processes the worker has started: a session lives in the one it was opened in."""
        self._lock = threading.Lock()

    def machine_optics(
        self,
        lattice: Lattice,
        options: Options,
        *,
        inside: bool = False,
        values: dict | None = None,
        branches: list[str] | None = None,
    ) -> Future[MachineOptics]:
        """The optics of every branch, or of ``branches`` (:func:`.model.machine_optics`), computed in
        the worker."""
        raw = self._submit(
            _machine_optics,
            self.name,
            lattice.to_json(),
            options.model_dump_json(),
            inside,
            dict(values or {}),
            branches,
        )
        result: Future[MachineOptics] = Future()

        def done(future: Future[str]) -> None:
            try:
                result.set_result(MachineOptics.model_validate_json(future.result()))
            except BaseException as error:  # noqa: BLE001 - handed to whoever waits
                result.set_exception(error)

        raw.add_done_callback(done)
        return result

    def route_optics(
        self,
        lattice: Lattice,
        route: Route,
        options: Options,
        *,
        inside: bool = False,
        values: dict | None = None,
    ) -> Future[MachineOptics]:
        """A route's optics (:func:`.model.route_optics`), as machine optics with one entry, route:<name>."""
        raw = self._submit(
            _route_optics,
            self.name,
            lattice.to_json(),
            route.model_dump_json(),
            options.model_dump_json(),
            inside,
            dict(values or {}),
        )
        result: Future[MachineOptics] = Future()

        def done(future: Future[str]) -> None:
            try:
                result.set_result(MachineOptics.model_validate_json(future.result()))
            except BaseException as error:  # noqa: BLE001 - handed to whoever waits
                result.set_exception(error)

        raw.add_done_callback(done)
        return result

    def route_start(
        self, lattice: Lattice, route: Route, options: Options, values: dict | None = None
    ) -> tuple[TwissInit, str]:
        """Where a route starts (:func:`.model.route_start`): its optics and orbit there, and their origin."""
        start, origin = self._submit(
            _route_start, self.name, lattice.to_json(), route.model_dump_json(), options, dict(values or {})
        ).result()
        return TwissInit.model_validate_json(start), origin

    def open_session(
        self, lattice: Lattice, branch: str, options: Options | None = None, values: dict | None = None
    ) -> RemoteSession:
        """A session on one branch, open in the worker until closed."""
        key = uuid.uuid4().hex
        options = options or Options()
        opened = self._submit(
            _open_session, key, self.name, lattice.to_json(), branch, options, dict(values or {})
        )
        generation = self._generation
        opened.result()
        return RemoteSession(self, key, lattice.layout(branch).id, generation)

    def _submit(self, function, *args, generation: int | None = None) -> Future:
        """Submit a call to the worker's process, starting one if there is none or it died; a session's
        call (its ``generation``) fails with BrokenProcessPool instead once its process is gone."""
        with self._lock:
            for attempt in (1, 2):
                if generation is not None and (self._pool is None or generation != self._generation):
                    raise BrokenProcessPool(f"{self.name}: the engine's process is gone, and its session")
                if self._pool is None:
                    self._generation += 1
                    if self.host is not None:
                        self._pool = self.host.connect()
                    else:
                        self._pool = ProcessPoolExecutor(
                            max_workers=1,
                            mp_context=multiprocessing.get_context("spawn"),
                            initializer=_watch_parent,
                        )
                try:
                    return self._pool.submit(function, *args)
                except BrokenProcessPool:  # the last worker died; start another once
                    self._pool = None
                    if attempt == 2 or generation is not None:
                        raise
        raise AssertionError("unreachable")

    def close(self) -> None:
        with self._lock:
            if self._pool is not None:
                self._pool.shutdown(wait=False, cancel_futures=True)
                self._pool = None


class RemoteSession:
    """A session open in an engine worker, used as a local one (:class:`.model.Session`)."""

    def __init__(self, worker: EngineWorker, key: str, branch: str, generation: int = 0) -> None:
        self.worker = worker
        self.key = key
        self.engine = worker.name
        self.branch = branch
        self.generation = generation
        """The worker's process it lives in."""

    def _call(self, method: str, *args) -> Any:
        return self.worker._submit(_call, self.key, method, *args, generation=self.generation).result()

    def update(self, values) -> None:
        self._call("update", dict(values))

    def optics(self, request: OpticsRequest) -> OpticsResult:
        return self._call("optics", request)

    def orbit(self, request: OrbitRequest) -> OrbitResult:
        return self._call("orbit", request)

    def track(self, request: TrackRequest) -> TrackResult:
        return self._call("track", request)

    def track_beam(
        self,
        coordinates,
        turns: int,
        observe,
        options: Options,
        start: str | None = None,
        end: str | None = None,
        ramps=(),
        momentum=None,
    ) -> BeamTrack:
        return self._call(
            "track_beam", coordinates, turns, tuple(observe), options, start, end, tuple(ramps), momentum
        )

    def close(self) -> None:
        try:
            self.worker._submit(_close_session, self.key, generation=self.generation).result(timeout=60)
        except Exception:  # noqa: BLE001 - a dead worker has nothing left to close
            pass


def _watch_parent() -> None:
    """In a worker's process: end it once the process that started it has ended."""
    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch() -> None:
        multiprocessing.connection.wait([parent.sentinel])  # ready when the parent is gone
        os._exit(0)

    threading.Thread(target=watch, name="parent-watch", daemon=True).start()


#: Sessions open in this worker process, by key.
_SESSIONS: dict[str, Any] = {}

#: Errors that cross back to the caller as they are; others become RuntimeError with their message.
_PASSED = (UnsupportedRequest, ParticleLost, EngineUnavailable, ValueError, KeyError)


def _open_session(
    key: str, name: str, lattice_json: str, branch: str, options: Options, values: dict
) -> None:
    from . import engine

    try:
        _SESSIONS[key] = engine(name).open(Lattice.from_json(lattice_json), branch, options, values)
    except _PASSED:
        raise
    except Exception as error:  # e.g. from Julia, which may not pickle
        raise RuntimeError(f"{name}: {error}") from None


def _call(key: str, method: str, *args) -> Any:
    session = _SESSIONS.get(key)
    if session is None:
        raise KeyError(f"no session {key} in this worker (it may have restarted)")
    try:
        return getattr(session, method)(*args)
    except _PASSED:
        raise
    except Exception as error:
        raise RuntimeError(f"{session.engine} {method}: {error}") from None


def _close_session(key: str) -> None:
    session = _SESSIONS.pop(key, None)
    if session is not None:
        session.close()


def _route_optics(
    name: str, lattice_json: str, route_json: str, options_json: str, inside: bool, values: dict | None = None
) -> str:
    """In the worker: a route's optics, with how it starts, as JSON."""
    from . import engine
    from .model import route_optics

    chosen = engine(name)
    route = Route.model_validate_json(route_json)
    options = Options.model_validate_json(options_json)
    notes: dict[str, str] = {}
    try:
        result = route_optics(
            chosen,
            Lattice.from_json(lattice_json),
            route,
            values=values,
            options=options,
            inside=inside,
            notes=notes,
        )
    except _PASSED:
        raise
    except Exception as error:  # e.g. from Julia, which may not pickle
        raise RuntimeError(f"{name}: {error}") from None
    return MachineOptics(
        engine=name,
        version=chosen.capabilities().version,
        fidelity=options.fidelity,
        results={f"route:{route.name}": result},
        notes=notes,
    ).model_dump_json()


def _route_start(
    name: str, lattice_json: str, route_json: str, options: Options, values: dict
) -> tuple[str, str]:
    """In the worker: where a route starts, as JSON, and the origin of its optics."""
    from . import engine
    from .model import route_start

    try:
        start, origin = route_start(
            engine(name),
            Lattice.from_json(lattice_json),
            Route.model_validate_json(route_json),
            values=values,
            options=options,
        )
    except _PASSED:
        raise
    except Exception as error:  # e.g. from Julia, which may not pickle
        raise RuntimeError(f"{name}: {error}") from None
    return start.model_dump_json(), origin


def _machine_optics(
    name: str,
    lattice_json: str,
    options_json: str,
    inside: bool = False,
    values: dict | None = None,
    branches: list[str] | None = None,
) -> str:
    """In the worker: machine optics of a lattice (or some of its branches), with notes, as JSON."""
    from . import engine
    from .model import machine_optics

    chosen = engine(name)
    lattice = Lattice.from_json(lattice_json)
    options = Options.model_validate_json(options_json)
    notes: dict[str, str] = {}
    results = machine_optics(
        chosen, lattice, values=values or {}, options=options, notes=notes, inside=inside, branches=branches
    )
    return MachineOptics(
        engine=name,
        version=chosen.capabilities().version,
        fidelity=options.fidelity,
        results=results,
        notes=notes,
    ).model_dump_json()
