"""An engine in a process of its own (ADR 0003, ADR 0006).

A server keeps one worker per engine. The worker loads the engine once and
keeps it loaded between requests: Xsuite's kernels, and for TrackPad the Julia
runtime with its compiled code, which takes tens of seconds to start. Julia,
reached through juliacall, wants the main thread of its process, and a crash
inside an engine ends the worker, not the server; the next request starts a
new one.

Requests and results cross the process boundary as JSON, the contract's plain
data, so nothing engine-specific leaves the worker.
"""

import multiprocessing
import threading
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from eicvibe.domain.lattice import Lattice

from .contracts import MachineOptics, Options


class EngineWorker:
    """One engine in its own process; requests run one at a time, in order."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._pool: ProcessPoolExecutor | None = None
        self._lock = threading.Lock()

    def machine_optics(
        self, lattice: Lattice, options: Options, *, inside: bool = False
    ) -> Future[MachineOptics]:
        """The optics of every branch (:func:`.model.machine_optics`), computed in the worker."""
        raw = self._submit(_machine_optics, self.name, lattice.to_json(), options.model_dump_json(), inside)
        result: Future[MachineOptics] = Future()

        def done(future: Future[str]) -> None:
            try:
                result.set_result(MachineOptics.model_validate_json(future.result()))
            except BaseException as error:  # noqa: BLE001 - handed to whoever waits
                result.set_exception(error)

        raw.add_done_callback(done)
        return result

    def _submit(self, function, *args) -> Future:
        with self._lock:
            for attempt in (1, 2):
                if self._pool is None:
                    self._pool = ProcessPoolExecutor(
                        max_workers=1, mp_context=multiprocessing.get_context("spawn")
                    )
                try:
                    return self._pool.submit(function, *args)
                except BrokenProcessPool:  # the last worker died; start another once
                    self._pool = None
                    if attempt == 2:
                        raise
        raise AssertionError("unreachable")

    def close(self) -> None:
        with self._lock:
            if self._pool is not None:
                self._pool.shutdown(wait=False, cancel_futures=True)
                self._pool = None


def _machine_optics(name: str, lattice_json: str, options_json: str, inside: bool = False) -> str:
    """In the worker: machine optics of a lattice, with notes, as JSON."""
    from . import engine
    from .model import machine_optics

    chosen = engine(name)
    lattice = Lattice.from_json(lattice_json)
    options = Options.model_validate_json(options_json)
    notes: dict[str, str] = {}
    results = machine_optics(chosen, lattice, options=options, notes=notes, inside=inside)
    return MachineOptics(
        engine=name,
        version=chosen.capabilities().version,
        fidelity=options.fidelity,
        results=results,
        notes=notes,
    ).model_dump_json()
