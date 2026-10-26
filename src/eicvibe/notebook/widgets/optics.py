"""A branch's or route's optics under its lattice strip, as the viewer draws them (ADR 0016).

The optics are computed by the engine chosen in the view, in its worker (:mod:`..workers`), with
an operating mode's settings and the view's values over the lattice's; the view is answered once
they are, the kernel free meanwhile, and each is kept for the view.
"""

import json
from concurrent.futures import Future
from typing import Any

from traitlets import Unicode

from eicvibe.api.optics import branch_optics, engine_infos
from eicvibe.domain.lattice import Lattice
from eicvibe.engines.contracts import Fidelity, MachineOptics, OpticsResult, Options
from eicvibe.store import Machine
from eicvibe.virtual import default_engine

from .. import workers
from ._base import AlongS, Values

type Key = tuple[str, str, str, str]
"""(branch, engine, fidelity, mode)"""


class OpticsView(AlongS):
    """Beta and dispersion along s, inside magnets where the engine samples them, with the ring's
    tunes and chromaticities; hover for values, drag to zoom, click an element, choose the engine,
    the fidelity and the mode in the view (``engine``, ``fidelity``, ``mode``: the first ones).
    ``result``, optics already computed (with ``values`` of your own, say), is shown first."""

    view = Unicode("optics").tag(sync=True)
    engine = Unicode().tag(sync=True)
    fidelity = Unicode("standard").tag(sync=True)
    mode = Unicode("").tag(sync=True)
    """The operating mode whose settings the optics have; empty: the lattice's."""

    def __init__(
        self,
        machine: Machine | Lattice,
        branch: str | None = None,
        *,
        engine: str | None = None,
        fidelity: str = "standard",
        mode: str | None = None,
        values: Values | None = None,
        result: OpticsResult | None = None,
        s: tuple[float, float] | None = None,
        strip: bool = True,
        height: int = 420,
    ) -> None:
        self._values = dict(values or {})
        self._computed: dict[Key, MachineOptics] = {}
        if result is not None:
            engine, fidelity = result.engine, result.fidelity.value
            self._computed[(result.branch, engine, fidelity, mode or "")] = MachineOptics(
                engine=engine,
                version=None,
                fidelity=result.fidelity,
                results={result.branch: result},
                notes={result.branch: "computed in this notebook"},
            )
        super().__init__(
            machine,
            branch or (result.branch if result is not None else None),
            s,
            engine=engine or default_engine(),
            fidelity=fidelity,
            mode=mode or "",
            strip=strip,
            height=height,
        )

    def _reply(self, ident: int, method: str, args: dict[str, Any]) -> str | None:
        if method == "engines":
            return json.dumps([info.model_dump() for info in engine_infos()])
        if method == "optics":
            key = (args["branch"], args["engine"], args["fidelity"], args.get("mode") or "")
            if key in self._computed:
                return self._branch_optics(key)
            self._compute(ident, key)
            return None
        return super()._reply(ident, method, args)

    def _compute(self, ident: int, key: Key) -> None:
        branch, engine, fidelity, mode = key
        if mode and mode not in self._modes:
            raise ValueError(f"no operating mode {mode}")
        values = (self._modes[mode] if mode else {}) | self._values
        computing = workers.optics(
            self._lattice, branch, engine, Options(fidelity=Fidelity(fidelity)), values, self._routes
        )

        def done(future: Future[MachineOptics]) -> None:
            try:
                self._computed[key] = future.result()
                text = self._branch_optics(key)
            except Exception as error:  # noqa: BLE001 - told to the view
                self._fail(ident, error)
            else:
                self.send({"response": ident, "json": text})

        computing.add_done_callback(done)

    def _branch_optics(self, key: Key) -> str:
        return branch_optics(self._lattice, self._computed[key], key[0], key[3] or None).model_dump_json()
