"""BPM orbits of virtual runs under the lattice strip, as the viewer draws them (ADR 0016).

The runs' orbits are read from their folders (:class:`~eicvibe.api.runs.RunCatalog`, as the server
reads them), turn by turn, the slow orbit or a capture; while a run goes on the view follows it, as
long as the kernel is free to answer.
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from traitlets import Dict, List, Unicode

from eicvibe.api.runs import RunCatalog
from eicvibe.domain.lattice import Lattice
from eicvibe.store import Machine
from eicvibe.virtual import BpmHistory, RingRun, RouteRun

from ._base import AlongS

type Run = RingRun | RouteRun | BpmHistory | str | Path


class OrbitView(AlongS):
    """The orbits ``runs`` recorded at their BPMs, x above y, some turns apart, the newest opaque;
    live, paused or stepped through, from the BPMs' buffer, the slow orbit or a capture. ``runs``
    are runs of this ring or route, going on or not: the runs themselves, their histories, or their
    folders."""

    view = Unicode("orbit").tag(sync=True)
    runs = List(Dict()).tag(sync=True)

    def __init__(
        self,
        machine: Machine | Lattice,
        branch: str | None = None,
        *,
        runs: Iterable[Run] = (),
        s: tuple[float, float] | None = None,
        strip: bool = True,
        height: int = 420,
    ) -> None:
        self._runs = RunCatalog([_folder(run) for run in runs])
        super().__init__(machine, branch, s, runs=self._run_entries(), strip=strip, height=height)

    def _reply(self, ident: int, method: str, args: dict[str, Any]) -> str | None:
        if method in ("orbits", "captures") and args["run"] not in {run["id"] for run in self.runs}:
            raise ValueError(f"no run {args['run']} in this view")
        if method == "orbits":
            payload = self._runs.orbits(
                args["run"],
                int(args["every"]),
                int(args["count"]),
                args.get("end"),
                args.get("source", "buffer"),
            )
            entries = self._run_entries()
            if entries != self.runs:  # a run goes on: its state and turns, as the listing has them
                self.runs = entries
            return payload.body.decode()
        if method == "captures":
            return json.dumps([c.model_dump() for c in self._runs.captures(args["run"])])
        return super()._reply(ident, method, args)

    def _run_entries(self) -> list[dict[str, Any]]:
        return [json.loads(entry.model_dump_json()) for entry in self._runs.entries()]


def _folder(run: Run) -> Path:
    """A run's folder, from the run, its history or the folder itself."""
    if isinstance(run, RingRun | RouteRun | BpmHistory):
        return Path(run.directory)
    return Path(run)
