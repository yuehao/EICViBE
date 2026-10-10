"""Virtual rings' runs for the viewer: their BPM orbits, read from the runs' folders (ADR 0007).

The server is given folders of runs (``eicvibe-web --runs DIR``) and reads them
as any notebook would (:class:`~eicvibe.virtual.BpmHistory`), while the runs go
on in their own processes: the server starts nothing and holds no state per
client. A run is found again on every listing, so runs started after the
server appear by themselves. Orbits of turns already written never change,
so those responses are kept; the latest are read afresh.
"""

import math
import threading
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from eicvibe.virtual import BpmHistory

from . import schemas
from .catalog import Payload

MAX_ORBITS = 50


class RunCatalog:
    """The runs in some folders, by id."""

    def __init__(self, paths: Iterable[Path | str], *, keep: int = 256) -> None:
        self._paths = [Path(p) for p in paths]
        self._keep = keep
        self._cache: OrderedDict[tuple, Payload] = OrderedDict()
        self._lock = threading.Lock()

    def _folders(self) -> dict[str, Path]:
        found: dict[str, Path] = {}
        for path in self._paths:
            candidates = [path] if (path / "run.json").is_file() else sorted(path.glob("*/run.json"))
            for candidate in candidates:
                folder = candidate if candidate.is_dir() else candidate.parent
                try:
                    run_id = BpmHistory.open(folder).info.id
                except OSError, ValueError:
                    continue
                name, n = run_id, 2
                while name in found:
                    name, n = f"{run_id}-{n}", n + 1
                found[name] = folder
        return found

    def entries(self) -> list[schemas.RunEntry]:
        out = []
        for run_id, folder in self._folders().items():
            history = BpmHistory.open(folder)
            info, (first, end), state = history.info, history.span(), history.state()
            out.append(
                schemas.RunEntry(
                    id=run_id,
                    kind=info.kind,
                    unit=info.unit,
                    lattice=info.lattice,
                    lattice_digest=info.lattice_digest.removeprefix("sha256:"),
                    branch=info.branch,
                    branch_id=info.branch_id,
                    engine=info.engine,
                    fidelity=info.fidelity,
                    start=info.start.isoformat(),
                    revolution_period=info.revolution_period,
                    bpms=len(info.bpms),
                    first_turn=first,
                    end_turn=end,
                    state=state.get("state", "unknown"),
                    updated=state.get("updated"),
                    machine=info.machine,
                    buffer_turns=info.buffer_turns,
                    slow_period=info.slow_period,
                    captures=len(history.captures()),
                )
            )
        return sorted(out, key=lambda e: e.start, reverse=True)

    def history(self, run_id: str) -> BpmHistory:
        folder = self._folders().get(run_id)
        if folder is None:
            raise KeyError(run_id)
        return BpmHistory.open(folder)

    def captures(self, run_id: str) -> list[schemas.CaptureEntry]:
        known = ("id", "kind", "trigger", "first", "turns", "wall", "partial")
        return [
            schemas.CaptureEntry(
                **{k: c[k] for k in known if k != "partial"},
                partial=bool(c.get("partial")),
                detail={k: v for k, v in c.items() if k not in known},
            )
            for c in self.history(run_id).captures()
        ]

    def orbits(
        self, run_id: str, every: int, count: int, end: int | None = None, source: str = "buffer"
    ) -> Payload:
        """``count`` orbits ``every`` apart, the newest at ``end - 1`` (by default the latest).

        ``source``: the BPMs' buffer (turns), ``slow`` (the slow orbit's samples), or
        ``capture:<id>`` (a capture's turns).
        """
        history = self.history(run_id)
        if source == "slow":
            return self._slow(run_id, history, every, count, end)
        if source.startswith("capture:"):
            ident = source.removeprefix("capture:")
            tbt = history.capture(ident)
            first, recorded = tbt.first_turn, tbt.first_turn + tbt.turns
            newest = recorded - 1 if end is None else min(max(end, first + 1), recorded) - 1
            oldest = max(newest - (count - 1) * every, first)
            oldest = newest - (newest - oldest) // every * every
            pick = np.arange(oldest - first, newest - first + 1, every)
            key = (run_id, source, every, count, newest)
            with self._lock:
                if key in self._cache:
                    return self._cache[key]
            payload = self._frames(
                run_id,
                source,
                history.info.unit,
                tbt,
                pick,
                every,
                first,
                recorded,
                "capture",
                history.events(oldest, newest + 1),
            )
            with self._lock:
                self._cache[key] = payload
            return payload
        return self._buffer(run_id, history, every, count, end)

    def _slow(self, run_id: str, history: BpmHistory, every: int, count: int, end: int | None) -> Payload:
        series = history.orbit()
        times = history.orbit_times()
        total = len(series.turn)
        if total == 0:
            raise ValueError("the run has no slow orbit yet")
        newest = total - 1 if end is None else min(end, total) - 1
        oldest = max(newest - (count - 1) * every, 0)
        oldest = newest - (newest - oldest) // every * every
        pick = np.arange(oldest, newest + 1, every)
        events = history.events(int(series.turn[oldest]), int(series.turn[newest] + series.turns[newest]))
        return Payload.of(
            schemas.OrbitFrames(
                run=run_id,
                source="slow",
                unit="sample",
                sample_period=history.info.slow_period or None,
                times=[float(times[k]) for k in pick],
                bpms=list(series.bpms),
                s=[float(v) for v in series.s],
                every=every,
                turns=[int(k) for k in pick],
                x=_rows(series.x.T, pick),
                y=_rows(series.y.T, pick),
                intensity=_rows(series.intensity.T, pick),
                first_turn=0,
                end_turn=total,
                state=history.state().get("state", "unknown"),
                events=_events(events),
            )
        )

    def _frames(self, run_id, source, unit, tbt, pick, every, first, recorded, state, events) -> Payload:
        return Payload.of(
            schemas.OrbitFrames(
                run=run_id,
                source=source,
                unit=unit,
                sample_period=None,
                times=None,
                bpms=list(tbt.x.bpms),
                s=[float(v) for v in tbt.x.s],
                every=every,
                turns=[int(tbt.first_turn + k) for k in pick],
                x=_rows(tbt.x.position, pick),
                y=_rows(tbt.y.position, pick),
                intensity=_rows(tbt.x.intensity, pick),
                first_turn=first,
                end_turn=recorded,
                state=state,
                events=_events(events),
            )
        )

    def _buffer(self, run_id: str, history: BpmHistory, every: int, count: int, end: int | None) -> Payload:
        """``count`` orbits ``every`` turns apart, the newest at turn ``end - 1`` (by default the latest)."""
        first, recorded = history.span()
        if recorded == first:
            raise ValueError("the run has recorded no turns yet")
        newest = recorded - 1 if end is None else min(end, recorded) - 1
        oldest = max(newest - (count - 1) * every, first)
        oldest = newest - (newest - oldest) // every * every  # whole steps back from the newest
        key = (run_id, every, count, newest)
        if end is not None and end <= recorded:
            with self._lock:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    return self._cache[key]
        tbt = history.turn_by_turn(oldest, newest - oldest + 1)
        pick = np.arange(0, tbt.turns, every)
        state = history.state().get("state", "unknown")
        payload = self._frames(
            run_id,
            "buffer",
            history.info.unit,
            tbt,
            pick,
            every,
            first,
            recorded,
            state,
            history.events(oldest, newest + 1),
        )
        if end is not None and end <= recorded:
            with self._lock:
                self._cache[key] = payload
                while len(self._cache) > self._keep:
                    self._cache.popitem(last=False)
        return payload


def _rows(matrix: np.ndarray, pick) -> list[list[float | None]]:
    """Columns ``pick`` of ``[bpm, n]`` as rows of JSON numbers (null where there is no reading)."""
    return [[None if not math.isfinite(v) else round(float(v), 6) for v in matrix[:, k]] for k in pick]


def _events(events: list[dict]) -> list[schemas.RunEvent]:
    return [
        schemas.RunEvent(
            turn=e["turn"],
            kind=e["kind"],
            detail={k: v for k, v in e.items() if k not in ("turn", "kind", "wall")},
        )
        for e in events
    ]
