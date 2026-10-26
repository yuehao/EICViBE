"""What a virtual ring's BPMs read, kept on disk the way a BPM system keeps it (ADR 0007).

A ring's BPMs, like a real ring's (NSLS-II, RHIC, the LHC), write every turn
into a buffer of the latest turns, overwritten as the beam goes round; what is
kept is a slow orbit, the readings averaged over each second of the run, and
the captures: the buffer frozen around a trigger (a kick, an injection, a beam
loss, or a request) and saved. A route's run records every shot.

A run's folder::

    run.json               what ran: lattice, branch, engine, fidelity, BPMs, beam, clock, buffer
    events.jsonl           kicks, setting changes, injections, losses, captures, errors, with their turn
    tbt/<first>-<n>.h5     turn-by-turn readings, a chunk of n turns from turn <first> (HDF5):
                           x, y (mm), intensity (nC), status ([bpm, turn]); float32 and uint8.
                           A ring keeps the chunks of its latest ``buffer_turns`` turns (the buffer)
    orbit.bin              the slow orbit: the mean of every BPM over each ``slow_period`` seconds of
                           the run (wall clock), or over each chunk (float64 records)
    captures/<id>.h5       a capture: the turns around a trigger, as tbt chunks are; <id>.json what it is
    hidden.json            the BPMs' hidden errors (offsets, gains): not part of the data
    state.json             running, stopped, done or failed, and the turn reached: for viewers

Files are written whole (a temporary file renamed) and the orbit file only
grows by whole records, so a reader in another process, a notebook say, sees
every turn written so far and never a partial one. Captures, and a route's
shots, are dropped oldest first beyond a size limit; the slow orbit is kept.
"""

import json
import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from pydantic import BaseModel, Field

from eicvibe.diagnostics.bpm import OrbitSeries, PlaneData, TurnByTurn


class RunInfo(BaseModel):
    """What a run is: written once, when it starts.

    A ring's run records turns (``unit`` turn, ``revolution_period`` the revolution's); a route's
    records shots, single passes along it (``unit`` shot, ``revolution_period`` the time between
    shots, ``branch`` the route's name and ``branch_id`` ``route:<name>``). Turn numbers in the
    history are then shot numbers.
    """

    id: str
    kind: str = Field("ring", description="ring (turn after turn) or route (shot after shot)")
    unit: str = Field("turn", description="What the history counts: turn or shot")
    lattice: str
    lattice_digest: str
    branch: str
    branch_id: str
    engine: str
    host: str | None = Field(None, description="The engine host it ran on (ADR 0015); None: here")
    fidelity: str
    effects: list[str]
    start: datetime = Field(description="When turn 0 was, on the run's own clock")
    revolution_period: float = Field(description="s")
    chunk: int = Field(
        description="Turns an engine call tracks, at most (calls are sized to the engine's pace)"
    )
    ramp_turns: int = Field(0, ge=0, description="Turns a setting change ramps over; 0: a step")
    bpms: list[str] = Field(description="BPM names, in beam order: EIC names where there is a registry")
    occurrences: list[str]
    s: list[float]
    beam: dict[str, Any]
    resolution: float = Field(description="BPM resolution, mm rms per turn at full intensity")
    orbit_resolution: float = Field(
        0.0,
        ge=0,
        description="The BPMs' closed-orbit acquisition noise, mm rms per sample (0: the turns' average)",
    )
    retention_bytes: int = Field(description="Captures (a route: shots) kept, at most; the oldest go first")
    machine: str | None = Field(None, description="The virtual accelerator's machine that ran it (ADR 0008)")
    buffer_turns: int = Field(0, ge=0, description="Turns the buffer keeps, the latest; 0: all, by size")
    slow_period: float = Field(0.0, ge=0, description="Slow orbit sample, s of wall clock; 0: one per chunk")
    values: dict[str, float] = Field(
        default_factory=dict,
        description="Settings it started with, beyond the lattice's: 'occurrence parameter' to value, as the "
        "settings events give changes",
    )
    ramp: dict[str, Any] | None = Field(
        None,
        description="A ring's ramp profile, which a ramp command runs (ADR 0013): its name, energies (eV), "
        "turns to the flattop and to extraction; the 'ramp' events give the turns it started at",
    )


@dataclass
class _Chunk:
    first: int
    count: int
    path: Path

    @property
    def end(self) -> int:
        return self.first + self.count


@dataclass
class BpmHistory:
    """The readings of a run, by turn: written by the run, read by anyone."""

    directory: Path
    info: RunInfo
    writable: bool = False
    protect_from: int | None = None
    """While a capture waits for its turns: the buffer keeps turns from here on."""
    slow_noise: Callable[[int], np.ndarray] | None = None
    """The closed-orbit acquisition's noise of one sample, ``[plane, bpm]`` (mm), added to the average
    of the readings it is given without noise (``append``'s ``slow``); None: the readings averaged."""
    _kept: int = field(default=0, repr=False)
    _slow: dict[str, Any] | None = field(default=None, repr=False)

    @classmethod
    def create(cls, directory: str | Path, info: RunInfo) -> BpmHistory:
        directory = Path(directory)
        if (directory / "run.json").exists():
            raise FileExistsError(f"{directory} already holds a run; give the new run its own folder")
        (directory / "tbt").mkdir(parents=True, exist_ok=True)
        _write_atomically(directory / "run.json", info.model_dump_json(indent=1))
        return cls(directory, info, writable=True)

    @classmethod
    def open(cls, directory: str | Path) -> BpmHistory:
        """A run's history, to read; the run may still be writing it."""
        directory = Path(directory)
        info = RunInfo.model_validate_json((directory / "run.json").read_text())
        return cls(directory, info)

    # -- writing (the run) ------------------------------------------------------------

    def append(
        self,
        first: int,
        x: np.ndarray,
        y: np.ndarray,
        intensity: np.ndarray,
        status: np.ndarray,
        wall: float | tuple[float, float],
        slow: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """A chunk of readings ``[bpm, turn]`` from turn ``first``, tracked in the wall-clock interval
        ``wall`` (or by the time ``wall``), and its share of the slow orbit: of ``slow`` (the readings
        without their turn-by-turn noise, x and y) where given, with :attr:`slow_noise` on each sample."""
        if not self.writable:
            raise PermissionError("this history is open for reading")
        count = x.shape[1]
        path = self.directory / "tbt" / f"{first:012d}-{count}.h5"
        _save(path, x, y, intensity, status)
        sx, sy = slow if slow is not None else (x, y)
        if self.info.slow_period > 0 and isinstance(wall, tuple):
            self._slow_orbit(first, sx, sy, intensity, wall)
        else:
            with warnings.catch_warnings():  # a BPM with no reading in a whole chunk averages to NaN
                warnings.simplefilter("ignore", RuntimeWarning)
                means = [np.nanmean(a, axis=1) for a in (sx, sy, intensity)]
            moment = wall[1] if isinstance(wall, tuple) else wall
            self._orbit_record(first, count, moment, means)
        if self.info.buffer_turns:
            self._keep_buffer(first + count)
        else:
            self._kept += path.stat().st_size
            self._retain()

    def flush(self) -> None:
        """Write the slow orbit's last, partial sample (when the run ends)."""
        if self._slow is not None and self._slow["turns"]:
            self._write_slow()
        self._slow = None

    def _slow_orbit(self, first: int, x, y, intensity, wall: tuple[float, float]) -> None:
        """Average the chunk's turns into samples of ``slow_period`` seconds of wall clock.

        The turns of a chunk are spread evenly over the interval it took, so each sample averages
        the turns tracked in its second, as a BPM system's slow acquisition averages its own.
        """
        period, count = self.info.slow_period, x.shape[1]
        start, end = wall
        moments = start + (np.arange(count) + 0.5) / count * max(end - start, 0.0)
        bins = np.floor(moments / period).astype(np.int64)
        for sample in np.unique(bins):
            if self._slow is not None and self._slow["bin"] != sample:
                self._write_slow()
            cut = bins == sample
            turns = np.flatnonzero(cut)
            if self._slow is None:
                n = x.shape[0]
                self._slow = {
                    "bin": int(sample),
                    "first": first + int(turns[0]),
                    "turns": 0,
                    "sums": np.zeros((3, n)),
                    "counts": np.zeros((3, n)),
                }
            acc = self._slow
            for k, matrix in enumerate((x, y, intensity)):
                part = matrix[:, cut]
                finite = np.isfinite(part)
                acc["sums"][k] += np.where(finite, part, 0.0).sum(axis=1)
                acc["counts"][k] += finite.sum(axis=1)
            acc["turns"] += len(turns)

    def _write_slow(self) -> None:
        acc = self._slow
        with np.errstate(invalid="ignore", divide="ignore"):
            means = [acc["sums"][k] / acc["counts"][k] for k in range(3)]
        self._orbit_record(acc["first"], acc["turns"], acc["bin"] * self.info.slow_period, means)
        self._slow = None

    def _orbit_record(self, first: int, turns: int, wall: float, means: list[np.ndarray]) -> None:
        if self.slow_noise is not None:
            noise = self.slow_noise(len(means[0]))
            means = [means[0] + noise[0], means[1] + noise[1], *means[2:]]
        record = np.concatenate([[first, turns, first * self.info.revolution_period, wall], *means])
        with open(self.directory / "orbit.bin", "ab") as out:
            out.write(record.astype(np.float64).tobytes())

    def _keep_buffer(self, end: int) -> None:
        """Drop chunks entirely older than the buffer (and than a waiting capture's first turn)."""
        keep_from = end - self.info.buffer_turns
        if self.protect_from is not None:
            keep_from = min(keep_from, self.protect_from)
        for chunk in self._chunks():
            if chunk.end <= keep_from:
                chunk.path.unlink(missing_ok=True)

    # -- captures ---------------------------------------------------------------------

    def save_capture(self, kind: str, trigger: int, first: int, turns: int, **detail: Any) -> dict[str, Any]:
        """Freeze turns ``[first, first + turns)`` of the buffer as a capture around ``trigger``."""
        tbt = self.turn_by_turn(first, turns)
        ident = f"{trigger:012d}-{kind}"
        directory = self.directory / "captures"
        directory.mkdir(exist_ok=True)
        _save(directory / f"{ident}.h5", tbt.x.position, tbt.y.position, tbt.x.intensity, tbt.x.status)
        meta = {
            "id": ident,
            "kind": kind,
            "trigger": trigger,
            "first": tbt.first_turn,
            "turns": tbt.turns,
            "wall": datetime.now(UTC).isoformat(),
            **detail,
        }
        _write_atomically(directory / f"{ident}.json", json.dumps(meta))
        self.event(trigger, "capture", id=ident, of=kind, first=tbt.first_turn, turns=tbt.turns)
        self._retain_captures()
        return meta

    def _retain_captures(self) -> None:
        captures = sorted((self.directory / "captures").glob("*.h5"))
        sizes = [c.stat().st_size for c in captures]
        total = sum(sizes)
        for path, size in zip(captures[:-1], sizes[:-1], strict=False):
            if total <= self.info.retention_bytes:
                break
            path.unlink(missing_ok=True)
            path.with_suffix(".json").unlink(missing_ok=True)
            total -= size

    def captures(self) -> list[dict[str, Any]]:
        """The run's captures, oldest first: what triggered each, at which turn, which turns it holds."""
        directory = self.directory / "captures"
        out = []
        for path in sorted(directory.glob("*.json")) if directory.exists() else []:
            if path.with_suffix(".h5").exists():
                out.append(json.loads(path.read_text(encoding="utf-8")))
        return out

    def capture(self, ident: str) -> TurnByTurn:
        """A capture's turns, as a :class:`TurnByTurn` (its events within)."""
        meta_path = self.directory / "captures" / f"{ident}.json"
        if not meta_path.exists():
            raise KeyError(f"no capture {ident}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with h5py.File(meta_path.with_suffix(".h5"), "r") as data:
            arrays = {k: data[k][()] for k in ("x", "y", "intensity", "status")}
        return self._turn_by_turn(meta["first"], arrays, f"capture {ident} ({meta['kind']})")

    def set_state(self, state: str, turn: int, **detail: Any) -> None:
        """Whether the run is running, stopped, done or failed, and how far it is."""
        payload = {"state": state, "turn": turn, "updated": datetime.now(UTC).isoformat(), **detail}
        _write_atomically(self.directory / "state.json", json.dumps(payload))

    def state(self) -> dict[str, Any]:
        """The run's state as it last wrote it; ``unknown`` for a run that never did."""
        try:
            return json.loads((self.directory / "state.json").read_text(encoding="utf-8"))
        except OSError, ValueError:
            return {"state": "unknown", "turn": self.turn, "updated": None}

    def event(self, turn: int, kind: str, **detail: Any) -> None:
        """Something that happened before ``turn``: a kick, a setting change, an injection, a loss."""
        line = json.dumps({"turn": turn, "kind": kind, "wall": datetime.now(UTC).isoformat(), **detail})
        with open(self.directory / "events.jsonl", "a", encoding="utf-8") as out:
            out.write(line + "\n")

    def _retain(self) -> None:
        if self._kept <= self.info.retention_bytes:
            return
        chunks = self._chunks()
        self._kept = sum(c.path.stat().st_size for c in chunks)
        for chunk in chunks[:-1]:
            if self._kept <= self.info.retention_bytes:
                break
            size = chunk.path.stat().st_size
            chunk.path.unlink(missing_ok=True)
            self._kept -= size

    # -- reading (anyone) -------------------------------------------------------------

    def _chunks(self) -> list[_Chunk]:
        out = []
        for path in (self.directory / "tbt").glob("*.h5"):
            if path.name.endswith(".part.h5"):
                continue
            first, _, count = path.stem.partition("-")
            out.append(_Chunk(int(first), int(count), path))
        return sorted(out, key=lambda c: c.first)

    def span(self) -> tuple[int, int]:
        """The turns kept turn by turn: ``(first, end)``, end exclusive; ``(0, 0)`` before the first chunk."""
        chunks = self._chunks()
        return (chunks[0].first, chunks[-1].end) if chunks else (0, 0)

    @property
    def turn(self) -> int:
        """Turns recorded so far."""
        return self.span()[1]

    def turn_by_turn(
        self, start: int | None = None, turns: int = 1024, bpms: list[str] | tuple[str, ...] | None = None
    ) -> TurnByTurn:
        """``turns`` turns from turn ``start`` (by default the latest), as a :class:`TurnByTurn`.

        Turns no longer kept, or not yet tracked, are left out; asking only for those is an error.
        """
        first_kept, end = self.span()
        if start is None:
            start = max(end - turns, first_kept)
        stop = min(start + turns, end)
        start = max(start, first_kept)
        if stop <= start:
            raise ValueError(
                f"turns {start} to {start + turns} are not kept; this run keeps {first_kept} to {end}"
            )
        parts: dict[str, list[np.ndarray]] = {k: [] for k in ("x", "y", "intensity", "status")}
        covered = start
        for chunk in self._chunks():
            if chunk.end <= start or chunk.first >= stop:
                continue
            if chunk.first > covered:
                break  # a gap: dropped while we read; keep what is contiguous
            with h5py.File(chunk.path, "r") as data:
                cut = slice(covered - chunk.first, min(stop, chunk.end) - chunk.first)
                for key in parts:
                    parts[key].append(data[key][:, cut])
            covered = min(stop, chunk.end)
        if covered == start:
            raise ValueError(f"turns from {start} were dropped while being read; ask again")
        arrays = {k: np.concatenate(parts[k], axis=1) for k in ("x", "y", "intensity", "status")}
        return self._turn_by_turn(start, arrays, None, bpms)

    def _turn_by_turn(
        self, start: int, arrays: dict[str, np.ndarray], what: str | None, bpms=None
    ) -> TurnByTurn:
        x = arrays["x"].astype(float)
        y = arrays["y"].astype(float)
        intensity = arrays["intensity"].astype(float)
        status = arrays["status"].astype(int)
        covered = start + x.shape[1]
        info = self.info
        names, s = tuple(info.bpms), np.asarray(info.s)
        planes = (PlaneData(names, s, x, intensity, status), PlaneData(names, s, y, intensity, status))
        if bpms is not None:
            planes = tuple(p.select(bpms) for p in planes)
        return TurnByTurn(
            ring=info.branch,
            x=planes[0],
            y=planes[1],
            first_turn=start,
            time=info.start + timedelta(seconds=start * info.revolution_period),
            revolution_period=info.revolution_period,
            source=f"virtual {info.branch} run {info.id} ({info.engine}, {info.fidelity})"
            + (f", {what}" if what else ""),
            metadata={
                "run": info.id,
                "engine": info.engine,
                "fidelity": info.fidelity,
                "lattice_digest": info.lattice_digest,
                "events": self.events(start, covered),
            },
        )

    def orbit(self, start: int | None = None, stop: int | None = None) -> OrbitSeries:
        """The slow orbit, for samples starting in turns ``[start, stop)``: the mean reading over each
        ``slow_period`` seconds of the run (or each chunk)."""
        n = len(self.info.bpms)
        width = 4 + 3 * n
        path = self.directory / "orbit.bin"
        raw = np.fromfile(path, dtype=np.float64) if path.exists() else np.zeros(0)
        records = raw[: len(raw) // width * width].reshape(-1, width)
        turn = records[:, 0].astype(int)
        keep = np.ones(len(records), bool)
        if start is not None:
            keep &= turn >= start
        if stop is not None:
            keep &= turn < stop
        records = records[keep]
        return OrbitSeries(
            ring=self.info.branch,
            bpms=tuple(self.info.bpms),
            s=np.asarray(self.info.s),
            turn=records[:, 0].astype(int),
            turns=records[:, 1].astype(int),
            time=records[:, 2],
            start=self.info.start,
            x=records[:, 4 : 4 + n],
            y=records[:, 4 + n : 4 + 2 * n],
            intensity=records[:, 4 + 2 * n :],
            source=f"virtual {self.info.branch} run {self.info.id}",
        )

    def orbit_times(self) -> np.ndarray:
        """The slow orbit's samples' wall-clock times (s, Unix), in the order of :meth:`orbit`."""
        width = 4 + 3 * len(self.info.bpms)
        path = self.directory / "orbit.bin"
        raw = np.fromfile(path, dtype=np.float64) if path.exists() else np.zeros(0)
        return raw[: len(raw) // width * width].reshape(-1, width)[:, 3]

    def events(self, start: int | None = None, stop: int | None = None) -> list[dict[str, Any]]:
        """What happened during the run, in order, with the turn before which it came."""
        path = self.directory / "events.jsonl"
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if (start is None or event["turn"] >= start) and (stop is None or event["turn"] < stop):
                out.append(event)
        return sorted(out, key=lambda e: e["turn"])  # a capture is written once its turns are, after others'


def _save(path: Path, x, y, intensity, status) -> None:
    temporary = path.with_suffix(".part.h5")
    with h5py.File(temporary, "w") as out:
        out["x"] = np.asarray(x).astype(np.float32)
        out["y"] = np.asarray(y).astype(np.float32)
        out["intensity"] = np.asarray(intensity).astype(np.float32)
        out["status"] = np.asarray(status).astype(np.uint8)
    os.replace(temporary, path)


def _write_atomically(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
