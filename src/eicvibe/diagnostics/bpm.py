"""BPM data as analysis reads it, whether a real BPM system or a virtual ring produced it (ADR 0007).

A :class:`TurnByTurn` acquisition holds, for each plane, the positions every
BPM read on every turn: ``position[bpm, turn]`` in mm, as BPM systems report
them (the RHIC turn-by-turn files, the ``:rdx``/``:rdy`` channels of ADR 0003),
with the BPMs' names and s positions, the sum signal and a status per
reading. The planes have their own BPM lists, because single-plane BPMs exist
(RHIC's are). An :class:`OrbitSeries` is the slow orbit: positions averaged over
many turns, sample after sample.

Readers of real data (:mod:`.formats`) and the virtual ring's history
(:mod:`eicvibe.virtual`) produce these, so the same analysis (:mod:`.analysis`)
runs on either.
"""

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

import numpy as np

#: A reading's status: valid, or why not.
VALID = 0
NO_BEAM = 1
"""No particle reached the BPM (lost beam)."""


@dataclass(frozen=True, eq=False)
class PlaneData:
    """One plane of a turn-by-turn acquisition."""

    bpms: tuple[str, ...]
    """BPM names: EIC names in EICViBE, the control system's elsewhere."""
    s: np.ndarray
    """s of every BPM, m."""
    position: np.ndarray
    """Readings ``[bpm, turn]``, mm; NaN where not valid."""
    intensity: np.ndarray | None = None
    """Sum signal ``[bpm, turn]``: bunch charge in nC from a virtual ring; raw units from a real system."""
    status: np.ndarray | None = None
    """``[bpm, turn]``: VALID, NO_BEAM, or a system's own code."""

    def __post_init__(self) -> None:
        n = len(self.bpms)
        if self.position.ndim != 2 or self.position.shape[0] != n or len(self.s) != n:
            raise ValueError(f"{n} BPMs, {len(self.s)} s values and positions of shape {self.position.shape}")
        for name in ("intensity", "status"):
            value = getattr(self, name)
            if value is not None and value.shape != self.position.shape:
                raise ValueError(f"{name} has shape {value.shape}, positions {self.position.shape}")

    @property
    def turns(self) -> int:
        return self.position.shape[1]

    def select(self, bpms: list[str] | tuple[str, ...]) -> PlaneData:
        """These BPMs only, in this order; BPMs the plane does not have are left out."""
        index = [self.bpms.index(b) for b in bpms if b in self.bpms]
        pick = (
            (lambda a: None if a is None else a[index]) if index else (lambda a: None if a is None else a[:0])
        )
        return PlaneData(
            tuple(self.bpms[i] for i in index),
            self.s[index],
            self.position[index],
            pick(self.intensity),
            pick(self.status),
        )

    def window(self, start: int, count: int) -> PlaneData:
        """Turns ``start`` to ``start + count`` of this acquisition (positions in it, not turn numbers)."""
        cut = slice(start, start + count)
        return PlaneData(
            self.bpms,
            self.s,
            self.position[:, cut],
            None if self.intensity is None else self.intensity[:, cut],
            None if self.status is None else self.status[:, cut],
        )


@dataclass(frozen=True, eq=False)
class TurnByTurn:
    """A turn-by-turn acquisition of a ring's BPMs."""

    ring: str
    """The ring, e.g. bar, or Blue for RHIC's blue ring."""
    x: PlaneData
    y: PlaneData
    first_turn: int
    """Turn number of the first turn, in the counting of whatever produced the data."""
    time: datetime | None
    """When the first turn was read; a virtual ring's own clock: its start plus turns times the period."""
    revolution_period: float | None = None
    """s"""
    source: str = ""
    """Where the data comes from: a file, or a virtual ring's run."""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def turns(self) -> int:
        return max(self.x.turns, self.y.turns)

    def plane(self, name: str) -> PlaneData:
        if name not in ("x", "y"):
            raise ValueError(f"no plane {name!r}; the planes are x and y")
        return getattr(self, name)

    def select(self, bpms: list[str] | tuple[str, ...]) -> TurnByTurn:
        return replace(self, x=self.x.select(bpms), y=self.y.select(bpms))

    def window(self, start: int, count: int) -> TurnByTurn:
        return replace(
            self,
            x=self.x.window(start, count),
            y=self.y.window(start, count),
            first_turn=self.first_turn + start,
        )


@dataclass(frozen=True, eq=False)
class OrbitSeries:
    """The slow orbit: readings averaged over many turns, one sample after another."""

    ring: str
    bpms: tuple[str, ...]
    s: np.ndarray
    turn: np.ndarray
    """First turn of every sample."""
    turns: np.ndarray
    """Turns averaged in every sample."""
    time: np.ndarray
    """Seconds since ``start``, at every sample's first turn."""
    start: datetime | None
    x: np.ndarray
    """``[sample, bpm]``, mm"""
    y: np.ndarray
    intensity: np.ndarray | None = None
    source: str = ""
