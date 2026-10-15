"""Beams a machine is given at a branch's start, by a code that tracked what comes before it (ADR 0012).

The EIN's LTR starts where its preinjector ends, and the preinjector is tracked by an injector
code, not by the lattice engines. The beam that code delivered is part of the machine's files:

    beams/<branch>.yaml   where it is delivered, its charge, the reference momentum its
                          coordinates are about, the particles' file and where they came from
    beams/<branch>.csv    the particles: x, px, y, py, zeta, delta (m, momentum over the
                          reference momentum, m, ditto, m ahead of the bunch centre, relative)

A route that starts at the start of such a branch shoots this beam, a sample of it per shot,
rather than a Gaussian matched to the branch's initial optics.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .yamlio import FileError, mapping, read_yaml, write_yaml

BEAMS_DIR = "beams"
COLUMNS = ("x", "px", "y", "py", "zeta", "delta")

_HEADER = """\
The beam delivered at the start of branch {branch} (ADR 0012): particles a code tracked up to it.
Its particles (CSV beside this file) are x, px, y, py, zeta, delta: m, momentum over the
reference momentum below, m, ditto, m ahead of the bunch centre, relative momentum deviation."""


@dataclass(frozen=True, eq=False)
class DeliveredBeam:
    """A bunch's particles at the start of a branch, as an injector code delivered them."""

    branch: str
    """Branch name."""
    particles: np.ndarray = field(repr=False)
    """``[n, 6]``: x, px, y, py, zeta, delta about :attr:`reference_pc`."""
    charge: float
    """C"""
    reference_pc: float
    """eV"""
    source: dict[str, Any] = field(default_factory=dict)
    """Where they came from: the code's file and its SHA-256, and notes."""

    def about(self, reference_pc: float) -> np.ndarray:
        """The particles about another reference momentum (eV)."""
        ratio = self.reference_pc / reference_pc
        p = self.particles.copy()
        p[:, 1] *= ratio
        p[:, 3] *= ratio
        p[:, 5] = (1.0 + p[:, 5]) * ratio - 1.0
        return p


def write_beam(folder: Path, beam: DeliveredBeam) -> list[Path]:
    """Write a delivered beam into a machine folder; returns the files written."""
    directory = Path(folder) / BEAMS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    csv = directory / f"{beam.branch}.csv"
    np.savetxt(csv, beam.particles, delimiter=",", header=",".join(COLUMNS), comments="", fmt="%.10g")
    data = mapping(
        [
            ("branch", beam.branch),
            ("charge", beam.charge),
            ("reference_pc", beam.reference_pc),
            ("particles", csv.name),
            ("count", int(len(beam.particles))),
            ("source", mapping(beam.source.items())),
        ]
    )
    return [write_yaml(directory / f"{beam.branch}.yaml", data, _HEADER.format(branch=beam.branch)), csv]


def read_beams(folder: Path) -> dict[str, DeliveredBeam]:
    """The beams delivered in a machine folder, by branch name (none without a beams folder)."""
    directory = Path(folder) / BEAMS_DIR
    beams = {}
    for path in sorted(directory.glob("*.yaml")) if directory.is_dir() else ():
        data = read_yaml(path)
        try:
            csv = directory / str(data["particles"])
            particles = np.loadtxt(csv, delimiter=",", skiprows=1, ndmin=2)
            branch = str(data["branch"])
            charge, reference = float(data["charge"]), float(data["reference_pc"])
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise FileError(path, f"not a delivered beam: {error}") from None
        if particles.shape[1] != len(COLUMNS):
            raise FileError(csv, f"expected the columns {', '.join(COLUMNS)}")
        source = dict(data.get("source") or {})
        beams[branch] = DeliveredBeam(branch, particles, charge, reference, source)
    return beams


def file_source(path: Path, **notes: Any) -> dict[str, Any]:
    """Where a beam came from: a file and its SHA-256, with notes."""
    return {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), **notes}
