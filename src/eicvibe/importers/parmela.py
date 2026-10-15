"""PARMELA's files, as EICViBE reads them: a deck's beamline, and a particle distribution (ADR 0012).

PARMELA (Los Alamos, here version 3.4) tracks an injector in time from its cathode, with field
maps and space charge. EICViBE does not run it; it reads what it describes and what it wrote:

- :func:`read_deck`: the beamline of an input deck, element after element with where each starts
  and ends (lengths in cm in the deck, metres here) and its numbers as written. Only the elements
  that make the beamline are kept (``drift``, ``solenoid``, ``quad``, ``bend``, ``steerer``,
  ``cell``, ``trwave``); field-map and run commands are not.
- :func:`read_field_maps`: the field maps a deck reads (``poisson``, a static field such as a DC
  gun's; ``cfield``, a cavity's), with their numbers and files: Poisson Superfish's (r, z) maps,
  written by its SF7 (``.T7``).
- :func:`read_distribution`: a particle file with the columns x (m), x momentum over mc, y (m),
  y momentum over mc, z (m, from the bunch centre, positive ahead) and total energy (MeV), the
  form of the EIN preinjector's output at the LTR. They become EICViBE's tracking coordinates
  about a reference momentum: x, px, y, py (momenta over the reference's), zeta and delta.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eicvibe.domain.lattice import SPECIES

#: The deck elements that make the beamline, each advancing it by its length.
BEAMLINE = ("drift", "solenoid", "quad", "bend", "steerer", "cell", "trwave")
#: The deck commands that read a field map, its file named on the next line.
FIELD_MAPS = ("poisson", "cfield")


@dataclass(frozen=True)
class DeckElement:
    """One beamline element of a PARMELA deck, where it is, and its numbers after the length."""

    kind: str
    start: float
    """m from the cathode"""
    end: float
    values: tuple[float, ...]
    line: int
    """Its line in the deck."""

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class FieldMap:
    """A field map a PARMELA deck reads: its command, the command's numbers and the map's file."""

    command: str
    values: tuple[float, ...]
    """As written: for ``poisson``, the map's z offset (cm) and its scale first; for ``cfield``,
    the number the deck's cells refer to it by."""
    file: str
    line: int
    """The command's line in the deck."""


def read_deck(path: str | Path) -> list[DeckElement]:
    """The beamline of a PARMELA deck, in order: every element with its position (comments, ``!``
    and ``;``, removed; the commands that set fields, errors and the run left out)."""
    elements = []
    z = 0.0
    for number, raw in enumerate(Path(path).read_text(errors="replace").splitlines(), start=1):
        words = _words(raw)
        if not words or words[0].lower() not in BEAMLINE:
            continue
        length = float(words[1]) / 100.0
        values = tuple(float(w) for w in words[2:] if _number(w))
        elements.append(DeckElement(words[0].lower(), round(z, 9), round(z + length, 9), values, number))
        z += length
    return elements


def read_field_maps(path: str | Path) -> list[FieldMap]:
    """The field maps a PARMELA deck reads, in its order, each with the file on the line after it."""
    lines = Path(path).read_text(errors="replace").splitlines()
    maps = []
    for k, raw in enumerate(lines[:-1]):
        words = _words(raw)
        if words and words[0].lower() in FIELD_MAPS and (file := _words(lines[k + 1])):
            values = tuple(float(w) for w in words[1:] if _number(w))
            maps.append(FieldMap(words[0].lower(), values, file[0], k + 1))
    return maps


def read_distribution(path: str | Path, *, reference_pc: float, species: str = "electron") -> np.ndarray:
    """Particles ``[n, 6]`` of a PARMELA-style file, as x, px, y, py, zeta, delta about ``reference_pc``
    (eV): the momenta over the reference momentum, zeta the distance ahead of the bunch centre."""
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] < 6:
        raise ValueError(f"{path}: expected 6 columns (x, px/mc, y, py/mc, z, E), found {data.shape[1]}")
    x, bgx, y, bgy, z, energy = data[:, :6].T
    mass = SPECIES[species].mass  # eV
    reference = reference_pc / mass  # the reference momentum over mc
    momentum = np.sqrt((energy * 1e6 / mass) ** 2 - 1.0)  # each particle's, over mc
    return np.column_stack([x, bgx / reference, y, bgy / reference, z, momentum / reference - 1.0])


def _words(line: str) -> list[str]:
    """A deck line's words, its comments (``!``, ``;``) removed."""
    return line.split("!")[0].split(";")[0].split()


def _number(word: str) -> bool:
    try:
        return math.isfinite(float(word))
    except ValueError:
        return False
