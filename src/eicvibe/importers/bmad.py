"""Bmad lattices, through Bmad's own MAD-X writer, corrected.

Bmad reads its lattice (CALLs, overlays, superposition, expressions) and
writes it as MAD-X (``bmad_to_mad_sad_elegant``), which
:func:`~eicvibe.importers.madx.import_madx` then imports: Bmad is the
authority on its own language, as MAD-X is on its own (ADR 0002). The writer
loses three things on the way, which :func:`bmad_to_madx` puts back:

- **Elements superimposed on a pipe.** Bmad makes each overlap a slave of
  two lords, the pipe and the element (``LTR_PIP01\\LTR_QUA01``), and the
  writer names it after the pipe (``LTR_PIP01_L1001``). Such pipes become
  drifts before Bmad reads the lattice; a pipe is a drift with an aperture,
  and apertures are not written, so nothing else changes (checked: every
  element keeps its kind, length and position), and each element keeps its
  own name.
- **Monitors and instruments**, which the writer writes as drifts. Bmad's
  element list (from Tao) names them, and they become MAD-X monitors and
  instruments again.
- **A patch's x_pitch.** The writer gives MAD-X ``YROTATION = -x_pitch``,
  but MAD-X turns the frame as Bmad does: ``x_pitch = +a`` and
  ``YROTATION = +a`` both turn it by ``a`` toward +x (survey theta = +a) and
  give a beam on the axis px = -a (checked with Tao and MAD-X 5.09). The
  angle's sign is corrected. The writer's ``XROTATION = -y_pitch`` and
  ``SROTATION = tilt`` are right.

A monitor that a zero-length element splits (a marker at its centre) keeps
its name on its first piece, which ends where the marker is; the rest is a
drift. Another element split that way stays in pieces, with a warning.
Apertures are left out (the MAD-X that cpymad embeds refuses ``apertype``).

The Bmad programs come from ``bin_dir``, else ``$EICVIBE_BMAD_BIN``, Bmad's
``$ACC_EXE``, the PATH, or ``~/packages/bmad-ecosystem/production/bin``.
"""

import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .madx import ImportNote, Severity

CONVERTER = "bmad_to_mad_sad_elegant"
TAO = "tao"
_DEFAULT_BIN = Path.home() / "packages/bmad-ecosystem/production/bin"
_READOUTS = {"monitor": "monitor", "detector": "monitor", "instrument": "instrument"}


class BmadUnavailable(RuntimeError):
    """The Bmad programs this needs are not installed."""


@dataclass(frozen=True, slots=True)
class BmadTools:
    converter: Path
    tao: Path


@dataclass(frozen=True, slots=True)
class BmadElement:
    """An element of the tracking lattice as Bmad has it: name, key, s at its exit, length (m)."""

    name: str
    key: str
    s: float
    length: float


@dataclass(frozen=True, slots=True)
class BmadConversion:
    """The MAD-X file written, the corrections made, and Bmad's own element list to check it by."""

    madx: Path
    notes: tuple[ImportNote, ...]
    elements: tuple[BmadElement, ...]

    @property
    def warnings(self) -> tuple[ImportNote, ...]:
        return tuple(note for note in self.notes if note.severity is Severity.WARNING)


def bmad_tools(bin_dir: str | Path | None = None) -> BmadTools:
    """The Bmad converter and Tao, from ``bin_dir`` or the usual places (see the module)."""
    candidates = [bin_dir, os.environ.get("EICVIBE_BMAD_BIN"), os.environ.get("ACC_EXE")]
    for folder in [Path(c) for c in candidates if c]:
        if (folder / CONVERTER).is_file() and (folder / TAO).is_file():
            return BmadTools(folder / CONVERTER, folder / TAO)
    if bin_dir is None and (converter := shutil.which(CONVERTER)) and (tao := shutil.which(TAO)):
        return BmadTools(Path(converter), Path(tao))
    if bin_dir is None and (_DEFAULT_BIN / CONVERTER).is_file() and (_DEFAULT_BIN / TAO).is_file():
        return BmadTools(_DEFAULT_BIN / CONVERTER, _DEFAULT_BIN / TAO)
    where = bin_dir or "$EICVIBE_BMAD_BIN, $ACC_EXE, the PATH or ~/packages/bmad-ecosystem/production/bin"
    raise BmadUnavailable(f"{CONVERTER} and {TAO} not found in {where}")


def tao_table(
    lattice: str | Path,
    columns: Sequence[str],
    tools: BmadTools | None = None,
    sources: Sequence[str] = ("*.bmad",),
) -> list[dict]:
    """One row per element of branch 0's tracking lattice: Tao's ``lat_list`` of ``columns``.

    Columns are Tao's (``ele.name``, ``ele.key``, ``ele.s``, ``ele.a.beta``,
    ``orbit.vec.1``, ``ele.x_position``, ...); names and keys stay strings,
    everything else becomes a float. Tao reads a copy of the lattice's folder
    (the files matching ``sources``), since Bmad writes beside what it reads.
    """
    tools, lattice = tools or bmad_tools(), Path(lattice)
    with tempfile.TemporaryDirectory(prefix="eicvibe-tao-") as scratch:
        _stage(lattice.parent, Path(scratch), sources)
        return _tao_rows(Path(scratch), lattice.name, columns, tools)


def bmad_elements(lattice: str | Path, tools: BmadTools | None = None) -> tuple[BmadElement, ...]:
    """Branch 0's tracking lattice as Bmad has it."""
    rows = tao_table(lattice, _ELEMENT_COLUMNS, tools)
    return tuple(BmadElement(r["ele.name"], r["ele.key"].lower(), r["ele.s"], r["ele.l"]) for r in rows)


_ELEMENT_COLUMNS = ("ele.name", "ele.key", "ele.s", "ele.l")


def _stage(folder: Path, stage: Path, sources: Sequence[str]) -> None:
    for pattern in sources:
        for path in folder.glob(pattern):
            shutil.copy(path, stage / path.name)


def _tao_rows(stage: Path, lattice: str, columns: Sequence[str], tools: BmadTools) -> list[dict]:
    """Tao's ``lat_list`` of ``columns``, run in ``stage`` on its file ``lattice``."""
    # One column a list: Tao puts a name on a line of its own and wraps long lines.
    names = [f"eicvibe_{i}.txt" for i in range(len(columns))]
    commands = [
        f"python -write {name} lat_list -track_only 1@0>>*|model {column}"
        for name, column in zip(names, columns, strict=True)
    ]
    (stage / "eicvibe.tao").write_text("\n".join(commands) + "\n")
    for name in names:
        (stage / name).unlink(missing_ok=True)
    run = subprocess.run(
        [str(tools.tao), "-lat", lattice, "-noplot", "-command", "call eicvibe.tao; quit"],
        cwd=stage,
        capture_output=True,
        text=True,
    )
    if not all((stage / name).is_file() for name in names):
        raise RuntimeError(f"Tao could not list {list(columns)} of {lattice}:\n{run.stdout[-2000:]}")
    outputs = [[line.strip() for line in (stage / n).read_text().splitlines() if line.strip()] for n in names]
    rows = [{} for _ in outputs[0]]
    for column, lines in zip(columns, outputs, strict=True):
        for row, line in zip(rows, lines, strict=True):
            row[column] = line if column in ("ele.name", "ele.key") else float(line)
    return rows


def _elements(stage: Path, lattice: str, tools: BmadTools) -> tuple[BmadElement, ...]:
    rows = _tao_rows(stage, lattice, _ELEMENT_COLUMNS, tools)
    return tuple(BmadElement(r["ele.name"], r["ele.key"].lower(), r["ele.s"], r["ele.l"]) for r in rows)


def bmad_to_madx(
    source: str | Path,
    madx: str | Path,
    *,
    tools: BmadTools | None = None,
    sources: Sequence[str] = ("*.bmad",),
) -> BmadConversion:
    """Convert the Bmad lattice file ``source`` to the MAD-X file ``madx``, corrected (see the module).

    The files of ``source``'s folder matching ``sources`` are copied to a
    scratch folder first, so that CALLs resolve and nothing is written beside
    the design files.
    """
    tools, source, madx = tools or bmad_tools(), Path(source), Path(madx)
    notes: list[ImportNote] = []
    with tempfile.TemporaryDirectory(prefix="eicvibe-bmad-") as scratch:
        stage = Path(scratch)
        _stage(source.parent, stage, sources)
        before = _elements(stage, source.name, tools)
        pipes = _pipes_under_superimposed(stage, before, tools)
        if pipes:
            _make_drifts(stage, pipes)
            for digested in stage.glob("*digested*"):
                digested.unlink()  # Bmad would read its digest of the files as they were
            notes.append(
                ImportNote(
                    "bmad-pipe-as-drift",
                    ", ".join(sorted(pipes)),
                    "pipes with elements superimposed on them read as drifts, so that the elements keep "
                    "their names (apertures are not written)",
                )
            )
        elements = _elements(stage, source.name, tools)
        _same_layout(before, elements, pipes)
        run = subprocess.run(
            [str(tools.converter), "-madx", "-noaperture", source.name],
            cwd=stage,
            capture_output=True,
            text=True,
        )
        written = stage / f"{source.stem}.madx"
        if not written.is_file():
            raise RuntimeError(
                f"{CONVERTER} wrote no MAD-X for {source}:\n{run.stdout[-3000:]}{run.stderr[-2000:]}"
            )
        text = written.read_text()
    text = _readouts(text, elements, notes)
    text = _pitch_signs(text, notes)
    for lords in sorted({e.name for e in elements if "\\" in e.name}):
        notes.append(
            ImportNote(
                "bmad-multiple-lords",
                lords,
                "an overlap of superimposed elements, named by the writer after its first lord",
                Severity.WARNING,
            )
        )
    header = (
        f"// Converted by eicvibe.importers.bmad from {source.name} with {CONVERTER}, corrected:\n"
        + "".join(f"//   {note}\n" for note in notes)
    )
    madx.write_text(header + text)
    return BmadConversion(madx, tuple(notes), elements)


def _pipes_under_superimposed(stage: Path, elements: Sequence[BmadElement], tools: BmadTools) -> set[str]:
    """Pipes that elements are superimposed on, and that are not superimposed themselves."""
    lords = {name for e in elements if "\\" in e.name for name in e.name.split("\\")}
    if not lords:
        return set()
    text = "\n".join(path.read_text() for path in stage.glob("*.bmad"))
    statements = re.sub(r"&\s*\n", " ", text)
    pipes = set()
    for name in lords:
        definition = re.search(rf"(?im)^\s*{re.escape(name)}\s*:\s*(\w+)", statements)
        if definition is None or definition.group(1).lower() != "pipe":
            continue
        superimposed = re.search(
            rf"(?im)(^\s*{re.escape(name)}\s*:.*\bsuperimpose\b)|(\bsuperimpose\s*,\s*element\s*=\s*{re.escape(name)}\b)",
            statements,
        )
        if superimposed is None:
            pipes.add(name)
    return pipes


def _make_drifts(stage: Path, pipes: set[str]) -> None:
    for path in stage.glob("*.bmad"):
        text = path.read_text()
        for name in pipes:
            text = re.sub(rf"(?im)^(\s*{re.escape(name)}\s*:\s*)pipe\b", r"\1DRIFT", text)
        path.write_text(text)


def _same_layout(before: Sequence[BmadElement], after: Sequence[BmadElement], pipes: set[str]) -> None:
    """Every element that is not a drift has its kind, length and exit in both lattices."""

    def placed(elements):
        out = {}
        for e in elements:
            lords = [n.split("#")[0] for n in e.name.upper().split("\\") if n.split("#")[0] not in drifts]
            if lords:
                out[(lords[-1], round(e.s, 9))] = round(e.length, 9)
        return out

    drifts = {p.upper() for p in pipes}
    old, new = placed(before), placed(after)
    if old != new or not math.isclose(before[-1].s, after[-1].s, abs_tol=1e-9):
        moved = sorted(set(old.items()) ^ set(new.items()))[:6]
        raise RuntimeError(f"making {sorted(pipes)} drifts changed the lattice: {moved}")


def _readouts(text: str, elements: Sequence[BmadElement], notes: list[ImportNote]) -> str:
    """Monitors and instruments, which the writer made drifts, back as what they are."""
    pieces: dict[str, list[str]] = defaultdict(list)
    kinds: dict[str, str] = {}
    for e in elements:
        if e.key in _READOUTS and "\\" not in e.name:
            lord = e.name.split("#")[0].upper()
            kinds[lord] = _READOUTS[e.key]
            if "#" in e.name and (piece := e.name.upper().replace("#", "_")) not in pieces[lord]:
                pieces[lord].append(piece)  # a piece of a split element (a repeated element is defined once)
    restored, split = [], []
    for lord in kinds:
        names = pieces.get(lord) or [lord]
        first = names[0]
        pattern = rf"(?m)^{re.escape(first)}: drift, l ="
        if not re.search(pattern, text):
            notes.append(ImportNote("bmad-readout", lord, "not found in the MAD-X file", Severity.WARNING))
            continue
        text = re.sub(pattern, f"{lord}: {kinds[lord]}, l =", text)
        if len(names) > 1:
            text = re.sub(rf"\b{re.escape(first)}\b", lord, text)  # in the lines too
            split.append(f"{lord} ({len(names)} pieces)")
        restored.append(lord)
    if restored:
        notes.append(ImportNote("bmad-readout", ", ".join(restored), "written as drifts; restored"))
    if split:
        notes.append(
            ImportNote(
                "bmad-split-readout",
                ", ".join(split),
                "split by a zero-length element: the first piece keeps the name, the rest are drifts",
            )
        )
    others = sorted(
        {e.name.split("#")[0] for e in elements if "#" in e.name and e.key not in ("drift", "pipe")}
    )
    others = [name for name in others if name.upper() not in kinds]
    if others:
        notes.append(
            ImportNote(
                "bmad-split",
                ", ".join(others),
                "split by a zero-length element into pieces, each its own element",
                Severity.WARNING,
            )
        )
    return text


def _pitch_signs(text: str, notes: list[ImportNote]) -> str:
    """``YROTATION = x_pitch``: the writer gives it the opposite sign."""
    corrected = []

    def flip(match: re.Match) -> str:
        corrected.append(f"{match.group(1)} ({-float(match.group(3)):+.6g} rad)")
        return f"{match.group(1)}{match.group(2)}{-float(match.group(3))!r};"

    text = re.sub(r"(?m)^([\w.]+__y)(: yrotation, angle = )\s*([-+0-9.Ee]+);", flip, text)
    if corrected:
        notes.append(
            ImportNote(
                "bmad-patch-pitch",
                ", ".join(corrected),
                "YROTATION = x_pitch: the writer wrote -x_pitch, which MAD-X applies as the opposite pitch",
            )
        )
    return text
