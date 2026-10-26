"""Turn-by-turn BPM files, read into and written from :class:`~.bpm.TurnByTurn`.

- **HDF5** (``write_hdf5``, ``read_hdf5``): EICViBE's own files, the default
  wherever it writes BPM data. Per plane (groups ``x`` and ``y``) the BPMs'
  names, s (m), the readings ``position[bpm, turn]`` (mm), the sum signal and
  the status; the ring, first turn, time, revolution period, source and
  metadata (JSON) as attributes of the file. An acquisition reads back as it
  was written. ``read_hdf5`` also reads HDF5 files of plain matrices,
  ``x0``/``y0`` (or ``X``/``Y``) of ``[bpm, turn]`` in nm, with the beam
  current and a comment; those carry no BPM names.
- **SDDS** (``read_sdds``, ``write_sdds``): turn-by-turn files in SDDS, the
  self-describing data set format of Argonne's APS. Parameters carry the time,
  fill number and trigger; arrays the BPM names, gains and averaging per plane;
  one row per reading, turn after turn, with s, position (mm), the raw A and B
  signals and a status. Each plane has its own BPM list. Writing an
  acquisition in this layout lets tools made for these files read it.

``read_sdds_page`` and ``write_sdds_page`` read and write the first page of any
SDDS file in its ASCII form, which is what these files use.
"""

import json
import os
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .bpm import NO_BEAM, VALID, PlaneData, TurnByTurn

# -- EICViBE's turn-by-turn HDF5 ----------------------------------------------------------

#: What an EICViBE turn-by-turn file says it is: its ``layout`` attribute.
HDF5_LAYOUT = "eicvibe turn-by-turn"
HDF5_VERSION = 1


def _plain(value: Any) -> Any:
    """What JSON cannot write as it is: numpy's numbers and arrays, times."""
    if isinstance(value, np.generic | np.ndarray):
        return value.tolist()
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def write_hdf5(tbt: TurnByTurn, path: str | Path) -> None:
    """A turn-by-turn acquisition in EICViBE's HDF5 layout, which ``read_hdf5`` reads back whole."""
    temporary = Path(f"{path}.part")
    with h5py.File(temporary, "w") as f:
        f.attrs["layout"] = HDF5_LAYOUT
        f.attrs["version"] = HDF5_VERSION
        f.attrs["ring"] = tbt.ring
        f.attrs["first_turn"] = tbt.first_turn
        f.attrs["source"] = tbt.source
        f.attrs["metadata"] = json.dumps(tbt.metadata, default=_plain)
        if tbt.time is not None:
            f.attrs["time"] = tbt.time.isoformat()
        if tbt.revolution_period is not None:
            f.attrs["revolution_period"] = tbt.revolution_period
        for name in ("x", "y"):
            data, group = tbt.plane(name), f.create_group(name)
            group["bpms"] = np.array(data.bpms, dtype=h5py.string_dtype())
            group["s"] = np.asarray(data.s, dtype=float)
            group["s"].attrs["units"] = "m"
            group["position"] = np.asarray(data.position, dtype=float)
            group["position"].attrs["units"] = "mm"
            if data.intensity is not None:
                group["intensity"] = np.asarray(data.intensity, dtype=float)
            if data.status is not None:
                group["status"] = np.asarray(data.status)
    os.replace(temporary, path)


def read_hdf5(path: str | Path, *, bpms: list[str] | None = None, s: list[float] | None = None) -> TurnByTurn:
    """A turn-by-turn HDF5 file: EICViBE's own, as ``write_hdf5`` wrote it, or plain matrices.

    Matrices (``x0``/``y0``, ``X``/``Y`` or ``x``/``y``: nm, ``[bpm, turn]``) name no BPMs; give their
    names and s if known, else they are ``bpm001``, ``bpm002``, ... An EICViBE file names its own.
    """
    with h5py.File(path, "r") as f:
        if f.attrs.get("layout") == HDF5_LAYOUT:
            return _eicvibe_layout(f)
        return _matrices(f, path, bpms, s)


def _eicvibe_layout(f: h5py.File) -> TurnByTurn:
    planes = {}
    for name in ("x", "y"):
        group = f[name]
        planes[name] = PlaneData(
            tuple(group["bpms"].asstr()[()]),
            group["s"][()],
            group["position"][()],
            group["intensity"][()] if "intensity" in group else None,
            group["status"][()] if "status" in group else None,
        )
    a = f.attrs
    return TurnByTurn(
        ring=str(a["ring"]),
        x=planes["x"],
        y=planes["y"],
        first_turn=int(a["first_turn"]),
        time=datetime.fromisoformat(a["time"]) if "time" in a else None,
        revolution_period=float(a["revolution_period"]) if "revolution_period" in a else None,
        source=str(a["source"]),
        metadata=json.loads(a["metadata"]),
    )


def _matrices(f: h5py.File, path: str | Path, bpms: list[str] | None, s: list[float] | None) -> TurnByTurn:
    keys = next(
        (
            pair
            for pair in (("x0", "y0"), ("X", "Y"), ("x", "y"))
            if all(isinstance(f.get(k), h5py.Dataset) for k in pair)
        ),
        None,
    )
    if keys is None:
        raise ValueError(f"{path} holds no turn-by-turn data: neither EICViBE's layout nor matrices x0, y0")
    x, y = (np.asarray(f[k][()], dtype=float) * 1e-6 for k in keys)  # nm to mm
    metadata = {
        k: (f[k][()].decode() if isinstance(f[k][()], bytes) else f[k][()].item())
        for k in ("Ib", "comment", "hor. volt", "ver. volt", "n")
        if k in f and f[k].shape == ()
    }
    names = tuple(bpms or (f"bpm{i + 1:03d}" for i in range(x.shape[0])))
    positions = np.asarray(s if s is not None else np.full(len(names), np.nan), dtype=float)
    return TurnByTurn(
        ring="",
        x=PlaneData(names, positions, x),
        y=PlaneData(names, positions, y),
        first_turn=0,
        time=None,
        source=f"HDF5 turn-by-turn file {path}",
        metadata=metadata,
    )


# -- SDDS (ASCII, one page) -------------------------------------------------------------


def _definitions(header: list[str]) -> list[tuple[str, dict[str, str]]]:
    out = []
    for line in header:
        line = line.strip()
        if not line.startswith("&") or line.startswith("&data") or line.startswith("&description"):
            continue
        kind, _, rest = line[1:].partition(" ")
        fields = {}
        for item in rest.replace("&end", "").split(","):
            key, sep, value = item.strip().partition("=")
            if sep:
                fields[key.strip()] = value.strip().strip('"')
        out.append((kind, fields))
    return out


_NUMERIC = {
    "double": float,
    "float": float,
    "long": int,
    "short": int,
    "long64": int,
    "ulong": int,
    "ushort": int,
}


def _value(text: str, kind: str) -> Any:
    if kind in _NUMERIC:
        return _NUMERIC[kind](float(text)) if _NUMERIC[kind] is int else float(text)
    return text


def read_sdds_page(path: str | Path) -> dict[str, Any]:
    """The first page of an ASCII SDDS file: ``{"parameters": {...}, "arrays": {...}, "columns": {...}}``."""
    text = Path(path).read_text(encoding="latin-1")
    if not text.startswith("SDDS"):
        raise ValueError(f"{path} is not an SDDS file")
    head, marker, body = text.partition("&data")
    if not marker:
        raise ValueError(f"{path} has no data section")
    mode_line, _, body = body.partition("\n")
    if "ascii" not in mode_line:
        raise ValueError(f"{path}: only ASCII SDDS is read here, not {mode_line.strip()}")
    definitions = _definitions(head.splitlines())
    lines = [line for line in body.splitlines() if line.strip() and not line.lstrip().startswith("!")]
    position = 0

    def next_line() -> str:
        nonlocal position
        position += 1
        return lines[position - 1]

    parameters, arrays, columns = {}, {}, {}
    for kind, fields in definitions:
        if kind == "parameter":
            raw = next_line().strip()
            value = shlex.split(raw)[0] if raw.startswith('"') else raw
            parameters[fields["name"]] = _value(value, fields.get("type", "double"))
    for kind, fields in definitions:
        if kind == "array":
            count = int(next_line().split()[0])
            tokens: list[str] = []
            while len(tokens) < count:
                tokens += shlex.split(next_line())
            convert = _NUMERIC.get(fields.get("type", "double"))
            arrays[fields["name"]] = (
                np.array([float(t) for t in tokens[:count]]).astype(convert) if convert else tokens[:count]
            )
    names = [
        (fields["name"], fields.get("type", "double")) for kind, fields in definitions if kind == "column"
    ]
    if names:
        rows = int(next_line().split()[0])
        table = [next_line().split() for _ in range(rows)]
        for k, (name, kind) in enumerate(names):
            convert = _NUMERIC.get(kind)
            values = [row[k] for row in table]
            columns[name] = np.array(values, dtype=float).astype(convert) if convert else values
    return {"parameters": parameters, "arrays": arrays, "columns": columns, "definitions": definitions}


def _format(value: Any, kind: str) -> str:
    if kind in ("double", "float"):
        return f"{float(value): .9e}"
    if kind in _NUMERIC:
        return str(int(value))
    text = str(value)
    return f'"{text}"' if (not text or " " in text) else text


def write_sdds_page(
    path: str | Path,
    definitions: list[tuple[str, dict[str, str]]],
    parameters: dict[str, Any],
    arrays: dict[str, Any],
    columns: dict[str, Any],
) -> None:
    """One page of ASCII SDDS with these definitions (as ``read_sdds_page`` returns them) and values."""
    out = ["SDDS1"]
    for kind, fields in definitions:
        items = ", ".join(
            f'{k}="{v}"' if (k == "description" or " " in v) else f"{k}={v}" for k, v in fields.items()
        )
        out.append(f"&{kind} {items}, &end")
    out += ["&data mode=ascii, &end", "! page number 1"]
    for kind, fields in definitions:
        if kind == "parameter":
            out.append(_format(parameters[fields["name"]], fields.get("type", "double")))
    for kind, fields in definitions:
        if kind == "array":
            values = list(arrays[fields["name"]])
            out.append(f"{len(values)}           ! 1-dimensional array {fields['name']}:")
            kind_ = fields.get("type", "double")
            for k in range(0, len(values), 6):
                out.append(" ".join(_format(v, kind_) for v in values[k : k + 6]))
    names = [
        (fields["name"], fields.get("type", "double")) for kind, fields in definitions if kind == "column"
    ]
    if names:
        rows = len(columns[names[0][0]])
        out.append(f"              {rows}")
        for r in range(rows):
            out.append(" ".join(_format(columns[name][r], kind) for name, kind in names) + " ")
    temporary = Path(f"{path}.part")
    temporary.write_text("\n".join(out) + "\n", encoding="latin-1")
    os.replace(temporary, path)


# -- SDDS turn-by-turn ----------------------------------------------------------------

#: The layout of the SDDS turn-by-turn files, as their headers define it.
SDDS_DEFINITIONS: list[tuple[str, dict[str, str]]] = [
    ("parameter", {"name": "datatype", "type": "long"}),
    ("parameter", {"name": "timestamp", "type": "long"}),
    ("parameter", {"name": "timestampns", "type": "long"}),
    ("parameter", {"name": "uid", "type": "long"}),
    ("parameter", {"name": "pid", "type": "long"}),
    ("parameter", {"name": "hostname", "type": "string"}),
    ("parameter", {"name": "filename", "type": "string"}),
    ("parameter", {"name": "rhicfillnum", "type": "long"}),
    ("parameter", {"name": "comment", "type": "string"}),
    ("parameter", {"name": "machine", "type": "short"}),
    ("parameter", {"name": "version", "type": "float"}),
    ("parameter", {"name": "trigevline", "type": "short"}),
    ("parameter", {"name": "trigev", "type": "short"}),
    ("parameter", {"name": "bucket", "type": "long"}),
    ("parameter", {"name": "fasttime", "type": "long"}),
    ("parameter", {"name": "timebegin", "type": "long"}),
    ("parameter", {"name": "timeend", "type": "long"}),
    ("parameter", {"name": "nBpmsX", "type": "long"}),
    ("parameter", {"name": "nTurnsX", "type": "long"}),
    ("parameter", {"name": "nBpmsY", "type": "long"}),
    ("parameter", {"name": "nTurnsY", "type": "long"}),
    *[
        ("array", {"name": f"{name}{plane}", **({"units": unit} if unit else {}), "type": kind})
        for plane in "XY"
        for name, unit, kind in (
            ("swName", "", "string"),
            ("gain", "dB", "short"),
            ("variance", "mm", "float"),
            ("bucket", "", "short"),
            ("skipFactor", "", "short"),
            ("numAveragedTurns", "", "long"),
        )
    ],
    *[
        (
            "column",
            {
                "name": f"{name}{plane}",
                **({"units": unit} if unit else {}),
                "description": text,
                "type": kind,
            },
        )
        for plane in "XY"
        for name, unit, text, kind in (
            ("s", "m", "S position", "float"),
            ("position", "mm", "BPM Position", "float"),
            ("araw", "", "BPM A Corr", "short"),
            ("braw", "", "BPM B Corr", "short"),
            ("status", "", "BPM Status", "long"),
        )
    ],
]


def _sdds_plane(data: dict[str, Any], plane: str) -> PlaneData:
    p, a, c = data["parameters"], data["arrays"], data["columns"]
    n, turns = int(p[f"nBpms{plane}"]), int(p[f"nTurns{plane}"])
    used = n * turns  # rows are turn after turn; the other plane's longer list pads this one's columns

    def matrix(name: str, dtype=float) -> np.ndarray:
        return np.asarray(c[f"{name}{plane}"][:used], dtype=dtype).reshape(turns, n).T

    # Positions are kept as read: what the status bits mean is the analysis's call.
    raw = matrix("araw") + matrix("braw")
    return PlaneData(
        tuple(a[f"swName{plane}"]), matrix("s")[:, 0], matrix("position"), raw, matrix("status", int)
    )


def read_sdds(path: str | Path) -> TurnByTurn:
    """An SDDS turn-by-turn file."""
    data = read_sdds_page(path)
    p = data["parameters"]
    seconds = p.get("timestamp", 0) + p.get("timestampns", 0) * 1e-9
    ring = (  # the ring, where the file was saved under a folder named for it
        "Blue"
        if "/Blue/" in str(p.get("filename", ""))
        else "Yellow"
        if "/Yellow/" in str(p.get("filename", ""))
        else ""
    )
    return TurnByTurn(
        ring=ring,
        x=_sdds_plane(data, "X"),
        y=_sdds_plane(data, "Y"),
        first_turn=0,
        time=datetime.fromtimestamp(seconds, UTC) if seconds else None,
        revolution_period=None,
        source=f"SDDS turn-by-turn file {path}",
        metadata={
            k: p[k]
            for k in ("rhicfillnum", "comment", "trigev", "trigevline", "bucket", "filename", "machine")
            if k in p
        },
    )


def write_sdds(tbt: TurnByTurn, path: str | Path) -> None:
    """A turn-by-turn acquisition in the SDDS turn-by-turn layout, so tools made for those files read it."""
    stamp = tbt.time.timestamp() if tbt.time else 0.0
    meta = tbt.metadata
    parameters: dict[str, Any] = {
        "datatype": meta.get("datatype", 0),
        "timestamp": int(stamp),
        "timestampns": int(round((stamp - int(stamp)) * 1e9)),
        "uid": 0,
        "pid": os.getpid(),
        "hostname": meta.get("hostname", "eicvibe"),
        "filename": str(meta.get("filename", Path(path).name)),
        "rhicfillnum": int(meta.get("rhicfillnum", 0)),
        "comment": str(meta.get("comment", tbt.source or "No comment")),
        "machine": int(meta.get("machine", 0)),
        "version": 1.1,
        "trigevline": int(meta.get("trigevline", 0)),
        "trigev": int(meta.get("trigev", 0)),
        "bucket": int(meta.get("bucket", 0)),
        "fasttime": 0,
        "timebegin": int(stamp),
        "timeend": int(stamp),
    }
    arrays: dict[str, Any] = {}
    columns: dict[str, Any] = {}
    rows = max(len(tbt.x.bpms) * tbt.x.turns, len(tbt.y.bpms) * tbt.y.turns)
    for plane, data in (("X", tbt.x), ("Y", tbt.y)):
        n, turns = len(data.bpms), data.turns
        parameters[f"nBpms{plane}"], parameters[f"nTurns{plane}"] = n, turns
        arrays[f"swName{plane}"] = list(data.bpms)
        for name in ("gain", "variance", "bucket", "skipFactor", "numAveragedTurns"):
            arrays[f"{name}{plane}"] = [0] * n
        status = data.status if data.status is not None else np.where(np.isnan(data.position), NO_BEAM, VALID)
        position = np.nan_to_num(data.position, nan=0.0)
        intensity = data.intensity if data.intensity is not None else np.zeros_like(position)
        half = np.clip(
            np.round(np.nan_to_num(intensity) / 2), -32768, 32767
        )  # the sum signal split into A and B

        s = np.repeat(np.asarray(data.s)[:, None], turns, axis=1)
        for name, matrix in (
            ("s", s),
            ("position", position),
            ("araw", half),
            ("braw", half),
            ("status", status),
        ):
            columns[f"{name}{plane}"] = _turn_after_turn(matrix, rows)
    write_sdds_page(path, SDDS_DEFINITIONS, parameters, arrays, columns)


def _turn_after_turn(matrix: np.ndarray, rows: int) -> np.ndarray:
    """``[bpm, turn]`` as the file's rows (turn after turn, BPM after BPM), padded to ``rows``."""
    values = np.asarray(matrix, dtype=float).T.reshape(-1)
    out = np.zeros(rows)
    out[: len(values)] = values
    return out
