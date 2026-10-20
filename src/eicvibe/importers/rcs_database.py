"""The RCS parameter database, an EIC workbook, as EICViBE reads it: the ring's ramp (ADR 0013).

The database (``03_RCS Parameter Database.xlsx`` in the EIN folder's ``einlattices``) has a sheet
per ramp. Its energy sheet (``9GeV Lat`` for the 40-cell ring) gives the designers' waveform by its
parameters, each labelled in a cell (``E0=``, ``Ef=``, ``EendLR``, ``RE=``, ``N1=``, the cycle's
times ``t1=`` to ``t7=`` in ms) and evaluated every millisecond in a table below them; its RF sheet
(``RF Ramp 9GeV Lat``) the cavities' voltage and synchronous phase they computed for it, in time
from the ramp's start. :func:`read_rcs_ramp` reads both, and checks the waveform it rebuilds from
the parameters against the sheet's table.
"""

import math
from pathlib import Path

import numpy as np

from eicvibe.domain.machine.ramps import EnergyRamp, RampProfile, RfProgram
from eicvibe.store.beams import file_source

from .xlsx import read_xlsx

C_LIGHT = 299_792_458.0  # m/s
GEV = 1e9
MS = 1e-3


def read_rcs_ramp(
    path: str | Path,
    *,
    energy_sheet: str = "9GeV Lat",
    rf_sheet: str = "RF Ramp 9GeV Lat",
    branch: str = "rcs",
    name: str = "rcs",
) -> RampProfile:
    """A ring's ramp from the RCS parameter database: the waveform of ``energy_sheet``, the RF program
    of ``rf_sheet``. Raises ValueError where the sheets are not as expected, or the waveform rebuilt
    from the sheet's parameters is not the sheet's own table."""
    path = Path(path)
    sheets = {title.strip(): rows for title, rows in read_xlsx(path).items()}
    for sheet in (energy_sheet, rf_sheet):
        if sheet not in sheets:
            raise ValueError(f"{path.name} has no sheet {sheet!r}; it has {', '.join(sheets)}")
    rows = sheets[energy_sheet]
    times = {f"t{k}": _labelled(rows, f"t{k}=", energy_sheet) * MS for k in range(1, 8)}
    energy = EnergyRamp(
        injection=_labelled(rows, "E0=", energy_sheet) * GEV,
        extraction=_labelled(rows, "Ef=", energy_sheet) * GEV,
        linear_end=_labelled(rows, "EendLR", energy_sheet) * GEV,
        rate=_labelled(rows, "RE=", energy_sheet) * GEV,
        smooth_turns=round(_labelled(rows, "N1=", energy_sheet)),
    )
    period = _labelled(rows, "Circ", energy_sheet) / C_LIGHT  # the sheet's turn
    _check_waveform(energy, period, rows, times, energy_sheet)
    rf = _rf_program(sheets[rf_sheet], rf_sheet)
    return RampProfile(
        name=name,
        branch=branch,
        energy=energy,
        rf=rf,
        porch=times["t1"],
        flattop=times["t3"],
        cycle=round(sum(times.values()), 9),
        source=file_source(
            path,
            sheets=[energy_sheet, rf_sheet],
            note="the energy program as its parameters; the RF program as its table",
        ),
    )


def _labelled(rows: list[list[str]], label: str, sheet: str) -> float:
    """The number in the first cell right of the first cell reading ``label``."""
    for row in rows:
        for k, cell in enumerate(row):
            if cell.strip() == label:
                for value in row[k + 1 :]:
                    if (number := _number(value)) is not None:
                        return number
                raise ValueError(f"sheet {sheet!r}: no number right of {label!r}")
    raise ValueError(f"sheet {sheet!r}: no cell {label!r}")


def _number(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _table(rows: list[list[str]], first: str, sheet: str) -> tuple[list[str], list[list[float | None]]]:
    """The table whose header row starts with ``first`` (case aside): its header and its rows of
    numbers, up to the first row without a number in its first column."""
    for k, row in enumerate(rows):
        if row and row[0].strip().lower().startswith(first):
            header = [cell.strip() for cell in row]
            table = []
            for below in rows[k + 1 :]:
                if not below or _number(below[0]) is None:
                    break
                table.append([_number(cell) for cell in below])
            return header, table
    raise ValueError(f"sheet {sheet!r}: no table starting with {first!r}")


def _column(header: list[str], table: list[list[float | None]], *names: str) -> np.ndarray:
    for name in names:
        for k, title in enumerate(header):
            if title.lower() == name:
                return np.array([row[k] if k < len(row) and row[k] is not None else np.nan for row in table])
    raise ValueError(f"no column {' or '.join(map(repr, names))} among {header}")


def _check_waveform(
    energy: EnergyRamp, period: float, rows: list[list[str]], times: dict[str, float], sheet: str
) -> None:
    """The waveform from the parameters is the sheet's own table, from the porch to extraction."""
    header, table = _table(rows, "time (ms)", sheet)
    time = _column(header, table, "time (ms)") * MS
    theirs = _column(header, table, "energy") * GEV
    inside = time <= times["t1"] + times["t2"] + times["t3"] + 1e-9
    turns = (time - times["t1"]) / period
    mine = np.array([energy.energy(n, period) for n in turns])
    worst = float(np.max(np.abs(mine[inside] - theirs[inside]) / theirs[inside]))
    if worst > 1e-9:
        raise ValueError(
            f"sheet {sheet!r}: its parameters do not give its table (relative difference {worst:.2g})"
        )
    ramp = energy.turns(period) * period
    if abs(ramp - times["t2"]) > 1e-9:
        raise ValueError(
            f"sheet {sheet!r}: its ramp lasts {ramp / MS:.6g} ms, not t2 = {times['t2'] / MS:.6g} ms"
        )


def _rf_program(rows: list[list[str]], sheet: str) -> RfProgram:
    header, table = _table(rows, "time", sheet)
    time = _column(header, table, "time (sec)")
    phase = np.radians(_column(header, table, "phase degrees", "phase (deg)"))
    return RfProgram(
        time=tuple(time),
        voltage=tuple(_column(header, table, "voltage")),
        phase=tuple(phase),
        energy=tuple(_column(header, table, "energy (ev)")),
    )
