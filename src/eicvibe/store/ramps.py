"""A ring's ramp profiles: its energy and RF programs in time (ADR 0013).

::

    ramps/<name>.yaml   the ring (branch), the energy waveform's parameters, the injection porch,
                        the flattop and the cycle, the cavities sharing the RF, and its source
    ramps/<name>.csv    the RF program: time (s from the ramp's start), voltage (V, the cavities'
                        sum), phase (the synchronous phase from the zero crossing, rad, positive where
                        the bunch gains energy), energy (eV, the designers' at that time, to check)

The energy program is kept as its designers' parameters (``eicvibe.domain.machine.ramps``), the RF
program as the table they computed.
"""

from pathlib import Path

from eicvibe.domain.machine.ramps import RampProfile

from .yamlio import FileError, flow, mapping, read_table, read_yaml, write_table, write_yaml

RAMPS_DIR = "ramps"
COLUMNS = ("time", "voltage", "phase", "energy")

_HEADER = """\
The ramp {name} of ring {branch} (ADR 0013): its energy and RF programs, the injection porch and
the flattop. Energies are total energies (eV); a turn is the revolution period at the speed of
light. The energy program is its designers' waveform, by its parameters: a smooth start over
smooth_turns turns up to the linear rate (eV/s), a linear ramp to linear_end, a smooth end into
extraction. The RF program (the CSV beside this file) is a table in time from the ramp's start;
cavities share its voltage equally (all of the ring's when none are listed)."""

_RF_NOTE = (
    "time (s from the ramp's start), voltage (V, all cavities), phase (synchronous, rad, from the zero "
    "crossing, positive accelerating), energy (eV, the designers', to check)"
)


def write_ramp(folder: Path, profile: RampProfile) -> list[Path]:
    """Write a ramp profile into a machine folder; returns the files written."""
    directory = Path(folder) / RAMPS_DIR
    csv = directory / f"{profile.name}.csv"
    rf = profile.rf
    energies = rf.energy or (float("nan"),) * len(rf.time)
    write_table(csv, list(COLUMNS), zip(rf.time, rf.voltage, rf.phase, energies, strict=True), _RF_NOTE)
    energy = profile.energy.model_dump()
    items = [
        ("branch", profile.branch),
        ("energy", flow(mapping(energy.items()))),
        ("rf", csv.name),
        ("porch", profile.porch),
        ("flattop", profile.flattop),
        ("cycle", profile.cycle),
    ]
    if profile.cavities:
        items.append(("cavities", flow(list(profile.cavities))))
    if profile.source:
        items.append(("source", mapping(profile.source.items())))
    header = _HEADER.format(name=profile.name, branch=profile.branch)
    return [write_yaml(directory / f"{profile.name}.yaml", mapping(items), header), csv]


def read_ramps(folder: Path) -> dict[str, RampProfile]:
    """The ramp profiles in a machine folder, by name (the file's stem); none without a ramps folder."""
    directory = Path(folder) / RAMPS_DIR
    ramps = {}
    for path in sorted(directory.glob("*.yaml")) if directory.is_dir() else ():
        data = read_yaml(path)
        if not isinstance(data, dict) or "rf" not in data:
            raise FileError(path, "not a ramp profile: it needs branch, energy and rf")
        header, rows = read_table(directory / str(data["rf"]))
        if missing := [c for c in COLUMNS[:3] if c not in header]:
            raise FileError(directory / str(data["rf"]), f"needs the columns {', '.join(missing)}")
        columns = {name: tuple(row[header.index(name)] for row in rows) for name in header}
        energy = columns.get("energy", ())
        try:
            ramps[path.stem] = RampProfile.model_validate(
                {
                    "name": path.stem,
                    "branch": data.get("branch"),
                    "energy": dict(data.get("energy") or {}),
                    "rf": {
                        "time": columns["time"],
                        "voltage": columns["voltage"],
                        "phase": columns["phase"],
                        "energy": () if all(e != e for e in energy) else energy,
                    },
                    "cavities": tuple(data.get("cavities") or ()),
                    "porch": data.get("porch", 0.0),
                    "flattop": data.get("flattop", 0.0),
                    "cycle": data.get("cycle", 0.0),
                    "source": dict(data.get("source") or {}),
                }
            )
        except ValueError as error:
            raise FileError(path, f"not a ramp profile: {error}") from None
    return ramps
