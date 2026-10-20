"""Machine folders: the lattice, devices, power supplies and settings as files people read and edit.

YAML for structured data, CSV for long tables; see :mod:`.machine` for the
layout and ADR 0005 for the reasons.
"""

from .beams import DeliveredBeam, read_beams, write_beam
from .lattice_files import name_ids
from .machine import (
    Machine,
    Setting,
    build_machine,
    check_machine,
    eic_names,
    is_machine,
    load_lattice,
    load_machine,
    read_settings,
    set_design,
    source_files,
    update_names,
    write_machine,
    write_settings,
)
from .ramps import read_ramps, write_ramp
from .yamlio import FileError

__all__ = [
    "DeliveredBeam",
    "read_beams",
    "write_beam",
    "FileError",
    "Machine",
    "Setting",
    "build_machine",
    "check_machine",
    "eic_names",
    "is_machine",
    "load_lattice",
    "load_machine",
    "name_ids",
    "read_ramps",
    "read_settings",
    "set_design",
    "source_files",
    "update_names",
    "write_machine",
    "write_ramp",
    "write_settings",
]
