"""The electron injector complex (EIN) as one machine.

The sources live in the EIN folder of the virtual accelerator project: MAD-X
files staged in ``EICVIBE_lattice/madx`` and the preinjector beam in
``einlattices/LTR``. The Bmad designs of the ``einlattices`` checkout (the
LTR and the RTE) are staged by :func:`convert_ein_bmad`, through Bmad's own
writer with its losses corrected (:mod:`eicvibe.importers.bmad`): the LTR's
magnets and BPMs keep their names, and its injection patch its sign. The
Elegant BAR and the MAD-X lines were staged by ``EIN_connection.ipynb``. They are the LTR, the BAR with its
injection and extraction lines (ITL, ETL), the 40-cell 9 GeV RCS
(RCS9GeV40CellV1) and the 10 GeV RTE, connected as in the legacy route
notebooks, with one difference. LTR_P1 and LTR_P2 are not separate
beamlines: they have the same elements as the start and the P2 section of
LTR_main and differ only in quadrupole strengths. They become the BAR-mode
settings of one LTR branch, so that each magnet exists once, as the virtual
accelerator needs one set of PVs per magnet.

Branches and connections, in beam order:

- ``ltr``: LTR_main, cut after MAR.INJ (the rest repeats the RCS injection section);
- a fork after LTR_MARK_P1 to ``itl``, for BAR operation; in direct
  operation the beam continues along ``ltr``;
- ``itl`` to ``bar`` through the septum: ISEPTUM and the ring SEPTUM are the
  injection and circulation traversals of the shared component ``bar_septum``;
- ``bar`` to ``etl`` through the same septum, whose extraction traversal is ESEPTUM.
  The septum is one magnet that each line models for its own beam. Its field
  is on for the injected and extracted beams, so ISEPTUM and ESEPTUM are
  bends; it is pulsed so that the stored beam does not see it, so the BAR's
  SEPTUM is a kicker. It is rectangular (the bends' faces are parallel, 0.4 m
  apart along the stored beam), and the floor geometry joins the lines at its
  faces, which closes the LTR-ITL-BAR-ETL loop exactly. Its one pulsed supply,
  ``bar-sptm18-ps`` with the temporary names, sets the field of ISEPTUM and
  ESEPTUM, which therefore give it explicitly (``Kn0``, not the reference
  field), so that a supply at 0 A means no field;
- ``etl`` back to ``ltr``, landing just after LTR_MARK_P2;
- ``ltr`` to ``rcs`` at the injection point, the marker right after MQA_US;
- a fork at the extraction point, the marker right after ES.PIP03, to
  ``rte``, whose start through MAR.SEP repeats the RCS extraction section and is cut.

The RCS file is written at its extraction energy (9 GeV, 30 MV per cavity).
As in EIN_connection.ipynb, the RCS is pinned at injection: 0.75 GeV and the
RF of the parameter database's ramp at t = 0. Which path the beam takes is a
scenario choice, not part of the lattice (ADR 0001). Not modelled yet: the
preinjector, which exists as GPT and PARMELA input only.

Names are temporary: SEG has not assigned EIC area prefixes to these
branches, so :data:`TEMPORARY_AREAS` holds placeholders that callers replace,
and :class:`~eicvibe.domain.machine.NamingScheme` holds the signal tokens.
"""

import argparse
import hashlib
import itertools
import math
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType

from eicvibe.domain.lattice import (
    SPECIES,
    BranchKind,
    Change,
    ElementKind,
    Lattice,
    LatticeBuilder,
    adopt_ids,
    derived_id,
    diff,
)
from eicvibe.domain.machine import (
    MachineDescription,
    MachineRegistry,
    NamingArea,
    NamingRepository,
    NamingScheme,
    SharedMagnet,
    assemble_registry,
    default_repository,
    describe_machine,
)
from eicvibe.importers.bmad import BmadConversion, BmadTools, bmad_to_madx
from eicvibe.importers.madx import ImportNote, import_madx
from eicvibe.store import (
    FileError,
    Setting,
    check_machine,
    is_machine,
    load_lattice,
    update_names,
    write_machine,
    write_settings,
)
from eicvibe.store.device_files import add_bpms, rename_references
from eicvibe.store.lattice_files import write_lattice
from eicvibe.store.machine import write_index, write_routes

_STAGED = "EICVIBE_lattice/madx"


@dataclass(frozen=True)
class EinSources:
    """EIN source files, relative to the EIN folder, and the elements the RCS connection points follow."""

    ltr: str = f"{_STAGED}/LTR_main.madx"
    ltr_p1: str = f"{_STAGED}/LTR_P1.madx"
    ltr_p2: str = f"{_STAGED}/LTR_P2.madx"
    ltr_beam: str = "einlattices/LTR/beam_pre_inj.bmad"
    itl: str = f"{_STAGED}/ITL.madx"
    bar: str = f"{_STAGED}/BAR.madx"
    etl: str = f"{_STAGED}/ETL.madx"
    rcs: str = f"{_STAGED}/RCS9GeV40CellV1.madx"
    rcs_injection_after: str = "mqa_us"
    """The LTR lands at the RCS marker right after this element."""
    rcs_extraction_after: str = "es.pip03"
    """The RTE leaves at the RCS marker right after this element."""
    rte: str = f"{_STAGED}/RTE_10GeV.madx"


#: The Bmad design files of the EIN and the staged MAD-X files they become (:func:`convert_ein_bmad`).
EIN_BMAD: Mapping[str, str] = MappingProxyType(
    {
        "einlattices/LTR/LTR_main.bmad": f"{_STAGED}/LTR_main.madx",
        "einlattices/LTR/LTR_P1.bmad": f"{_STAGED}/LTR_P1.madx",
        "einlattices/LTR/LTR_P2.bmad": f"{_STAGED}/LTR_P2.madx",
        "einlattices/RTE/RTE_10GeV.bmad": f"{_STAGED}/RTE_10GeV.madx",
    }
)

#: The RCS of the earlier route notebook, RCSV5S1, as written in the einlattices checkout.
RCSV5S1 = {"rcs": "einlattices/RCS/RCSV5S1.madx", "rcs_extraction_after": "es_pip03"}

#: RCS injection energy (total, eV) and total RF voltage (V) at t = 0 of the 9 GeV ramp
#: (RCS parameter database), as pinned in EIN_connection.ipynb.
RCS_INJECTION_ENERGY = 0.75e9
RCS_INJECTION_RF_VOLTAGE = 1.69758e6

#: Temporary EIC area prefixes of the EIN branches, until SEG assigns them. Override any of them.
TEMPORARY_AREAS: Mapping[str, str] = MappingProxyType(
    {"ltr": "ltr", "itl": "itl", "bar": "bar", "etl": "etl", "rcs": "rcs", "rte": "rte"}
)

_RINGS = {"bar", "rcs"}
#: The septum ITL, BAR and ETL share: a bend for injected and extracted beam, a kicker for the stored beam.
SEPTUM = "bar_septum"
_SEPTUM_DRIVEN = ("injection", "extraction")
#: Magnet kinds wired in series strings, by branch: in the rings, dipoles and quadrupoles of the same
#: length and design strength most likely share a supply. An assumption until the EIN power design
#: is known; ein_registry takes others.
EIN_SERIES: Mapping[str, tuple[ElementKind, ...]] = MappingProxyType(
    {"bar": (ElementKind.BEND, ElementKind.QUADRUPOLE), "rcs": (ElementKind.BEND, ElementKind.QUADRUPOLE)}
)
_BRANCHES = ("ltr", "itl", "bar", "etl", "rcs", "rte")

#: The beam's paths (routes.yaml): straight from the linac into the RCS, or through the BAR
#: (into it by the ITL, out by the ETL, back onto the LTR) first. Both go once round the RCS.
EIN_ROUTES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {"direct": ("ltr", "rcs"), "bar": ("ltr", "itl", "bar", "etl", "ltr", "rcs")}
)


@dataclass(frozen=True)
class SettingValue:
    """One setting of an operating mode: a parameter value on an occurrence."""

    occurrence: str
    parameter: str
    value: float


@dataclass(frozen=True)
class Ein:
    """The assembled EIN lattice, its BAR-mode settings, and how it was made."""

    lattice: Lattice
    bar_mode: tuple[SettingValue, ...]
    """Values that switch the LTR from direct to BAR operation; the lattice holds the direct ones."""
    notes: tuple[ImportNote, ...]
    sources: Mapping[str, Path]
    """Every file read, by role."""
    madx_version: str


def build_ein(
    folder: str | Path,
    *,
    sources: EinSources | None = None,
    rcs_energy: float | None = RCS_INJECTION_ENERGY,
    rcs_rf_voltage: float | None = RCS_INJECTION_RF_VOLTAGE,
    workdir: str | Path | None = None,
    name: str = "ein",
) -> Ein:
    """Import the EIN sources under ``folder`` and connect them into one lattice.

    ``rcs_energy`` (total energy, eV) and ``rcs_rf_voltage`` (total over the
    RCS cavities, V) replace the values of the RCS file; None keeps them.
    MAD-X runs in ``workdir``, by default a temporary directory, so that the
    scripts' TWISS and SURVEY output does not land beside the sources. For
    another RCS, pass e.g. ``EinSources(**RCSV5S1)``.
    """
    folder, sources = Path(folder), sources or EinSources()
    with tempfile.TemporaryDirectory(prefix="eicvibe-ein-") as scratch:
        run_in = workdir or scratch

        def load(key: str):
            kind = BranchKind.PERIODIC if key in _RINGS else BranchKind.LINAC
            return import_madx(folder / getattr(sources, key), kind=kind, name=key, workdir=run_in)

        imports = {key: load(key) for key in _BRANCHES}
        modes = {key: load(key).lattice for key in ("ltr_p1", "ltr_p2")}

    counter = itertools.count()
    builder = LatticeBuilder(
        name, ids=lambda prefix: derived_id(prefix, name, "assembly", str(next(counter)))
    )
    for key, result in imports.items():
        builder.include(result.lattice, prefix=f"{key}.")
        builder.update_branch(f"{key}.{result.lattice.branches[0].name}", name=key)

    _cut_after(builder, "ltr", "mar.inj")
    _cut_through(builder, "rte", "mar.sep")
    injection = _marker_after(builder.build(), "rcs", f"rcs.{sources.rcs_injection_after}")
    extraction = _marker_after(builder.build(), "rcs", f"rcs.{sources.rcs_extraction_after}")
    builder.add_fork("ltr", "itl", after="ltr_mark_p1")
    builder.connect_injection("itl", "bar", via=("iseptum", "septum"), component="bar_septum")
    builder.connect_extraction("bar", "etl", via=("septum", "eseptum"), component="bar_septum")
    builder.add_fork("etl", "ltr", to="ltr_mark_p2", to_side="exit")
    builder.add_fork("ltr", "rcs", to=injection, to_side="exit")
    builder.add_fork("rcs", "rte", after=extraction)
    _drop_unplaced(builder)

    draft = builder.build()
    builder.update_element(
        draft.layout("ltr").placements[0].definition.id, **_bmad_beginning(folder / sources.ltr_beam)
    )
    if rcs_energy is not None:
        mass = SPECIES["electron"].mass
        momentum = math.sqrt((rcs_energy - mass) * (rcs_energy + mass))
        reference = {"species_ref": "electron", "pc_ref": momentum, "E_tot_ref": None}
        builder.update_element(draft.layout("rcs").placements[0].definition.id, ReferenceP=reference)
    if rcs_rf_voltage is not None:
        cavities = [p for p in draft.layout("rcs").placements if p.kind is ElementKind.RFCAVITY]
        for definition in {p.definition.id for p in cavities}:
            builder.update_element(definition, RFP={"voltage": rcs_rf_voltage / len(cavities)})
    # The septum's own supply sets its field for the injected and extracted beams: state it, so
    # that zero means no field instead of the reference field.
    [septum] = draft.shared_components
    for traversal in septum.traversals:
        if traversal.role in _SEPTUM_DRIVEN:
            placed = draft.placement(traversal.occurrence)
            builder.update_element(
                placed.definition.id,
                MagneticMultipoleP={"Kn0": placed.element.BendP.angle_ref / placed.length},
                BendP={"Kn0_from_g_ref": False},
            )
    lattice = builder.build()

    bar_mode = (
        *_mode_settings(lattice, "ltr", modes["ltr_p1"], None, "ltr_mark_p1"),
        *_mode_settings(lattice, "ltr", modes["ltr_p2"], "ltr_mark_p2", "mar.inj"),
    )
    notes = tuple(note for result in imports.values() for note in result.notes)
    roles = (*_BRANCHES, "ltr_p1", "ltr_p2", "ltr_beam")
    read = {role: (folder / getattr(sources, role)).resolve() for role in roles}
    return Ein(lattice, bar_mode, notes, read, imports["ltr"].madx_version)


def convert_ein_bmad(folder: str | Path, *, tools: BmadTools | None = None) -> dict[str, BmadConversion]:
    """Stage the EIN's Bmad design files (:data:`EIN_BMAD`) as MAD-X, corrected (ADR 0002).

    Bmad runs on copies of the design folders; the MAD-X files replace those
    in ``EICVIBE_lattice/madx``. Returns each conversion, with its notes.
    """
    folder = Path(folder)
    (folder / _STAGED).mkdir(parents=True, exist_ok=True)
    return {
        source: bmad_to_madx(folder / source, folder / madx, tools=tools) for source, madx in EIN_BMAD.items()
    }


def ein_areas(
    lattice: Lattice, areas: Mapping[str, str | Iterable[NamingArea]] | None = None
) -> dict[str, tuple[NamingArea, ...]]:
    """Naming areas of each EIN branch: :data:`TEMPORARY_AREAS`, with ``areas`` replacing any of them.

    A prefix makes the whole branch one area with positions from its start;
    a sequence of :class:`~eicvibe.domain.machine.NamingArea` splits it, e.g.
    the RCS into its sections.
    """
    chosen = {**TEMPORARY_AREAS, **(areas or {})}
    result = {}
    for branch, value in chosen.items():
        if isinstance(value, str):
            value = (NamingArea(prefix=value, start=0.0, end=lattice.layout(branch).length + 1.0),)
        result[branch] = tuple(value)
    return result


def ein_registry(
    ein: Ein | Lattice,
    *,
    areas: Mapping[str, str | Iterable[NamingArea]] | None = None,
    scheme: NamingScheme | None = None,
    series: Mapping[str, Iterable[ElementKind]] | None = None,
) -> MachineRegistry:
    """Element names, devices and channels of every EIN branch: the registry of :func:`ein_description`."""
    lattice = ein.lattice if isinstance(ein, Ein) else ein
    return assemble_registry(lattice, ein_description(lattice, areas=areas, scheme=scheme, series=series))


def ein_description(
    ein: Ein | Lattice,
    *,
    areas: Mapping[str, str | Iterable[NamingArea]] | None = None,
    scheme: NamingScheme | None = None,
    series: Mapping[str, Iterable[ElementKind]] | None = None,
) -> MachineDescription:
    """The starting naming rules and devices of the EIN, named with ``areas`` and ``scheme``.

    Every magnet is on a power circuit. The kinds in ``series`` (by branch;
    default :data:`EIN_SERIES`) are wired in strings: magnets of the same
    length and design strength share a supply. Other magnets have their own.

    The shared septum is one magnet, named once, with one pulsed supply that
    sets the field of the injection and extraction bends and never that of
    the BAR's kicker. Device ids do not depend on the names, so a registry
    with official names keeps the identity of every device.
    """
    lattice = ein.lattice if isinstance(ein, Ein) else ein
    return describe_machine(
        lattice,
        ein_areas(lattice, areas),
        scheme=scheme,
        series=EIN_SERIES if series is None else series,
        shared={SEPTUM: SharedMagnet(home="circulation", driven=_SEPTUM_DRIVEN)},
    )


def ein_repository(areas: Mapping[str, str] | None = None) -> NamingRepository:
    """The naming repository extended with the temporary EIN area prefixes.

    With it, :meth:`MachineRegistry.naming_problems` reports what is not
    temporary by choice, such as signal tokens awaiting approval.
    """
    prefixes = {**TEMPORARY_AREAS, **(areas or {})}.values()
    return default_repository().extended(areas=prefixes, note="temporary EIN area prefixes")


#: The machine folder, inside the EIN folder.
MACHINE_DIR = "machine"


def ein_source(ein: Ein) -> dict:
    """Where the lattice came from, for machine.yaml: importer, MAD-X, each file read and its SHA-256."""
    return {
        "importer": "eicvibe.machines.ein",
        "eicvibe": version("eicvibe"),
        "madx": ein.madx_version,
        "files": {
            role: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for role, path in ein.sources.items()
        },
        "notes": dict(Counter(f"{n.severity.value}: {n.code}" for n in ein.notes)),
    }


def write_ein_machine(
    ein: Ein, folder: str | Path, description: MachineDescription | None = None
) -> list[Path]:
    """Write a new EIN machine folder: lattice and device files, and the BAR-mode settings.

    The device files start from :func:`ein_description`; from then on people
    edit them, and :func:`update_ein_machine` never touches them.
    """
    folder = Path(folder)
    written = write_machine(folder, ein.lattice, description or ein_description(ein), source=ein_source(ein))
    written.append(_write_bar_mode(folder, ein, ein.lattice))
    if routes := write_routes(folder, {name: list(b) for name, b in EIN_ROUTES.items()}):
        written.append(routes)
    return written


def update_ein_machine(
    ein: Ein, folder: str | Path, *, write: bool = False, devices: bool = False
) -> tuple[Change, ...]:
    """What a fresh import changes in ``folder``'s lattice files; with ``write``, apply it.

    The import keeps the ids of every element, occurrence and branch whose
    name it keeps, or that it only renamed (:func:`renamed_occurrences`), so
    devices stay attached. Writing replaces the lattice files, the provenance
    in machine.yaml and the BAR-mode settings; naming, device and other
    settings files are left as people edited them, unless ``devices``: then
    their references follow the renamed occurrences, and instruments the
    import adds become BPMs (:func:`update_ein_devices`).
    """
    folder = Path(folder)
    old = load_lattice(folder)
    renamed = renamed_occurrences(old, ein.lattice)
    new = adopt_ids(ein.lattice, old, renamed)
    changes = diff(old, new)
    if write and devices:
        update_ein_devices(folder, old, new, renamed)
    if write:
        write_index(folder, new, ein_source(ein))
        write_lattice(new, folder)
        _write_bar_mode(folder, ein, new)
        write_routes(folder, {name: list(b) for name, b in EIN_ROUTES.items()})  # only if there is none
        update_names(folder)  # the EIC names, by the naming files people keep
    return changes


def renamed_occurrences(old: Lattice, new: Lattice) -> dict[tuple[str, str], str]:
    """Occurrences ``new`` has under another name than ``old``: ``(branch, new name)`` to the old name.

    An occurrence is the same when it is of the same kind and length and ends
    at the same place in the same branch, and only its name differs (drifts
    aside, whose names follow the elements around them).
    """

    def unnamed(lattice: Lattice, others: Lattice) -> dict[tuple, str]:
        out = {}
        branches = {b.name for b in others.branches}
        for branch in lattice.branches:
            if branch.name not in branches:
                continue
            names = {p.name for p in others.layout(branch.name).placements}
            for placed in lattice.layout(branch.name).placements:
                if placed.name in names or placed.kind is ElementKind.DRIFT:
                    continue
                key = (branch.name, placed.kind, round(placed.length, 9), round(placed.s_exit, 6))
                out[key] = placed.name
        return out

    before, after = unnamed(old, new), unnamed(new, old)
    return {(key[0], name): before[key] for key, name in after.items() if key in before}


def update_ein_devices(
    folder: str | Path, old: Lattice, new: Lattice, renamed: Mapping[tuple[str, str], str]
) -> list[Path]:
    """Keep the device files on the lattice: references follow ``renamed``, new instruments become BPMs.

    Everything else people wrote in the naming and device files stays.
    Returns the files changed.
    """
    folder = Path(folder)
    changed = rename_references(folder, {f"{b}/{was}": f"{b}/{name}" for (b, name), was in renamed.items()})
    instruments = {
        (branch.name, p.name)
        for branch in old.branches
        for p in old.layout(branch.name).placements
        if p.kind is ElementKind.INSTRUMENT
    }
    added = [
        f"{branch.name}/{p.name}"
        for branch in new.branches
        for p in new.layout(branch.name).placements
        if p.kind is ElementKind.INSTRUMENT
        and (branch.name, p.name) not in instruments
        and (branch.name, p.name) not in renamed
    ]
    if added and (path := add_bpms(folder, added, [b.name for b in new.branches])):
        changed.append(path)
    return changed


def _write_bar_mode(folder: Path, ein: Ein, lattice: Lattice) -> Path:
    """The BAR-mode settings, on ``lattice`` (which may carry adopted ids)."""
    settings = []
    for value in ein.bar_mode:
        placed = ein.lattice.placement(value.occurrence)
        target = lattice.placement(placed.name, branch=ein.lattice.branch(placed.branch_id).name)
        settings.append(Setting(target.id, value.parameter, value.value))
    description = (
        "BAR mode: the LTR quadrupole strengths for the route through the BAR (from LTR_P1 and LTR_P2).\n"
        "The lattice holds the direct-mode strengths. Rewritten when the lattice is imported again."
    )
    return write_settings(folder, "bar_mode", settings, lattice, description)


def main(argv: list[str] | None = None) -> int:
    """Import the EIN from its design files into its machine folder.

    ``python -m eicvibe.machines.ein FOLDER`` writes FOLDER/machine the first
    time. Later it shows how the design files differ from the lattice files,
    and ``--write`` replaces the lattice files (devices, naming and settings
    people edited are kept).
    """
    parser = argparse.ArgumentParser(
        description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "folder", type=Path, help="the EIN folder holding EICVIBE_lattice/madx and einlattices"
    )
    parser.add_argument("--machine", type=Path, help=f"the machine folder (default: FOLDER/{MACHINE_DIR})")
    parser.add_argument("--write", action="store_true", help="replace the lattice files with the import")
    parser.add_argument(
        "--devices",
        action="store_true",
        help="with --write: device references follow renamed magnets, and new instruments become BPMs",
    )
    parser.add_argument(
        "--convert-bmad",
        action="store_true",
        help="first stage the Bmad designs (LTR, RTE) as MAD-X again, through Bmad's writer, corrected",
    )
    parser.add_argument(
        "--bmad-bin", type=Path, help="the folder of Bmad's programs (tao, bmad_to_mad_sad_elegant)"
    )
    args = parser.parse_args(argv)
    machine = args.machine or args.folder / MACHINE_DIR
    if args.convert_bmad:
        from eicvibe.importers.bmad import bmad_tools

        for source, conversion in convert_ein_bmad(args.folder, tools=bmad_tools(args.bmad_bin)).items():
            print(f"{source} -> {conversion.madx.relative_to(args.folder)}")
            for note in conversion.notes:
                print(f"  {note}")
    ein = build_ein(args.folder)
    if not is_machine(machine):
        written = write_ein_machine(ein, machine)
        print(f"wrote a new machine folder, {machine} ({len(written)} files)")
    else:
        changes = update_ein_machine(ein, machine, write=args.write, devices=args.devices)
        if not changes:
            print(f"{machine}: the lattice files match the design files")
        else:
            counts = Counter(f"{c.kind.value} {c.entity}" for c in changes)
            print(f"{machine}: the design files differ from the lattice files:")
            print("  " + ", ".join(f"{n} {what}" for what, n in sorted(counts.items())))
            for change in changes[:40]:
                print(f"  {change}")
            if len(changes) > 40:
                print(f"  ... and {len(changes) - 40} more")
            print(
                ("replaced the lattice files" + (" and updated the device files" if args.devices else ""))
                if args.write
                else "nothing written; --write replaces the lattice files, --devices also the device files"
            )
    try:
        problems = check_machine(machine)
    except FileError as error:
        print(f"error: {error}")
        return 1
    for problem in problems:
        print(f"  {problem}")
    return 1 if problems else 0


def _cut_after(builder: LatticeBuilder, branch: str, occurrence: str) -> None:
    """Remove everything after ``occurrence``."""
    placements = builder.build().layout(branch).placements
    index = next(i for i, p in enumerate(placements) if p.name == occurrence)
    for placed in placements[index + 1 :]:
        builder.remove(placed.id)


def _cut_through(builder: LatticeBuilder, branch: str, occurrence: str) -> None:
    """Remove everything between the BeginningEle and ``occurrence``, inclusive."""
    placements = builder.build().layout(branch).placements
    index = next(i for i, p in enumerate(placements) if p.name == occurrence)
    for placed in placements[1 : index + 1]:
        builder.remove(placed.id)


def _marker_after(lattice: Lattice, branch: str, definition: str) -> str:
    """Name of the marker right after the one placement of ``definition`` in ``branch``."""
    placements = lattice.layout(branch).placements
    indices = [i for i, p in enumerate(placements) if p.definition.name == definition]
    if len(indices) != 1:
        raise ValueError(f"{branch} places {definition!r} {len(indices)} times; expected once")
    following = placements[indices[0] + 1]
    if following.kind is not ElementKind.MARKER:
        raise ValueError(
            f"{branch}: {following.name} after {definition!r} is a {following.kind.value}, not a marker"
        )
    return following.name


def _drop_unplaced(builder: LatticeBuilder) -> None:
    lattice = builder.build()
    placed = {o.element for b in lattice.branches for o in b.occurrences}
    for element in lattice.elements:
        if element.id not in placed:
            builder.remove_element(element.id)


def _bmad_beginning(path: Path) -> dict:
    """TwissP and FloorP from the BEGINNING[...] lines of a Bmad beam file."""
    values = {
        key.lower(): float(value)
        for key, value in re.findall(r"^\s*BEGINNING\[(\w+)\]\s*=\s*(\S+)", path.read_text(), re.MULTILINE)
    }
    twiss_keys = ("beta_a", "alpha_a", "beta_b", "alpha_b", "eta_x", "etap_x", "eta_y", "etap_y")
    twiss = {key: values[key] for key in twiss_keys if key in values}
    floor_keys = (("x_position", "x"), ("y_position", "y"), ("z_position", "z"), ("theta_position", "theta"))
    floor = {target: values[source] for source, target in floor_keys if source in values}
    return {"TwissP": twiss, **({"FloorP": floor} if floor else {})}


def _mode_settings(
    lattice: Lattice, branch: str, mode: Lattice, start: str | None, stop: str
) -> list[SettingValue]:
    """Magnet strengths of ``mode`` that differ from ``branch``, matched element by element.

    Both are compared from the occurrence named ``start`` (from the beginning
    if None) through the one named ``stop``; forks are skipped. The elements
    in that stretch must have the same kinds and lengths.
    """

    def stretch(layout) -> list:
        placements = [p for p in layout.placements[1:] if p.kind is not ElementKind.FORK]
        names = [p.name for p in placements]
        first = 0 if start is None else names.index(start)
        return placements[first : names.index(stop) + 1]

    ours, theirs = stretch(lattice.layout(branch)), stretch(mode.layout(mode.root_branch))
    settings = []
    for a, b in zip(ours, theirs, strict=True):
        if a.kind != b.kind or not math.isclose(a.length, b.length, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{mode.name}: {b.name} does not match {a.name} of {branch}")
        if a.kind in (ElementKind.QUADRUPOLE, ElementKind.SEXTUPOLE, ElementKind.KICKER):
            ours_m, theirs_m = a.element.MagneticMultipoleP, b.element.MagneticMultipoleP
            a_values = ours_m.model_dump() if ours_m else {}
            b_values = theirs_m.model_dump() if theirs_m else {}
            for key, value in b_values.items():
                if value != a_values.get(key, 0.0):
                    settings.append(SettingValue(a.id, f"MagneticMultipoleP.{key}", value))
    return settings


if __name__ == "__main__":
    raise SystemExit(main())
