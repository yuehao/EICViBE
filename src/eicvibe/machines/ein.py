"""The electron injector complex (EIN) as one machine.

The sources live in the EIN folder of the virtual accelerator project: MAD-X
files staged in ``EICVIBE_lattice/madx`` (converted from the Bmad and Elegant
designs of the ``einlattices`` checkout by ``EIN_connection.ipynb``) and the
preinjector beam in ``einlattices/LTR``. They are the LTR, the BAR with its
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
import json
import math
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType

from eicvibe.domain.lattice import SPECIES, BranchKind, ElementKind, Lattice, LatticeBuilder, derived_id
from eicvibe.domain.machine import (
    MachineRegistry,
    NamingArea,
    NamingRepository,
    NamingScheme,
    add_shared_magnet_supply,
    default_repository,
    generate_machine_registry,
)
from eicvibe.importers.madx import ImportNote, import_madx

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
_BRANCHES = ("ltr", "itl", "bar", "etl", "rcs", "rte")


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
) -> MachineRegistry:
    """Devices and channels of every EIN branch, named with ``areas`` and ``scheme``.

    The shared septum is one device at its three traversals, with one pulsed
    supply that sets the field of the injection and extraction bends and
    never that of the BAR's kicker. Device ids do not depend on the names, so
    a registry regenerated with official names keeps the identity of every device.
    """
    lattice = ein.lattice if isinstance(ein, Ein) else ein
    named = ein_areas(lattice, areas)
    registry = generate_machine_registry(lattice, named, scheme=scheme)
    return add_shared_magnet_supply(
        registry,
        lattice,
        SEPTUM,
        home="circulation",
        driven=_SEPTUM_DRIVEN,
        areas=named["bar"],
        scheme=scheme,
    )


def ein_repository(areas: Mapping[str, str] | None = None) -> NamingRepository:
    """The naming repository extended with the temporary EIN area prefixes.

    With it, :meth:`MachineRegistry.naming_problems` reports what is not
    temporary by choice, such as signal tokens awaiting approval.
    """
    prefixes = {**TEMPORARY_AREAS, **(areas or {})}.values()
    return default_repository().extended(areas=prefixes, note="temporary EIN area prefixes")


#: Files :func:`save_ein` writes.
OUTPUT_FILES = {
    "lattice": "ein_lattice.json",
    "bar_mode": "ein_bar_mode.json",
    "registry": "ein_registry.json",
    "manifest": "ein_manifest.json",
}


def save_ein(ein: Ein, folder: str | Path, *, registry: MachineRegistry | None = None) -> dict[str, Path]:
    """Write the lattice, the BAR-mode settings, the device registry and a manifest into ``folder``.

    The lattice is its canonical JSON (``Lattice.from_json`` reads it back).
    The manifest records every source file with its SHA-256, the MAD-X and
    EICViBE versions and the lattice digest, so a file can be traced to its inputs.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    registry = registry or ein_registry(ein)
    paths = {key: folder / name for key, name in OUTPUT_FILES.items()}
    paths["lattice"].write_text(ein.lattice.to_json(indent=1) + "\n")
    bar_mode = {"lattice_digest": ein.lattice.digest, "settings": [asdict(s) for s in ein.bar_mode]}
    paths["bar_mode"].write_text(json.dumps(bar_mode, indent=1) + "\n")
    paths["registry"].write_text(registry.model_dump_json(indent=1) + "\n")
    manifest = {
        "lattice": OUTPUT_FILES["lattice"],
        "lattice_digest": ein.lattice.digest,
        "eicvibe_version": version("eicvibe"),
        "madx_version": ein.madx_version,
        "sources": {
            role: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for role, path in ein.sources.items()
        },
        "branches": {b.name: round(ein.lattice.layout(b.name).length, 9) for b in ein.lattice.branches},
        "notes": dict(Counter(f"{n.severity.value}: {n.code}" for n in ein.notes)),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=1) + "\n")
    return paths


def main(argv: list[str] | None = None) -> None:
    """Build the EIN from an EIN folder and save it there: ``python -m eicvibe.machines.ein FOLDER``."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "folder", type=Path, help="the EIN folder holding EICVIBE_lattice/madx and einlattices"
    )
    parser.add_argument("--out", type=Path, help="output folder (default: FOLDER/EICVIBE_lattice)")
    args = parser.parse_args(argv)
    ein = build_ein(args.folder)
    for path in save_ein(ein, args.out or args.folder / "EICVIBE_lattice").values():
        print(path)


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
    main()
