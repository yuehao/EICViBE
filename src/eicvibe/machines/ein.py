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
notebooks, with one difference: there is one LTR and one setting of it for
both paths. The designers' LTR_P1 and LTR_P2 (the LTR retuned for the BAR)
are not imported; the ITL and ETL are matched instead (:func:`match_ein`), so
that the beam reaches P2 with the same optics through the BAR or past it, and
each magnet exists once with one strength.

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
scenario choice, not part of the lattice (ADR 0001).

Before the LTR, the preinjector (:mod:`.preinjector`, ADR 0012): ``prj`` from
the cathode to the LTR, which it forks into at its start, and its gun and spin
diagnostic lines ``prj_gun`` and ``prj_spin``, from its engineers' survey and
PARMELA deck. Its magnets are given by their fields, and the lattice engines do
not track it: the routes start at the LTR, with the beam the preinjector
delivers there (``beams/ltr.csv``, from the deck's run).

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
from dataclasses import dataclass, replace
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
    Route,
    adopt_ids,
    derived_id,
    diff,
    route_through,
    with_design,
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
from eicvibe.engines.match import Matched
from eicvibe.importers.bmad import BmadConversion, BmadTools, bmad_to_madx
from eicvibe.importers.madx import ImportNote, import_madx
from eicvibe.store import (
    FileError,
    check_machine,
    is_machine,
    load_lattice,
    set_design,
    update_names,
    write_machine,
)
from eicvibe.store.beams import DeliveredBeam, file_source, write_beam
from eicvibe.store.device_files import add_bpms, read_devices, rename_references, write_devices
from eicvibe.store.lattice_files import branch_order, write_lattice
from eicvibe.store.machine import eic_names, read_index, write_index, write_routes

from .preinjector import (
    BRANCHES as PREINJECTOR_BRANCHES,
)
from .preinjector import (
    DEVICES as PREINJECTOR_DEVICES,
)
from .preinjector import (
    GUN_DIPOLE,
    SPIN_KICKER,
    PreinjectorSources,
    build_preinjector,
    preinjector_beam,
)

_STAGED = "EICVIBE_lattice/madx"


@dataclass(frozen=True)
class EinSources:
    """EIN source files, relative to the EIN folder, and the elements the RCS connection points follow."""

    ltr: str = f"{_STAGED}/LTR_main.madx"
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
        "einlattices/RTE/RTE_10GeV.bmad": f"{_STAGED}/RTE_10GeV.madx",
    }
)

#: The RCS of the earlier route notebook, RCSV5S1, as written in the einlattices checkout.
RCSV5S1 = {"rcs": "einlattices/RCS/RCSV5S1.madx", "rcs_extraction_after": "es_pip03"}

#: The preinjector's design files (ADR 0012), included in the EIN by default.
PREINJECTOR = PreinjectorSources()

#: RCS injection energy (total, eV) and total RF voltage (V) at t = 0 of the 9 GeV ramp
#: (RCS parameter database), as pinned in EIN_connection.ipynb.
RCS_INJECTION_ENERGY = 0.75e9
RCS_INJECTION_RF_VOLTAGE = 1.69758e6

#: Temporary EIC area prefixes of the EIN branches, until SEG assigns them. Override any of them.
TEMPORARY_AREAS: Mapping[str, str] = MappingProxyType(
    {
        "prj": "prj",
        "prj_gun": "prj",
        "prj_spin": "prj",
        "ltr": "ltr",
        "itl": "itl",
        "bar": "bar",
        "etl": "etl",
        "rcs": "rcs",
        "rte": "rte",
    }
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
#: (into it by the ITL, out by the ETL, back onto the LTR) first; then once round the RCS, on to its
#: extraction (about half a turn more) and along the RTE.
EIN_ROUTES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {"direct": ("ltr", "rcs", "rte"), "bar": ("ltr", "itl", "bar", "etl", "ltr", "rcs", "rte")}
)
#: Full turns round a ring before the route goes on: the RCS's circulating beam, shown once round.
EIN_TURNS: Mapping[str, int] = MappingProxyType({"rcs": 1})
#: Rings the beam is accelerated in: the RCS ramps it to the RTE's momentum.
EIN_ACCELERATED: tuple[str, ...] = ("rcs",)
#: Rings each route stores its beam in: the BAR accumulates and damps, so its beam leaves on the
#: BAR's own optics.
EIN_STORED: Mapping[str, tuple[str, ...]] = MappingProxyType({"bar": ("bar",)})
#: Where the two routes meet: the merge after the ETL's D2 dipole, where the LTR's last part, to
#: the RCS, begins; its quadrupoles match the beam, whichever way it came, to the RCS.
EIN_MERGE = "ltr_mark_p2"
#: The LTR's quadrupoles between its last bend and P1, where the ITL leaves: they set the optics
#: at P1, and across the bypass at P2, without touching the dispersion, for both routes alike.
EIN_P1_QUADS = ("ltr_qua07", "ltr_qua08", "ltr_qua09", "ltr_qua10")


def _routes() -> dict[str, list[str] | dict]:
    """The routes as routes.yaml writes them."""
    return {
        name: {
            "branches": list(branches),
            "stored": list(EIN_STORED.get(name, ())),
            "turns": {r: n for r, n in EIN_TURNS.items() if r in branches},
            "accelerated": [r for r in EIN_ACCELERATED if r in branches],
        }
        for name, branches in EIN_ROUTES.items()
    }


def ein_route(lattice: Lattice, name: str) -> Route:
    """One of the EIN's routes (:data:`EIN_ROUTES`) on ``lattice``."""
    branches = EIN_ROUTES[name]
    return route_through(
        lattice, name, branches, EIN_STORED.get(name, ()),
        {r: n for r, n in EIN_TURNS.items() if r in branches}, [r for r in EIN_ACCELERATED if r in branches],
    )  # fmt: skip


@dataclass(frozen=True)
class Ein:
    """The assembled EIN lattice and how it was made."""

    lattice: Lattice
    notes: tuple[ImportNote, ...]
    sources: Mapping[str, Path]
    """Every file read, by role."""
    madx_version: str
    names: Mapping[tuple[str, str], str] = MappingProxyType({})
    """EIC names the design files give (the preinjector's), by (branch, occurrence name)."""


def build_ein(
    folder: str | Path,
    *,
    sources: EinSources | None = None,
    rcs_energy: float | None = RCS_INJECTION_ENERGY,
    rcs_rf_voltage: float | None = RCS_INJECTION_RF_VOLTAGE,
    workdir: str | Path | None = None,
    name: str = "ein",
    preinjector: PreinjectorSources | None = PREINJECTOR,
) -> Ein:
    """Import the EIN sources under ``folder`` and connect them into one lattice.

    ``rcs_energy`` (total energy, eV) and ``rcs_rf_voltage`` (total over the
    RCS cavities, V) replace the values of the RCS file; None keeps them.
    MAD-X runs in ``workdir``, by default a temporary directory, so that the
    scripts' TWISS and SURVEY output does not land beside the sources. For
    another RCS, pass e.g. ``EinSources(**RCSV5S1)``. ``preinjector``: its
    design files (None: without it).
    """
    folder, sources = Path(folder), sources or EinSources()
    with tempfile.TemporaryDirectory(prefix="eicvibe-ein-") as scratch:
        run_in = workdir or scratch

        def load(key: str):
            kind = BranchKind.PERIODIC if key in _RINGS else BranchKind.LINAC
            return import_madx(folder / getattr(sources, key), kind=kind, name=key, workdir=run_in)

        imports = {key: load(key) for key in _BRANCHES}

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
    pre = build_preinjector(folder, preinjector) if preinjector is not None else None
    if pre is not None:
        builder.include(pre.lattice)
        builder.add_fork("prj", "ltr")  # the preinjector's end is where the LTR starts
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
    [septum] = [c for c in draft.shared_components if c.name == SEPTUM]
    for traversal in septum.traversals:
        if traversal.role in _SEPTUM_DRIVEN:
            placed = draft.placement(traversal.occurrence)
            builder.update_element(
                placed.definition.id,
                MagneticMultipoleP={"Kn0": placed.element.BendP.angle_ref / placed.length},
                BendP={"Kn0_from_g_ref": False},
            )
    lattice = builder.build()

    notes = tuple(note for result in imports.values() for note in result.notes)
    roles = (*_BRANCHES, "ltr_beam")
    read = {role: (folder / getattr(sources, role)).resolve() for role in roles}
    if pre is None:
        return Ein(lattice, notes, read, imports["ltr"].madx_version)
    return Ein(lattice, notes + pre.notes, read | pre.files, imports["ltr"].madx_version, pre.names)


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
    present = {b.name for b in lattice.branches}
    result = {}
    for branch, value in chosen.items():
        if branch not in present:  # an EIN without its preinjector, say
            continue
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
    return assemble_registry(lattice, ein_description(ein, areas=areas, scheme=scheme, series=series))


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
    names = ein.names if isinstance(ein, Ein) else {}
    present = {b.name for b in lattice.branches}
    return describe_machine(
        lattice,
        {b: a for b, a in ein_areas(lattice, areas).items() if b in present},
        scheme=scheme,
        series=EIN_SERIES if series is None else series,
        shared={k: v for k, v in EIN_SHARED.items() if k in {c.name for c in lattice.shared_components}},
        names={lattice.placement(n, branch=b).id: eic for (b, n), eic in names.items()},
    )


#: The magnets several EIN branches pass through, how each is named and which traversals its
#: supply sets: the BAR septum, the gun dipole (the main line's bend; reversed, the gun diagnostic
#: line's) and the spin kicker (off for the main line, the spin diagnostic line's bend when pulsed).
EIN_SHARED: Mapping[str, SharedMagnet] = MappingProxyType(
    {
        SEPTUM: SharedMagnet(home="circulation", driven=_SEPTUM_DRIVEN),
        GUN_DIPOLE: SharedMagnet(home="circulation", driven=("circulation",), abbreviation="dh"),
        SPIN_KICKER: SharedMagnet(home="circulation", driven=("extraction",), abbreviation="kck"),
    }
)


def ein_repository(areas: Mapping[str, str] | None = None) -> NamingRepository:
    """The naming repository extended with the temporary EIN area prefixes.

    With it, :meth:`MachineRegistry.naming_problems` reports what is not
    temporary by choice, such as signal tokens awaiting approval.
    """
    prefixes = {**TEMPORARY_AREAS, **(areas or {})}.values()
    return default_repository().extended(
        areas=prefixes,
        devices=PREINJECTOR_DEVICES,
        note="temporary EIN area prefixes and the preinjector's device abbreviations",
    )


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
    """Write a new EIN machine folder: lattice and device files, and the routes.

    The device files start from :func:`ein_description`; from then on people
    edit them, and :func:`update_ein_machine` never touches them.
    """
    folder = Path(folder)
    written = write_machine(folder, ein.lattice, description or ein_description(ein), source=ein_source(ein))
    if routes := write_routes(folder, _routes()):
        written.append(routes)
    if "preinjector_beam" in ein.sources:
        written += write_beam(folder, ein_delivered_beam(ein.sources["preinjector_beam"], ein.lattice))
    return written


def ein_delivered_beam(path: Path, lattice: Lattice, charge: float = PREINJECTOR.charge) -> DeliveredBeam:
    """The beam the preinjector delivers to the LTR (PARMELA's run), about the LTR's reference momentum."""
    reference = lattice.initial_conditions(lattice.layout("ltr").id).reference.pc_ref
    particles = preinjector_beam(path.parent, reference, PreinjectorSources(beam=path.name))
    note = "PARMELA 3.4 run of the preinjector deck, at its interface to the LTR (ADR 0012)"
    return DeliveredBeam("ltr", particles, charge, reference, file_source(path, code=note))


def add_ein_preinjector(
    machine: str | Path, folder: str | Path, sources: PreinjectorSources = PREINJECTOR
) -> list[Path]:
    """Add the preinjector to an EIN machine folder that has none (ADR 0012).

    Its branches, forked into the LTR; their devices, EIC names and the shared gun dipole and spin
    kicker; and the beam it delivers to the LTR (``beams/ltr``). The other branches, devices, names
    and settings stay as the folder has them: they are read and written back. Returns the files
    written.
    """
    machine = Path(machine)
    old = load_lattice(machine)
    if present := sorted({b.name for b in old.branches} & set(PREINJECTOR_BRANCHES)):
        raise ValueError(f"{machine} already has {', '.join(present)}")
    pre = build_preinjector(folder, sources)
    builder = LatticeBuilder.from_lattice(old)
    builder.include(pre.lattice)
    builder.add_fork("prj", "ltr")  # the preinjector's end is where the LTR starts
    new = builder.build()
    added = Ein(new, pre.notes, pre.files, "", pre.names)
    part = ein_description(added)
    kept = read_devices(machine, old)
    ours = set(PREINJECTOR_BRANCHES)
    prj_ids = {new.layout(b).id for b in ours}
    naming = kept.naming.model_dump()
    naming["areas"] |= {k: v for k, v in part.naming.model_dump()["areas"].items() if k in prj_ids}
    naming["names"] |= part.naming.model_dump()["names"]
    components = {c.id for c in new.shared_components if c.name in (GUN_DIPOLE, SPIN_KICKER)}
    naming["shared"] = [
        *naming["shared"],
        *(s for s in part.naming.model_dump()["shared"] if s["component"] in components),
    ]
    in_prj = {p.id for b in ours for p in new.layout(b).placements}
    supplies = [
        s
        for s in part.supplies
        if any(o in in_prj for c in s.circuits for m in c.magnets for o in m.occurrences)
    ]
    description = type(kept).model_validate(
        {
            "lattice": new.name,
            "naming": naming,
            "supplies": [*(s.model_dump() for s in kept.supplies), *(s.model_dump() for s in supplies)],
            "bpms": [
                *(b.model_dump() for b in kept.bpms),
                *(b.model_dump() for b in part.bpms if b.occurrence in in_prj),
            ],
            "rf": [
                *(r.model_dump() for r in kept.rf),
                *(r.model_dump() for r in part.rf if r.occurrence in in_prj),
            ],
            "tune_monitors": [m.model_dump() for m in kept.tune_monitors],
        }
    )
    source = dict(read_index(machine).get("source") or {})
    files = dict(source.get("files") or {})
    files |= {
        role: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for role, p in pre.files.items()
    }
    source["files"] = files
    order = branch_order(new)
    written = [write_index(machine, new, source, order)]
    written += write_lattice(new, machine, order, eic_names(assemble_registry(new, description)))
    written += write_devices(description, new, machine)
    written += write_beam(machine, ein_delivered_beam(pre.files["preinjector_beam"], new, sources.charge))
    return written


def update_ein_machine(
    ein: Ein, folder: str | Path, *, write: bool = False, devices: bool = False
) -> tuple[Change, ...]:
    """What a fresh import changes in ``folder``'s lattice files; with ``write``, apply it.

    The import keeps the ids of every element, occurrence and branch whose
    name it keeps, or that it only renamed (:func:`renamed_occurrences`), so
    devices stay attached. Writing replaces the lattice files and the provenance
    in machine.yaml; naming, device and settings files are left as people
    edited them, unless ``devices``: then
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
        write_routes(folder, _routes())  # only if there is none
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


@dataclass(frozen=True)
class EinMatch:
    """The EIN's design matched (:func:`match_ein_lattice`): its steps, and what is left at BAR injection."""

    steps: dict[str, Matched]
    """``p2``, ``itl`` and ``rcs``, in order."""
    injection: tuple[float, float, float]
    """The dispersion the beam enters the BAR with less the BAR's own there, and its slope's
    (m, rad), and their invariant H (m): what the ITL cannot match (see :func:`match_ein_lattice`)."""

    @property
    def values(self) -> dict[tuple[str, str], float]:
        """Every value found, the later steps last."""
        return {k: v for m in self.steps.values() for k, v in m.values.items()}

    @property
    def success(self) -> bool:
        return all(m.success for m in self.steps.values())


def match_ein_lattice(lattice: Lattice, *, engine: str = "xsuite", fidelity: str = "standard") -> EinMatch:
    """The EIN's design matched so that the BAR is transparent: one LTR setting for both routes.

    In order: ``p2``, the ETL brings the BAR's beam (stored, so on the BAR's periodic optics where
    it leaves) to P2 with the optics the beam has there past the BAR, all of them (beta, alpha,
    dispersion and its slope), with the ETL's quadrupoles and the LTR's before P1
    (:data:`EIN_P1_QUADS`, which choose the optics at P1 and P2 for both routes); ``itl``, the
    ITL's quadrupoles bring the beam from P1 to the BAR's periodic beta and alpha where it lands;
    ``rcs``, the LTR's quadrupoles after the merge (:data:`EIN_MERGE`) bring the beam, which is
    the same whichever way it came, to the RCS's periodic optics where it lands.

    The ITL cannot match the dispersion as well. Between its bends (D1 after P1, the septum at the
    BAR) the dispersion invariant H is kept by the quadrupoles, and the septum undoes D1 only
    for a beam with the BAR's beta at D1 (about 16 m); the ETL needs the same at D2, before P2.
    The beam past the BAR crosses about 23 m of drift between D1 and D2, so it cannot have that
    beta at both: the dispersion error goes where the BAR damps it, at injection, and
    :attr:`EinMatch.injection` says how large it is. Every quadrupole keeps its polarity, and beta
    in each stretch stays below 1.2 times its design peak (or 120 m). The values are for the
    design (:func:`~eicvibe.store.set_design`, :func:`~eicvibe.domain.lattice.with_design`).
    """
    from eicvibe.engines import engine as find_engine
    from eicvibe.engines import machine_optics, route_optics
    from eicvibe.engines.contracts import Fidelity, Options
    from eicvibe.engines.match import At, Stretch, match_lines
    from eicvibe.engines.model import route_start

    model, options = find_engine(engine), Options(fidelity=Fidelity(fidelity))
    routes = {n: ein_route(lattice, n) for n in EIN_ROUTES}
    whole = machine_optics(model, lattice, options=options)
    ltr = lattice.layout("ltr")
    ids = {b: [p.id for p in lattice.layout(b).placements] for b in _BRANCHES}

    def name(branch_id: str) -> str:
        return lattice.branch(branch_id).name

    def entering(branch: str, occurrence: str):
        """A ring's periodic optics at an occurrence's entrance."""
        at = ids[branch].index(occurrence)
        optics = whole[lattice.layout(branch).id]
        return optics.start if at == 0 else optics.at(ids[branch][at - 1])

    def quads(branch: str, chosen=None) -> list[tuple[str, str]]:
        return [
            (p.id, "MagneticMultipoleP.Kn1")
            for p in lattice.layout(branch).placements
            if p.kind is ElementKind.QUADRUPOLE and (chosen is None or p.name in chosen)
        ]

    def limit(route, inside: set[str]) -> float:
        optics = route_optics(model, lattice, route, options=options)
        betas = [b for o, b in zip(optics.occurrences, optics.beta_x, strict=True) if o in inside]
        betas += [b for o, b in zip(optics.occurrences, optics.beta_y, strict=True) if o in inside]
        return 1.2 * max(max(betas), 100.0)

    def segments(route, branch: str):
        return [s for s in route.segments if name(s.branch) == branch]

    bar, direct = routes["bar"], routes["direct"]
    [itl], [etl] = segments(bar, "itl"), segments(bar, "etl")
    to_itl, merge = segments(bar, "ltr")  # the LTR to P1, and after P2
    stored = segments(bar, "bar")
    p1, p2 = to_itl.end, ids["ltr"][ids["ltr"].index(merge.start) - 1]  # P2: where the ETL's beam joins
    start, _ = route_start(model, lattice, direct, options=options)
    first = ids["ltr"][0]
    to_p2 = Stretch(
        "ltr", ltr.id, first, p2, start, beta_limit=limit(direct, set(ids["ltr"][: ids["ltr"].index(p2) + 1]))
    )
    landing = entering("bar", stored[0].start)
    steps = {
        "p2": match_lines(
            model, lattice,
            [to_p2, Stretch(
                "etl", etl.branch, etl.start, etl.end, _init(whole[stored[-1].branch].at(stored[-1].end)),
                At("ltr", p2), beta_limit=limit(bar, set(ids["etl"])),
            )],
            quads("ltr", EIN_P1_QUADS) + quads("etl"), options=options, keep_signs=True,
        )
    }  # fmt: skip
    steps["itl"] = match_lines(
        model, lattice,
        [Stretch("ltr", ltr.id, first, p1, start), Stretch(
            "itl", itl.branch, itl.start, itl.end, At("ltr", p1),
            {k: getattr(landing, k) for k in ("beta_x", "alpha_x", "beta_y", "alpha_y")},
            beta_limit=limit(bar, set(ids["itl"])),
        )],
        quads("itl"), values=steps["p2"].values, options=options, keep_signs=True, beta_limit=None,
    )  # fmt: skip
    after = set(ids["ltr"][ids["ltr"].index(merge.start) :])
    rcs = segments(direct, "rcs")[0]
    steps["rcs"] = match_lines(
        model, lattice,
        [to_p2, Stretch(
            "merge", ltr.id, merge.start, merge.end, At("ltr", p2),
            _optics(entering("rcs", rcs.start)), beta_limit=limit(direct, after),
        )],
        [v for v in quads("ltr") if v[0] in after], values=steps["p2"].values, options=options,
        keep_signs=True, beta_limit=None,
    )  # fmt: skip
    injected = steps["itl"].stretches["itl"].after
    d_eta, d_etap = injected.eta_x - landing.eta_x, injected.etap_x - landing.etap_x
    gamma = (1 + landing.alpha_x**2) / landing.beta_x
    h = gamma * d_eta**2 + 2 * landing.alpha_x * d_eta * d_etap + landing.beta_x * d_etap**2
    return EinMatch(steps, (d_eta, d_etap, h))


def match_ein(folder: str | Path, *, engine: str = "xsuite", fidelity: str = "standard") -> EinMatch:
    """:func:`match_ein_lattice` of a machine folder's lattice; nothing is written."""
    return match_ein_lattice(load_lattice(Path(folder)), engine=engine, fidelity=fidelity)


def matched_ein(ein: Ein, *, engine: str = "xsuite", fidelity: str = "standard") -> tuple[Ein, EinMatch]:
    """``ein`` with its design matched (:func:`match_ein_lattice`), and the match."""
    found = match_ein_lattice(ein.lattice, engine=engine, fidelity=fidelity)
    return replace(ein, lattice=with_design(ein.lattice, found.values)), found


def _optics(at) -> dict[str, float]:
    return {k: getattr(at, k) for k in ("beta_x", "alpha_x", "beta_y", "alpha_y", "eta_x", "etap_x")}


def _init(at):
    from eicvibe.engines.contracts import TwissInit

    return TwissInit(**(at.model_dump() | {"x": 0.0, "px": 0.0, "y": 0.0, "py": 0.0}))


def _match_report(found: EinMatch, lattice: Lattice) -> list[str]:
    lines = []
    for key, m in found.steps.items():
        for stretch, s in m.stretches.items():
            if not s.targets:
                continue
            lines.append(
                f"{key}, {stretch}: Bmag x {s.bmag_before[0]:.4g} -> {s.bmag_after[0]:.6g}, "
                f"y {s.bmag_before[1]:.4g} -> {s.bmag_after[1]:.6g}; peak beta {s.beta_max[0]:.4g} m, "
                f"{s.beta_max[1]:.4g} m" + ("" if m.success else "  NOT MATCHED within tolerance")
            )
        for (occurrence, parameter), value in m.values.items():
            start = m.start[(occurrence, parameter)]
            if value != start:
                lines.append(f"  {lattice.placement(occurrence).name:12s} Kn1 {start:+.6g} -> {value:+.6g}")
    d_eta, d_etap, h = found.injection
    lines.append(
        f"left at BAR injection (the BAR damps it): dispersion {1e3 * d_eta:+.1f} mm, "
        f"its slope {1e3 * d_etap:+.1f} mrad, H {h:.3g} m"
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    """Import the EIN from its design files into its machine folder, matched.

    ``python -m eicvibe.machines.ein FOLDER`` writes FOLDER/machine the first
    time. Later it shows how the design files differ from the lattice files,
    and ``--write`` replaces the lattice files (devices, naming and settings
    people edited are kept). The import is matched first (match_ein_lattice:
    the BAR transparent, both routes into the RCS) unless --no-match;
    ``--match`` matches the machine folder's own lattice files in place.
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
    parser.add_argument(
        "--match",
        action="store_true",
        help="match the machine folder's lattice files in place (the ITL, ETL and LTR quadrupoles)",
    )
    parser.add_argument(
        "--no-match", action="store_true", help="import the design files' strengths as they are"
    )
    parser.add_argument("--engine", default="xsuite", help="the engine that matches: xsuite or trackpad")
    args = parser.parse_args(argv)
    machine = args.machine or args.folder / MACHINE_DIR
    if args.match:
        if not is_machine(machine):
            parser.error(f"{machine} is not a machine folder yet: import the EIN first")
        lattice = load_lattice(machine)
        found = match_ein_lattice(lattice, engine=args.engine)
        print("\n".join(_match_report(found, lattice)))
        changes = set_design(machine, found.values, lattice)
        print(
            f"set {len(changes)} strengths in the lattice files"
            if changes
            else "already matched: nothing written"
        )
        return 0 if found.success else 1
    if args.convert_bmad:
        from eicvibe.importers.bmad import bmad_tools

        for source, conversion in convert_ein_bmad(args.folder, tools=bmad_tools(args.bmad_bin)).items():
            print(f"{source} -> {conversion.madx.relative_to(args.folder)}")
            for note in conversion.notes:
                print(f"  {note}")
    ein = build_ein(args.folder)
    if not args.no_match:
        ein, found = matched_ein(ein, engine=args.engine)
        print("matched the design (--no-match imports it as it is):")
        print("\n".join(_match_report(found, ein.lattice)))
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


if __name__ == "__main__":
    raise SystemExit(main())
