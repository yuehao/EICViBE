"""Import the final lattice of a MAD-X script through cpymad.

MAD-X itself runs the script, so CALLs, macros, MATCH, SEQEDIT and every
other command act as they do in MAD-X. Each requested sequence is then read
back in expanded form, with its implicit drifts and with every expression
evaluated at its final value, and converted to the PALS-style description of
:mod:`eicvibe.domain.lattice`:

- each MAD-X element becomes one element definition, placed once per
  sequence node; element classes defined in the script become tags;
- the sequence's ``$start`` marker becomes the branch's ``BeginningEle``,
  carrying the reference particle of the sequence's BEAM;
- ids derive from the lattice name and the MAD-X names, so re-importing an
  edited script keeps the ids of unchanged elements and
  :func:`~eicvibe.domain.lattice.diff` shows only what changed.

Every conversion that is not a plain copy (a sign flip, an RBEND turned into
a Bend, a dropped attribute, an unsupported element) is recorded as an
:class:`ImportNote`. Field and alignment errors (EFCOMP, EALIGN) are not
imported, since cpymad does not expose them per element.

Conventions, beyond those of :mod:`eicvibe.domain.lattice.conventions`:

- kicks: ``Kn0L = -HKICK`` and ``Ks0L = VKICK``;
- RF: ``phase = 2 pi LAG`` with ``zero_phase`` BELOW_TRANSITION, because
  LAG = 0 is the zero crossing that is stable below transition and LAG = 0.5
  the one stable above it; VOLT in MV and FREQ in MHz become V and Hz;
- XROTATION, YROTATION and SROTATION are right-handed, like PALS ``x_rot``,
  ``y_rot`` and ``z_rot`` (checked against MAD-X SURVEY);
- RBEND: with ``option, rbarc`` (the default) L is the chord, so the length
  becomes the arc length; E1 and E2 gain ANGLE/2 to become sector-face angles.
"""

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from eicvibe.domain.lattice import (
    ALLOWED_GROUPS,
    MAX_MULTIPOLE_ORDER,
    SPECIES,
    BranchKind,
    ElementKind,
    Lattice,
    LatticeBuilder,
    derived_id,
)

if TYPE_CHECKING:
    from cpymad.madx import Element, Madx, Sequence


class Severity(StrEnum):
    INFO = "info"
    """A conversion that keeps the physics, such as a sign flip or a unit change."""
    WARNING = "warning"
    """Something was dropped or approximated, so the imported physics differs."""


@dataclass(frozen=True, slots=True)
class ImportNote:
    """One conversion made while importing."""

    code: str
    subject: str
    """MAD-X element or sequence name(s)."""
    message: str
    severity: Severity = Severity.INFO

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.code}: {self.subject}: {self.message}"


@dataclass(frozen=True, slots=True)
class MadxImport:
    """An imported lattice and the notes made while converting it."""

    lattice: Lattice
    notes: tuple[ImportNote, ...]
    madx_version: str

    @property
    def warnings(self) -> tuple[ImportNote, ...]:
        return tuple(note for note in self.notes if note.severity is Severity.WARNING)


class UnsupportedElementError(ValueError):
    """A MAD-X element without a counterpart here, raised in strict mode."""


type KindSpec = BranchKind | str | Mapping[str, BranchKind | str]


def import_madx(
    path: str | Path,
    sequences: Iterable[str] | None = None,
    *,
    kind: KindSpec = BranchKind.PERIODIC,
    name: str | None = None,
    strict: bool = False,
    stdout: Any = False,
    workdir: str | Path | None = None,
) -> MadxImport:
    """Run the MAD-X script at ``path`` and import its sequences in their final state.

    By default the script runs in its own directory, so relative CALLs resolve
    as in a MAD-X session started there, and any files it writes (TWISS or
    SURVEY output) land there too. ``workdir`` runs it elsewhere, e.g. in a
    temporary directory, which keeps a self-contained script from writing
    beside its source; relative CALLs then resolve from ``workdir``. A STOP
    ends the script. ``stdout`` goes to cpymad: False hides the MAD-X output,
    None shows it, a file captures it. ``name`` defaults to the file name
    without its extension; the other arguments are those of :func:`import_cpymad`.
    """
    from cpymad.madx import Madx

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"no MAD-X script at {path}")
    with Madx(stdout=stdout) as madx:
        try:
            with madx.chdir(str(Path(workdir).resolve() if workdir else path.parent)):
                madx.call(str(path))
        except RuntimeError as error:
            raise RuntimeError(
                f"MAD-X stopped while running {path}; rerun with stdout=None to see why"
            ) from error
        return import_cpymad(madx, sequences, kind=kind, name=name or path.stem, strict=strict)


def import_cpymad(
    madx: Madx,
    sequences: Iterable[str] | None = None,
    *,
    kind: KindSpec = BranchKind.PERIODIC,
    name: str = "madx",
    strict: bool = False,
) -> MadxImport:
    """Import sequences of a cpymad session in its current state.

    ``sequences`` defaults to MAD-X's current sequence, the one last USEd or
    defined; the first one becomes the root branch. ``kind`` is "periodic"
    (a ring, the default) or "linac", for every sequence or by sequence name.
    A sequence that is not expanded yet is expanded with USE, which leaves
    other sequences alone; if it has no BEAM, it first gets MAD-X's default
    one. With ``strict``, an element without a counterpart here raises
    :class:`UnsupportedElementError` instead of becoming a drift or marker.
    """
    converter = _Converter(madx, _LABEL_CHARS.sub("_", name), strict)
    for sequence in _requested(madx, sequences):
        converter.add_sequence(sequence, _branch_kind(kind, sequence))
    lattice = converter.builder.build()
    converter.check_positions(lattice)
    return MadxImport(lattice, converter.finish(), str(madx.version))


def _requested(madx: Madx, sequences: Iterable[str] | None) -> list[str]:
    defined = list(madx.sequence)
    if sequences is None:
        active = madx.sequence()
        if active is None:
            raise ValueError("the MAD-X session has no sequence")
        return [active.name]
    names = [s.lower() for s in sequences]  # MAD-X names are case-insensitive and stored in lower case
    if missing := [s for s in names if s not in defined]:
        raise KeyError(f"no MAD-X sequence {', '.join(missing)} (defined: {', '.join(defined)})")
    return names


def _has_beam(madx: Madx, sequence: str) -> bool:
    """Whether a BEAM command set the beam of ``sequence``: its own, or the global one.

    cpymad has no public list of BEAM commands, hence ``_libmadx``. The global
    beam's SEQUENCE parameter stays empty until a BEAM without SEQUENCE runs.
    """
    return sequence in madx._libmadx.get_beam_names() or bool(madx.beam.sequence)


def _branch_kind(kind: KindSpec, sequence: str) -> BranchKind:
    if isinstance(kind, Mapping):
        by_name = {k.lower(): v for k, v in kind.items()}
        return BranchKind(by_name.get(sequence, BranchKind.PERIODIC))
    return BranchKind(kind)


_LABEL_CHARS = re.compile(r"[^A-Za-z0-9_.\-]")
_IMPLICIT_DRIFT = re.compile(r"drift_\d+")


class _NamedIds:
    """Id factory keyed by the MAD-X name of the entity being created, so that re-imports keep ids."""

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self.key = ""

    def __call__(self, prefix: str) -> str:
        return derived_id(prefix, self.namespace, self.key)


@dataclass
class _Spec:
    kind: ElementKind
    length: float = 0.0
    groups: dict[str, dict[str, Any]] = field(default_factory=dict)
    tags: tuple[str, ...] = ()


#: Conventions noted once per import rather than once per element.
_CONVENTIONS = {
    "kick-sign": "kicks became Kn0L = -HKICK and Ks0L = VKICK",
    "rf-phase": "RF phase became 2 pi LAG mod 2 pi, zero_phase BELOW_TRANSITION; VOLT, FREQ became V, Hz",
    "rbend": "RBEND became a Bend with the arc length, and E1, E2 measured from the sector faces",
    "spread": "KNL/KSL of an order that also has a per-length strength was spread over the length",
    "drift-like": "imported as a Drift (a Marker if zero length); PALS has no counterpart",
}


class _Converter:
    def __init__(self, madx: Madx, name: str, strict: bool) -> None:
        self.madx = madx
        self.strict = strict
        self.ids = _NamedIds(name)
        self.builder = LatticeBuilder(name, ids=self.ids)
        self.notes: list[ImportNote] = []
        self.applied: dict[str, dict[str, None]] = {}  # convention code -> MAD-X element names, in order
        self.definitions: dict[str, str | None] = {}  # MAD-X element key -> definition name, None if skipped
        self.element_labels: set[str] = set()
        self.branch_labels: set[str] = set()
        self.positions: dict[str, list[float]] = {}  # branch -> MAD-X s of each occurrence, in order
        self.lengths: dict[str, float] = {}

    # -- notes ---------------------------------------------------------------

    def warn(self, code: str, subject: str, message: str) -> None:
        self.notes.append(ImportNote(code, subject, message, Severity.WARNING))

    def apply(self, convention: str, subject: str) -> None:
        self.applied.setdefault(convention, {})[subject] = None

    def finish(self) -> tuple[ImportNote, ...]:
        for code, names in self.applied.items():
            subjects = list(names)
            shown = ", ".join(subjects[:5]) + (f" and {len(subjects) - 5} more" if len(subjects) > 5 else "")
            self.notes.append(ImportNote(code, shown, _CONVENTIONS[code]))
        return tuple(self.notes)

    # -- sequences -----------------------------------------------------------

    def add_sequence(self, name: str, kind: BranchKind) -> None:
        sequence = self.madx.sequence[name]
        has_beam = _has_beam(self.madx, name)
        if not sequence.is_expanded:
            if not has_beam:
                self.madx.command.beam(sequence=name)  # USE stops MAD-X for a sequence without a beam
            self.madx.use(sequence=name)
            self.notes.append(
                ImportNote("expanded", name, "the sequence was not in use; it was expanded by USE")
            )
        elements = list(sequence.expanded_elements)
        positions = list(sequence.expanded_element_positions())
        branch = self._label(name, self.branch_labels)
        self.ids.key = f"branch/{name}"
        self.builder.add_branch(branch, kind)
        self.lengths[branch] = sequence.length

        start = elements[0].name if elements and elements[0].name.endswith("$start") else f"{name}$start"
        groups: dict[str, dict[str, Any]] = {}
        if has_beam:
            groups["ReferenceP"] = self._reference(sequence, name)
        else:
            self.warn("no-beam", name, "the sequence has no BEAM, so the reference particle is not set")
        if kind is BranchKind.LINAC:
            self.notes.append(
                ImportNote("no-twiss", name, "set the initial optics as TwissP on the BeginningEle")
            )
        beginning = self._define(start, _Spec(ElementKind.BEGINNING_ELE, groups=groups))
        used = {beginning}
        self._place(branch, beginning, beginning, 0.0)

        # $start and $end markers, including those of nested sequences, are MAD-X bookkeeping.
        placed = [
            (self._definition(e, name), s)
            for e, s in zip(elements, positions, strict=True)
            if "$" not in e.name
        ]
        placed = [(definition, s) for definition, s in placed if definition is not None]
        counts = Counter(definition for definition, _ in placed)
        seen: Counter[str] = Counter()
        for definition, s in placed:
            seen[definition] += 1
            occurrence = definition if counts[definition] == 1 else f"{definition}_{seen[definition]}"
            self._place(branch, definition, self._free(occurrence, used), s)

    def _place(self, branch: str, definition: str, occurrence: str, s: float) -> None:
        self.ids.key = f"occurrence/{branch}/{occurrence}"
        self.builder.append(branch, definition, name=occurrence)
        self.positions.setdefault(branch, []).append(s)

    def check_positions(self, lattice: Lattice) -> None:
        for branch, length in self.lengths.items():
            layout = lattice.layout(branch)
            # Occurrences were appended in MAD-X order, so they pair with the recorded positions.
            for placed, s in zip(layout.placements, self.positions[branch], strict=True):
                if not math.isclose(placed.s_enter, s, rel_tol=1e-9, abs_tol=1e-9):
                    where, message = f"{branch}/{placed.name}", f"starts at s = {placed.s_enter} m here"
                    self.warn("position", where, f"{message}, {s} m in MAD-X")
            ours = layout.length
            if not math.isclose(ours, length, rel_tol=1e-9, abs_tol=1e-9):
                self.warn("length", branch, f"is {ours} m long here, {length} m in MAD-X")

    def _reference(self, sequence: Sequence, subject: str) -> dict[str, Any]:
        """The BEAM as ReferenceP. Only pc_ref is kept: MAD-X masses differ slightly from SPECIES."""
        beam = sequence.beam
        particle = str(beam.particle).lower()
        species = particle if particle in SPECIES else None
        if species is None:
            mass, charge = beam.mass * 1e9, beam.charge
            matches = [
                s for s in SPECIES.values() if s.charge == charge and abs(s.mass - mass) < 1e-3 * s.mass
            ]
            if len(matches) == 1:
                species = matches[0].name
                message = f"BEAM {particle} with mass {mass:.9g} eV and charge {charge:g} is {species}"
                self.notes.append(ImportNote("species", subject, message))
            else:
                species = particle
                self.warn(
                    "species", subject, f"BEAM {particle} matches no known species; its energy is unchecked"
                )
        return {"species_ref": species, "pc_ref": beam.pc * 1e9}

    # -- definitions ---------------------------------------------------------

    def _definition(self, element: Element, sequence: str) -> str | None:
        name = element.name
        # MAD-X numbers implicit drifts per sequence, so drift_0 of two sequences are different drifts.
        key = f"{sequence}/{name}" if _IMPLICIT_DRIFT.fullmatch(name) else name
        if key not in self.definitions:
            spec = self._convert(element)
            self.definitions[key] = None if spec is None else self._define(name, spec, key)
        return self.definitions[key]

    def _define(self, name: str, spec: _Spec, key: str | None = None) -> str:
        label = self._label(name, self.element_labels)
        self.ids.key = f"element/{key or name}"
        try:
            self.builder.define(label, spec.kind, length=spec.length, tags=spec.tags, **spec.groups)
        except ValueError as error:  # including pydantic's ValidationError
            raise ValueError(f"MAD-X element {name!r} gives an invalid {spec.kind.value}: {error}") from error
        return label

    def _label(self, name: str, used: set[str]) -> str:
        label = self._free(_LABEL_CHARS.sub("_", name)[:120] or "_", used)
        if label != name and "$" not in name and not _IMPLICIT_DRIFT.fullmatch(name):
            self.notes.append(ImportNote("renamed", name, f"renamed to {label!r}"))
        return label

    @staticmethod
    def _free(candidate: str, used: set[str]) -> str:
        label, n = candidate, 1
        while label in used:
            n += 1
            label = f"{candidate}_{n}"
        used.add(label)
        return label

    def _convert(self, element: Element) -> _Spec | None:
        base = element.base_type.name
        handler = _HANDLERS.get(base)
        if handler is None:
            if self.strict:
                raise UnsupportedElementError(
                    f"MAD-X {base.upper()} {element.name!r} has no counterpart here"
                )
            spec = self._drift_or_marker(element)
            message = f"{base.upper()} has no counterpart here; imported as a {spec.kind.value}"
            self.warn("unsupported", element.name, message)
        else:
            spec = handler(self, element)
            if spec is None:
                return None
        if "ApertureP" in ALLOWED_GROUPS[spec.kind] and (aperture := self._aperture(element)):
            spec.groups["ApertureP"] = aperture
        if base != "translation":
            self._ignored(element, ("dx", "dy", "ds", "dtheta", "dphi", "dpsi"))
        spec.tags = (*spec.tags, *self._classes(element))
        return spec

    def _classes(self, element: Element) -> list[str]:
        """Element classes between the element and its MAD-X base type, as tags."""
        tags = []
        parent = element.parent
        while parent.name != parent.parent.name:
            tags.append(_LABEL_CHARS.sub("_", parent.name))
            parent = parent.parent
        return tags

    def _ignored(self, element: Element, attributes: Iterable[str]) -> None:
        """Warn about attributes that differ from the base type's default but are not imported."""
        base = element.base_type
        for attribute in attributes:
            if attribute in element and element[attribute] != base[attribute]:
                self.warn(
                    "dropped", element.name, f"{attribute.upper()} = {element[attribute]} is not imported"
                )

    # -- element kinds -------------------------------------------------------

    def _drift_or_marker(self, e: Element) -> _Spec:
        return _Spec(ElementKind.DRIFT, e.l) if e.l > 0 else _Spec(ElementKind.MARKER)

    def _drift(self, e: Element) -> _Spec | None:
        return _Spec(ElementKind.DRIFT, e.l) if e.l > 0 else None

    def _marker(self, e: Element) -> _Spec:
        return _Spec(ElementKind.MARKER)

    def _instrument(self, e: Element) -> _Spec:
        return _Spec(ElementKind.INSTRUMENT, e.l)

    def _drift_like(self, e: Element) -> _Spec:
        self.apply("drift-like", e.name)
        spec = self._drift_or_marker(e)
        if e.base_type.name in ("rcollimator", "ecollimator") and (e.xsize > 0 or e.ysize > 0):
            # Legacy XSIZE/YSIZE; an APERTYPE aperture, if given, replaces this one.
            shape = "RECTANGULAR" if e.base_type.name == "rcollimator" else "ELLIPTICAL"
            x, y = e.xsize or None, e.ysize or None
            spec.groups["ApertureP"] = {
                "shape": shape,
                **({"x_min": -x, "x_max": x} if x else {}),
                **({"y_min": -y, "y_max": y} if y else {}),
            }
        return spec

    def _sbend(
        self, e: Element, *, length: float | None = None, e1: float | None = None, e2: float | None = None
    ):
        length = e.l if length is None else length
        fintx = e.fintx if e.fintx >= 0 else e.fint  # MAD-X: a negative FINTX means FINTX = FINT
        bend = {
            "angle_ref": e.angle,
            "e1": e.e1 if e1 is None else e1,
            "e2": e.e2 if e2 is None else e2,
            "edge1_int": e.fint * e.hgap,
            "edge2_int": fintx * e.hgap,
            "h1": e.h1,
            "h2": e.h2,
            "tilt_ref": e.get("tilt", 0.0),
        }
        per_length = {
            "Ks0": e.k0s,
            "Kn1": e.k1,
            "Ks1": e.k1s,
            "Kn2": e.k2,
            "Ks2": e.k2s,
            "Kn3": e.k3,
            "Ks3": e.k3s,
        }
        if e.k0 != 0.0 and not math.isclose(e.k0, e.angle / length, rel_tol=1e-12):
            per_length["Kn0"] = e.k0  # a field that differs from the reference curvature
        self._ignored(e, ("ktap",))
        # Multipoles are given in the bend's frame, which TILT already rotates through tilt_ref.
        return _Spec(ElementKind.BEND, length, {"BendP": bend, **self._magnetic(e, per_length, length, 0.0)})

    def _rbend(self, e: Element) -> _Spec:
        half = e.angle / 2
        length = e.l * half / math.sin(half) if self.madx.options.rbarc and half != 0.0 else e.l
        if e.get("true_rbend"):
            self.warn("dropped", e.name, "TRUE_RBEND is not imported; the magnet is a regular bend here")
        self.apply("rbend", e.name)
        return self._sbend(e, length=length, e1=e.e1 + half, e2=e.e2 + half)

    def _quadrupole(self, e: Element) -> _Spec:
        self._ignored(e, ("e1", "e2", "fint", "fintx", "hgap", "h1", "h2", "ktap"))
        per_length = {"Kn0": e.get("k0", 0.0), "Kn1": e.k1, "Ks1": e.k1s}
        return _Spec(ElementKind.QUADRUPOLE, e.l, self._magnetic(e, per_length, e.l, e.get("tilt", 0.0)))

    def _sextupole(self, e: Element) -> _Spec:
        self._ignored(e, ("ktap",))
        return _Spec(
            ElementKind.SEXTUPOLE,
            e.l,
            self._magnetic(e, {"Kn2": e.k2, "Ks2": e.k2s}, e.l, e.get("tilt", 0.0)),
        )

    def _octupole(self, e: Element) -> _Spec:
        return _Spec(
            ElementKind.OCTUPOLE, e.l, self._magnetic(e, {"Kn3": e.k3, "Ks3": e.k3s}, e.l, e.get("tilt", 0.0))
        )

    def _multipole(self, e: Element) -> _Spec:
        if e.angle != 0.0:
            self.warn("dropped", e.name, "ANGLE (a thin bend of the reference orbit) is not imported")
        self._ignored(e, ("ksi", "ktap"))
        return _Spec(ElementKind.MULTIPOLE, 0.0, self._magnetic(e, {}, 0.0, e.get("tilt", 0.0)))

    def _solenoid(self, e: Element) -> _Spec:
        self._ignored(e, ("ksi",))  # a thin solenoid; PALS has only the per-length Ksol
        groups = {"SolenoidP": {"Ksol": e.ks}, **self._magnetic(e, {}, e.l, e.get("tilt", 0.0))}
        return _Spec(ElementKind.SOLENOID, e.l, groups)

    def _kicker(self, e: Element) -> _Spec:
        base = e.base_type.name
        if base == "hkicker":
            h, v, tags = e.kick or e.hkick, 0.0, ("HCOR",)
        elif base == "vkicker":
            h, v, tags = 0.0, e.kick or e.vkick, ("VCOR",)
        else:
            h, v, tags = e.hkick, e.vkick, ()
        self.apply("kick-sign", e.name)
        groups = self._magnetic(e, {}, e.l, e.get("tilt", 0.0), integrated={"Kn0L": -h, "Ks0L": v})
        return _Spec(ElementKind.KICKER, e.l, groups, tags)

    def _rf(self, e: Element, kind: ElementKind) -> _Spec:
        voltage, phase = e.volt * 1e6, 2 * math.pi * e.lag
        if voltage < 0:
            voltage, phase = -voltage, phase + math.pi
        phase %= 2 * math.pi  # LAG is periodic, and a LAG above 1 would look like degrees
        rf: dict[str, Any] = {"voltage": voltage, "phase": phase, "zero_phase": "BELOW_TRANSITION"}
        if e.harmon:
            rf["harmon"] = e.harmon
        elif e.freq > 0:
            rf["frequency"] = e.freq * 1e6  # MAD-X fills FREQ from HARMON, so only an explicit one is kept
        self.apply("rf-phase", e.name)
        return _Spec(kind, e.l, {"RFP": rf})

    def _rfcavity(self, e: Element) -> _Spec:
        return self._rf(e, ElementKind.RFCAVITY)

    def _crabcavity(self, e: Element) -> _Spec:
        if e.get("tilt", 0.0):
            self.warn(
                "dropped", e.name, f"TILT = {e.get('tilt', 0.0)} (the crabbing plane) has no PALS counterpart"
            )
        return self._rf(e, ElementKind.CRABCAVITY)

    def _rotation(self, e: Element) -> _Spec:
        rotation = {"xrotation": "x_rot", "yrotation": "y_rot", "srotation": "z_rot"}[e.base_type.name]
        return _Spec(ElementKind.PATCH, 0.0, {"PatchP": {rotation: e.angle}})

    def _translation(self, e: Element) -> _Spec:
        return _Spec(
            ElementKind.PATCH, 0.0, {"PatchP": {"x_offset": e.dx, "y_offset": e.dy, "z_offset": e.ds}}
        )

    def _beambeam(self, e: Element) -> _Spec:
        self._ignored(e, ("xma", "yma", "bbshape", "width", "bbdir"))
        beam = {"sigma_x": e.sigx or None, "sigma_y": e.sigy or None, "charge": e.charge}
        if e.get("npart"):
            beam["N_particle"] = e.npart
        return _Spec(ElementKind.BEAMBEAM, 0.0, {"BeamBeamP": beam})

    # -- shared groups -------------------------------------------------------

    def _magnetic(
        self,
        e: Element,
        per_length: Mapping[str, float],
        length: float,
        tilt: float,
        integrated: Mapping[str, float] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """MagneticMultipoleP from strengths and the element's KNL/KSL."""
        values = {k: v for k, v in (*per_length.items(), *(integrated or {}).items()) if v != 0.0}
        dropped = []
        for part, attribute in (("n", "knl"), ("s", "ksl")):
            for order, value in enumerate(e.get(attribute) or ()):
                if value == 0.0:
                    continue
                if order > MAX_MULTIPOLE_ORDER:
                    dropped.append(f"{attribute.upper()}[{order}]")
                elif f"Kn{order}" in values or f"Ks{order}" in values:  # this order is per length
                    if length == 0.0:
                        dropped.append(f"{attribute.upper()}[{order}]")
                        continue
                    values[f"K{part}{order}"] = values.get(f"K{part}{order}", 0.0) + value / length
                    self.apply("spread", e.name)
                else:
                    values[f"K{part}{order}L"] = values.get(f"K{part}{order}L", 0.0) + value
        if dropped:
            self.warn(
                "dropped", e.name, f"{', '.join(dropped)} not imported (orders up to {MAX_MULTIPOLE_ORDER})"
            )
        if tilt:
            for order in sorted({int(match[1]) for k in values if (match := re.match(r"K[ns](\d+)", k))}):
                values[f"tilt{order}"] = tilt
        return {"MagneticMultipoleP": values} if values else {}

    def _aperture(self, e: Element) -> dict[str, Any] | None:
        """ApertureP from APERTYPE, APERTURE, APER_OFFSET and the polygon APER_VX, APER_VY."""
        size = [float(v) for v in e.get("aperture") or ()]
        vx, vy = list(e.get("aper_vx") or ()), list(e.get("aper_vy") or ())
        polygon = len(vx) >= 3 and len(vx) == len(vy)
        if not polygon and not any(size):
            return None
        dx, dy = [*(e.get("aper_offset") or ()), 0.0, 0.0][:2]
        self._ignored(e, ("aper_tilt",))
        if polygon:
            points = list(zip(vx, vy, strict=True))
            edges = zip(points, points[1:] + points[:1], strict=True)
            if sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in edges) < 0:  # clockwise
                points.reverse()  # PALS lists vertices counterclockwise
            return {
                "shape": "VERTICES",
                "vertices": {"center": (dx, dy), "list": [{"point": p} for p in points]},
            }
        a, b, c, d = [*size, 0.0, 0.0, 0.0][:4]
        apertype = str(e.apertype).lower()
        if apertype == "circle":
            return _ellipse(a, a, dx, dy)
        if apertype == "ellipse" and a > 0 and b > 0:
            return _ellipse(a, b, dx, dy)
        if apertype == "rectangle":
            return _rectangle(a, b, dx, dy)
        if apertype in ("rectellipse", "lhcscreen") and min(a, b, c, d) > 0:
            return _rectellipse(a, b, c, d, dx, dy)
        if apertype == "rectcircle" and min(a, b, c) > 0:
            return _rectellipse(a, b, c, c, dx, dy)
        self.warn("dropped", e.name, f"aperture {apertype.upper()} {size} is not imported")
        return None


_HANDLERS: dict[str, Callable[[_Converter, Any], _Spec | None]] = {
    "drift": _Converter._drift,
    "marker": _Converter._marker,
    "sbend": _Converter._sbend,
    "rbend": _Converter._rbend,
    "quadrupole": _Converter._quadrupole,
    "sextupole": _Converter._sextupole,
    "octupole": _Converter._octupole,
    "multipole": _Converter._multipole,
    "solenoid": _Converter._solenoid,
    "hkicker": _Converter._kicker,
    "vkicker": _Converter._kicker,
    "kicker": _Converter._kicker,
    "tkicker": _Converter._kicker,
    "rfcavity": _Converter._rfcavity,
    "crabcavity": _Converter._crabcavity,
    "instrument": _Converter._instrument,
    "monitor": _Converter._instrument,
    "hmonitor": _Converter._instrument,
    "vmonitor": _Converter._instrument,
    "placeholder": _Converter._drift_like,
    "collimator": _Converter._drift_like,
    "rcollimator": _Converter._drift_like,
    "ecollimator": _Converter._drift_like,
    "xrotation": _Converter._rotation,
    "yrotation": _Converter._rotation,
    "srotation": _Converter._rotation,
    "translation": _Converter._translation,
    "beambeam": _Converter._beambeam,
}


def _ellipse(a: float, b: float, dx: float, dy: float) -> dict[str, Any]:
    return {"x_min": dx - a, "x_max": dx + a, "y_min": dy - b, "y_max": dy + b}


def _rectangle(a: float, b: float, dx: float, dy: float) -> dict[str, Any]:
    """A zero half width is unlimited."""
    x = {"x_min": dx - a, "x_max": dx + a} if a > 0 else {}
    y = {"y_min": dy - b, "y_max": dy + b} if b > 0 else {}
    return {"shape": "RECTANGULAR", **x, **y}


def _rectellipse(a: float, b: float, c: float, d: float, dx: float, dy: float) -> dict[str, Any]:
    """The rectangle |x| <= a, |y| <= b cut by the ellipse with half axes c, d, as an outline."""
    if a >= c and b >= d:
        return _ellipse(c, d, dx, dy)
    if (a / c) ** 2 + (b / d) ** 2 <= 1.0:
        return _rectangle(a, b, dx, dy)
    xb = c * math.sqrt(max(0.0, 1.0 - (b / d) ** 2))  # where the top and bottom edges meet the ellipse
    ya = d * math.sqrt(max(0.0, 1.0 - (a / c) ** 2))  # where the side edges meet it
    # The straight edges counterclockwise from +x; the ellipse joins each to the next.
    edges = [
        ((a, -ya), (a, ya)) if a < c else None,
        ((xb, b), (-xb, b)) if b < d else None,
        ((-a, ya), (-a, -ya)) if a < c else None,
        ((-xb, -b), (xb, -b)) if b < d else None,
    ]
    vertices = []
    for start, end in filter(None, edges):
        vertices += [{"point": start, "radius": (c, d)}, {"point": end}]
    return {"shape": "VERTICES", "vertices": {"center": (dx, dy), "list": vertices}}
