"""Viewer payloads computed from one immutable lattice revision.

Nothing here depends on the web framework: a :class:`RevisionViews` turns a
lattice (and optionally its machine registry) into the response models of
:mod:`.schemas`, and is equally usable from a notebook or a test.
"""

from collections import Counter, defaultdict
from functools import cached_property

from eicvibe.domain.lattice import (
    ALLOWED_GROUPS,
    GROUP_TYPES,
    ElementKind,
    Lattice,
    PlacedElement,
    parameter_info,
)
from eicvibe.domain.machine import MachineRegistry
from eicvibe.domain.machine.power import (
    PolynomialTransfer,
    TableTransfer,
    TransferFunction,
    design_value,
    magnet_current,
)
from eicvibe.domain.machine.registry import CircuitBinding, ObservableBinding, ParameterBinding
from eicvibe.geometry import FloorGeometry, Frame, element_path, floor_geometry

from . import schemas

#: Positions and s are rounded to this many decimals of a metre (1 nm) ...
POSITION_DIGITS = 9
#: ... and floor paths, which are for drawing, to 1 um; angles to 1 nrad.
PATH_DIGITS = 6
ANGLE_DIGITS = 9

_MAIN_ORDER = {
    ElementKind.KICKER: 0,
    ElementKind.QUADRUPOLE: 1,
    ElementKind.SEXTUPOLE: 2,
    ElementKind.OCTUPOLE: 3,
}

COLUMNS = (
    schemas.Column(key="angle", label="angle", unit="rad", description="Bend angle, BendP.angle_ref"),
    schemas.Column(
        key="Kn1", label="Kn1", unit="1/m^2", description="Normal quadrupole strength, Kn1 + Kn1L/L"
    ),
    schemas.Column(
        key="Kn1L", label="Kn1L", unit="1/m", description="Integrated quadrupole strength, Kn1 L + Kn1L"
    ),
    schemas.Column(
        key="Kn2", label="Kn2", unit="1/m^3", description="Normal sextupole strength, Kn2 + Kn2L/L"
    ),
    schemas.Column(
        key="Kn2L", label="Kn2L", unit="1/m^2", description="Integrated sextupole strength, Kn2 L + Kn2L"
    ),
    schemas.Column(
        key="Kn3L", label="Kn3L", unit="1/m^3", description="Integrated octupole strength, Kn3 L + Kn3L"
    ),
    schemas.Column(key="Kn0L", label="Kn0L", unit="rad", description="Horizontal kick; > 0 toward -x"),
    schemas.Column(key="Ks0L", label="Ks0L", unit="rad", description="Vertical kick, skew dipole"),
    schemas.Column(key="Ksol", label="Ksol", unit="1/m", description="Solenoid strength, B_s / (B rho)"),
    schemas.Column(key="voltage", label="voltage", unit="V", description="RF peak voltage"),
    schemas.Column(key="tilt", label="tilt", unit="rad", description="Rotation about s of the main field"),
)


def _position(value: float, digits: int = POSITION_DIGITS) -> float:
    return round(value, digits) + 0.0  # + 0.0 turns -0.0 into 0.0


def pose(frame: Frame) -> schemas.Pose:
    theta, phi, psi = (_position(a, ANGLE_DIGITS) for a in frame.angles)
    x, y, z = (_position(c) for c in frame.r)
    return schemas.Pose(x=x, y=y, z=z, theta=theta, phi=phi, psi=psi)


def _strength(placed: PlacedElement, order: int) -> tuple[float | None, float | None]:
    """(per length, integrated) normal strength of ``order``; a thin element has no per-length value."""
    group = placed.element.MagneticMultipoleP
    if group is None:
        return (None, 0.0) if placed.length == 0.0 else (0.0, 0.0)
    per_length, integrated = getattr(group, f"Kn{order}"), getattr(group, f"Kn{order}L")
    total = per_length * placed.length + integrated
    return (total / placed.length if placed.length else None), total


def _row_values(placed: PlacedElement) -> dict[str, float | None]:
    """Numeric table columns of one occurrence; a key is missing where it does not apply."""
    element, kind = placed.element, placed.kind
    values: dict[str, float | None] = {}
    if kind is ElementKind.BEND and element.BendP is not None:
        values["angle"] = element.BendP.angle_ref
        if element.BendP.tilt_ref:
            values["tilt"] = element.BendP.tilt_ref
    for order, (thick, thin), always in (
        (1, ("Kn1", "Kn1L"), kind is ElementKind.QUADRUPOLE),
        (2, ("Kn2", "Kn2L"), kind is ElementKind.SEXTUPOLE),
        (3, (None, "Kn3L"), kind is ElementKind.OCTUPOLE),
    ):
        per_length, integrated = _strength(placed, order)
        if always or integrated:
            if thick is not None:
                values[thick] = per_length
            values[thin] = integrated
    multipoles = element.MagneticMultipoleP
    if multipoles is not None:
        # A bend's Kn0 is its bending field, not a kick.
        for key in ("Ks0L",) if kind is ElementKind.BEND else ("Kn0L", "Ks0L"):
            kick = getattr(multipoles, key) + getattr(multipoles, key.removesuffix("L")) * placed.length
            if kick or kind is ElementKind.KICKER:
                values[key] = kick
        order = _MAIN_ORDER.get(kind)
        if order is not None and getattr(multipoles, f"tilt{order}"):
            values["tilt"] = getattr(multipoles, f"tilt{order}")
    elif kind is ElementKind.KICKER:
        values["Kn0L"] = values["Ks0L"] = 0.0
    if element.SolenoidP is not None and (element.SolenoidP.Ksol or kind is ElementKind.SOLENOID):
        values["Ksol"] = element.SolenoidP.Ksol
    if kind in (ElementKind.RFCAVITY, ElementKind.CRABCAVITY):
        values["voltage"] = element.RFP.voltage if element.RFP is not None else 0.0
    return values


def _transfer(transfer: TransferFunction) -> schemas.Transfer:
    return schemas.Transfer(
        kind=transfer.kind,
        summary=_transfer_summary(transfer),
        engineering_unit=transfer.engineering_unit,
        field_unit=transfer.field_unit,
        integrated=transfer.integrated,
        source=transfer.source.value,
    )


def _transfer_summary(transfer) -> str:
    """A transfer function in words, e.g. "field = 0.001 T/A x I"."""
    field = "integrated field" if transfer.integrated else "field"
    unit = f"{transfer.field_unit}/{transfer.engineering_unit}"
    if isinstance(transfer, PolynomialTransfer):
        c = transfer.coefficients
        if len(c) == 2:
            offset = f" {'+' if c[0] >= 0 else '-'} {abs(c[0]):g} {transfer.field_unit}" if c[0] else ""
            return f"{field} = {c[1]:g} {unit} × I{offset}"
        return f"{field}: polynomial of degree {len(c) - 1} in I"
    if isinstance(transfer, TableTransfer):
        return (
            f"{field}: measured curve of {len(transfer.currents)} points, "
            f"{transfer.currents[0]:g} to {transfer.currents[-1]:g} {transfer.engineering_unit}"
        )
    text = f"{field} from service {transfer.service!r}, magnet {transfer.magnet!r}"
    if transfer.hysteresis:
        text += ", with hysteresis"
    if transfer.fallback is not None:
        text += f"; offline: {_transfer_summary(transfer.fallback).split(' = ', 1)[-1]}"
    return text


class RevisionViews:
    """Payloads of one lattice revision; each is computed once, on first use."""

    def __init__(
        self, lattice: Lattice, registry: MachineRegistry | None = None, *, source: str = ""
    ) -> None:
        self.lattice = lattice
        self.registry = registry
        self.source = source or lattice.name
        self._path_cache: dict[str, list[tuple]] = {}

    @cached_property
    def geometry(self) -> FloorGeometry:
        return floor_geometry(self.lattice)

    @cached_property
    def _devices_at(self) -> dict[str, list]:
        """Registry devices by occurrence id."""
        found = defaultdict(list)
        for device in self.registry.devices if self.registry else ():
            for occurrence in device.occurrences:
                found[occurrence].append(device)
        return found

    @cached_property
    def _channels_of(self) -> dict[str, list]:
        found = defaultdict(list)
        for channel in self.registry.channels if self.registry else ():
            found[channel.device].append(channel)
        return found

    def _eic_name(self, occurrence: str) -> str | None:
        slot = self.registry.slot_of(occurrence) if self.registry else None
        return slot.name if slot else None

    def _paths(self, branch_id: str) -> list[tuple]:
        """Floor path of every occurrence of a branch (by id)."""
        if branch_id not in self._path_cache:
            frames = self.geometry.branches[branch_id].frames
            placements = self.lattice.layout(branch_id).placements
            self._path_cache[branch_id] = [
                element_path(f.entrance, p.element) for f, p in zip(frames, placements, strict=True)
            ]
        return self._path_cache[branch_id]

    def _bounds(self, branch_id: str) -> schemas.Bounds:
        points = [point for path in self._paths(branch_id) for point in path]
        if not points:
            return schemas.Bounds(x_min=0.0, x_max=0.0, z_min=0.0, z_max=0.0)
        xs, zs = [p[0] for p in points], [p[2] for p in points]
        return schemas.Bounds(
            x_min=_position(min(xs)),
            x_max=_position(max(xs)),
            z_min=_position(min(zs)),
            z_max=_position(max(zs)),
        )

    def summary(self) -> schemas.LatticeSummary:
        lattice = self.lattice
        branches = []
        for layout in (lattice.layout(b.id) for b in lattice.branches):
            conditions = lattice.initial_conditions(layout.id)
            reference = conditions.reference if conditions else None
            branches.append(
                schemas.BranchSummary(
                    id=layout.id,
                    name=layout.branch.name,
                    kind=layout.branch.kind.value,
                    length=layout.length,
                    occurrences=len(layout.placements),
                    placed_by=self.geometry.branches[layout.id].placed_by,
                    bounds=self._bounds(layout.id),
                    reference=reference
                    and schemas.Reference(
                        species=reference.species_ref, pc=reference.pc_ref, E_tot=reference.E_tot_ref
                    ),
                )
            )
        forks = [
            schemas.ForkSummary(
                id=edge.id,
                name=lattice.placement(edge.id).name,
                from_branch=edge.from_branch,
                s_from=edge.s_from,
                to_branch=edge.to_branch,
                to_occurrence=edge.to_occurrence,
                to_side=edge.to_side.value,
                s_to=edge.s_to,
            )
            for edge in lattice.forks
        ]
        kinds = Counter(p.kind.value for b in lattice.branches for p in lattice.layout(b.id).placements)
        return schemas.LatticeSummary(
            source=self.source,
            name=lattice.name,
            digest=self.digest,
            root_branch=lattice.root_branch,
            branches=branches,
            forks=forks,
            closures=[
                schemas.Closure(fork=c.fork, position_error=c.position_error, angle_error=c.angle_error)
                for c in self.geometry.closures
            ],
            shared_components=[self._component(c) for c in lattice.shared_components],
            diagnostics=[
                *(
                    schemas.Diagnostic(code=d.code, message=d.message, subject=d.subject)
                    for d in lattice.diagnose()
                ),
                *(
                    schemas.Diagnostic(code="power-string", message=problem, subject=lattice.name)
                    for problem in (self.registry.power_problems(lattice) if self.registry else ())
                ),
            ],
            kinds=dict(sorted(kinds.items())),
            devices=len(self.registry.devices) if self.registry else 0,
            channels=len(self.registry.channels) if self.registry else 0,
            circuits=len(self.registry.circuits) if self.registry else 0,
            strings=sum(c.series for c in self.registry.circuits) if self.registry else 0,
        )

    @property
    def digest(self) -> str:
        return self.lattice.digest.removeprefix("sha256:")

    def _component(self, component) -> schemas.SharedComponent:
        return schemas.SharedComponent(
            id=component.id,
            name=component.name,
            traversals=[
                schemas.Traversal(
                    role=t.role,
                    occurrence=t.occurrence,
                    branch=self.lattice.placement(t.occurrence).branch_id,
                )
                for t in component.traversals
            ],
        )

    def branch(self, branch_id: str) -> schemas.BranchData:
        """Table and floor data of one branch (by id or name)."""
        lattice = self.lattice
        layout = lattice.layout(branch_id)
        placements = layout.placements
        rows = [_row_values(p) for p in placements]
        used = {key for row in rows for key in row}
        columns = [c for c in COLUMNS if c.key in used]
        offsets, xs, ys, zs = [0], [], [], []
        for path in self._paths(layout.id):
            for x, y, z in path:
                xs.append(_position(x, PATH_DIGITS))
                ys.append(_position(y, PATH_DIGITS))
                zs.append(_position(z, PATH_DIGITS))
            offsets.append(len(xs))
        frames = self.geometry.branches[layout.id].frames
        components = {}
        for component in lattice.shared_components:
            for traversal in component.traversals:
                components[traversal.occurrence] = component.name
        return schemas.BranchData(
            branch=layout.id,
            table=schemas.BranchTable(
                id=[p.id for p in placements],
                name=[p.name for p in placements],
                eic_name=[self._eic_name(p.id) for p in placements],
                kind=[p.kind.value for p in placements],
                element=[p.definition.name for p in placements],
                s=[_position(p.s_enter) for p in placements],
                length=[p.length for p in placements],
                values={c.key: [row.get(c.key) for row in rows] for c in columns},
                columns=columns,
                device=[
                    ", ".join(d.name for d in self._devices_at[p.id]) if p.id in self._devices_at else None
                    for p in placements
                ],
                fork_to=[p.occurrence.fork.to_branch if p.occurrence.fork else None for p in placements],
                component=[components.get(p.id) for p in placements],
            ),
            floor=schemas.BranchFloor(
                offsets=offsets,
                x=xs,
                y=ys,
                z=zs,
                theta=[_position(f.entrance.angles[0], ANGLE_DIGITS) for f in frames],
            ),
        )

    def occurrence(self, occurrence_id: str) -> schemas.OccurrenceDetail:
        lattice = self.lattice
        placed = lattice.placement(occurrence_id)
        overridden = placed.occurrence.override_map()
        groups = []
        for group_name in (g for g in GROUP_TYPES if g in ALLOWED_GROUPS[placed.kind]):
            group = getattr(placed.element, group_name, None)
            if group is None:
                continue
            values = group.model_dump(mode="json", exclude_defaults=True)
            parameters = []
            for name, value in values.items():
                info = parameter_info(group_name, name)
                parameters.append(
                    schemas.Parameter(
                        name=name,
                        value=value,
                        unit=info.unit,
                        description=info.description,
                        overridden=name in overridden.get(group_name, {}),
                    )
                )
            groups.append(schemas.Group(group=group_name, parameters=parameters))
        fork = placed.occurrence.fork
        component = lattice.component_of(placed.id)
        frames = self.geometry.branches[placed.branch_id].frames[placed.index]
        return schemas.OccurrenceDetail(
            id=placed.id,
            name=placed.name,
            eic_name=self._eic_name(placed.id),
            kind=placed.kind.value,
            element=placed.definition.name,
            element_id=placed.definition.id,
            branch=placed.branch_id,
            index=placed.index,
            s_enter=placed.s_enter,
            s_exit=placed.s_exit,
            length=placed.length,
            tags=sorted(placed.tags),
            groups=groups,
            fork=fork
            and schemas.ForkLink(
                to_branch=fork.to_branch, to_occurrence=fork.to_occurrence, to_side=fork.to_side.value
            ),
            component=component and self._component(component),
            entrance=pose(frames.entrance),
            exit=pose(frames.exit),
            devices=[self._device(d, placed) for d in self._devices_at.get(placed.id, ())],
        )

    def _device(self, device, placed: PlacedElement) -> schemas.Device:
        channels = []
        for channel in self._channels_of.get(device.id, ()):
            binding = channel.binding
            channels.append(
                schemas.Channel(
                    name=channel.name,
                    role=channel.role.value,
                    unit=channel.unit,
                    control_limits=channel.control_limits,
                    parameter=binding.parameter if isinstance(binding, ParameterBinding) else None,
                    targets=list(binding.targets) if isinstance(binding, ParameterBinding) else [],
                    circuit=binding.circuit if isinstance(binding, CircuitBinding) else None,
                    observable=binding.observable.value if isinstance(binding, ObservableBinding) else None,
                )
            )
        power = [self._power(c, placed) for c in self.registry.circuits if c.device == device.id]
        return schemas.Device(
            id=device.id, name=device.name, type=device.type.value, channels=channels, power=power
        )

    def _power(self, circuit, placed: PlacedElement) -> schemas.Power:
        setting = self.registry.setting_of(circuit.id)
        magnet = next((m for m in circuit.magnets if placed.id in m.occurrences), None)
        design = None
        if magnet is not None and setting is not None and setting.control_limits is not None:
            try:
                value = design_value(placed, magnet.parameter)
                design = magnet_current(
                    self.lattice, magnet, value, setting.control_limits, occurrence=placed.id
                )
            except ValueError, LookupError:
                design = None
        return schemas.Power(
            circuit=circuit.id,
            name=circuit.name,
            setting=setting.name if setting else None,
            series=circuit.series,
            magnets=len(circuit.magnets),
            occurrences=[o for m in circuit.magnets for o in m.occurrences],
            parameter=magnet.parameter if magnet else None,
            polarity=magnet.polarity if magnet else None,
            transfer=_transfer(magnet.transfer) if magnet else None,
            design_current=design,
        )
