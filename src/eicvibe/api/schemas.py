"""Response bodies of the lattice viewer API, version 1.

They are the contract with the web client: ``web/src/api/schema.d.ts`` is
generated from the OpenAPI document of these models (``npm run
generate-types``), and a test fails when the committed document is stale.

Large per-occurrence data is sent as parallel columns (one list per field,
one entry per occurrence in beam order), which is several times smaller than
a list of objects and is what plotting and table code consume directly.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

#: Bumped when a response changes shape; part of every ETag.
SCHEMA_VERSION = 1


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True)


class Bounds(_Model):
    """Extent of a branch on the floor, m."""

    x_min: float
    x_max: float
    z_min: float
    z_max: float


class Pose(_Model):
    """Global position (m) and orientation (rad, ``W = R_y(theta) R_x(phi) R_z(psi)``)."""

    x: float
    y: float
    z: float
    theta: float
    phi: float
    psi: float


class Reference(_Model):
    species: str | None
    pc: float | None = Field(description="Reference momentum times c, eV")
    E_tot: float | None = Field(description="Reference total energy, eV")


class LatticeEntry(_Model):
    """One lattice the server offers."""

    source: str = Field(description="Catalog name, from the file name")
    name: str = Field(description="Lattice name")
    digest: str = Field(description="Hex SHA-256 of the canonical lattice; the revision key in URLs")
    branches: int
    occurrences: int
    registry: bool = Field(description="Whether devices and channels are available")
    loaded_at: str = Field(description="When the server loaded this revision, ISO 8601")


class BranchSummary(_Model):
    id: str
    name: str
    kind: str
    length: float
    occurrences: int
    placed_by: str | None = Field(description="Fork occurrence that places this branch on the floor")
    bounds: Bounds
    reference: Reference | None = Field(description="Reference particle at the branch start, if given")


class ForkSummary(_Model):
    id: str
    name: str
    from_branch: str
    s_from: float
    to_branch: str
    to_occurrence: str | None
    to_side: str
    s_to: float


class Closure(_Model):
    """A fork into a branch already placed elsewhere, and how far apart the two frames are."""

    fork: str
    position_error: float = Field(description="m")
    angle_error: float = Field(description="rad")


class Traversal(_Model):
    role: str
    occurrence: str
    branch: str


class SharedComponent(_Model):
    id: str
    name: str
    traversals: list[Traversal]


class Diagnostic(_Model):
    code: str
    message: str
    subject: str


class LatticeSummary(_Model):
    schema_version: int = SCHEMA_VERSION
    source: str
    name: str
    digest: str
    root_branch: str | None
    branches: list[BranchSummary]
    forks: list[ForkSummary]
    closures: list[Closure]
    shared_components: list[SharedComponent]
    diagnostics: list[Diagnostic]
    kinds: dict[str, int] = Field(description="Occurrence count by element kind")
    devices: int = Field(description="Devices in the registry; 0 without one")
    channels: int


class Column(_Model):
    """How to label one numeric column of :class:`BranchTable`."""

    key: str
    label: str
    unit: str | None
    description: str


class BranchTable(_Model):
    """Per-occurrence values in beam order, one list per field.

    ``Kn1`` and ``Kn1L`` (and the same for ``Kn2``) combine both ways a lattice
    can give a strength: ``Kn1L = Kn1 * length + Kn1L`` and, for an element
    with length, ``Kn1 = Kn1L / length``. A value is null where it does not
    apply to the element's kind.
    """

    id: list[str]
    name: list[str]
    kind: list[str]
    element: list[str] = Field(description="Name of the placed element definition")
    s: list[float] = Field(description="s at the entrance, m")
    length: list[float]
    values: dict[str, list[float | None]] = Field(description="Numeric columns by key; see ``columns``")
    columns: list[Column]
    device: list[str | None] = Field(description="Names of the devices at the occurrence, comma separated")
    fork_to: list[str | None] = Field(description="Branch a fork leads to")
    component: list[str | None] = Field(description="Shared component an occurrence is a traversal of")


class BranchFloor(_Model):
    """Reference orbit on the floor: occurrence i spans points ``offsets[i]`` to ``offsets[i+1] - 1``."""

    offsets: list[int]
    x: list[float]
    y: list[float]
    z: list[float]
    theta: list[float] = Field(description="Heading at each occurrence's entrance, rad")


class BranchData(_Model):
    schema_version: int = SCHEMA_VERSION
    branch: str
    table: BranchTable
    floor: BranchFloor


class Parameter(_Model):
    name: str
    value: Any
    unit: str | None
    description: str | None
    overridden: bool = Field(description="Set on this occurrence rather than on the element definition")


class Group(_Model):
    group: str
    parameters: list[Parameter]


class Channel(_Model):
    name: str
    role: str
    unit: str
    control_limits: tuple[float, float] | None
    parameter: str | None = Field(description="Lattice parameter a setting drives")
    targets: list[str] = Field(description="Occurrences or definitions whose parameter the setting drives")
    observable: str | None = Field(description="Beam quantity a measurement reads")


class Device(_Model):
    id: str
    name: str
    type: str
    channels: list[Channel]


class ForkLink(_Model):
    to_branch: str
    to_occurrence: str | None
    to_side: str


class OccurrenceDetail(_Model):
    schema_version: int = SCHEMA_VERSION
    id: str
    name: str
    kind: str
    element: str
    element_id: str
    branch: str
    index: int
    s_enter: float
    s_exit: float
    length: float
    tags: list[str]
    groups: list[Group]
    fork: ForkLink | None
    component: SharedComponent | None
    entrance: Pose
    exit: Pose
    devices: list[Device]
