"""Response bodies of the lattice viewer API, schema version 2.

They are the contract with the web client: ``web/src/api/schema.d.ts`` is
generated from the OpenAPI document of these models (``npm run
generate-types``), and a test fails when the committed document is stale.

Large per-occurrence data is sent as parallel columns (one list per field,
one entry per occurrence in beam order), which is several times smaller than
a list of objects and is what plotting and table code consume directly.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

#: Bumped when a response changes shape; the web client refuses other versions.
#: 2: element EIC names (``eic_name``) and setting targets. 3: power circuits. 4: optics. 5: BPM orbits.
SCHEMA_VERSION = 5


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
    circuits: int = Field(description="Power circuits in the registry")
    strings: int = Field(description="Circuits that drive several magnets in series")
    routes: list[RouteSummary] = Field(default_factory=list, description="The machine's routes")
    modes: list[str] = Field(
        default_factory=list, description="Its operating modes (settings/<mode>.yaml), for optics in a mode"
    )


class RouteSegment(_Model):
    branch: str = Field(description="Branch id")
    start: str = Field(description="First occurrence (from its entrance)")
    end: str = Field(description="Last occurrence (to its exit)")
    s_start: float = Field(description="s along the route where the segment starts, m")
    length: float


class RouteSummary(_Model):
    """A route: the beam's path through several branches, joined at forks (routes.yaml)."""

    id: str = Field(description="route:<name>, as views address it where they take a branch id")
    name: str
    branches: list[str] = Field(description="The branch names it was given")
    segments: list[RouteSegment]
    length: float
    problems: list[str] = Field(description="What keeps it from being one pass, e.g. an energy change")


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
    name: list[str] = Field(description="Lattice (design) name of the occurrence")
    eic_name: list[str | None] = Field(
        description="EIC name (EIC-SEG-RSI-009) of the element, from the machine registry; null without one"
    )
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
    circuit: str | None = Field(description="Power circuit whose current the setting is")
    observable: str | None = Field(description="Beam quantity a measurement reads")


class Transfer(_Model):
    """A magnet's transfer function, from its supply's current to its field."""

    kind: str = Field(description="polynomial, table or service")
    summary: str = Field(description="E.g. field = 0.001 T/A x I")
    engineering_unit: str
    field_unit: str
    integrated: bool = Field(description="Field integrated over the magnet")
    source: str = Field(description="placeholder or measured")


class Power(_Model):
    """One power circuit of a supply, seen from one occurrence."""

    circuit: str
    name: str = Field(description="The family a string powers, or the magnet")
    setting: str | None = Field(description="The setting channel, in the supply's engineering unit")
    series: bool = Field(description="Several magnets in series on the supply")
    magnets: int
    occurrences: list[str] = Field(description="Every occurrence the circuit sets")
    parameter: str | None = Field(description="The parameter it sets here; null where it sets none")
    polarity: int | None
    transfer: Transfer | None = Field(description="This magnet's transfer function")
    design_current: float | None = Field(description="Current giving this magnet its design value")


class Device(_Model):
    id: str
    name: str
    type: str
    channels: list[Channel]
    power: list[Power] = Field(description="The power circuits of a supply")


class ForkLink(_Model):
    to_branch: str
    to_occurrence: str | None
    to_side: str


class OccurrenceDetail(_Model):
    schema_version: int = SCHEMA_VERSION
    id: str
    name: str = Field(description="Lattice (design) name")
    eic_name: str | None = Field(description="EIC name of the element, from the machine registry")
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


class EngineInfo(_Model):
    """A physics engine the server can compute optics with (ADR 0006)."""

    name: str
    available: bool = Field(description="Installed here; starting it may still fail")
    version: str | None = Field(description="Known once the engine has run, for engines started on demand")
    fidelities: list[str]
    method: str = Field(description="How it computes optics and chromaticity")


class OpticsProfile(_Model):
    """Optics inside elements too: the branch start, then the end of every piece the engine sampled."""

    s: list[float] = Field(description="m")
    index: list[int] = Field(
        description="For each point after the start, the occurrence (its position in ``occurrences``) "
        "whose piece ends there"
    )
    values: dict[str, list[float]] = Field(description="Columns by key, as in ``BranchOptics.values``")


class BranchOptics(_Model):
    """Optics of one branch, from the optics of the whole machine (lines start where their forks are).

    Columns hold the value at the branch start and then at the exit of every
    occurrence: one more entry than the branch has occurrences.
    """

    mode: str | None = Field(
        None, description="The operating mode whose settings they have; none: the lattice's"
    )
    schema_version: int = SCHEMA_VERSION
    branch: str
    engine: str
    version: str | None
    fidelity: str
    method: str | None = Field(description="How the maps and chromaticity were computed, e.g. TPSA")
    periodic: bool
    available: bool = Field(description="Whether the branch has optics; see ``origin`` when not")
    origin: str = Field(
        description="Periodic solution, initial conditions, or the fork the line starts from; "
        "otherwise why it has no optics"
    )
    occurrences: list[str] = Field(
        description="Occurrence ids in beam order; column entry i + 1 is at the exit of i"
    )
    s: list[float] = Field(description="m")
    values: dict[str, list[float]] = Field(description="Columns by key; see ``columns``")
    columns: list[Column]
    tune_x: float | None = Field(description="A ring's tunes; null for a line")
    tune_y: float | None
    chroma_x: float | None = Field(description="dQ/d(delta)")
    chroma_y: float | None
    momentum_compaction: float | None
    profile: OpticsProfile | None = Field(
        description="Optics inside elements, where the engine samples them (TrackPad: its integrator steps)"
    )


class RunEntry(_Model):
    """A virtual machine's run (ADR 0007): what ran, and how far it has come.

    A ring's run records turns; a route's records shots, single passes along
    the route, and its turn numbers count shots.
    """

    id: str
    kind: str = Field(description="ring or route")
    unit: str = Field(description="What its turn numbers count: turn or shot")
    lattice: str = Field(description="Lattice name")
    lattice_digest: str
    branch: str = Field(description="Branch name, or the route's name")
    branch_id: str = Field(description="Branch id, or route:<name>")
    engine: str
    fidelity: str
    start: str = Field(description="Turn 0 on the run's own clock, ISO 8601")
    revolution_period: float = Field(description="s per turn, or per shot")
    bpms: int
    first_turn: int = Field(description="Earliest turn still kept turn by turn")
    end_turn: int = Field(description="Turns recorded so far")
    state: str = Field(description="running, stopped, done, failed or unknown")
    updated: str | None = Field(description="When the run last wrote, ISO 8601")
    machine: str | None = Field(None, description="The virtual accelerator's machine that runs it (ADR 0008)")
    buffer_turns: int = Field(description="Turns its BPMs' buffer keeps (a ring); 0: every turn or shot")
    slow_period: float = Field(description="Its slow orbit's sample, s; 0: one per chunk")
    captures: int = Field(
        description="Captures saved: the buffer frozen by a kick, injection, loss or request"
    )


class CaptureEntry(_Model):
    """The BPM buffer frozen around a trigger and saved, as a real BPM system's acquisitions are."""

    id: str
    kind: str = Field(description="kick, inject, post-mortem (the beam was lost) or request")
    trigger: int = Field(description="The turn of the trigger")
    first: int = Field(description="Its first turn")
    turns: int
    wall: str = Field(description="When it was saved, ISO 8601")
    partial: bool = Field(description="The run ended before all its turns were tracked")
    detail: dict[str, Any]


class RunEvent(_Model):
    turn: int
    kind: str = Field(description="start, inject, kick, settings, loss, warning, error")
    detail: dict[str, Any]


class OrbitFrames(_Model):
    """BPM readings ``every`` turns (shots, samples) apart, oldest first: one orbit, or trajectory, each.

    From the BPMs' buffer (the latest turns), from a capture, or from the slow orbit, whose
    frames are samples, each the mean over ``sample_period`` seconds of the run.
    """

    schema_version: int = SCHEMA_VERSION
    run: str
    source: str = Field(description="buffer, slow, or capture:<id>")
    unit: str = Field(description="What turn numbers count: turn, shot, or sample (the slow orbit)")
    sample_period: float | None = Field(description="s between the slow orbit's samples")
    times: list[float] | None = Field(description="The slow orbit's samples: their wall-clock time, s (Unix)")
    bpms: list[str] = Field(description="BPM names in beam order")
    s: list[float] = Field(description="s of every BPM, m")
    every: int
    turns: list[int] = Field(description="The turn of every orbit, oldest first")
    x: list[list[float | None]] = Field(description="[orbit][bpm], mm; null where the BPM read nothing")
    y: list[list[float | None]]
    intensity: list[list[float | None]] = Field(description="[orbit][bpm], nC")
    first_turn: int = Field(description="Earliest turn kept")
    end_turn: int = Field(description="Turns recorded so far")
    state: str
    events: list[RunEvent] = Field(description="What happened before the orbits shown, from the oldest on")


class PlaneCheck(_Model):
    """One plane of a run's check: what its BPMs measured against what the model gives."""

    measured_tune: float | None = Field(
        description="Fractional tune of the readings, 0 to 1: the line seen, or 1 minus it where the phase "
        "advance along the ring says so"
    )
    measured_full: float | None = Field(description="With the model's integer part")
    model_tune: float | None = Field(description="The model's tune (optics), integer part included")
    map_tune: float | None = Field(
        description="The run's own one-turn map's fractional tune, when it started"
    )
    amplitude: float | None = Field(description="Oscillation amplitude, median over the BPMs, mm")
    initial_amplitude: float | None = Field(
        description="The same over the first 32 turns: before the bunch decoheres, its particles' amplitude"
    )
    beam_size: float | None = Field(
        description="The beam's rms size from the model's beta, median over the BPMs, mm"
    )
    kick_sigma: float | None = Field(description="initial_amplitude in beam sizes: a pinger kicks a few")
    spectrum: list[float] = Field(
        description="Line amplitudes at RunCheck.frequencies, mean over the BPMs, mm"
    )
    phase_error: list[float | None] = Field(
        description="Measured phase advance from the first BPM minus the model's, degrees"
    )
    phase_rms: float | None = Field(description="rms of phase_error, degrees")
    beta_beating: list[float | None] = Field(
        description="Relative beta error from the oscillation amplitudes (BPM gain errors included)"
    )
    beta_beating_rms: float | None
    orbit: list[float | None] = Field(description="Mean reading of every BPM over the stretch, mm")
    model_orbit: list[float | None] = Field(description="The model's closed orbit at every BPM, mm")
    drift_tune: list[float | None] = Field(description="Tune of each stretch of RunCheck.drift_turns")
    drift_amplitude: list[float | None] = Field(description="Amplitude of each stretch, mm")


class Detuning(_Model):
    """Tune against amplitude in the model: single particles kicked in one plane, tracked by the engine."""

    plane: str = Field(description="The plane kicked: the one the beam oscillates in most")
    amplitude: list[float] = Field(description="Largest oscillation amplitude over the BPMs, mm")
    tune_x: list[float]
    tune_y: list[float]
    turns: int
    expected_x: float | None = Field(
        None, description="The model's tune at the amplitude measured after the kick"
    )
    expected_y: float | None = None


class RunCheck(_Model):
    """A ring run's tracking checked against the model of its lattice and settings, over some turns."""

    run: str
    engine: str = Field(description="The engine of the model")
    fidelity: str
    source: str = Field(description="buffer or capture:<id>")
    first_turn: int
    turns: int
    trigger: int | None = Field(description="A capture's trigger turn (a kick)")
    kick: dict[str, float] | None = Field(None, description="The kick of a kick's capture: x and y, rad")
    settings: str = Field(description="Which settings the model has")
    bpms: list[str]
    s: list[float]
    frequencies: list[float] = Field(description="Of the spectra, fractions of the revolution")
    drift_turns: list[int] = Field(description="First turn of each stretch of the tune drift")
    drift_window: int
    synchrotron_tune: float | None = Field(description="The run's own one-turn map's, when it started")
    x: PlaneCheck
    y: PlaneCheck
    detuning: Detuning | None
    notes: list[str]


class AiStatus(_Model):
    """The AI features (ADR 0009): whether they are on, which model answers, and whether it does."""

    enabled: bool = Field(description="On (the default), or turned off by an administrator or --no-ai")
    available: bool = Field(description="The model answers")
    provider: str | None = Field(description="local, openai or anthropic")
    model: str | None
    where: str | None = Field(description="Where the questions go: 'on this server', 'by OpenAI', ...")
    reason: str


class AskRequest(_Model):
    """A question about what the viewer shows; the server gathers the facts itself."""

    question: str = Field(min_length=1, max_length=500)
    digest: str = Field(description="The revision shown")
    branch: str = Field(description="The branch shown (id or name), or route:<name>")
    tab: str | None = None
    occurrence: str | None = Field(None, description="The selected occurrence's id")
    engine: str | None = Field(None, description="The Optics tab's engine, for the optics already computed")
    fidelity: str | None = None
    run: str | None = Field(None, description="The run the Orbit or Check tab shows")
    machine: str | None = Field(None, description="The virtual accelerator's machine chosen")


class ExplainCheck(_Model):
    """The check to explain, as the Check tab asked for it."""

    run: str
    source: str = "kick"
    first: int | None = None
    turns: int = Field(1024, ge=64, le=8192)
    engine: str | None = None
    detuning: bool = False
