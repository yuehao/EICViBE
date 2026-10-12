"""The virtual accelerator over HTTP (ADR 0008): ``/api/v1/va/...``, for the viewer, admin page and notebooks.

Browsers log in (``POST /va/login``) when someone first changes something, and
carry a session cookie; notebooks send a personal token (``Authorization:
Bearer ...``, made in the viewer's account menu or with ``eicvibe-va token``).
Machines and their settings are read without a login; changing anything needs
one, and the service checks the role (ADR 0003's viewer, physicist, operator,
admin). The write log is for users who are logged in; the action log and users
for operators and administrators. A server without a virtual accelerator
answers 404 here, except ``/va/status``.
"""

import time
from collections.abc import Callable
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from eicvibe.ai import PROVIDERS, AiConfig, AiUnavailable

from .service import Conflict, Forbidden, VirtualAccelerator
from .store import ANONYMOUS, SESSION_HOURS, Role, User

COOKIE = "eicvibe_session"


class AiProvider(BaseModel):
    label: str
    url: str = Field(description="Its URL unless another is given")
    key_variable: str | None = Field(description="The environment variable a key may come from")
    model: str = Field(description="Its model unless another is chosen; empty: choose one")


class AiSettings(BaseModel):
    """The AI features' configuration (ADR 0009), without the keys."""

    enabled: bool
    provider: Literal["local", "openai", "anthropic"]
    model: str
    url: str
    key: str | None = Field(
        description="Where the provider's API key comes from: stored, a variable, or none"
    )
    key_hint: str | None = Field(description="The stored key's last four characters")
    saved: bool = Field(description="Chosen here; otherwise the server's defaults (its command line)")
    providers: dict[str, AiProvider]


class AiChange(BaseModel):
    enabled: bool = True
    provider: Literal["local", "openai", "anthropic"] = "local"
    model: str = ""
    url: str = ""
    api_key: str | None = Field(None, description="A new key for the provider; empty keeps the stored one")
    clear_key: bool = False
    reason: str = ""


class AiTest(BaseModel):
    """What a configuration gives, tried before saving it."""

    available: bool
    reason: str
    models: list[str] = Field(description="The models the provider offers, where it says")
    answer: str | None = Field(description="A few words from the model")
    seconds: float | None = Field(description="How long those took")


class VaStatus(BaseModel):
    enabled: bool
    source: str | None = Field(None, description="The machine files it runs")


class Me(BaseModel):
    name: str
    role: str = Field(description="viewer, physicist, operator or admin")
    logged_in: bool
    agent: bool = Field(False, description="Calling with an agent's token")


class Login(BaseModel):
    name: str
    password: str


class EngineChoice(BaseModel):
    name: str
    available: bool


class VaOptions(BaseModel):
    """What machines can be made of."""

    source: str
    rings: list[str]
    routes: list[str]
    modes: list[str]
    engines: list[EngineChoice]
    fidelities: list[str]
    sandbox_quota: int


class MachineEntry(BaseModel):
    id: str
    name: str
    kind: str = Field(description="ring or route")
    target: str = Field(description="The ring's branch name, or the route's name")
    engine: str | None
    fidelity: str
    mode: str | None
    shared: bool
    owner: str
    parent: str | None
    particles: int
    rate: float | None
    ramp_turns: int = Field(description="Turns a ring's setting change ramps over; 0 steps")
    autostart: bool
    state: str = Field(description="stopped, starting, running, stopping or failed")
    error: str | None
    run: str | None = Field(description="The run it writes now (its BPM history, /runs/{run})")
    position: int | None = Field(description="Its run's turn (ring) or shot (route) now")
    started: str | None
    changes: int = Field(description="Setting channels away from their original values")
    warnings: list[str] = Field(description="What the run found wrong, e.g. an RF phase that holds no bunch")


class NewMachine(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    kind: Literal["ring", "route"]
    target: str
    engine: str | None = None
    fidelity: str = "standard"
    mode: str | None = None
    shared: bool = True
    particles: int = Field(50, ge=1, le=100_000)
    rate: float | None = Field(None, gt=0)
    ramp_turns: int = Field(1024, ge=0, le=1_000_000, description="Turns a ring's setting change ramps over")
    autostart: bool = False
    reason: str = ""


class Fork(BaseModel):
    name: str | None = None
    reason: str = ""


class Reason(BaseModel):
    reason: str = ""


class SettingRow(BaseModel):
    channel: str
    device: str
    unit: str
    kind: str = Field(description="circuit (a supply's current) or parameter (e.g. an RF voltage)")
    limits: list[float] | None
    original: float = Field(description="From the machine files: design, and the mode")
    value: float = Field(description="Now: the live value, or the original")
    changed: bool
    by: str | None
    at: str | None
    occurrences: list[str] = Field(description="Lattice occurrences it sets")


class Change(BaseModel):
    channel: str
    value: float


class WriteRequest(BaseModel):
    changes: list[Change] = Field(min_length=1)
    reason: str = ""
    atomic: bool = Field(True, description="One refused write refuses them all")


class WriteEntry(BaseModel):
    channel: str
    unit: str
    old: float | None
    new: float
    applied: bool
    refusal: str | None


class WriteResult(BaseModel):
    correlation: str | None = Field(description="Groups the writes of one change in the log")
    applied: bool
    writes: list[WriteEntry]


class ResetRequest(BaseModel):
    channels: list[str] | None = Field(None, description="None: every changed channel")
    reason: str = ""


class SnapshotRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class SnapshotEntry(BaseModel):
    id: int
    name: str
    time: str | None = None
    user: str | None = None
    changes: int


class BeamRequest(BaseModel):
    action: Literal["kick", "inject"]
    x: float = Field(0.0, description="Kick angle, rad")
    y: float = 0.0
    reason: str = ""


class LogEntry(BaseModel):
    id: int
    time: str
    user: str
    machine: str
    channel: str
    unit: str
    old: float | None
    new: float
    applied: bool
    refusal: str | None
    correlation: str
    reason: str
    run: str | None
    turn: int | None = Field(description="The run's turn or shot when it was written")


class ActionEntry(BaseModel):
    id: int
    time: str
    user: str
    machine: str | None
    action: str
    detail: dict[str, Any]
    reason: str


class UserEntry(BaseModel):
    name: str
    role: str
    created: str
    disabled: bool


class NewUser(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    role: str


class UserCreated(BaseModel):
    name: str
    role: str
    password: str = Field(description="Shown once")


class UserChange(BaseModel):
    role: str | None = None
    disabled: bool | None = None


class PasswordChange(BaseModel):
    old: str
    new: str = Field(min_length=8)


class TokenRequest(BaseModel):
    label: str = Field(min_length=1, max_length=80)
    kind: Literal["api", "agent"] = Field(
        "api",
        description="api: a notebook's, with its user's rights; agent: a model's harness, which on shared "
        "machines only proposes (a person applies)",
    )


class TokenIssued(BaseModel):
    token: str = Field(description="Shown once: send it as Authorization: Bearer <token>")
    label: str


class TokenEntry(BaseModel):
    label: str
    created: str
    kind: str = "api"


class ProposalRequest(BaseModel):
    """An operation to compute on a machine as it is set (ADR 0010); nothing is written."""

    operation: Literal["orbit_correction", "local_bump", "tune_move"]
    plane: Literal["x", "y"] = "x"
    source: Literal["measured", "model"] = Field(
        "measured", description="Orbit correction: the BPMs' latest slow orbit, or the model's own"
    )
    method: Literal["svd", "micado"] = "svd"
    damping: float = Field(0.01, gt=0, description="SVD: Tikhonov damping, of the largest singular value")
    count: int = Field(3, ge=1, le=50, description="MICADO: correctors to use")
    at: str | None = Field(None, description="Bump: a BPM's name (or an element's)")
    amplitude_mm: float | None = Field(None, description="Bump: the offset there, mm")
    angle_mrad: float | None = Field(None, description="Bump: the angle there, mrad (four correctors)")
    dqx: float = Field(0.0, description="Tune move: the change of the horizontal tune")
    dqy: float = Field(0.0, description="Tune move: the change of the vertical tune")
    correctors: list[str] | None = Field(None, description="Only these corrector channels")
    quadrupoles: list[str] | None = Field(None, description="Tune move: only these quadrupole channels")
    reason: str = ""


class ProposalChange(BaseModel):
    channel: str
    device: str
    unit: str
    before: float
    after: float
    limits: list[float] | None = None


class ProposalCheck(BaseModel):
    name: str
    ok: bool
    value: float | None = None
    limit: float | None = None
    note: str = ""


class ProposalBody(BaseModel):
    operation: str
    machine: str
    summary: str
    changes: list[ProposalChange]
    predicted: dict[str, Any]
    checks: list[ProposalCheck]
    method: str
    parameters: dict[str, Any]
    engine: str
    references: list[str]
    context: dict[str, Any] = Field(default_factory=dict, description="What was measured when proposed")


class ProposalEntry(BaseModel):
    id: str
    machine: str
    created: str
    user: str
    agent: bool = Field(description="Proposed through an agent's token")
    operation: str
    request: dict[str, Any]
    proposal: ProposalBody
    state: str = Field(description="proposed, applied, verified, discarded or stale")
    turn: int | None
    decided: str | None
    decided_by: str | None
    applied_turn: int | None
    correlation: str | None
    verification: dict[str, Any] | None


class TuneReading(BaseModel):
    """A ring's tune monitor: fractional tunes folded into 0 to 0.5, as one pickup sees them."""

    x: float
    y: float
    turn: int
    turns: int = Field(description="Turns the spectrum took")
    settled: bool = Field(description="No setting ramped during those turns")


def va_router(get: Callable[[], VirtualAccelerator | None]) -> APIRouter:
    """The ``/va`` routes, on whatever virtual accelerator ``get`` returns (None: none here)."""
    router = APIRouter(prefix="/va", tags=["virtual accelerator"])

    def service() -> VirtualAccelerator:
        va = get()
        if va is None:
            raise HTTPException(
                404, "this server runs no virtual accelerator; start eicvibe-web with --va STATE"
            )
        return va

    def user(request: Request) -> User:
        va = get()
        if va is None:
            return ANONYMOUS
        header = request.headers.get("authorization", "")
        token = (
            header.removeprefix("Bearer ").strip()
            if header.startswith("Bearer ")
            else request.cookies.get(COOKIE)
        )
        found = va.store.resolve_token(token) if token else None
        return found or ANONYMOUS

    Caller = Annotated[User, Depends(user)]

    def call(function: Callable[[], Any]) -> Any:
        try:
            return function()
        except Forbidden as error:
            raise HTTPException(401 if str(error) == "log in first" else 403, str(error)) from None
        except Conflict as error:
            raise HTTPException(409, str(error)) from None
        except KeyError as error:
            raise HTTPException(404, str(error.args[0]) if error.args else "not found") from None
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    def needs(me: User, role: Role) -> None:
        if me is ANONYMOUS:
            raise HTTPException(401, "log in first")
        if not me.can(role):
            raise HTTPException(403, f"{me.name} is a {me.role.label}; this needs a {role.label}")

    def admin(me: User) -> None:
        needs(me, Role.ADMIN)
        if me.agent:
            raise HTTPException(403, "an agent's token cannot administer; a person does")

    # -- who --------------------------------------------------------------------------------

    @router.get("/status", response_model=VaStatus)
    def status() -> VaStatus:
        va = get()
        return VaStatus(enabled=va is not None, source=va.source.name if va else None)

    @router.get("/me", response_model=Me)
    def whoami(me: Caller) -> Me:
        service()
        return Me(name=me.name, role=me.role.label, logged_in=me is not ANONYMOUS, agent=me.agent)

    @router.post("/login", response_model=Me)
    def login(body: Login, response: Response) -> Me:
        va = service()
        found = va.store.check_password(body.name, body.password)
        if found is None:
            va.store.act(body.name, "login refused")
            raise HTTPException(401, "wrong name or password")
        token = va.store.issue_token(found.name, "session", "browser", hours=SESSION_HOURS)
        response.set_cookie(
            COOKIE, token, max_age=SESSION_HOURS * 3600, httponly=True, samesite="lax", path="/api/"
        )
        va.store.act(found.name, "login")
        return Me(name=found.name, role=found.role.label, logged_in=True)

    @router.post("/logout", status_code=204)
    def logout(request: Request, response: Response) -> Response:
        va = service()
        token = request.cookies.get(COOKIE)
        if token:
            va.store.revoke_token(token)
        response = Response(status_code=204)
        response.delete_cookie(COOKIE, path="/api/")
        return response

    @router.post("/me/password", status_code=204)
    def change_password(me: Caller, body: PasswordChange) -> Response:
        va = service()
        if me is ANONYMOUS or va.store.check_password(me.name, body.old) is None:
            raise HTTPException(401, "wrong password")
        va.store.set_password(me.name, body.new)
        va.store.act(me.name, "change password")
        return Response(status_code=204)

    @router.get("/me/tokens", response_model=list[TokenEntry])
    def my_tokens(me: Caller) -> list[TokenEntry]:
        va = service()
        if me is ANONYMOUS:
            raise HTTPException(401, "log in first")
        return [TokenEntry(**t) for t in va.store.tokens(me.name)]

    @router.post("/me/tokens", response_model=TokenIssued)
    def new_token(me: Caller, body: TokenRequest) -> TokenIssued:
        va = service()
        if me is ANONYMOUS:
            raise HTTPException(401, "log in first")
        if me.agent:
            raise HTTPException(403, "an agent's token cannot make tokens")
        token = va.store.issue_token(me.name, body.kind, body.label)
        va.store.act(me.name, "issue token", label=body.label, kind=body.kind)
        return TokenIssued(token=token, label=body.label)

    # -- machines -------------------------------------------------------------------------------

    @router.get("/options", response_model=VaOptions)
    def options() -> VaOptions:
        return VaOptions(**service().options())

    @router.get("/machines", response_model=list[MachineEntry])
    def machines() -> list[MachineEntry]:
        return [MachineEntry(**m) for m in service().machines()]

    @router.post("/machines", response_model=MachineEntry, status_code=201)
    def create(me: Caller, body: NewMachine) -> MachineEntry:
        va = service()
        return MachineEntry(**call(lambda: va.create(me, **body.model_dump())))

    @router.get("/machines/{machine}", response_model=MachineEntry)
    def machine(machine: str) -> MachineEntry:
        va = service()
        return MachineEntry(**call(lambda: va.machine(machine)))

    @router.post("/machines/{machine}/start", response_model=MachineEntry)
    def start(me: Caller, machine: str, body: Reason | None = None) -> MachineEntry:
        va = service()
        return MachineEntry(**call(lambda: va.start(me, machine, (body or Reason()).reason)))

    @router.post("/machines/{machine}/stop", response_model=MachineEntry)
    def stop(me: Caller, machine: str, body: Reason | None = None) -> MachineEntry:
        va = service()
        return MachineEntry(**call(lambda: va.stop(me, machine, (body or Reason()).reason)))

    @router.post("/machines/{machine}/fork", response_model=MachineEntry, status_code=201)
    def fork(me: Caller, machine: str, body: Fork | None = None) -> MachineEntry:
        va = service()
        body = body or Fork()
        return MachineEntry(**call(lambda: va.fork(me, machine, body.name, body.reason)))

    @router.delete("/machines/{machine}", status_code=204)
    def remove(me: Caller, machine: str, reason: str = "") -> Response:
        va = service()
        call(lambda: va.remove(me, machine, reason))
        return Response(status_code=204)

    @router.post("/machines/{machine}/beam", status_code=204)
    def beam(me: Caller, machine: str, body: BeamRequest) -> Response:
        va = service()
        call(lambda: va.beam(me, machine, body.action, body.x, body.y, body.reason))
        return Response(status_code=204)

    @router.post("/machines/{machine}/acquire", status_code=204)
    def acquire(me: Caller, machine: str, body: Reason | None = None) -> Response:
        """Capture a ring's BPM buffer now; it appears among the run's captures."""
        va = service()
        call(lambda: va.acquire(me, machine, (body or Reason()).reason))
        return Response(status_code=204)

    # -- settings ---------------------------------------------------------------------------------

    @router.get("/machines/{machine}/settings", response_model=list[SettingRow])
    def settings(machine: str, changed: bool = False) -> list[SettingRow]:
        va = service()
        return [SettingRow(**r) for r in call(lambda: va.settings(machine, changed))]

    @router.post("/machines/{machine}/settings", response_model=WriteResult)
    def write(me: Caller, machine: str, body: WriteRequest) -> WriteResult:
        va = service()
        changes = {c.channel: c.value for c in body.changes}
        return WriteResult(**call(lambda: va.write(me, machine, changes, body.reason, atomic=body.atomic)))

    @router.get("/machines/{machine}/tunes", response_model=TuneReading)
    def tunes(machine: str) -> TuneReading:
        """The ring's tune monitor now."""
        reading = call(lambda: service().tunes(machine))
        if reading is None:
            raise HTTPException(
                404, f"{machine} has no tune reading yet: a running ring has one after 1024 turns"
            )
        return TuneReading(**reading)

    @router.post("/machines/{machine}/proposals", response_model=ProposalEntry, status_code=201)
    def propose(me: Caller, machine: str, body: ProposalRequest) -> ProposalEntry:
        """Compute an operation (orbit correction, a closed bump, a tune move) on the machine as it is set
        and as its BPMs read; nothing is written. A large ring's first orbit correction takes a while
        (its response matrix)."""
        va = service()
        return ProposalEntry(**call(lambda: va.propose(me, machine, body.model_dump())))

    @router.get("/machines/{machine}/proposals", response_model=list[ProposalEntry])
    def proposals(machine: str) -> list[ProposalEntry]:
        va = service()
        return [ProposalEntry(**p) for p in call(lambda: va.proposals(machine))]

    @router.get("/proposals/{proposal}", response_model=ProposalEntry)
    def proposal(proposal: str) -> ProposalEntry:
        va = service()
        return ProposalEntry(**call(lambda: va.proposal(proposal)))

    @router.post("/proposals/{proposal}/apply", response_model=ProposalEntry)
    def apply(me: Caller, proposal: str, body: Reason | None = None) -> ProposalEntry:
        """Write a proposal's changes: on a shared machine by a person (an agent's token cannot)."""
        va = service()
        return ProposalEntry(**call(lambda: va.apply(me, proposal, (body or Reason()).reason)))

    @router.post("/proposals/{proposal}/verify", response_model=ProposalEntry)
    def verify(me: Caller, proposal: str) -> ProposalEntry:
        """What the BPMs or the tune monitor show since it was applied, against the prediction."""
        va = service()
        return ProposalEntry(**call(lambda: va.verify(me, proposal)))

    @router.post("/proposals/{proposal}/discard", response_model=ProposalEntry)
    def discard(me: Caller, proposal: str, body: Reason | None = None) -> ProposalEntry:
        va = service()
        return ProposalEntry(**call(lambda: va.discard(me, proposal, (body or Reason()).reason)))

    @router.post("/machines/{machine}/reset", response_model=WriteResult)
    def reset(me: Caller, machine: str, body: ResetRequest) -> WriteResult:
        va = service()
        return WriteResult(**call(lambda: va.reset(me, machine, body.channels, body.reason)))

    @router.get("/machines/{machine}/snapshots", response_model=list[SnapshotEntry])
    def snapshots(machine: str) -> list[SnapshotEntry]:
        va = service()
        call(lambda: va.machine(machine))
        return [SnapshotEntry(**s) for s in va.store.snapshots(machine)]

    @router.post("/machines/{machine}/snapshots", response_model=SnapshotEntry, status_code=201)
    def snapshot(me: Caller, machine: str, body: SnapshotRequest) -> SnapshotEntry:
        va = service()
        return SnapshotEntry(**call(lambda: va.snapshot(me, machine, body.name)))

    @router.post("/machines/{machine}/snapshots/{snapshot}/restore", response_model=WriteResult)
    def restore(me: Caller, machine: str, snapshot: int, body: Reason | None = None) -> WriteResult:
        va = service()
        return WriteResult(**call(lambda: va.restore(me, machine, snapshot, (body or Reason()).reason)))

    # -- the logs -----------------------------------------------------------------------------------

    @router.get("/log", response_model=list[LogEntry])
    def log(
        me: Caller,
        machine: str | None = None,
        user_name: str | None = Query(None, alias="user"),
        channel: str | None = None,
        correlation: str | None = None,
        before: int | None = None,
        limit: int = Query(200, ge=1, le=2000),
    ) -> list[LogEntry]:
        """Every write, newest first; ``before`` an id pages back. For users who are logged in."""
        va = service()
        needs(me, Role.VIEWER)
        rows = va.store.writes(
            machine=machine,
            user=user_name,
            channel=channel,
            correlation=correlation,
            before=before,
            limit=limit,
        )
        return [LogEntry(**r) for r in rows]

    @router.get("/actions", response_model=list[ActionEntry])
    def actions(
        me: Caller,
        machine: str | None = None,
        user_name: str | None = Query(None, alias="user"),
        before: int | None = None,
        limit: int = Query(200, ge=1, le=2000),
    ) -> list[ActionEntry]:
        """Starts, stops, forks, snapshots, kicks, logins and user changes, newest first; for operators."""
        va = service()
        needs(me, Role.OPERATOR)
        return [
            ActionEntry(**a)
            for a in va.store.actions(machine=machine, user=user_name, before=before, limit=limit)
        ]

    # -- users --------------------------------------------------------------------------------------

    @router.get("/users", response_model=list[UserEntry])
    def users(me: Caller) -> list[UserEntry]:
        va = service()
        admin(me)
        return [UserEntry(**u) for u in va.store.users()]

    @router.post("/users", response_model=UserCreated, status_code=201)
    def add_user(me: Caller, body: NewUser) -> UserCreated:
        va = service()
        admin(me)
        role = call(lambda: Role.named(body.role))
        password = call(lambda: va.store.add_user(body.name, role))
        va.store.act(me.name, "add user", name=body.name, role=role.label)
        return UserCreated(name=body.name, role=role.label, password=password)

    @router.patch("/users/{name}", response_model=UserEntry)
    def change_user(me: Caller, name: str, body: UserChange) -> UserEntry:
        va = service()
        admin(me)
        demoted = body.role is not None and call(lambda: Role.named(body.role)) < Role.ADMIN
        if name == me.name and (body.disabled or demoted):
            raise HTTPException(409, "an administrator cannot lock themselves out")
        if body.role is not None:
            role = call(lambda: Role.named(body.role))
            call(lambda: va.store.set_role(name, role))
            va.store.act(me.name, "change role", name=name, role=role.label)
        if body.disabled is not None:
            call(lambda: va.store.set_disabled(name, body.disabled))
            va.store.act(me.name, "disable user" if body.disabled else "enable user", name=name)
        found = next((u for u in va.store.users() if u["name"] == name), None)
        if found is None:
            raise HTTPException(404, f"no user {name}")
        return UserEntry(**found)

    @router.post("/users/{name}/password", response_model=UserCreated)
    def reset_password(me: Caller, name: str) -> UserCreated:
        va = service()
        admin(me)
        password = call(lambda: va.store.set_password(name))
        va.store.act(me.name, "reset password", name=name)
        role = next(u["role"] for u in va.store.users() if u["name"] == name)
        return UserCreated(name=name, role=role, password=password)

    # -- the AI features (ADR 0009) ------------------------------------------------------------

    def ai_settings(va: VirtualAccelerator) -> AiSettings:
        config = va.ai_config()
        key, source = config.key()
        return AiSettings(
            enabled=config.enabled,
            provider=config.provider,
            model=config.model,
            url=config.url,
            key=source,
            key_hint=f"…{key[-4:]}" if source == "stored" and len(key) >= 8 else None,
            saved=va.ai_saved(),
            providers={name: AiProvider(**info) for name, info in PROVIDERS.items()},
        )

    @router.get("/ai", response_model=AiSettings)
    def get_ai(me: Caller) -> AiSettings:
        """The AI features' configuration, for administrators; keys are never sent."""
        va = service()
        admin(me)
        return ai_settings(va)

    @router.put("/ai", response_model=AiSettings)
    def set_ai(me: Caller, body: AiChange) -> AiSettings:
        """Choose the AI features' model; logged as an action, without the key."""
        va = service()
        admin(me)
        call(lambda: va.set_ai(me, **body.model_dump()))
        return ai_settings(va)

    @router.post("/ai/test", response_model=AiTest)
    def test_ai(me: Caller, body: AiChange) -> AiTest:
        """Try a configuration without saving it: does the model answer, and how fast?"""
        va = service()
        admin(me)
        keys = dict(va.ai_config().keys)
        if body.api_key:
            keys[body.provider] = body.api_key.strip()
        config = AiConfig(provider=body.provider, model=body.model.strip(), url=body.url.strip(), keys=keys)
        model = config.make()
        available, reason = model.status()
        try:
            models = model.models()
        except AiUnavailable:
            models = []
        answer = seconds = None
        if available:
            start = time.monotonic()
            messages = [
                {"role": "system", "content": "Answer with one word."},
                {"role": "user", "content": "Say: ready"},
            ]
            try:
                answer = "".join(model.stream(messages, max_tokens=16)).strip()
                seconds = round(time.monotonic() - start, 2)
            except AiUnavailable as error:
                available, reason = False, str(error)
        return AiTest(available=available, reason=reason, models=models, answer=answer, seconds=seconds)

    return router
