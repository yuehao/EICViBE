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

from collections.abc import Callable
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from .service import Conflict, Forbidden, VirtualAccelerator
from .store import ANONYMOUS, SESSION_HOURS, Role, User

COOKIE = "eicvibe_session"


class VaStatus(BaseModel):
    enabled: bool
    source: str | None = Field(None, description="The machine files it runs")


class Me(BaseModel):
    name: str
    role: str = Field(description="viewer, physicist, operator or admin")
    logged_in: bool


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


class TokenIssued(BaseModel):
    token: str = Field(description="Shown once: send it as Authorization: Bearer <token>")
    label: str


class TokenEntry(BaseModel):
    label: str
    created: str


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

    # -- who --------------------------------------------------------------------------------

    @router.get("/status", response_model=VaStatus)
    def status() -> VaStatus:
        va = get()
        return VaStatus(enabled=va is not None, source=va.source.name if va else None)

    @router.get("/me", response_model=Me)
    def whoami(me: Caller) -> Me:
        service()
        return Me(name=me.name, role=me.role.label, logged_in=me is not ANONYMOUS)

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
        token = va.store.issue_token(me.name, "api", body.label)
        va.store.act(me.name, "issue token", label=body.label)
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

    return router
