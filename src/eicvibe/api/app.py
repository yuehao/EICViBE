"""HTTP API of the lattice viewer, and the web client it serves.

Every endpoint is a read of an immutable revision, addressed by digest, so
the server keeps no per-client state: any number of browsers, notebooks and
server worker processes can use it at once without affecting each other.
Responses are built once per revision, kept as bytes (and gzip bytes), and
revalidated with ETags, so repeated and concurrent requests cost almost
nothing. What a user looks at (branch, zoom, selection) lives in the client.
Optics are computed by engine worker processes (:mod:`.optics`), also once
per revision, engine and fidelity.
"""

import os
from concurrent.futures.process import BrokenProcessPool
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import RootModel

from eicvibe.engines.contracts import EngineUnavailable, Fidelity, UnsupportedRequest
from eicvibe.va.api import va_router
from eicvibe.va.service import VirtualAccelerator

from . import schemas
from .catalog import Catalog, Payload, Revision
from .checks import RunChecks
from .optics import OpticsService, UnknownEngine
from .runs import MAX_ORBITS, RunCatalog

#: Where ``npm run build`` in ``web/`` puts the client.
STATIC_DIR = Path(__file__).parent / "static"
#: Lattice files and folders for :func:`app_from_environment`, separated by ``os.pathsep``.
LATTICES_VARIABLE = "EICVIBE_LATTICES"
#: Folders of virtual rings' runs (ADR 0007), likewise.
RUNS_VARIABLE = "EICVIBE_RUNS"
#: The virtual accelerator's state folder (ADR 0008), and the machine files it runs.
VA_VARIABLE = "EICVIBE_VA"
VA_SOURCE_VARIABLE = "EICVIBE_VA_SOURCE"
#: The first administrator's password, for automated setups; otherwise one is made and printed.
VA_ADMIN_VARIABLE = "EICVIBE_VA_ADMIN_PASSWORD"

_MISSING_CLIENT = """<!doctype html><meta charset="utf-8"><title>EICViBE</title>
<body style="font-family: system-ui; max-width: 40em; margin: 3em auto; line-height: 1.5">
<h1>EICViBE lattice viewer</h1>
<p>The API is running (<a href="/api/docs">documentation</a>), but the web client has not been built.
Build it with <code>npm --prefix web install &amp;&amp; npm --prefix web run build</code>
and reload this page.</p>"""


def _send(request: Request, payload: Payload) -> Response:
    headers = {"ETag": payload.etag, "Cache-Control": "no-cache", "Vary": "Accept-Encoding"}
    if payload.etag in request.headers.get("if-none-match", ""):
        return Response(status_code=304, headers=headers)
    if "gzip" in request.headers.get("accept-encoding", ""):
        headers["Content-Encoding"] = "gzip"
        return Response(payload.gzipped, media_type="application/json", headers=headers)
    return Response(payload.body, media_type="application/json", headers=headers)


def _not_modified(request: Request, etag: str) -> Response | None:
    if etag in request.headers.get("if-none-match", ""):
        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": "no-cache"})
    return None


def create_app(
    catalog: Catalog,
    *,
    static_dir: Path | None = STATIC_DIR,
    optics: OpticsService | None = None,
    runs: RunCatalog | None = None,
    va: VirtualAccelerator | None = None,
) -> FastAPI:
    optics = optics or OpticsService()
    runs = runs or RunCatalog([])
    checks = RunChecks(runs, catalog, optics.worker)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if va is not None:
            va.autostart()
        yield
        optics.close()  # stops the engine workers
        if va is not None:
            va.close()  # stops the machines; their settings and the log stay

    app = FastAPI(
        title="EICViBE lattice viewer",
        version=version("eicvibe"),
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    api = APIRouter(prefix="/api/v1")

    def revision(digest: str) -> Revision:
        found = catalog.get(digest)
        if found is None:
            raise HTTPException(404, f"no lattice revision {digest}")
        return found

    @api.get("/lattices", response_model=list[schemas.LatticeEntry])
    def lattices(request: Request) -> Response:
        """Lattices this server offers, at their current revisions."""
        entries = RootModel[list[schemas.LatticeEntry]]([r.entry() for r in catalog.current()])
        return _send(request, Payload.of(entries))

    @api.get("/lattices/{digest}", response_model=schemas.LatticeSummary)
    def lattice(digest: str, request: Request) -> Response:
        """Branches, forks, floor closures, shared components and diagnostics of a revision."""
        found = revision(digest)
        return _not_modified(request, found.etag("summary")) or _send(
            request, found.payload("summary", found.views.summary)
        )

    @api.get("/lattices/{digest}/branches/{branch}", response_model=schemas.BranchData)
    def branch(digest: str, branch: str, request: Request) -> Response:
        """Element table and floor coordinates of one branch (by id or name)."""
        found = revision(digest)
        try:
            branch_id = found.lattice.layout(branch).id
        except KeyError:
            raise HTTPException(404, f"no branch {branch}") from None
        key = f"branch/{branch_id}"
        return _not_modified(request, found.etag(key)) or _send(
            request, found.payload(key, lambda: found.views.branch(branch_id))
        )

    @api.get("/lattices/{digest}/occurrences/{occurrence}", response_model=schemas.OccurrenceDetail)
    def occurrence(digest: str, occurrence: str, request: Request) -> Response:
        """Parameters, floor pose, fork, shared component and devices of one occurrence."""
        found = revision(digest)
        key = f"occurrence/{occurrence}"
        if (cached := _not_modified(request, found.etag(key))) is not None:
            return cached
        try:
            payload = found.payload(key, lambda: found.views.occurrence(occurrence), keep=False)
        except KeyError:
            raise HTTPException(404, f"no occurrence {occurrence}") from None
        return _send(request, payload)

    @api.get("/lattices/{digest}/branches/{branch}/optics", response_model=schemas.BranchOptics)
    async def branch_optics(
        digest: str,
        branch: str,
        request: Request,
        engine: str = Query("xsuite", description="Engine name; see /engines"),
        fidelity: Fidelity = Fidelity.STANDARD,
    ) -> Response:
        """Design optics of one branch: beta functions, dispersion, phase advance, orbit, tunes, chromaticity.

        The first request for a revision computes the whole machine (lines start from the
        optics at their forks), which can take a while: TrackPad starts Julia on first use.
        """
        found = revision(digest)
        if branch.startswith("route:"):
            if branch.removeprefix("route:") not in found.views.routes:
                raise HTTPException(404, f"no {branch}")
        else:
            try:
                found.lattice.layout(branch)
            except KeyError:
                raise HTTPException(404, f"no branch {branch}") from None
        try:
            payload = await optics.branch(found, engine, fidelity, branch)
        except UnknownEngine:
            raise HTTPException(404, f"no engine {engine}") from None
        except UnsupportedRequest as error:
            raise HTTPException(422, str(error)) from None
        except EngineUnavailable as error:
            raise HTTPException(503, str(error)) from None
        except BrokenProcessPool:
            raise HTTPException(503, f"the {engine} worker stopped; try again") from None
        return _send(request, payload)

    @api.get("/runs", response_model=list[schemas.RunEntry])
    def run_list() -> list[schemas.RunEntry]:
        """Virtual rings' runs in the folders the server was given, newest first."""
        return runs.entries()

    @api.get("/runs/{run}/orbits", response_model=schemas.OrbitFrames)
    def orbits(
        run: str,
        request: Request,
        every: int = Query(5, ge=1, le=1_000_000, description="Turns between orbits"),
        count: int = Query(5, ge=1, le=MAX_ORBITS, description="Orbits"),
        end: int | None = Query(
            None, ge=1, description="The newest orbit is at turn end - 1; default: the latest"
        ),
        source: str = Query(
            "buffer",
            pattern=r"^(buffer|slow|capture:[\w.-]+)$",
            description="buffer (the latest turns), slow (the slow orbit's samples) or capture:<id>",
        ),
    ) -> Response:
        """BPM readings along the ring ``every`` turns (or samples) apart, from the BPMs' buffer, a
        capture, or the slow orbit."""
        try:
            payload = runs.orbits(run, every, count, end, source)
        except KeyError:
            raise HTTPException(404, f"no run {run}") from None
        except ValueError as error:
            raise HTTPException(404, str(error)) from None
        return _send(request, payload)

    @api.get("/runs/{run}/captures", response_model=list[schemas.CaptureEntry])
    def captures(run: str) -> list[schemas.CaptureEntry]:
        """A run's captures, oldest first: its BPM buffer frozen by a kick, injection, loss or request."""
        try:
            return runs.captures(run)
        except KeyError:
            raise HTTPException(404, f"no run {run}") from None

    @api.get("/runs/{run}/check", response_model=schemas.RunCheck)
    def check(
        run: str,
        source: str = Query(
            "kick",
            pattern=r"^(kick|buffer|capture:[\w.-]+)$",
            description="kick (the latest kick's capture, from the kick on), buffer (the latest turns) or "
            "capture:<id>",
        ),
        first: int | None = Query(
            None, ge=0, description="First turn; default: the kick, or the latest turns"
        ),
        turns: int = Query(1024, ge=64, le=8192, description="Turns to analyse"),
        engine: str | None = Query(None, description="The model's engine; default: the run's"),
        detuning: bool = Query(False, description="Also track single particles for tune against amplitude"),
    ) -> schemas.RunCheck:
        """A ring run's tracking against the model: tunes, phase advance, beta and orbit at its BPMs."""
        try:
            return checks.check(run, source, first, turns, engine, detuning)
        except KeyError:
            raise HTTPException(404, f"no run {run}") from None
        except UnknownEngine:
            raise HTTPException(404, f"no engine {engine}") from None
        except (ValueError, UnsupportedRequest) as error:
            raise HTTPException(422, str(error)) from None
        except EngineUnavailable as error:
            raise HTTPException(503, str(error)) from None
        except BrokenProcessPool:
            raise HTTPException(503, f"the {engine} worker stopped; try again") from None

    @api.get("/engines", response_model=list[schemas.EngineInfo])
    def engines() -> list[schemas.EngineInfo]:
        """Physics engines that can compute optics here."""
        return optics.engines()

    @api.get("/lattices/{digest}/lattice.json")
    def download(digest: str, request: Request) -> Response:
        """The revision itself, in canonical JSON, to open with ``Lattice.from_json``."""
        found = revision(digest)
        response = _not_modified(request, found.etag("lattice.json"))
        if response is None:
            response = Response(found.lattice.to_json(), media_type="application/json")
            response.headers["ETag"] = found.etag("lattice.json")
        response.headers["Content-Disposition"] = f'attachment; filename="{found.source}_{digest[:12]}.json"'
        return response

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    api.include_router(va_router(lambda: va))
    app.include_router(api)

    if static_dir is not None and (static_dir / "index.html").is_file():

        @app.middleware("http")
        async def cache_assets(request: Request, call_next):
            response = await call_next(request)
            if request.url.path.startswith("/assets/"):  # file names carry content hashes
                response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            elif not request.url.path.startswith("/api/"):
                response.headers.setdefault("Cache-Control", "no-cache")
            return response

        app.mount("/", StaticFiles(directory=static_dir, html=True), name="client")
    else:

        @app.get("/", include_in_schema=False)
        def missing_client() -> HTMLResponse:
            return HTMLResponse(_MISSING_CLIENT)

    return app


def app_from_environment() -> FastAPI:
    """The app for lattices listed in ``EICVIBE_LATTICES``; the factory server worker processes use."""
    paths = [p for p in os.environ.get(LATTICES_VARIABLE, "").split(os.pathsep) if p]
    runs = [p for p in os.environ.get(RUNS_VARIABLE, "").split(os.pathsep) if p]
    va = None
    if state := os.environ.get(VA_VARIABLE):
        from eicvibe.va import load_source

        va = VirtualAccelerator(load_source(os.environ[VA_SOURCE_VARIABLE]), state)
        runs.append(str(va.runs))
        given = os.environ.get(VA_ADMIN_VARIABLE)
        if (password := va.first_admin(given)) and not given:
            print(
                f"Virtual accelerator: a first user, admin, with password {password} "
                "(shown once; change it on the admin page)",
                flush=True,
            )
    return create_app(Catalog(paths), runs=RunCatalog(runs), va=va)
