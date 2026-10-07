"""HTTP API of the lattice viewer, and the web client it serves.

Every endpoint is a read of an immutable revision, addressed by digest, so
the server keeps no per-client state: any number of browsers, notebooks and
server worker processes can use it at once without affecting each other.
Responses are built once per revision, kept as bytes (and gzip bytes), and
revalidated with ETags, so repeated and concurrent requests cost almost
nothing. What a user looks at (branch, zoom, selection) lives in the client.
"""

import os
from importlib.metadata import version
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import RootModel

from . import schemas
from .catalog import Catalog, Payload, Revision

#: Where ``npm run build`` in ``web/`` puts the client.
STATIC_DIR = Path(__file__).parent / "static"
#: Lattice files and folders for :func:`app_from_environment`, separated by ``os.pathsep``.
LATTICES_VARIABLE = "EICVIBE_LATTICES"

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


def create_app(catalog: Catalog, *, static_dir: Path | None = STATIC_DIR) -> FastAPI:
    app = FastAPI(
        title="EICViBE lattice viewer",
        version=version("eicvibe"),
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
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
    return create_app(Catalog(paths))
