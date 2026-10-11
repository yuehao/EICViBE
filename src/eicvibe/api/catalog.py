"""The lattices a server offers, addressed by content digest.

A catalog is configured with lattice files or folders when the server starts;
clients cannot make it read arbitrary server paths. Each file is loaded into
an immutable revision keyed by its digest, so every request for that digest
gets the same answer no matter who else is using the server, and responses
can be computed once and shared. When a file changes on disk the catalog
loads the new revision on its next refresh; recent older revisions stay
available, so a client looking at one is not disturbed.

A source is a machine folder (with ``machine.yaml``; :mod:`eicvibe.store`),
whose lattice and registry are read from the files people edit, so a saved
edit shows on the next refresh; or a folder of machine folders; or a lattice
file ``<stem>_lattice.json``, paired with the machine registry
``<stem>_registry.json`` beside it if that registry describes the lattice;
or a folder of such files.
"""

import gzip
import hashlib
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from eicvibe.domain.lattice import Lattice
from eicvibe.domain.machine import MachineRegistry
from eicvibe.store import FileError, is_machine, load_machine, read_settings, source_files
from eicvibe.store.machine import read_index, read_routes_file

from .schemas import SCHEMA_VERSION, LatticeEntry
from .views import RevisionViews

log = logging.getLogger(__name__)

LATTICE_SUFFIX = "_lattice.json"
REGISTRY_SUFFIX = "_registry.json"
ROUTES_SUFFIX = "_routes.yaml"


@dataclass(frozen=True, slots=True)
class Payload:
    """A serialized response, with its gzip form and validator."""

    body: bytes
    gzipped: bytes
    etag: str

    @classmethod
    def of(cls, model: BaseModel, *, etag: str | None = None) -> Payload:
        body = model.model_dump_json().encode()
        etag = etag or f'"{hashlib.sha256(body).hexdigest()[:24]}"'
        return cls(body, gzip.compress(body, compresslevel=6, mtime=0), etag)


class _Once:
    """A value computed at most once, even when many threads ask for it together."""

    __slots__ = ("lock", "value")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.value: Payload | None = None


def _code_tag() -> str:
    """Changes whenever the code that builds responses does.

    Part of every ETag, so a browser that cached a response from another
    server version revalidates to the new one instead of getting 304.
    """
    from eicvibe.geometry import floor

    from . import schemas, views

    digest = hashlib.sha256(str(SCHEMA_VERSION).encode())
    for module in (schemas, views, floor):
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()[:10]


CODE_TAG = _code_tag()


@dataclass(eq=False)
class Revision:
    """One loaded lattice revision, with the registry it was loaded with, and its cached responses."""

    source: str
    path: Path
    views: RevisionViews
    loaded_at: datetime
    registry_tag: str = "none"
    """Hash of the registry file, or "none": names and devices are part of every response."""
    modes: dict[str, dict] = field(default_factory=dict)
    """Operating modes of a machine folder (settings/<mode>.yaml): their values over the lattice's."""
    _cache: dict[str, _Once] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def digest(self) -> str:
        return self.views.digest

    @property
    def lattice(self) -> Lattice:
        return self.views.lattice

    def etag(self, key: str) -> str:
        """Validator of the response ``key``, known without building it."""
        return f'"{self.digest[:16]}-{self.registry_tag[:8]}-{_short_hash(key)}-{CODE_TAG}"'

    def payload(self, key: str, build: Callable[[], BaseModel], *, keep: bool = True) -> Payload:
        """The response ``key``, built by ``build`` on first use and, if ``keep``, shared afterwards."""
        if not keep:
            return Payload.of(build(), etag=self.etag(key))
        with self._lock:
            once = self._cache.setdefault(key, _Once())
        with once.lock:
            if once.value is None:
                once.value = Payload.of(build(), etag=self.etag(key))
            return once.value

    def entry(self) -> LatticeEntry:
        lattice = self.lattice
        return LatticeEntry(
            source=self.source,
            name=lattice.name,
            digest=self.digest,
            branches=len(lattice.branches),
            occurrences=sum(len(b.occurrences) for b in lattice.branches),
            registry=self.views.registry is not None,
            loaded_at=self.loaded_at.isoformat(timespec="seconds"),
        )


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class _Source:
    name: str
    lattice: Path
    """The lattice file, or the machine folder."""
    registry: Path | None
    machine: bool = False
    routes: Path | None = None
    """A lattice file's routes file, beside it."""


def _beside(path: Path, suffix: str) -> Path | None:
    """The file beside a ``<stem>_lattice.json`` with ``suffix`` in place of ``_lattice.json``, if any."""
    if not path.name.endswith(LATTICE_SUFFIX):
        return None
    candidate = path.with_name(path.name.removesuffix(LATTICE_SUFFIX) + suffix)
    return candidate if candidate.is_file() else None


def _source_name(path: Path) -> str:
    return path.name.removesuffix(LATTICE_SUFFIX).removesuffix(".json")


def discover(paths: Iterable[Path | str]) -> list[_Source]:
    """Sources named by ``paths``: machine folders, folders of them, lattice files, folders of those."""
    found: list[tuple[str, Path, bool]] = []
    for item in map(Path, paths):
        if is_machine(item):
            found.append((_machine_name(item), item, True))
        elif item.is_dir():
            found += [(_machine_name(m), m, True) for m in sorted(item.iterdir()) if is_machine(m)]
            found += [(_source_name(p), p, False) for p in sorted(item.glob(f"*{LATTICE_SUFFIX}"))]
        elif item.is_file():
            found.append((_source_name(item), item, False))
        else:
            raise FileNotFoundError(f"no machine folder, lattice file or folder at {item}")
    sources, names = [], set()
    for base, path, machine in found:
        name, n = base, 2
        while name in names:
            name, n = f"{base}-{n}", n + 1
        names.add(name)
        registry = None if machine else _beside(path, REGISTRY_SUFFIX)
        routes = None if machine else _beside(path, ROUTES_SUFFIX)
        sources.append(_Source(name, path.resolve(), registry, machine, routes))
    return sources


def _machine_name(folder: Path) -> str:
    try:
        return str(read_index(folder)["name"])
    except FileError:
        return folder.name


def load_revision(source: _Source) -> Revision:
    if source.machine:
        machine = load_machine(source.lattice)
        modes = {
            path.stem: {
                (s.occurrence, s.parameter): s.value
                for s in read_settings(source.lattice, path.stem, machine.lattice)
            }
            for path in sorted((source.lattice / "settings").glob("*.yaml"))
        }
        # What besides the lattice is in responses: a mode's optics are cached with the revision.
        tagged = (
            machine.registry.model_dump_json()
            + "".join(r.model_dump_json() for r in machine.routes)
            + repr(sorted((mode, sorted(values.items())) for mode, values in modes.items()))
        )
        registry_tag = hashlib.sha256(tagged.encode()).hexdigest()[:12]
        views = RevisionViews(
            machine.lattice, machine.registry, source=source.name, routes=machine.routes, modes=tuple(modes)
        )
        return Revision(source.name, source.lattice, views, datetime.now(UTC), registry_tag, modes=modes)
    lattice = Lattice.from_json(source.lattice.read_text())
    registry, registry_tag = None, "none"
    if source.registry is not None:
        text = source.registry.read_bytes()
        candidate = MachineRegistry.model_validate_json(text)
        problems = (
            candidate.check_against(lattice) if candidate.lattice == lattice.name else ["other lattice"]
        )
        if problems:
            log.warning(
                "ignoring %s: it does not match %s (%s)", source.registry, source.lattice, problems[0]
            )
        else:
            registry, registry_tag = candidate, hashlib.sha256(text).hexdigest()[:12]
    routes = read_routes_file(source.routes, lattice) if source.routes is not None else ()
    if routes:
        tagged = registry_tag + "".join(r.model_dump_json() for r in routes)
        registry_tag = hashlib.sha256(tagged.encode()).hexdigest()[:12]
    views = RevisionViews(lattice, registry, source=source.name, routes=routes)
    return Revision(source.name, source.lattice, views, datetime.now(UTC), registry_tag)


class Catalog:
    """Current revision of every configured lattice file, plus recent older revisions."""

    def __init__(self, paths: Iterable[Path | str], *, keep: int = 16, refresh_interval: float = 2.0) -> None:
        self._sources = discover(paths)
        self._keep = keep
        self._refresh_interval = refresh_interval
        self._lock = threading.Lock()
        self._refreshing = threading.Lock()
        self._stamps: dict[str, tuple] = {}
        self._current: dict[str, Revision] = {}
        self._revisions: OrderedDict[str, Revision] = OrderedDict()
        self._checked = 0.0
        self.refresh(force=True)

    @staticmethod
    def _stamp(source: _Source) -> tuple:
        if source.machine:  # every file people edit; adding or removing one counts too
            return tuple(
                (str(f), f.stat().st_mtime_ns, f.stat().st_size) for f in source_files(source.lattice)
            )
        files = [f for f in (source.lattice, source.registry, source.routes) if f is not None]
        return tuple((s.st_mtime_ns, s.st_size) for s in (f.stat() for f in files if f.exists()))

    def refresh(self, *, force: bool = False) -> None:
        """Load files that changed on disk, at most once per refresh interval unless forced.

        Requests are not held up by a reload: while one thread loads, others
        keep serving the revisions already loaded.
        """
        if not force and time.monotonic() - self._checked < self._refresh_interval:
            return
        if not self._refreshing.acquire(blocking=force):
            return
        try:
            self._checked = time.monotonic()
            for source in self._sources:
                try:
                    stamp = self._stamp(source)
                    if self._stamps.get(source.name) == stamp:
                        continue
                    revision = load_revision(source)
                except (OSError, ValueError) as error:
                    log.error("cannot load %s: %s", source.lattice, error)
                    continue
                self._install(source.name, stamp, revision)
        finally:
            self._refreshing.release()

    def _install(self, name: str, stamp: tuple, revision: Revision) -> None:
        with self._lock:
            self._stamps[name] = stamp
            known = self._revisions.get(revision.digest)
            if known is not None and known.registry_tag == revision.registry_tag:
                revision = known  # the same lattice and registry: keep its cached responses
            revisions = OrderedDict(self._revisions)
            revisions[revision.digest] = revision
            revisions.move_to_end(revision.digest)
            current = {**self._current, name: revision}
            keep = {r.digest for r in current.values()}
            for digest in list(revisions):
                if len(revisions) <= max(self._keep, len(keep)):
                    break
                if digest not in keep:
                    del revisions[digest]
            # Readers take these without the lock, so replace them rather than mutate them.
            self._revisions, self._current = revisions, current

    def current(self) -> list[Revision]:
        self.refresh()
        return list(self._current.values())

    def get(self, digest: str) -> Revision | None:
        """A revision by digest, with or without the ``sha256:`` prefix."""
        self.refresh()
        return self._revisions.get(digest.removeprefix("sha256:"))
