"""The virtual accelerator service (ADR 0008): machines people start, change and watch, and the log.

:class:`VirtualAccelerator` holds the machines (rings and routes of ADR 0007
that keep running), their settings in layers (the machine files' design and
mode, under a live layer of what people wrote), the users and their roles, and
the append-only log of every change. :mod:`eicvibe.va.api` serves it over HTTP
(the admin page and the notebook client, :mod:`eicvibe.va.client`).
"""

from .service import Conflict, Forbidden, MachineSpec, Source, VirtualAccelerator, load_source
from .settings import MachineSettings, Setting
from .store import ANONYMOUS, Role, Store, User

__all__ = [
    "ANONYMOUS",
    "Conflict",
    "Forbidden",
    "MachineSettings",
    "MachineSpec",
    "Role",
    "Setting",
    "Source",
    "Store",
    "User",
    "VirtualAccelerator",
    "load_source",
]
