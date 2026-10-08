"""Web service of EICViBE: the lattice viewer API and the web client (``web/``) it serves.

Needs the ``web`` extra (FastAPI, uvicorn). Run it with ``eicvibe-web`` or
``python -m eicvibe.api``; see :mod:`.__main__`.
"""

from .app import create_app
from .catalog import Catalog, Revision
from .views import RevisionViews

__all__ = ["Catalog", "Revision", "RevisionViews", "create_app"]
