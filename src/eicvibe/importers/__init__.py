"""Importers from other lattice formats into :mod:`eicvibe.domain.lattice`.

Each importer returns the lattice together with notes on every conversion
that is not a plain copy. Engine packages (cpymad, ...) are optional
dependencies and are imported only when an importer runs.
"""

from .madx import ImportNote, MadxImport, Severity, UnsupportedElementError, import_cpymad, import_madx

__all__ = ["ImportNote", "MadxImport", "Severity", "UnsupportedElementError", "import_cpymad", "import_madx"]
