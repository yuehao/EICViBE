"""The floor plan of every branch, as the viewer draws it (ADR 0016)."""

from traitlets import Unicode

from eicvibe.domain.lattice import Lattice
from eicvibe.store import Machine

from ._base import LatticeWidget


class FloorView(LatticeWidget):
    """The machine's floor plan: pan, zoom, hover, select. ``branch`` (or a route) is drawn over the
    others; picking an element of another branch makes it the branch shown, so that views linked to
    this one (:func:`~._base.link`) follow."""

    view = Unicode("floor").tag(sync=True)

    def __init__(self, machine: Machine | Lattice, branch: str | None = None, *, height: int = 380) -> None:
        super().__init__(machine, branch or "", height=height)
