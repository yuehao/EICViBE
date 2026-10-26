"""The viewer's own views in a notebook, each a widget of its own that the kernel feeds (ADR 0016).

- :class:`FloorView` (:mod:`.floor`): the floor plan of every branch.
- :class:`OpticsView` (:mod:`.optics`): a branch's or route's optics under its lattice strip,
  computed by the engines' workers.
- :class:`OrbitView` (:mod:`.orbit`): BPM orbits of virtual runs under the lattice strip, live
  while they run.
- :class:`Chart` (:mod:`.chart`): the viewer's chart, for any x-y data (:func:`chart`).

Each draws with the page's own components and their interaction (hover, select, drag to zoom along
s, follow a fork), and each stands alone. What a view shows (``branch``, ``selection``, its
``selected`` occurrence, ``s_range``) is shared with Python: read it, set it, or ``observe`` it.
:func:`link` ties views together as the page's are, in the browser.

The widgets' module is built with the page: ``npm run build`` in ``web/``.
"""

from ._base import LINKED, STATIC, AlongS, LatticeWidget, link
from .chart import PALETTE, Chart, chart
from .floor import FloorView
from .optics import OpticsView
from .orbit import OrbitView

__all__ = [
    "LINKED",
    "PALETTE",
    "STATIC",
    "AlongS",
    "Chart",
    "FloorView",
    "LatticeWidget",
    "OpticsView",
    "OrbitView",
    "chart",
    "link",
]
