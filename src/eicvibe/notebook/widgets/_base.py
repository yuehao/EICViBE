"""What every view of a machine has, and how views are linked (ADR 0016).

A view is a widget of its own: the lattice's summary, the branch shown, the occurrence selected,
its height, and the requests its page components make, answered with the server's own JSON
(:class:`~eicvibe.api.views.RevisionViews`). Views along s add the s range. The widgets share
one built module (``npm run build`` in ``web/``), which draws the view the widget names.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import anywidget
from traitlets import Bool, Dict, Float, Int, List, Unicode, validate

from eicvibe.api.views import RevisionViews
from eicvibe.domain.lattice import Lattice, PlacedElement
from eicvibe.store import Machine, read_settings

STATIC = Path(__file__).parents[1] / "static"
MODULE, STYLES = STATIC / "widget.js", STATIC / "widget.css"
if not MODULE.is_file():
    raise ImportError(
        f"the notebook widgets are not built ({MODULE} is missing): run `npm run build` in EICViBE's web/"
    )

type Values = Mapping[tuple[str, str], float]


class LatticeWidget(anywidget.AnyWidget):
    """A view of a machine (or a lattice): see the module."""

    _esm = MODULE
    _css = STYLES

    view = Unicode().tag(sync=True)
    """What the module draws: floor, optics or orbit."""
    summary = Dict().tag(sync=True)
    branch = Unicode().tag(sync=True)
    """The branch shown, by id (a name is taken too), or ``route:<name>``; empty: none."""
    selection = Unicode(None, allow_none=True).tag(sync=True)
    """The occurrence selected, by id (a name in the branch shown is taken too); None: none."""
    height = Int(400).tag(sync=True)

    def __init__(self, machine: Machine | Lattice, branch: str, **traits: Any) -> None:
        if isinstance(machine, Machine):
            lattice, routes, registry = machine.lattice, machine.routes, machine.registry
            modes = {
                path.stem: {
                    (v.occurrence, v.parameter): v.value
                    for v in read_settings(machine.folder, path.stem, lattice)
                }
                for path in sorted((machine.folder / "settings").glob("*.yaml"))
            }
        elif isinstance(machine, Lattice):
            lattice, routes, registry, modes = machine, (), None, {}
        else:
            raise TypeError(f"a machine or a lattice, not {type(machine).__name__}")
        self._lattice = lattice
        self._routes = {route.name: route for route in routes}
        self._modes = modes
        self._views = RevisionViews(lattice, registry, source=lattice.name, routes=routes, modes=tuple(modes))
        summary = json.loads(self._views.summary().model_dump_json())
        super().__init__(summary=summary, branch=self._branch_id(branch), **traits)
        self.on_msg(self._answer)

    @property
    def selected(self) -> PlacedElement | None:
        """The occurrence selected, placed in its branch; None when none is."""
        return self._lattice.placement(self.selection) if self.selection else None

    @validate("branch")
    def _valid_branch(self, proposal: Any) -> str:
        return self._branch_id(proposal["value"])

    @validate("selection")
    def _valid_selection(self, proposal: Any) -> str | None:
        value = proposal["value"]
        if value is None or value in self._lattice.topology.placements:
            return value
        if not self.branch or self.branch.startswith("route:"):
            raise ValueError(f"select an occurrence by its id here, not {value!r}")
        return self._lattice.placement(value, branch=self.branch).id

    def _branch_id(self, branch: str) -> str:
        if not branch:
            return ""
        if branch.startswith("route:"):
            if (name := branch.removeprefix("route:")) not in self._routes:
                raise ValueError(f"no route {name}: a machine's routes.yaml has them")
            return branch
        return self._lattice.layout(branch).id

    # -- what the view asks ---------------------------------------------------------------------

    def _answer(self, _widget: Any, content: Any, _buffers: Any) -> None:
        if not isinstance(content, dict) or "request" not in content:
            return
        ident = content["request"]
        try:
            text = self._reply(ident, str(content.get("method")), content.get("args") or {})
        except Exception as error:  # noqa: BLE001 - the view shows it where the data would be
            self._fail(ident, error)
        else:
            if text is not None:
                self.send({"response": ident, "json": text})

    def _fail(self, ident: int, error: BaseException) -> None:
        message = error.args[0] if isinstance(error, KeyError) and error.args else str(error)
        self.send({"response": ident, "error": str(message) or type(error).__name__})

    def _reply(self, ident: int, method: str, args: dict[str, Any]) -> str | None:
        """The answer, or None when it comes later; views add their own requests."""
        if method == "branch":
            return self._views.branch(self._lattice.layout(args["branch"]).id).model_dump_json()
        raise ValueError(f"no request {method!r}")


class AlongS(LatticeWidget):
    """A view along s, under the lattice strip (unless not ``strip``): it adds the s range."""

    s_range = List(Float()).tag(sync=True)
    """The s range shown, ``[s0, s1]`` (m)."""
    strip = Bool(True).tag(sync=True)

    def __init__(
        self, machine: Machine | Lattice, branch: str | None, s: tuple[float, float] | None, **traits: Any
    ) -> None:
        root = machine.lattice if isinstance(machine, Machine) else machine
        shown = branch or (root.root_branch or root.branches[0].id)
        super().__init__(machine, shown, s_range=list(s) if s is not None else [], **traits)

    def zoom(self, s0: float, s1: float) -> None:
        """Show s from ``s0`` to ``s1`` (m) of the branch shown."""
        self.s_range = [float(s0), float(s1)]


#: What linked views share: the branch shown, the selection, the s range (where both have it).
LINKED = ("branch", "selection", "s_range")


def link(*views: LatticeWidget) -> list[Any]:
    """Tie views together as the page's are: what one shows of ``LINKED`` the others show too.

    The link is in the browser (``ipywidgets.jslink``), so it holds while the kernel is busy. The
    views start from the first's values (the first that has one). Returns the links; ``unlink()``
    on one undoes it.
    """
    from ipywidgets import jslink

    links = []
    for trait in LINKED:
        having = [view for view in views if view.has_trait(trait)]
        if len(having) < 2:
            continue
        source = next((view for view in having if getattr(view, trait)), having[0])
        for view in having:
            if view is not source:
                setattr(view, trait, getattr(source, trait))
                links.append(jslink((source, trait), (view, trait)))
    return links
