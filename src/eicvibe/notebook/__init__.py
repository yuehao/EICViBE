"""EICViBE in a notebook: the machine, its optics, virtual rings and the virtual accelerator (ADR 0016).

    import eicvibe.notebook as ev

    m = ev.machine("EIN/machine")                     # a machine folder (ADR 0005), shown as its branches
    rcs = ev.optics(m, "rcs")                         # a branch's optics, as the viewer computes them
    ev.plot_optics(rcs, m, s=(0, 200))                # beta and dispersion under the element bar
    ev.table(rcs, m)                                  # the same, an occurrence a row (pandas)
    v = ev.OpticsView(m, "rcs", result=rcs)           # the viewer's own optics view, fed by this kernel
    v.selected, v.s_range; v.zoom(0, 200)             # what was picked, where the view is; move it
    ev.FloorView(m)                                   # the floor plan, a view of its own
    ev.chart({"βx": (rcs.s, rcs.beta_x)}, x_label="s (m)")  # the viewer's chart of any data

    with ev.session(m, "rcs", engine="xsuite") as s:  # an engine's session (ADR 0006)
        orbit = s.orbit(ev.OrbitRequest())
    ev.plot_orbit(orbit, machine=m)

    run = ev.start_ring(m, "rcs", "runs/rcs-1")       # a virtual ring tracking in the background (ADR 0007)
    tbt = run.history.turn_by_turn()                  # its BPMs' latest turns, as a file reads
    ev.write_hdf5(tbt, "rcs-1.h5"); ev.read_hdf5("rcs-1.h5")  # to a file and back (ev.write_sdds: SDDS)
    ev.plot_turns(tbt), ev.plot_spectrum(tbt), ev.tune(tbt, "x")
    ev.OrbitView(m, "rcs", runs=[run])                # its BPM orbits, live, under the lattice strip
    ev.show(m, "rcs", runs=[run])                     # floor plan, optics and orbits, linked
    run.ramp(); run.stop()

    va = ev.Client()                                  # the virtual accelerator's settings (ADR 0008)
    ev.viewer(branch="rcs")                           # the running viewer, in the notebook

What these return is the library's own objects, and the library is all here (engines, sessions,
virtual runs, BPM data and its analysis, the VA client), so a notebook's code moves into a module
or a tool as it is. The helpers save the steps every notebook would repeat: reading a machine and
its operating modes, choosing the engine, starting a ring with the machine's devices and ramp.
Importing this module also has IPython show EICViBE's objects as short summaries (:mod:`.display`)
and the plots (:mod:`.plots`, matplotlib) as figures. The widgets (:mod:`.widgets`) are the
viewer's own views, interactive, each a view of its own with its data from this kernel, linked
when you want (:func:`link`).

The kernel: the ``notebook`` extra puts Jupyter in EICViBE's environment, JupyterLab and the
widgets' front end with it (``uv sync`` installs it with the dev group; else ``uv sync --extra
notebook --extra xsuite --extra trackpad``). In the checkout, ``uv run jupyter lab``; VS Code's
notebooks take the checkout's ``.venv`` as their kernel. For a Jupyter of another environment,

    uv run python -m ipykernel install --user --name eicvibe --display-name EICViBE

makes this one its EICViBE kernel; that Jupyter draws the widgets once it has anywidget too
(``pip install anywidget`` there).

The engines run in worker processes of their own (:mod:`.workers`), as the server's do, never in
the kernel: TrackPad's Julia there can leave Python's GIL held and the kernel hung. TrackPad's
worker starts Julia on its first use (half a minute), once.
"""

import os
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from eicvibe.diagnostics import (
    OrbitSeries,
    TurnByTurn,
    amplitudes,
    beta_beating,
    closed_orbit,
    harmonics,
    phase_errors,
    read_hdf5,
    read_sdds,
    read_sdds_page,
    spectrum,
    tune,
    tune_drift,
    tunes,
    write_hdf5,
    write_sdds,
    write_sdds_page,
)
from eicvibe.domain.lattice import Lattice
from eicvibe.engines import (
    Effect,
    EngineUnavailable,
    Fidelity,
    OpticsRequest,
    OpticsResult,
    Options,
    OrbitRequest,
    OrbitResult,
    Ramp,
    Session,
    TrackRequest,
    TrackResult,
    TwissInit,
    UnsupportedRequest,
    engine,
    machine_optics,
    route_optics,
    select,
)
from eicvibe.store import Machine, check_machine, load_machine, read_settings
from eicvibe.va.client import Client, Refused, VaError
from eicvibe.virtual import (
    Beam,
    BpmHistory,
    BpmModel,
    ErrorSpec,
    MachineErrors,
    RingRun,
    RouteRun,
    default_engine,
    ring_bpms,
)

from . import workers
from .display import machine_rows, register, remember
from .plots import plot_elements, plot_optics, plot_orbit, plot_ramp, plot_spectrum, plot_turns

type Values = Mapping[tuple[str, str], float]


def machine(folder: str | Path) -> Machine:
    """The machine in a folder (ADR 0005), read afresh: an edit counts as soon as it is saved."""
    found = load_machine(Path(folder).expanduser())
    remember(found.lattice)
    return found


def mode_values(machine: Machine, mode: str) -> dict[tuple[str, str], float]:
    """An operating mode's settings (settings/<mode>.yaml): (occurrence id, parameter) to value."""
    settings = read_settings(machine.folder, mode, machine.lattice)
    return {(s.occurrence, s.parameter): s.value for s in settings}


def optics(
    machine: Machine | Lattice | str | Path,
    branch: str,
    *,
    engine: str | None = None,
    mode: str | None = None,
    values: Values | None = None,
    options: Options | None = None,
    inside: bool = True,
) -> OpticsResult:
    """A branch's optics, or a route's (``route:<name>``), as the viewer computes them.

    A ring has its periodic solution; a line its own initial conditions, or else the optics where
    a fork leads into it, computed with the machine. ``mode`` takes an operating mode's settings,
    ``values`` more on top ((occurrence id, parameter) to value). ``engine`` is ``trackpad`` or
    ``xsuite``; by default TrackPad where it is installed. With ``inside``, the optics inside
    magnets too where the engine samples them (TrackPad does). The engine computes in a worker
    process of its own (:mod:`.workers`), started on first use.
    """
    lattice, found = _lattice(machine)
    name = engine or default_engine()
    routes = {route.name: route for route in (found.routes if found is not None else ())}
    computed = workers.optics(
        lattice, branch, name, options or Options(), _values(found, mode, values), routes, inside=inside
    ).result()
    key = branch if branch.startswith("route:") else lattice.layout(branch).id
    if key not in computed.results:
        what = branch if branch.startswith("route:") else lattice.branch(branch).name
        raise UnsupportedRequest(name, [f"{what}: {computed.notes.get(key, 'no optics')}"])
    return computed.results[key]


@contextmanager
def session(
    machine: Machine | Lattice | str | Path,
    branch: str,
    *,
    engine: str | None = None,
    mode: str | None = None,
    values: Values | None = None,
    options: Options | None = None,
) -> Iterator[Session]:
    """An engine's session on a branch (ADR 0006), closed at the end of the ``with`` block.

    ``update`` changes settings in place; ``optics``, ``orbit``, ``track`` and ``track_beam`` compute.
    It starts with an operating mode's settings and ``values``, as :func:`optics` does, and lives in
    the engine's worker process (:mod:`.workers`).
    """
    lattice, found = _lattice(machine)
    name = engine or default_engine()
    opened = workers.session(lattice, branch, name, options or Options(), _values(found, mode, values))
    try:
        yield opened
    finally:
        opened.close()


def start_ring(
    machine: Machine | Lattice | str | Path,
    branch: str,
    directory: str | Path,
    *,
    mode: str | None = None,
    values: Values | None = None,
    ramp: bool = True,
    **options: Any,
) -> RingRun:
    """A virtual ring tracking in the background (ADR 0007), its BPMs recorded into ``directory``.

    Its BPMs are the machine's (their EIC names), it starts at an operating mode's settings and
    ``values``, and with ``ramp`` it takes the ring's ramp profile (ramps/), which ``run.ramp()``
    runs (ADR 0013). ``options`` are :class:`RingRun`'s: ``engine``, ``beam``, ``turns``, ``errors``,
    ``host`` and the rest. ``run.stop()`` ends it; its folder stays, for :class:`BpmHistory`.
    """
    lattice, found = _lattice(machine)
    profile = None
    if ramp and found is not None:
        name = lattice.branch(branch).name
        profile = next((r for r in found.ramps.values() if r.branch == name), None)
    chosen = {
        "registry": found.registry if found is not None else None,
        "values": _values(found, mode, values),
        "ramp": profile,
    }
    return RingRun(lattice, branch, directory, **(chosen | options)).start()


def table(data: Machine | OpticsResult | OrbitResult, machine: Machine | Lattice | None = None) -> Any:
    """A pandas DataFrame: a machine's branches, or a result's values an occurrence a row.

    Given the machine, a result's rows carry the occurrences' names and kinds (and an orbit's,
    their s); the index is the occurrence ids, which settings and other results use.
    """
    import pandas as pd

    if isinstance(data, Machine):
        return pd.DataFrame(machine_rows(data)).set_index("branch")
    if isinstance(data, OpticsResult):
        columns = ["s", "beta_x", "alpha_x", "beta_y", "alpha_y", "mu_x", "mu_y"]
        columns += ["eta_x", "etap_x", "eta_y", "etap_y", "x", "px", "y", "py"]
    elif isinstance(data, OrbitResult):
        columns = ["x", "px", "y", "py", "zeta", "delta"]
    else:
        raise TypeError(f"no table of a {type(data).__name__}")
    frame = pd.DataFrame(
        {c: getattr(data, c) for c in columns}, index=pd.Index(data.occurrences, name="occurrence")
    )
    if machine is not None:
        lattice = _lattice(machine)[0]
        placed = [lattice.placement(o) for o in data.occurrences]
        if isinstance(data, OrbitResult):
            frame.insert(0, "s", [p.s_exit for p in placed])
        frame.insert(0, "kind", [p.kind.value for p in placed])
        frame.insert(0, "name", [p.name for p in placed])
    return frame


def viewer(
    url: str | None = None,
    *,
    branch: str | None = None,
    tab: str | None = None,
    mode: str | None = None,
    engine: str | None = None,
    fidelity: str | None = None,
    run: str | None = None,
    source: str | None = None,
    s: tuple[float, float] | None = None,
    height: int = 640,
) -> Any:
    """The running viewer (``eicvibe-web``, ``eicvibe-va``), in the notebook as it is in a browser.

    It shares nothing with the kernel (:func:`show` draws the kernel's own machine and runs): runs
    a notebook writes appear in it when the server reads their folder (``--runs``). ``branch``
    (or ``route:<name>``), ``tab`` (optics, orbit or check), ``mode``, ``engine``, ``fidelity``,
    ``run``, ``source`` (the lattice, among several) and ``s`` (a range, m) choose what it shows,
    as a link to it does. ``url``: by default ``EICVIBE_URL``, else http://localhost:8000.
    """
    from IPython.display import IFrame

    base = (url or os.environ.get("EICVIBE_URL", "http://localhost:8000")).rstrip("/")
    chosen = {"source": source, "branch": branch, "engine": engine, "fidelity": fidelity}
    chosen |= {"mode": mode, "tab": tab, "run": run}
    state = {k: v for k, v in chosen.items() if v}
    if s is not None:
        state["s"] = f"{s[0]:.8g},{s[1]:.8g}"
    return IFrame(f"{base}/#{urlencode(state).replace('%2C', ',')}", width="100%", height=height)


def show(
    machine: Machine | Lattice | str | Path,
    branch: str | None = None,
    *,
    runs: Iterable[Any] = (),
    floor: bool = True,
    engine: str | None = None,
    fidelity: str = "standard",
    mode: str | None = None,
    values: Values | None = None,
    result: OpticsResult | None = None,
    s: tuple[float, float] | None = None,
    height: int = 420,
) -> Any:
    """The viewer's views of a branch or route (``route:<name>``), one under the other and linked as
    on the page (:func:`link`): the floor plan (unless not ``floor``), the optics, and given ``runs``
    their BPM orbits, each view along s ``height`` pixels. Each is a view of its own
    (``.children``); make them yourself for any other choice: :class:`FloorView`,
    :class:`OpticsView`, :class:`OrbitView`.
    """
    from ipywidgets import VBox

    from .widgets import FloorView, OpticsView, OrbitView, link

    lattice, found = _lattice(machine)
    source = found if found is not None else lattice
    views = [FloorView(source, branch)] if floor else []
    views.append(
        OpticsView(
            source,
            branch,
            engine=engine,
            fidelity=fidelity,
            mode=mode,
            values=values,
            result=result,
            s=s,
            height=height,
        )
    )
    if runs:
        views.append(OrbitView(source, branch, runs=runs, s=s, height=height))
    link(*views)
    return VBox(views)


def chart(series: Any, **options: Any) -> Any:
    """The viewer's chart of ``series`` (``{label: (x, y)}``), interactive; see :func:`.widgets.chart`."""
    from .widgets import chart as make

    return make(series, **options)


def link(*views: Any) -> list[Any]:
    """Tie views together as the page's are: the branch shown, the selection, the s range; see
    :func:`.widgets.link`."""
    from .widgets import link as tie

    return tie(*views)


#: The views, from :mod:`.widgets`, imported when first asked for: they need the built module.
_VIEWS = ("FloorView", "OpticsView", "OrbitView", "Chart")


def __getattr__(name: str) -> Any:
    if name in _VIEWS:
        from . import widgets

        return getattr(widgets, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _lattice(source: Machine | Lattice | str | Path) -> tuple[Lattice, Machine | None]:
    """The lattice, and the machine if there is one, of a machine, a lattice or a machine folder."""
    if isinstance(source, Lattice):
        remember(source)
        return source, None
    found = source if isinstance(source, Machine) else machine(source)
    remember(found.lattice)
    return found.lattice, found


def _values(found: Machine | None, mode: str | None, values: Values | None) -> dict[tuple[str, str], float]:
    if mode is not None and found is None:
        raise ValueError(f"mode {mode}: a machine folder holds the modes, not a lattice")
    return (mode_values(found, mode) if mode else {}) | dict(values or {})


register()

__all__ = [
    "Beam",
    "BpmHistory",
    "BpmModel",
    "Client",
    "Effect",
    "EngineUnavailable",
    "ErrorSpec",
    "Fidelity",
    "Machine",
    "MachineErrors",
    "OpticsRequest",
    "OpticsResult",
    "Options",
    "OrbitRequest",
    "OrbitResult",
    "OrbitSeries",
    "Ramp",
    "Refused",
    "RingRun",
    "RouteRun",
    "TrackRequest",
    "TrackResult",
    "TurnByTurn",
    "TwissInit",
    "UnsupportedRequest",
    "VaError",
    "amplitudes",
    "beta_beating",
    "Chart",
    "FloorView",
    "OpticsView",
    "OrbitView",
    "chart",
    "check_machine",
    "closed_orbit",
    "engine",
    "harmonics",
    "link",
    "load_machine",
    "machine",
    "machine_optics",
    "mode_values",
    "optics",
    "phase_errors",
    "plot_elements",
    "plot_optics",
    "plot_orbit",
    "plot_ramp",
    "plot_spectrum",
    "plot_turns",
    "read_hdf5",
    "read_sdds",
    "read_sdds_page",
    "register",
    "remember",
    "ring_bpms",
    "route_optics",
    "select",
    "session",
    "show",
    "spectrum",
    "start_ring",
    "table",
    "tune",
    "tune_drift",
    "tunes",
    "viewer",
    "write_hdf5",
    "write_sdds",
    "write_sdds_page",
]
