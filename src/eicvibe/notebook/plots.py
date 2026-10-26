"""EICViBE's data in matplotlib figures, drawn as the viewer draws it (ADR 0016).

Figures are made without pyplot, so they hold no global state: a notebook shows the figure a
function returns, a script saves it (``fig.savefig``). To draw into a pyplot figure instead, pass
``figure=plt.figure()``.

- :func:`plot_elements`: a branch's or a route's elements along s, on any axes, as the viewer's
  linear view draws them: magnets as blocks, the focusing ones above the axis and the defocusing
  ones below, instruments as ticks, forks as dashed flags.
- :func:`plot_optics`: beta and dispersion under the element bar.
- :func:`plot_orbit`: orbits along s: an engine's, BPM readings averaged over turns, a slow orbit's sample.
- :func:`plot_turns`, :func:`plot_spectrum`: one BPM's readings turn by turn; the planes' spectra and tunes.
- :func:`plot_ramp`: a ring's ramp: its energy, RF voltage and synchronous phase in time.

Beta and dispersion, like x and y, have panels of their own, sharing s: two scales are never one
axis's. Within a panel the planes keep the viewer's colours, and the vertical plane's dispersion
its dashes.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from eicvibe.diagnostics import closed_orbit, spectrum, tune
from eicvibe.diagnostics.bpm import OrbitSeries, TurnByTurn
from eicvibe.domain.lattice import Lattice, PlacedElement, Route, has_fields, route_placements
from eicvibe.domain.machine.ramps import RampProfile, ramp_period
from eicvibe.engines.contracts import OpticsResult, OrbitResult
from eicvibe.engines.convert import reference
from eicvibe.store import Machine


@dataclass(frozen=True)
class _Style:
    color: str
    dark: str
    height: float
    """Block height, as a fraction of a dipole's."""
    lane: str
    """straddle (centred on the axis), polarity (above when focusing, below when defocusing), tick or none."""


#: How each element kind is drawn: the viewer's colours, heights and lanes (web/src/model/kinds.ts).
STYLES = {
    "Bend": _Style("#5B7FA6", "#86A9D1", 1.0, "straddle"),
    "Quadrupole": _Style("#4A6D8C", "#7397B8", 0.6, "polarity"),
    "Sextupole": _Style("#284153", "#5F8299", 0.5, "polarity"),
    "Octupole": _Style("#192935", "#4E6678", 0.42, "polarity"),
    "Multipole": _Style("#192935", "#4E6678", 0.42, "polarity"),
    "RFCavity": _Style("#7A6E9E", "#A397C9", 0.8, "straddle"),
    "CrabCavity": _Style("#7A6E9E", "#A397C9", 0.8, "straddle"),
    "EGun": _Style("#8A5A7E", "#B886AA", 0.9, "straddle"),
    "Kicker": _Style("#9E6060", "#C98989", 0.5, "straddle"),
    "Solenoid": _Style("#4E8A8A", "#76B3B3", 0.75, "straddle"),
    "Instrument": _Style("#6A7D6A", "#93A893", 0.35, "tick"),
    "BeamBeam": _Style("#5A6473", "#8A95A6", 0.5, "tick"),
    "Fork": _Style("#C2185B", "#F06292", 1.0, "tick"),
    "Marker": _Style("#7A7A7A", "#A3A3A3", 0.25, "tick"),
    "Patch": _Style("#A07848", "#C9A070", 0.4, "tick"),
    "ReferenceChange": _Style("#A07848", "#C9A070", 0.4, "tick"),
    "BeginningEle": _Style("#7A7A7A", "#A3A3A3", 0.25, "none"),
    "Drift": _Style("#9AA0A6", "#5B6270", 0.0, "none"),
}
_UNKNOWN = _Style("#5A6473", "#8A95A6", 0.5, "straddle")

#: The strengths whose sign puts a magnet above or below the axis, the first that is not zero
#: (web/src/model/branch.ts). A magnet given by its fields (an injector's) is centred, as there.
_MAIN_ORDERS = {"Quadrupole": (1,), "Sextupole": (2,), "Octupole": (3,), "Multipole": (1, 2, 3)}

#: The viewer's colours for the optics' columns (web/src/model/optics.ts), light and dark, and dashes.
SERIES = {
    "beta_x": ("βx", "#1f5fbf", "#6ea8ff", "-"),
    "beta_y": ("βy", "#c7362f", "#ff7b72", "-"),
    "eta_x": ("ηx", "#2b8a3e", "#69db7c", "-"),
    "eta_y": ("ηy", "#8a5bd6", "#c4a7ff", "--"),
}
_PLANES = {"x": ("#1f5fbf", "#6ea8ff"), "y": ("#c7362f", "#ff7b72")}
_GREY = ("#6b7280", "#9ca3af")
_OTHERS = ("#1f5fbf", "#c7362f", "#2b8a3e", "#8a5bd6")
"""Several orbits on one panel: these in turn."""

_COLUMNS = {
    "alpha_x": "αx",
    "alpha_y": "αy",
    "mu_x": "μx (2π)",
    "mu_y": "μy (2π)",
    "etap_x": "ηx'",
    "etap_y": "ηy'",
    "x": "x (m)",
    "y": "y (m)",
}

#: Events shown on turn-by-turn plots: those that change the beam.
_EVENTS = ("kick", "inject", "settings", "ramp", "flattop", "extract", "loss")


def plot_elements(
    ax: Axes,
    machine: Machine | Lattice,
    branch: str,
    *,
    s: tuple[float, float] | None = None,
    labels: bool = True,
) -> None:
    """Draw a branch's elements along s on ``ax`` (a route's: ``route:<name>``, with a machine).

    ``s`` limits them to a range (m), which becomes the axes' limits; labels name the magnets wide
    enough to carry one.
    """
    lattice, routes = _source(machine)
    placed, length, flags = _placements(lattice, routes, branch)
    lo, hi = s if s is not None else (0.0, length)
    dark = _dark(ax)
    shown = [(p, s0) for p, s0 in placed if s0 + p.length >= lo and s0 <= hi]
    crowded = len(shown) > 400  # markers would hide the magnets, as the viewer leaves them out
    blocks: dict[str, list] = {}
    ticks: dict[str, list] = {}
    names: list[tuple[float, float, str, int]] = []
    for p, s0 in shown:
        style = STYLES.get(p.kind.value, _UNKNOWN)
        if style.lane == "none" or (crowded and p.kind.value == "Marker"):
            continue
        color = style.dark if dark else style.color
        h = 0.9 * style.height
        if style.lane == "tick" or p.length == 0.0:
            ticks.setdefault(color, []).append([(s0, -h / 2), (s0, h / 2)])
            continue
        polarity = _polarity(p) if style.lane == "polarity" else 0
        y0, y1 = (0.0, h) if polarity > 0 else (-h, 0.0) if polarity < 0 else (-h / 2, h / 2)
        s1 = s0 + p.length
        blocks.setdefault(color, []).append([(s0, y0), (s1, y0), (s1, y1), (s0, y1)])
        if labels and p.length >= 0.03 * (hi - lo):
            names.append((s0 + p.length / 2, y1 + 0.05 if polarity >= 0 else y0 - 0.05, p.name, polarity))
    axis = _GREY[dark]
    ax.plot([max(lo, 0.0), min(hi, length)], [0.0, 0.0], color=axis, lw=0.8, zorder=1)
    for color, rectangles in blocks.items():
        ax.add_collection(PolyCollection(rectangles, facecolors=color, edgecolors="none", alpha=0.9))
    for color, lines in ticks.items():
        ax.add_collection(LineCollection(lines, colors=color, linewidths=1.0, zorder=2))
    right = -math.inf
    for x, y, name, polarity in sorted(names):
        half = 0.0045 * (hi - lo) * len(name)  # about a character's width at 7 points
        if x - half < right + 0.01 * (hi - lo):
            continue
        ax.text(x, y, name, fontsize=7, ha="center", va="top" if polarity < 0 else "bottom", clip_on=True)
        right = x + half
    fork = STYLES["Fork"].dark if dark else STYLES["Fork"].color
    for at, label in flags:
        if lo <= at <= hi:
            ax.axvline(at, color=fork, lw=0.9, ls=(0, (4, 3)), zorder=3)
            ax.text(at, 1.0, f" {label}", color=fork, fontsize=7, va="top", clip_on=True)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)


def plot_optics(
    result: OpticsResult,
    machine: Machine | Lattice | None = None,
    *,
    s: tuple[float, float] | None = None,
    beta: Sequence[str] = ("beta_x", "beta_y"),
    dispersion: Sequence[str] = ("eta_x", "eta_y"),
    figure: Figure | None = None,
) -> Figure:
    """A branch's (or route's) optics: beta, and dispersion below, under its elements (given the machine).

    Inside magnets too where the engine sampled the optics there (TrackPad, with ``inside``).
    ``s`` limits the plot to a range (m); ``beta`` and ``dispersion`` choose the columns of each
    panel (any of the result's, such as ``alpha_x`` or ``mu_y``; an empty one leaves its panel out).
    """
    lattice = _source(machine)[0] if machine is not None else None
    s_all, column = _optics_columns(result, lattice)
    lo, hi = s if s is not None else (float(s_all[0]), float(s_all[-1]))
    panels = [list(keys) for keys in (beta, dispersion) if keys]
    fig, axes = _figure(figure, [2.6, 1.8][: len(panels)], bar=lattice is not None)
    if lattice is not None:
        plot_elements(axes[0], machine, result.branch, s=(lo, hi))
    near = _within(s_all, lo, hi)
    for ax, keys in zip(axes[lattice is not None :], panels, strict=True):
        dark = _dark(ax)
        for key in keys:
            label, light, darker, dash = SERIES.get(key, (_COLUMNS.get(key, key), None, None, "-"))
            color = (darker if dark else light) if light else None
            ax.plot(s_all[near], column(key)[near], color=color, ls=dash, lw=1.6, label=label)
        ax.set_ylabel(_axis_label(keys))
        _style(ax, legend=len(keys) > 1)
    axes[-1].set_xlabel("s (m)")
    axes[-1].set_xlim(lo, hi)
    axes[0].set_title(_optics_title(result), loc="left", fontsize=10)
    return fig


def plot_orbit(
    *data: OrbitResult | OpticsResult | TurnByTurn | OrbitSeries,
    machine: Machine | Lattice | None = None,
    s: tuple[float, float] | None = None,
    labels: Sequence[str] | None = None,
    sample: int = -1,
    figure: Figure | None = None,
) -> Figure:
    """Orbits along s, in mm: x above y, under the elements (given the machine).

    Each of ``data`` is an engine's orbit (a closed one, or the optics' orbit) drawn as a line, or
    BPM readings drawn as points: a turn-by-turn acquisition's mean over its turns, or a slow
    orbit's ``sample`` (the latest by default). Several share the panels, model and readings
    together, named by ``labels`` or by what they are. An engine's orbit needs the machine for
    its s unless it comes with the optics.
    """
    if not data:
        raise ValueError("plot_orbit wants an orbit, an acquisition or a slow orbit")
    lattice = _source(machine)[0] if machine is not None else None
    series = [_orbit_series(d, lattice, sample) for d in data]
    if labels is not None:
        series = [(label, *rest[1:]) for label, rest in zip(labels, series, strict=True)]
    branch = _resolve(machine, _branch_of(data[0])) if machine is not None else None
    length = max((float(np.nanmax(row[1])) for row in series if len(row[1])), default=1.0)
    if branch is not None:
        length = _placements(*_source(machine), branch)[1]
    lo, hi = s if s is not None else (0.0, length)
    fig, axes = _figure(figure, [2.0, 2.0], bar=branch is not None)
    if branch is not None:
        plot_elements(axes[0], machine, branch, s=(lo, hi))
    for ax, plane in zip(axes[branch is not None :], "xy", strict=True):
        dark = _dark(ax)
        for k, (label, s_values, x, y, points) in enumerate(series):
            values = x if plane == "x" else y
            near = _within(s_values, lo, hi)
            # one orbit takes its plane's colour; several keep one colour each, the same in both planes
            color = _PLANES[plane][dark] if len(series) == 1 else _OTHERS[k % len(_OTHERS)]
            if points:
                ax.plot(s_values[near], values[near], "o", ms=4, color=color, label=label)
            else:
                ax.plot(s_values[near], values[near], color=color, lw=1.6, label=label)
        ax.axhline(0.0, color=_GREY[dark], lw=0.6, zorder=0)
        ax.set_ylabel(f"{plane} (mm)")
        _style(ax, legend=len(series) > 1)
    axes[-1].set_xlabel("s (m)")
    axes[-1].set_xlim(lo, hi)
    axes[0].set_title(series[0][0] if len(series) == 1 else "Orbits", loc="left", fontsize=10)
    return fig


def plot_turns(tbt: TurnByTurn, bpm: str | int = 0, *, figure: Figure | None = None) -> Figure:
    """One BPM's readings turn by turn, x above y (mm), with the events of a virtual run in them.

    ``bpm`` is its name or its position in the x plane's list.
    """
    name = tbt.x.bpms[bpm] if isinstance(bpm, int) else bpm
    fig, axes = _figure(figure, [2.0, 2.0])
    turns = tbt.first_turn + np.arange(tbt.turns)
    for ax, plane in zip(axes, "xy", strict=True):
        dark = _dark(ax)
        data = tbt.plane(plane)
        if name in data.bpms:
            readings = data.position[data.bpms.index(name)]
            marker = "o" if len(readings) <= 200 else None
            ax.plot(turns[: len(readings)], readings, color=_PLANES[plane][dark], lw=1.0, marker=marker, ms=3)
        else:
            ax.text(0.5, 0.5, f"{name} reads no {plane}", transform=ax.transAxes, ha="center", va="center")
        for event in tbt.metadata.get("events", ()):
            if event["kind"] in _EVENTS:
                ax.axvline(event["turn"], color=_GREY[dark], lw=0.8, ls=(0, (4, 3)))
                if plane == "x":  # along its line, so that events close together stay legible
                    ax.text(event["turn"], 0.98, f"{event['kind']} ", transform=ax.get_xaxis_transform(),
                            fontsize=7, rotation=90, ha="right", va="top")  # fmt: skip
        ax.set_ylabel(f"{plane} (mm)")
        _style(ax, legend=False)
    axes[-1].set_xlabel("turn")
    axes[0].set_title(f"{name} · {tbt.source or tbt.ring}", loc="left", fontsize=10)
    return fig


def plot_spectrum(tbt: TurnByTurn, *, figure: Figure | None = None) -> Figure:
    """The planes' spectra, the mean of their BPMs' (mm, logarithmic), and their tunes."""
    fig, axes = _figure(figure, [3.2])
    ax = axes[0]
    dark = _dark(ax)
    for plane in "xy":
        frequency, amplitude = spectrum(tbt.plane(plane))
        q = tune(tbt, plane)
        ax.semilogy(frequency, amplitude, color=_PLANES[plane][dark], lw=1.4, label=f"{plane}   q = {q:.4f}")
        ax.axvline(q, color=_PLANES[plane][dark], lw=0.8, ls=":")
    ax.set_xlim(0.0, 0.5)
    ax.set_xlabel("frequency (of the revolution's)")
    ax.set_ylabel("amplitude (mm)")
    _style(ax, legend=True)
    ax.set_title(f"Spectra of {tbt.turns} turns · {tbt.source or tbt.ring}", loc="left", fontsize=10)
    return fig


def plot_ramp(
    profile: RampProfile, machine: Machine | Lattice, *, samples: int = 241, figure: Figure | None = None
) -> Figure:
    """A ring's ramp in time, as the admin page's Ramp tab charts it: energy, the RF voltage of all its
    cavities, and the synchronous phase from the zero crossing, from the ramp's start to extraction."""
    lattice = _source(machine)[0]
    period = ramp_period(profile, lattice)
    mass = reference("notebook", lattice, lattice.layout(profile.branch).id)["mass"]
    extraction = profile.extraction_turn(period)
    turns = np.linspace(0.0, extraction, samples)
    rows = [profile.at(t, period, mass) for t in turns]
    ms = turns * period * 1e3
    fig, axes = _figure(figure, [1.8, 1.8, 1.8])
    flattop = profile.ramp_turns(period) * period * 1e3
    panels = (
        ("energy (GeV)", [r["energy"] / 1e9 for r in rows]),
        ("RF voltage (MV)", [r["voltage"] / 1e6 for r in rows]),
        ("phase (deg)", [math.degrees(r["phase"]) for r in rows]),
    )
    for k, (ax, (label, values)) in enumerate(zip(axes, panels, strict=True)):
        dark = _dark(ax)
        ax.plot(ms, values, color=_PLANES["x"][dark], lw=1.6)
        ax.axvline(flattop, color=_GREY[dark], lw=0.9, ls=(0, (4, 3)))
        if k == 0:
            ax.text(flattop, 0.02, " flattop", transform=ax.get_xaxis_transform(), fontsize=8, va="bottom")
        ax.set_ylabel(label)
        _style(ax, legend=False)
    axes[-1].set_xlabel("time from the ramp's start (ms)")
    axes[-1].set_xlim(0.0, float(ms[-1]))
    e = profile.energy
    axes[0].set_title(
        f"Ramp {profile.name} of {profile.branch}: {e.injection / 1e9:.4g} → {e.extraction / 1e9:.4g} GeV, "
        f"extraction at {ms[-1]:.4g} ms ({extraction:,} turns)",
        loc="left",
        fontsize=10,
    )
    return fig


# -- helpers --------------------------------------------------------------------------------------


def _source(machine: Machine | Lattice) -> tuple[Lattice, Sequence[Route]]:
    if isinstance(machine, Machine):
        return machine.lattice, machine.routes
    if isinstance(machine, Lattice):
        return machine, ()
    raise TypeError(f"a machine or a lattice, not {type(machine).__name__}")


def _placements(
    lattice: Lattice, routes: Sequence[Route], branch: str
) -> tuple[list[tuple[PlacedElement, float]], float, list[tuple[float, str]]]:
    """A branch's or a route's occurrences with the s of their entrances, its length, and its fork flags."""
    if branch.startswith("route:"):
        name = branch.removeprefix("route:")
        route = next((r for r in routes if r.name == name), None)
        if route is None:
            raise KeyError(f"no route {name}; give the machine, whose routes.yaml has them")
        placed = [(p, s0) for _, p, s0 in route_placements(lattice, route)]
        return placed, placed[-1][1] + placed[-1][0].length if placed else 0.0, []
    layout = lattice.layout(branch)
    flags = [(e.s_from, f"→ {lattice.branch(e.to_branch).name}") for e in lattice.forks_from(layout.id)]
    flags += [(e.s_to, f"← {lattice.branch(e.from_branch).name}") for e in lattice.forks_into(layout.id)]
    return [(p, p.s_enter) for p in layout.placements], layout.length, flags


def _polarity(placed: PlacedElement) -> int:
    multipoles = placed.element.MagneticMultipoleP
    if multipoles is None or has_fields(multipoles):
        return 0
    for order in _MAIN_ORDERS.get(placed.kind.value, ()):
        total = getattr(multipoles, f"Kn{order}") * placed.length + getattr(multipoles, f"Kn{order}L")
        if total:
            return 1 if total > 0 else -1
    return 0


def _dark(ax: Axes) -> bool:
    """Whether the axes are dark (a dark style), so that their lines take the viewer's dark colours."""
    r, g, b = to_rgb(ax.get_facecolor())
    return 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5


def _figure(figure: Figure | None, heights: list[float], *, bar: bool = False) -> tuple[Figure, list[Axes]]:
    """A figure of panels sharing x, ``heights`` inches tall, under a slim element bar if ``bar``."""
    ratios = ([0.9] if bar else []) + heights
    figure = figure or Figure(figsize=(10.0, 0.8 + sum(ratios)), layout="constrained")
    axes = figure.subplots(len(ratios), 1, sharex=True, squeeze=False, gridspec_kw={"height_ratios": ratios})
    return figure, list(axes[:, 0])


def _style(ax: Axes, *, legend: bool) -> None:
    ax.grid(True, lw=0.6, alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if legend:  # above the panel's right end, clear of the curves
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            borderaxespad=0.1,
            fontsize=8,
            frameon=False,
            ncols=4,
        )


def _within(s: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """The points in [lo, hi] and one either side, so that lines run to the edges."""
    inside = (s >= lo) & (s <= hi)
    near = inside.copy()
    near[1:] |= inside[:-1]
    near[:-1] |= inside[1:]
    return near


def _optics_columns(result: OpticsResult, lattice: Lattice | None) -> tuple[np.ndarray, Any]:
    """s, and a column's values, of the optics inside elements where sampled, else at their exits."""
    if result.profile is not None:
        profile = result.profile

        def inside(key: str) -> np.ndarray:
            if key in type(profile).model_fields:
                return np.asarray(getattr(profile, key), dtype=float)
            # a column only the exits have (the orbit): each piece's end takes its occurrence's exit
            exits = np.asarray(getattr(result, key), dtype=float)
            return np.concatenate([[getattr(result.start, key)], exits[list(profile.index)]])

        return np.asarray(profile.s, dtype=float), inside
    first = 0.0
    if lattice is not None and not result.branch.startswith("route:"):
        first = lattice.placement(result.occurrences[0]).s_enter
    s = np.concatenate([[first], np.asarray(result.s, dtype=float)])

    def exits(key: str) -> np.ndarray:
        return np.concatenate([[getattr(result.start, key)], np.asarray(getattr(result, key), dtype=float)])

    return s, exits


def _axis_label(keys: Sequence[str]) -> str:
    if all(k.startswith("beta") for k in keys):
        return "β (m)"
    if all(k.startswith("eta_") for k in keys):
        return "η (m)"
    return ", ".join(SERIES[k][0] if k in SERIES else _COLUMNS.get(k, k) for k in keys)


def _optics_title(result: OpticsResult) -> str:
    from .display import branch_name

    parts = [branch_name(result.branch), f"{result.engine} {result.fidelity.value}"]
    if result.tune_x is not None and result.tune_y is not None:
        parts.append(f"Q {result.tune_x:.4f}, {result.tune_y:.4f}")
        if result.chroma_x is not None and result.chroma_y is not None:
            parts.append(f"ξ {result.chroma_x:.2f}, {result.chroma_y:.2f}")
    else:
        parts.append(f"μ {result.mu_x[-1]:.4f}, {result.mu_y[-1]:.4f} (2π)")
    return " · ".join(parts)


def _branch_of(data: Any) -> str:
    if isinstance(data, OrbitResult | OpticsResult):
        return data.branch
    return data.ring  # BPM data names its branch, or its route


def _resolve(machine: Machine | Lattice, branch: str) -> str | None:
    """The branch or route (``route:<name>``) of data that names one; None if the machine has neither."""
    lattice, routes = _source(machine)
    if branch.startswith("route:"):
        return branch if any(r.name == branch.removeprefix("route:") for r in routes) else None
    try:
        return lattice.layout(branch).id
    except KeyError:
        return f"route:{branch}" if any(r.name == branch for r in routes) else None


def _orbit_series(
    data: Any, lattice: Lattice | None, sample: int
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, bool]:
    """(label, s, x, y, whether points) of one orbit, in m and mm."""
    if isinstance(data, OpticsResult):
        s = np.asarray(data.s, dtype=float)
        x, y = np.asarray(data.x) * 1e3, np.asarray(data.y) * 1e3
        return f"orbit ({data.engine})", s, x, y, False
    if isinstance(data, OrbitResult):
        if lattice is None:
            raise ValueError("an engine's orbit has no s: give the machine (machine=)")
        s = np.array([lattice.placement(o).s_exit for o in data.occurrences])
        what = "closed orbit" if data.closed else "orbit"
        return f"{what} ({data.engine})", s, np.asarray(data.x) * 1e3, np.asarray(data.y) * 1e3, False
    if isinstance(data, TurnByTurn):
        x, y = closed_orbit(data)
        if data.x.bpms != data.y.bpms:  # single-plane BPMs: each plane at its own BPMs
            both = sorted(set(data.x.bpms) | set(data.y.bpms), key=lambda b: _s_of(data, b))
            s = np.array([_s_of(data, b) for b in both])
            x = np.array([x[data.x.bpms.index(b)] if b in data.x.bpms else np.nan for b in both])
            y = np.array([y[data.y.bpms.index(b)] if b in data.y.bpms else np.nan for b in both])
        else:
            s = np.asarray(data.x.s, dtype=float)
        last = data.first_turn + data.turns
        return f"BPMs, turns {data.first_turn:,} to {last:,}", s, x, y, True
    if isinstance(data, OrbitSeries):
        if not len(data.turn):
            raise ValueError("the slow orbit has no samples yet")
        first, count = int(data.turn[sample]), int(data.turns[sample])
        label = f"slow orbit, turns {first:,} to {first + count:,}"
        return label, np.asarray(data.s), data.x[sample], data.y[sample], True
    raise TypeError(f"no orbit in a {type(data).__name__}")


def _s_of(tbt: TurnByTurn, bpm: str) -> float:
    plane = tbt.x if bpm in tbt.x.bpms else tbt.y
    return float(plane.s[plane.bpms.index(bpm)])
