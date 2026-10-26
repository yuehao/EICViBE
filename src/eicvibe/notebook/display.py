"""How a notebook shows EICViBE's objects: a few facts and a small table, never every number (ADR 0016).

An optics result holds a dozen columns for thousands of occurrences; shown as Python shows a
model, it would fill the notebook. :func:`register` gives IPython a summary of each kind of object
instead, in HTML for notebooks and in text for terminals, and has it show the figures of
:mod:`.plots` as images. The objects themselves are unchanged: display is IPython's business,
not theirs. Results name branches by id; a branch whose lattice went through this module
(:func:`remember`) is shown by its name.
"""

import html
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from eicvibe.diagnostics.bpm import OrbitSeries, TurnByTurn
from eicvibe.domain.lattice import ElementKind, Lattice
from eicvibe.domain.machine.ramps import RampProfile
from eicvibe.engines.contracts import OpticsResult, OrbitResult, TrackResult, UnsupportedRequest
from eicvibe.engines.convert import reference
from eicvibe.store import Machine
from eicvibe.store.machine import SETTINGS_DIR, read_index
from eicvibe.virtual import BpmHistory, RingRun, ring_bpms

_NAMES: dict[str, str] = {}
"""Branch names by id, of the lattices seen; ids are digests, so lattices never clash."""


def remember(lattice: Lattice) -> None:
    """Show this lattice's branches by name."""
    _NAMES.update({b.id: b.name for b in lattice.branches})


def branch_name(branch: str) -> str:
    """A branch's name if its lattice was seen, else its id (a route's ``route:<name>`` as it is)."""
    return _NAMES.get(branch, branch)


# -- summaries: (title, facts, table) -------------------------------------------------------------

type Facts = list[tuple[str, str]]
type Table = tuple[list[str], list[list[Any]]] | None


def machine_rows(machine: Machine) -> list[dict[str, Any]]:
    """A machine's branches, in machine.yaml's order: kind, length, elements, BPMs, energy, ramp and the
    beam delivered at its start."""
    lattice = machine.lattice
    ramps = {r.branch: name for name, r in machine.ramps.items()}
    rows = []
    for name in read_index(machine.folder)["branches"]:
        layout = lattice.layout(name)
        try:
            particle = reference("notebook", lattice, layout.id)
            energy = math.hypot(particle["pc"], particle["mass"]) / 1e9
        except UnsupportedRequest:
            energy = None
        beam = machine.beams.get(name)
        rows.append(
            {
                "branch": name,
                "kind": "ring" if layout.periodic else "line",
                "length (m)": round(layout.length, 3),
                "elements": sum(
                    p.kind not in (ElementKind.DRIFT, ElementKind.BEGINNING_ELE) for p in layout.placements
                ),
                "BPMs": len(ring_bpms(lattice, layout.id, machine.registry)),
                "energy (GeV)": energy,
                "ramp": ramps.get(name),
                "beam": f"{len(beam.particles):,} particles, {beam.charge * 1e9:.3g} nC" if beam else None,
            }
        )
    return rows


def _machine(m: Machine) -> tuple[str, Facts, Table]:
    rows = machine_rows(m)
    modes = sorted(p.stem for p in (m.folder / SETTINGS_DIR).glob("*.yaml"))
    facts = [("folder", str(m.folder))]
    if m.routes:
        facts.append(("routes", ", ".join(r.name for r in m.routes)))
    if modes:
        facts.append(("operating modes", ", ".join(modes)))
    facts.append(("devices", f"{len(m.registry.devices):,}, {len(m.registry.channels):,} channels"))
    header = list(rows[0]) if rows else []
    return f"Machine {m.lattice.name}", facts, (header, [list(r.values()) for r in rows])


def _optics(r: OpticsResult) -> tuple[str, Facts, Table]:
    facts = [("engine", f"{r.engine} {r.fidelity.value}: {r.method}")]
    if r.effects:
        facts.append(("effects", ", ".join(sorted(e.value for e in r.effects))))
    if r.tune_x is not None:
        facts.append(("tunes", f"{r.tune_x:.5f}, {r.tune_y:.5f}"))
    else:
        facts.append(("phase advance", f"{r.mu_x[-1]:.5f}, {r.mu_y[-1]:.5f} (2π)"))
    if r.chroma_x is not None:
        facts.append(("chromaticity", f"{r.chroma_x:.4g}, {r.chroma_y:.4g}"))
    if r.momentum_compaction is not None:
        facts.append(("momentum compaction", f"{r.momentum_compaction:.5g}"))
    where = f"{len(r.occurrences):,} occurrences, {r.s[-1]:.6g} m"
    if r.profile is not None:
        where += f"; {len(r.profile.s):,} points inside them"
    facts.append(("at", where))
    rows = [
        [name, _range(getattr(r, f"{name}_x")), _range(getattr(r, f"{name}_y"))]
        for name in ("beta", "alpha", "eta")
    ] + [["orbit (mm)", _range(np.asarray(r.x) * 1e3), _range(np.asarray(r.y) * 1e3)]]
    kind = "ring" if r.periodic else "line"
    return f"Optics of {branch_name(r.branch)} ({kind})", facts, (["", "x: min … max", "y"], rows)


def _orbit(r: OrbitResult) -> tuple[str, Facts, Table]:
    facts = [("engine", f"{r.engine} {r.fidelity.value}"), ("at", f"{len(r.occurrences):,} occurrences")]
    if r.effects:
        facts.append(("effects", ", ".join(sorted(e.value for e in r.effects))))
    rows = [
        [plane, f"{_rms(np.asarray(v) * 1e3):.4g}", _range(np.asarray(v) * 1e3)]
        for plane, v in (("x", r.x), ("y", r.y))
    ]
    rows.append(["delta", f"{_rms(r.delta):.4g}", _range(r.delta)])
    what = "Closed orbit" if r.closed else "Orbit"
    return f"{what} of {branch_name(r.branch)}", facts, (["", "rms (mm)", "min … max"], rows)


def _track(r: TrackResult) -> tuple[str, Facts, Table]:
    facts = [
        ("engine", f"{r.engine} {r.fidelity.value}"),
        ("particles", f"{len(r.lost):,}, {sum(r.lost):,} lost"),
        ("turns", f"{r.turns:,}"),
        ("observed at", f"{len(r.observe):,} occurrences"),
    ]
    return f"Tracking in {branch_name(r.branch)}", facts, None


def _ramp(p: RampProfile) -> tuple[str, Facts, Table]:
    e = p.energy
    source = p.source.get("path")
    linear = f"{e.rate / 1e9:.5g} GeV/s to {e.linear_end / 1e9:.5g} GeV, after {e.smooth_turns} turns"
    facts = [
        ("energy", f"{e.injection / 1e9:.5g} → {e.extraction / 1e9:.5g} GeV"),
        ("linear ramp", linear),
        ("flattop", f"{p.flattop * 1e3:.4g} ms"),
        ("RF voltage", f"{min(p.rf.voltage) / 1e6:.4g} … {max(p.rf.voltage) / 1e6:.4g} MV"),
        ("synchronous phase", f"up to {math.degrees(max(p.rf.phase)):.3g}°"),
        ("cavities", ", ".join(p.cavities) or "all of the ring's"),
    ]
    if source:
        facts.append(("source", Path(str(source)).name))
    return f"Ramp {p.name} of {p.branch}", facts, None


def _tbt(t: TurnByTurn) -> tuple[str, Facts, Table]:
    valid = np.isfinite(t.x.position).mean() if t.x.position.size else 0.0
    facts = [
        ("source", t.source or "–"),
        ("turns", f"{t.first_turn:,} to {t.first_turn + t.turns:,} ({t.turns:,})"),
        ("BPMs", f"{len(t.x.bpms)} x, {len(t.y.bpms)} y; {valid:.1%} of the x readings valid"),
    ]
    if t.time is not None:
        facts.append(("first turn at", f"{t.time:%Y-%m-%d %H:%M:%S}"))
    if events := t.metadata.get("events"):
        kinds = sorted({e["kind"] for e in events})
        facts.append(("events", f"{len(events)}: {', '.join(kinds)}"))
    return f"Turn by turn, {t.ring}" if t.ring else "Turn by turn", facts, None


def _series(o: OrbitSeries) -> tuple[str, Facts, Table]:
    facts = [("source", o.source or "–"), ("BPMs", str(len(o.bpms))), ("samples", f"{len(o.turn):,}")]
    if len(o.turn):
        facts.append(("turns", f"{int(o.turn[0]):,} to {int(o.turn[-1] + o.turns[-1]):,}"))
    return f"Slow orbit, {o.ring}", facts, None


def _history(h: BpmHistory) -> tuple[str, Facts, Table]:
    info = h.info
    first, end = h.span()
    state = h.state()
    facts = [
        ("folder", str(h.directory)),
        ("engine", f"{info.engine} {info.fidelity}" + (f" on {info.host}" if info.host else "")),
        ("state", f"{state['state']} at {info.unit} {state.get('turn', end):,}"),
        ("kept turn by turn", f"{first:,} to {end:,}"),
        ("captures", str(len(h.captures()))),
        ("BPMs", str(len(info.bpms))),
    ]
    if info.ramp:
        facts.append(("ramp", str(info.ramp.get("name"))))
    return f"Run {info.id} of {info.branch} ({info.kind})", facts, None


def _run(r: RingRun) -> tuple[str, Facts, Table]:
    state = r.history.state()["state"] if (r.directory / "run.json").exists() else "not started"
    facts = [
        ("folder", str(r.directory)),
        ("engine", r.engine + (f" on {r.host.name}" if r.host else "")),
        ("state", f"{state} at turn {r.turn:,}" + (f": {r.error}" if r.error else "")),
    ]
    if (cycle := r.cycle) is not None:
        facts.append(("ramp", f"{cycle['ramp']}: {cycle['state']}, {cycle['energy'] / 1e9:.5g} GeV"))
    if r.tunes is not None:
        facts.append(("tune monitor", f"{r.tunes['x']:.4f}, {r.tunes['y']:.4f}"))
    facts += [("warning", w) for w in r.warnings]
    return f"Ring run of {r.lattice.branch(r.branch).name}", facts, None


SUMMARIES = {
    Machine: _machine,
    OpticsResult: _optics,
    OrbitResult: _orbit,
    TrackResult: _track,
    RampProfile: _ramp,
    TurnByTurn: _tbt,
    OrbitSeries: _series,
    BpmHistory: _history,
    RingRun: _run,
}


def summary(obj: Any) -> tuple[str, Facts, Table]:
    """What a notebook shows of ``obj``: a title, facts and perhaps a table."""
    for kind, make in SUMMARIES.items():
        if isinstance(obj, kind):
            return make(obj)
    raise TypeError(f"no summary of a {type(obj).__name__}")


def to_html(obj: Any) -> str:
    title, facts, table = summary(obj)
    left = "style='text-align:left'"
    out = [f"<div><strong>{html.escape(title)}</strong>", "<table>"]
    out += [f"<tr><th {left}>{html.escape(k)}</th><td {left}>{html.escape(v)}</td></tr>" for k, v in facts]
    out.append("</table>")
    if table is not None:
        header, rows = table
        out.append("<table><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in header))
        out.append("</tr></thead><tbody>")
        for row in rows:
            out.append("<tr>" + "".join(_cell(v) for v in row) + "</tr>")
        out.append("</tbody></table>")
    out.append("</div>")
    return "\n".join(out)


def to_text(obj: Any) -> str:
    title, facts, table = summary(obj)
    width = max((len(k) for k, _ in facts), default=0)
    lines = [title] + [f"  {k.ljust(width)}  {v}" for k, v in facts]
    if table is not None:
        header, rows = table
        cells = [header] + [[_text(v) for v in row] for row in rows]
        widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
        lines += ["  " + "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in cells]
    return "\n".join(lines)


# -- IPython --------------------------------------------------------------------------------------


def register(shell: Any = None) -> bool:
    """Have IPython show EICViBE's objects as summaries and :mod:`.plots`' figures as images.

    Does nothing outside IPython (False). Figures are left to a format already set for them, such
    as ``%config InlineBackend.figure_formats = ["svg"]``.
    """
    if shell is None:
        if "IPython" not in sys.modules:  # a script: no IPython to tell, and none to import
            return False
        shell = sys.modules["IPython"].get_ipython()
    if shell is None:
        return False
    formatters = shell.display_formatter.formatters
    for kind in SUMMARIES:
        formatters["text/html"].for_type(kind, to_html)
        formatters["text/plain"].for_type(kind, lambda obj, p, cycle: p.text(to_text(obj)))
    from matplotlib.figure import Figure

    if not any(_handles(formatters[m], Figure) for m in ("image/png", "image/svg+xml", "application/pdf")):
        from IPython.core.pylabtools import retina_figure

        formatters["image/png"].for_type(Figure, retina_figure)
    return True


def _handles(formatter: Any, kind: type) -> bool:
    try:
        formatter.lookup_by_type(kind)
        return True
    except KeyError:
        return False


def _range(values: Iterable[float]) -> str:
    v = np.asarray(list(values), dtype=float)
    if not v.size or not np.isfinite(v).any():
        return "–"
    return f"{np.nanmin(v):.4g} … {np.nanmax(v):.4g}"


def _rms(values: Iterable[float]) -> float:
    v = np.asarray(list(values), dtype=float)
    return float(np.sqrt(np.nanmean(v**2))) if v.size else math.nan


def _text(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _cell(value: Any) -> str:
    align = "right" if isinstance(value, int | float) and value is not None else "left"
    return f"<td style='text-align:{align}'>{html.escape(_text(value))}</td>"
