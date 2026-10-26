"""The notebook's views (ADR 0016): each a widget of its own, answered by the kernel as the server would
answer the page, and linked when asked to."""

import json
import math
import time
from pathlib import Path

import pytest

pytest.importorskip("xtrack")
pytest.importorskip("anywidget")
if not (Path(__file__).parents[2] / "src/eicvibe/notebook/static/widget.js").is_file():
    pytest.skip("the widgets are not built: npm run build in web/", allow_module_level=True)

import eicvibe.notebook as ev  # noqa: E402
from eicvibe.api import Catalog, schemas  # noqa: E402
from eicvibe.notebook import workers  # noqa: E402
from eicvibe.notebook.widgets import PALETTE, LatticeWidget  # noqa: E402
from tests.notebook.test_notebook import folder  # noqa: E402, F401 - the demo machine


def asking(widget: LatticeWidget):
    """Ask a view's kernel side as its page components do; the JSON answer (waited for: optics come
    from the engine's worker), or the error's text."""
    sent = []
    widget.send = lambda content, buffers=None: sent.append(content)

    def ask(method: str, **args):
        sent.clear()
        widget._answer(widget, {"request": 7, "method": method, "args": args}, [])
        for _ in range(6000):
            if sent:
                break
            time.sleep(0.05)
        [reply] = sent
        assert reply["response"] == 7
        return json.loads(reply["json"]) if "json" in reply else reply["error"]

    return ask


def test_the_floor_plan_is_a_view_of_its_own_with_the_servers_branches(folder):  # noqa: F811
    m = ev.machine(folder)
    floor = ev.FloorView(m)
    assert (floor.view, floor.branch, floor.has_trait("s_range")) == ("floor", "", False)
    [revision] = Catalog([folder]).current()
    assert floor.summary == json.loads(revision.views.summary().model_dump_json())
    ask = asking(floor)
    for branch in m.lattice.branches:
        assert ask("branch", branch=branch.id) == json.loads(
            revision.views.branch(branch.id).model_dump_json()
        )
    assert ask("optics", branch=branch.id, engine="xsuite", fidelity="standard") == "no request 'optics'"
    floor.branch = "extraction"  # drawn over the others
    assert floor.branch == m.lattice.layout("extraction").id


def test_the_optics_view_computes_in_the_engines_worker(folder):  # noqa: F811
    m = ev.machine(folder)
    optics = ev.OpticsView(m, "ring", engine="xsuite")
    assert (optics.view, optics.strip) == ("optics", True) and optics.branch == m.lattice.layout("ring").id
    ask = asking(optics)
    assert {e["name"] for e in ask("engines")} == {"xsuite", "trackpad"}
    ring = ask("optics", branch=optics.branch, engine="xsuite", fidelity="standard")
    ring = schemas.BranchOptics.model_validate(ring)
    assert ring.available and ring.tune_x == pytest.approx(ev.optics(m, "ring", engine="xsuite").tune_x)
    assert ring.origin == "periodic solution"
    line = ask(
        "optics", branch=m.lattice.layout("extraction").id, engine="xsuite", fidelity="standard", mode=""
    )
    assert line["available"] and line["origin"].startswith("fork")  # from the ring's optics
    route = ask("optics", branch="route:transfer", engine="xsuite", fidelity="standard", mode="")
    assert route["branch"] == "route:transfer" and route["available"]
    stronger = ask("optics", branch=optics.branch, engine="xsuite", fidelity="standard", mode="stronger")
    assert stronger["mode"] == "stronger" and stronger["tune_x"] > ring.tune_x
    nowhere = ask("optics", branch=optics.branch, engine="nowhere", fidelity="standard")
    assert nowhere.startswith("no engine 'nowhere'")
    none = ask("optics", branch=optics.branch, engine="xsuite", fidelity="standard", mode="none")
    assert none == "no operating mode none"
    assert ask("bogus") == "no request 'bogus'"


def test_optics_computed_in_the_notebook_are_shown_first_and_values_count(folder, monkeypatch):  # noqa: F811
    m = ev.machine(folder)
    qf = m.lattice.placement("QF_1", branch="ring").id
    values = {(qf, "MagneticMultipoleP.Kn1"): 0.42}
    mine = ev.optics(m, "ring", engine="xsuite", values=values)
    optics = ev.OpticsView(m, result=mine, values=values, engine="trackpad")
    assert (optics.branch, optics.engine, optics.fidelity) == (
        mine.branch,
        "xsuite",
        "standard",
    )  # the result's
    monkeypatch.setattr(workers, "optics", lambda *args, **kw: pytest.fail("computed again"))
    shown = asking(optics)("optics", branch=optics.branch, engine="xsuite", fidelity="standard", mode="")
    assert shown["tune_x"] == mine.tune_x and shown["origin"] == "computed in this notebook"
    monkeypatch.undo()
    fast = asking(optics)("optics", branch=optics.branch, engine="xsuite", fidelity="fast", mode="")
    assert fast["tune_x"] == pytest.approx(mine.tune_x, abs=1e-3)  # with the view's values, as before


def test_python_reads_and_moves_what_a_view_shows(folder):  # noqa: F811
    m = ev.machine(folder)
    optics = ev.OpticsView(m, s=(2.0, 30.0), strip=False)
    assert optics.branch == m.lattice.layout("ring").id and optics.s_range == [2.0, 30.0]  # the root branch
    optics.branch = "extraction"  # a name: kept as the id the view uses
    assert optics.branch == m.lattice.layout("extraction").id
    optics.selection = "Q1_1"  # a name in the branch shown
    assert optics.selection == m.lattice.placement("Q1_1", branch="extraction").id
    assert optics.selected.name == "Q1_1"
    optics.selection = None
    assert optics.selected is None
    optics.zoom(1, 4.5)
    assert optics.s_range == [1.0, 4.5]
    optics.branch = "route:fill"
    with pytest.raises(ValueError, match="no route nowhere"):
        optics.branch = "route:nowhere"
    with pytest.raises(ValueError, match="by its id here"):
        ev.FloorView(m).selection = "Q1_1"  # no branch shown: a name says nothing


def test_the_orbit_view_reads_its_runs_folders(folder, tmp_path):  # noqa: F811
    m = ev.machine(folder)
    run = ev.start_ring(m, "ring", tmp_path / "run", ramp=False, engine="xsuite", fidelity=ev.Fidelity.FAST,
                        beam=ev.Beam(particles=4), chunk=128, turns=256)  # fmt: skip
    try:
        run.wait(256, timeout=300)
    finally:
        run.stop()
    orbit = ev.OrbitView(m, "ring", runs=[run])
    assert orbit.view == "orbit" and not orbit.has_trait("engine")
    [entry] = orbit.runs
    assert entry["branch_id"] == orbit.branch and entry["lattice_digest"] == orbit.summary["digest"]
    ask = asking(orbit)
    frames = schemas.OrbitFrames.model_validate(ask("orbits", run=entry["id"], every=5, count=3, end=None))
    assert frames.turns == [245, 250, 255] and frames.state == "done"
    assert [c["kind"] for c in ask("captures", run=entry["id"])] == ["inject"]
    assert ask("orbits", run="nowhere", every=1, count=1, end=None) == "no run nowhere in this view"
    assert ask("branch", branch=orbit.branch)["table"]["id"][0] == m.lattice.layout("ring").placements[0].id


def test_views_link_as_the_pages_do(folder):  # noqa: F811
    m = ev.machine(folder)
    floor, optics, orbit = ev.FloorView(m), ev.OpticsView(m, "ring", s=(2.0, 30.0)), ev.OrbitView(m, "ring")
    links = ev.link(floor, optics, orbit)
    tied = sorted((link.source[1], link.source[0].view, link.target[0].view) for link in links)
    assert tied == [
        ("branch", "optics", "floor"),  # the first branch shown: the floor plan shows none of its own
        ("branch", "optics", "orbit"),
        ("s_range", "optics", "orbit"),
        ("selection", "floor", "optics"),
        ("selection", "floor", "orbit"),
    ]
    assert floor.branch == optics.branch == orbit.branch and orbit.s_range == [2.0, 30.0]  # the first's
    shown = ev.show(m, "ring", floor=False)
    assert [view.view for view in shown.children] == ["optics"]


def test_a_chart_takes_the_viewers_colours_in_order():
    rows = {f"s{k}": ([0, 1], [k, math.nan]) for k in range(6)}
    c = ev.chart(rows, x_label="turn", markers=[{"x": 0.5, "label": "kick"}], log=(-3, 0))
    assert [s["color"] for s in c.series] == [p[0] for p in PALETTE] + [PALETTE[0][0], PALETTE[1][0]]
    assert [s.get("dash") for s in c.series][3:] == [None, [6, 3], [6, 3]]  # again, dashed
    assert c.series[2]["y"] == [2.0, None] and c.log == [-3.0, 0.0]
    assert c.markers == [{"x": 0.5, "label": "kick", "color": "#6b7280", "darkColor": "#9ca3af"}]
