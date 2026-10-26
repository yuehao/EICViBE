"""EICViBE in a notebook (ADR 0016): the helpers, the summaries shown in place of every number, the plots."""

import numpy as np
import pytest

pytest.importorskip("xtrack")
pytest.importorskip("matplotlib")

from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import eicvibe.notebook as ev  # noqa: E402
from eicvibe.api.demo import ROUTES, demo_lattice  # noqa: E402
from eicvibe.diagnostics import PlaneData, TurnByTurn, closed_orbit  # noqa: E402
from eicvibe.notebook.display import register, to_html, to_text  # noqa: E402
from eicvibe.store import Setting, write_machine, write_ramp, write_settings  # noqa: E402
from eicvibe.store.machine import write_routes  # noqa: E402
from tests.virtual.test_ring import demo_ramp  # noqa: E402


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    """The demo machine as a folder: its routes, a ramp of its ring and an operating mode."""
    folder = tmp_path_factory.mktemp("machine")
    lattice = demo_lattice()
    write_machine(folder, lattice)
    write_routes(folder, ROUTES)
    write_ramp(folder, demo_ramp(lattice))
    qf = lattice.placement("QF_1", branch="ring").id
    write_settings(folder, "stronger", [Setting(qf, "MagneticMultipoleP.Kn1", 0.42)], lattice, "QF_1 up")
    return folder


def acquisition(machine) -> TurnByTurn:
    """256 turns of the ring's BPMs: betatron lines at 0.21 and 0.31, and an orbit across them."""
    bpms = ev.ring_bpms(machine.lattice, "ring", machine.registry)
    names, s = tuple(b[1] for b in bpms), np.array([b[2] for b in bpms])
    turns = np.arange(256)
    x = 0.5 * np.cos(2 * np.pi * 0.21 * turns)[None, :] + np.linspace(-0.1, 0.1, len(names))[:, None]
    y = 0.3 * np.cos(2 * np.pi * 0.31 * turns)[None, :] + np.zeros((len(names), 1))
    events = [{"turn": 100, "kind": "kick"}, {"turn": 120, "kind": "capture"}]
    return TurnByTurn(
        "ring", PlaneData(names, s, x), PlaneData(names, s, y), 0, None, metadata={"events": events}
    )


def test_optics_of_a_ring_a_line_from_its_fork_and_a_route_in_an_operating_mode(folder):
    m = ev.machine(folder)
    ring = ev.optics(m, "ring", engine="xsuite")
    assert ring.periodic and ring.tune_x == pytest.approx(2.21, abs=0.01)
    # The extraction line has no initial conditions: its optics start where the ring's fork leads.
    whole = ev.machine_optics(ev.engine("xsuite"), m.lattice)
    extraction = ev.optics(m, "extraction", engine="xsuite")
    assert extraction.beta_x == pytest.approx(whole[m.lattice.layout("extraction").id].beta_x)
    assert ev.optics(m, "route:transfer", engine="xsuite").branch == "route:transfer"
    stronger = ev.optics(m, "ring", engine="xsuite", mode="stronger")
    assert stronger.tune_x > ring.tune_x and ev.mode_values(m, "stronger")
    with pytest.raises(ValueError, match="a machine folder holds the modes"):
        ev.optics(m.lattice, "ring", engine="xsuite", mode="stronger")
    with pytest.raises(KeyError, match="no route nowhere"):
        ev.optics(m, "route:nowhere", engine="xsuite")


def test_tables_name_the_occurrences_and_a_machine_lists_its_branches(folder):
    m = ev.machine(folder)
    with ev.session(m, "ring", engine="xsuite") as session:
        optics = session.optics(ev.OpticsRequest())
        orbit = session.orbit(ev.OrbitRequest())
    frame = ev.table(optics, m)
    assert list(frame.columns[:4]) == ["name", "kind", "s", "beta_x"]
    assert frame.loc[optics.occurrences[3], ["name", "kind"]].tolist() == ["QF_1", "Quadrupole"]
    assert frame["beta_x"].tolist() == list(optics.beta_x)
    assert ev.table(orbit, m)["s"].iloc[-1] == pytest.approx(m.lattice.layout("ring").length)
    branches = ev.table(m)
    assert branches.loc["ring", ["kind", "ramp"]].tolist() == ["ring", "demo"]
    assert list(branches.index) == ["ring", "extraction", "injection"]  # machine.yaml's order


def test_a_notebook_shows_summaries_not_every_number(folder):
    m = ev.machine(folder)
    optics = ev.optics(m, "ring", engine="xsuite")
    text = to_text(optics)
    assert text.startswith("Optics of ring (ring)") and f"{optics.tune_x:.5f}" in text and len(text) < 1200
    assert "<strong>Optics of ring (ring)</strong>" in to_html(optics)
    assert to_text(m).startswith("Machine demo") and "routes" in to_text(m)
    assert "energy             2 → 2.01 GeV" in to_text(m.ramps["demo"])
    assert "0 to 256 (256)" in to_text(acquisition(m)) and "events  2: capture, kick" in to_text(
        acquisition(m)
    )


def test_ipython_shows_them_as_html_and_the_figures_as_images(folder):
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    try:
        assert register(shell)
        m = ev.machine(folder)
        data, _ = shell.display_formatter.format(m)
        assert data["text/plain"].startswith("Machine demo") and "<table>" in data["text/html"]
        data, _ = shell.display_formatter.format(ev.plot_ramp(m.ramps["demo"], m))
        assert data["image/png"][:8] == b"\x89PNG\r\n\x1a\n"
    finally:
        InteractiveShell.clear_instance()


def test_the_element_bar_puts_focusing_magnets_above_the_axis_and_flags_forks(folder):
    m = ev.machine(folder)
    ax = Figure().subplots()
    ev.plot_elements(ax, m, "ring", s=(0.0, 20.0))
    blocks = {}
    for collection in ax.collections:
        if isinstance(collection, PolyCollection):
            for path in collection.get_paths():
                v = path.vertices
                blocks[round(v[:, 0].min(), 9)] = (v[:, 1].min(), v[:, 1].max())
    at = {
        name: blocks[round(m.lattice.placement(name, branch="ring").s_enter, 9)]
        for name in ("QF_1", "QD_1", "B_1")
    }
    assert at["QF_1"][0] == 0.0 < at["QF_1"][1]  # focusing: above the axis
    assert at["QD_1"][0] < at["QD_1"][1] == 0.0  # defocusing: below
    assert at["B_1"][0] == -at["B_1"][1]  # a dipole straddles it
    assert ax.get_xlim() == (0.0, 20.0)
    ax = Figure().subplots()
    ev.plot_elements(ax, m, "ring")
    assert {"→ extraction", "← injection"} <= {t.get_text().strip() for t in ax.texts}


def test_the_plots_draw_optics_orbits_turns_spectra_and_ramps(folder):
    m = ev.machine(folder)
    optics = ev.optics(m, "ring", engine="xsuite")
    _, beta, eta = ev.plot_optics(optics, m, s=(10.0, 30.0)).axes
    assert beta.get_xlim() == (10.0, 30.0) and [line.get_label() for line in beta.get_lines()] == ["βx", "βy"]
    shown = beta.get_lines()[0].get_xdata()
    assert len(shown) < len(optics.s) and shown.min() < 10.0 < 30.0 < shown.max()  # to the edges, no further
    assert eta.get_lines()[1].get_linestyle() == "--"  # the vertical dispersion, dashed as in the viewer

    tbt = acquisition(m)
    with ev.session(m, "ring", engine="xsuite") as session:
        orbit = session.orbit(ev.OrbitRequest())
    _, xs, ys = ev.plot_orbit(orbit, tbt, machine=m).axes
    assert xs.get_lines()[1].get_ydata() == pytest.approx(closed_orbit(tbt)[0])
    assert [line.get_color() for line in xs.get_lines()[:2]] == [
        line.get_color() for line in ys.get_lines()[:2]
    ]
    with pytest.raises(ValueError, match="give the machine"):
        ev.plot_orbit(orbit)

    texts = {t.get_text().strip() for t in ev.plot_turns(tbt, 2).axes[0].texts}
    assert "kick" in texts and "capture" not in texts
    legend = ev.plot_spectrum(tbt).axes[0].get_legend()
    seen = [t.get_text().split("q = ") for t in legend.get_texts()]
    assert [(plane.strip(), float(q)) for plane, q in seen] == [
        ("x", pytest.approx(0.21, abs=5e-4)),
        ("y", pytest.approx(0.31, abs=5e-4)),
    ]
    energy = ev.plot_ramp(m.ramps["demo"], m).axes[0].get_lines()[0].get_ydata()
    assert energy[0] == pytest.approx(2.0, abs=1e-3) and energy[-1] == pytest.approx(2.01, abs=1e-3)


def test_a_ring_started_from_a_notebook_has_the_machines_mode_and_ramp(folder, tmp_path):
    m = ev.machine(folder)
    run = ev.start_ring(
        m,
        "ring",
        tmp_path / "run",
        mode="stronger",
        engine="xsuite",
        fidelity=ev.Fidelity.FAST,
        beam=ev.Beam(particles=4),
        chunk=128,
        turns=256,
    )
    try:
        run.wait(256, timeout=300)
    finally:
        run.stop()
    assert run.ramp_profile == m.ramps["demo"] and run.values == ev.mode_values(m, "stronger")
    assert "done at turn 256" in to_text(run)


def test_the_viewer_opens_where_a_link_to_it_would(monkeypatch):
    monkeypatch.setenv("EICVIBE_URL", "http://va.example:8000/")
    frame = ev.viewer(branch="route:transfer", tab="orbit", s=(10, 25.5))
    assert frame.src == "http://va.example:8000/#branch=route%3Atransfer&tab=orbit&s=10,25.5"
