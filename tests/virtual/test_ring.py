"""A virtual ring tracked turn after turn: BPM data that reads as a real system's does."""

import math
import time
from datetime import timedelta
from importlib.util import find_spec

import numpy as np
import pytest

pytest.importorskip("xtrack")

from eicvibe.api.demo import demo_lattice  # noqa: E402
from eicvibe.diagnostics import NO_BEAM, read_rhic, tune, write_rhic  # noqa: E402
from eicvibe.domain.lattice import ElementKind  # noqa: E402
from eicvibe.domain.machine import NamingArea, generate_machine_registry  # noqa: E402
from eicvibe.engines import engine  # noqa: E402
from eicvibe.engines.contracts import Fidelity, OpticsRequest, Options  # noqa: E402
from eicvibe.virtual import Beam, BpmHistory, BpmModel, RingRun, ring_bpms  # noqa: E402


@pytest.fixture(scope="module")
def demo():
    lattice = demo_lattice()
    areas = {"ring": (NamingArea(prefix="dr", start=0.0, end=lattice.layout("ring").length + 1.0),)}
    registry = generate_machine_registry(
        lattice, areas, series={"ring": (ElementKind.BEND, ElementKind.QUADRUPOLE)}
    )
    cavity = next(p.id for p in lattice.layout("ring").placements if p.kind is ElementKind.RFCAVITY)
    # The demo cavity's phase is 0 from the crest (PALS's ACCELERATING); a stored beam above transition
    # wants the falling zero crossing, pi/2 further on.
    return lattice, registry, {(cavity, "RFP.phase"): math.pi / 2}


def fractional(q: float) -> float:
    q %= 1.0
    return min(q, 1.0 - q)


def run(demo, folder, **options) -> RingRun:
    lattice, registry, stable = demo
    options = {
        "engine": "xsuite",
        "fidelity": Fidelity.FAST,
        "beam": Beam(particles=16),
        "chunk": 256,
    } | options
    options.setdefault("values", stable)
    return RingRun(lattice, "ring", folder, registry=registry, **options)


def test_a_ring_run_records_every_turn_as_bpms_read_it(demo, tmp_path):
    lattice, _, stable = demo
    ring = run(demo, tmp_path / "run", turns=1280)
    ring.kick(x=1e-4, y=5e-5)  # queued before starting: it comes before the first turn
    ring.start()
    ring.wait()
    ring.stop()
    assert ring.warnings == [] and ring.error is None
    history = BpmHistory.open(tmp_path / "run")
    assert history.span() == (0, 1280)  # within the 4096-turn buffer
    events = [(e["turn"], e["kind"], e.get("of")) for e in history.events()]
    assert events[:3] == [(0, "start", None), (0, "inject", None), (0, "kick", None)]
    # The injection and the kick each triggered a capture; the run ended first, so they hold what it has.
    assert sorted(events[3:]) == [(0, "capture", "inject"), (0, "capture", "kick")]
    assert {(c["kind"], c["turns"], c.get("partial")) for c in history.captures()} == {
        ("inject", 1280, True),
        ("kick", 1280, True),
    }
    tbt = history.turn_by_turn(256, 1024)
    assert tbt.x.bpms[0].startswith("dr-b") and len(tbt.x.bpms) == 11  # the registry's EIC names
    assert tbt.x.position.shape == (11, 1024) and tbt.x.status.max() == 0
    assert tbt.time == history.info.start + timedelta(seconds=256 * history.info.revolution_period)
    optics = (
        engine("xsuite")
        .open(lattice, "ring", Options(fidelity=Fidelity.FAST), stable)
        .optics(OpticsRequest(options=Options(fidelity=Fidelity.FAST)))
    )
    assert tune(tbt, "x") == pytest.approx(fractional(optics.tune_x), abs=2e-3)
    assert tune(tbt, "y") == pytest.approx(fractional(optics.tune_y), abs=2e-3)
    # Written in RHIC's layout and read back, it is analysed as a RHIC acquisition would be.
    write_rhic(tbt, tmp_path / "acquisition.sdds")
    assert tune(read_rhic(tmp_path / "acquisition.sdds"), "x") == pytest.approx(tune(tbt, "x"), abs=1e-6)
    orbit = history.orbit()  # a sample per second of the run, the last one when it ended
    assert orbit.turn[0] == 0 and orbit.turns.sum() == 1280 and orbit.intensity[0, 0] == pytest.approx(1.0)


def test_the_rf_phase_is_checked_against_transition(demo, tmp_path):
    _, _, stable = demo
    with pytest.warns(UserWarning, match="unstable above transition"):
        crest = {key: 0.0 for key in stable}  # PALS's ACCELERATING zero: the crest
        ring = run(demo, tmp_path / "crest", values=crest, turns=256).start()
    ring.wait()
    ring.stop()
    assert ring.warnings[0].startswith("ring: cavity CAV_1 at RF phase 0 degrees")
    assert "a stored beam needs 90 degrees" in ring.warnings[0]  # pi/2 on, by the cavity's zero phase
    with pytest.warns(UserWarning, match="unstable above transition"):
        ring = run(demo, tmp_path / "fixed", values=crest, turns=None).start()
    ring.update(stable)  # an operator sets the phase: the warning goes
    ring.wait(ring.turn + 2 * ring.chunk)
    ring.stop()
    assert ring.warnings == []
    assert [e["kind"] for e in BpmHistory.open(tmp_path / "crest").events()][:2] == ["start", "warning"]


def test_a_lost_beam_reads_nothing_until_the_next_injection(demo, tmp_path):
    ring = run(demo, tmp_path / "lost", bpm=BpmModel(resolution=0.0), turns=768)
    ring.kick(x=0.5)  # far beyond any aperture
    ring.start()
    ring.wait(256)
    ring.inject()
    ring.wait()
    ring.stop()
    history = BpmHistory.open(tmp_path / "lost")
    kinds = [e["kind"] for e in history.events()]
    assert "loss" in kinds and kinds.count("inject") == 2
    again = [e["turn"] for e in history.events() if e["kind"] == "inject"][1]  # when the run took it up
    dark = history.turn_by_turn(64, again - 64)
    assert np.isnan(dark.x.position).all() and (dark.x.status == NO_BEAM).all()
    assert not np.isnan(history.turn_by_turn(again, 768 - again).x.position).any()


def test_settings_ramp_in_so_the_orbit_moves_without_ringing(demo, tmp_path):
    lattice, _, _ = demo
    corrector = next(p.id for p in lattice.layout("ring").placements if p.kind is ElementKind.KICKER)
    key = (corrector, "MagneticMultipoleP.Kn0L")
    ringing = {}
    for ramp in (0, 256):  # a step, as between engine calls; a ramp, turn by turn
        one = BpmModel(resolution=0.0)
        ring = run(
            demo, tmp_path / f"ramp{ramp}", bpm=one, beam=Beam(particles=1), turns=None, ramp_turns=ramp
        )
        ring.start()
        ring.wait(512)
        ring.update({key: 5e-5})
        ring.wait(ring.turn + 100)
        ring.update({key: -2e-5})  # changed again while still ramping: it goes on from where it is
        ring.wait(ring.turn + ramp + 768)
        ring.stop()
        assert ring.values[key] == -2e-5
        history = BpmHistory.open(tmp_path / f"ramp{ramp}")
        changes = [e for e in history.events() if e["kind"] == "settings"]
        assert [e["ramp_turns"] for e in changes] == [ramp, ramp]
        before = history.turn_by_turn(changes[0]["turn"] - 256, 256).x.position
        after = history.turn_by_turn(changes[1]["turn"] + ramp, 512).x.position  # ramped all the way
        moved = np.abs(after.mean(axis=1) - before.mean(axis=1)).max()
        ringing[ramp] = after.std(axis=1).max() / moved
    # One centroid on the closed orbit: after steps it rings about the new orbit by as much as that
    # moved; after the ramps, by a few thousandths of it (a 1024-turn ramp leaves a ten-thousandth).
    assert ringing[0] > 0.5 and ringing[256] < 0.01


def test_a_run_slowed_to_its_rate_takes_a_change_within_a_call_not_a_chunk(demo, tmp_path):
    lattice, _, _ = demo
    corrector = next(p.id for p in lattice.layout("ring").placements if p.kind is ElementKind.KICKER)
    key = (corrector, "MagneticMultipoleP.Kn0L")
    # The engine goes thousands of turns a second here; the run is held to 100. Calls sized to the
    # engine would track the whole 1024-turn chunk and then wait ten seconds before taking a change.
    ring = run(demo, tmp_path / "paced", turns=None, chunk=1024, max_rate=100.0, ramp_turns=0)
    ring.start()
    ring.wait(100)
    asked = time.monotonic()
    ring.update({key: 1e-5})
    while ring.values.get(key) != 1e-5 and time.monotonic() - asked < 15:
        time.sleep(0.01)
    taken = time.monotonic() - asked
    ring.stop()
    assert ring.values.get(key) == 1e-5 and taken < 1.5


def test_bpms_come_from_the_registry_or_else_the_instruments(demo):
    lattice, registry, _ = demo
    with_names = ring_bpms(lattice, "ring", registry)
    plain = ring_bpms(lattice, "ring")
    assert [b[0] for b in with_names] == [b[0] for b in plain]  # the same BPMs, in beam order
    assert with_names[0][1].startswith("dr-b") and plain[0][1].startswith("BPM")


@pytest.mark.skipif(find_spec("juliacall") is None, reason="juliacall is not installed")
def test_trackpad_keeps_a_ring_running_too(demo, tmp_path):
    lattice, _, stable = demo
    ring = run(demo, tmp_path / "tp", engine="trackpad", fidelity=Fidelity.STANDARD, turns=1024)
    ring.kick(x=1e-4, y=5e-5)
    ring.start()
    ring.wait()
    ring.stop()
    tbt = BpmHistory.open(tmp_path / "tp").turn_by_turn(0, 1024)
    optics = engine("xsuite").open(lattice, "ring", values=stable).optics(OpticsRequest())
    assert tune(tbt, "x") == pytest.approx(fractional(optics.tune_x), abs=2e-3)


def test_a_kick_a_request_and_a_beam_loss_freeze_the_buffer(demo, tmp_path):
    ring = run(demo, tmp_path / "triggers", buffer_turns=512, chunk=256, bpm=BpmModel(resolution=0.0))
    ring.start()
    ring.wait(1024)
    ring.kick(x=1e-4)
    ring.acquire()
    ring.wait(ring.turn + 1024)
    ring.kick(x=0.5)  # far beyond any aperture: the beam is lost
    ring.wait(ring.turn + 1024)
    ring.stop()
    history = BpmHistory.open(tmp_path / "triggers")
    captures = {c["kind"]: c for c in history.captures()}
    assert set(captures) == {"inject", "kick", "request", "post-mortem"}
    kick = history.capture(captures["kick"]["id"])
    assert kick.turns == 512 and captures["kick"]["trigger"] - kick.first_turn == 64  # 1/8 before it
    lattice, _, stable = demo
    optics = (
        engine("xsuite")
        .open(lattice, "ring", Options(fidelity=Fidelity.FAST), stable)
        .optics(OpticsRequest())
    )
    assert tune(kick, "x") == pytest.approx(fractional(optics.tune_x), abs=3e-3)  # what the kick set ringing
    request = captures["request"]
    assert request["first"] + request["turns"] == request["trigger"]  # the buffer as it was
    pm = captures["post-mortem"]
    lost = history.capture(pm["id"])
    assert pm["trigger"] - lost.first_turn == 384  # 3/4 before the beam went
    gone = pm["trigger"] - lost.first_turn
    assert np.isfinite(lost.x.position[:, : gone - 1]).all() and np.isnan(lost.x.position[:, gone:]).all()
    assert history.span()[1] - history.span()[0] <= 512 + 256  # only the buffer is kept turn by turn
