"""Operations as accelerator physicists do them: orbit correction, closed bumps, tune moves (ADR 0010)."""

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xtrack")

from eicvibe.api.demo import demo_lattice  # noqa: E402
from eicvibe.domain.machine import generate_registry  # noqa: E402
from eicvibe.operations.bump import local_bump  # noqa: E402
from eicvibe.operations.model import MachineModel  # noqa: E402
from eicvibe.operations.orbit import correct_orbit  # noqa: E402
from eicvibe.operations.proposal import Proposal  # noqa: E402
from eicvibe.operations.tune import move_tunes  # noqa: E402
from eicvibe.va.settings import MachineSettings  # noqa: E402

EIN = (
    Path(os.environ.get("EICVIBE_EIN_FOLDER", "/Users/haoyue/Research/EIC/virtual_accelerator/EIN"))
    / "machine"
)


@pytest.fixture(scope="module")
def settings():
    lattice = demo_lattice()
    return MachineSettings(lattice, generate_registry(lattice, "ring"), {})


@pytest.fixture(scope="module")
def ring(settings):
    with MachineModel(settings, "ring") as model:
        yield model


def errors(model: MachineModel) -> dict[str, float]:
    """Two horizontal correctors set wrong: the orbit errors to find."""
    x = model.correctors["x"]
    return {x[2].channel: model.value(x[2].channel) + 3.0, x[7].channel: model.value(x[7].channel) - 2.0}


def test_the_model_knows_the_rings_bpms_correctors_and_quadrupoles(ring):
    assert len(ring.bpms) == 11 and [m.s for m in ring.bpms] == sorted(m.s for m in ring.bpms)
    assert len(ring.correctors["x"]) == len(ring.correctors["y"]) == 11
    assert all(k.channel.endswith(":sp:h") for k in ring.correctors["x"])
    assert len(ring.quadrupoles) == 22
    orbit = ring.orbit()
    assert np.max(np.abs(orbit["x"])) < 1e-8  # the design ring's closed orbit is on axis (to its tolerance)


def test_svd_correction_takes_out_an_orbit_error_and_checks_itself_in_the_model(settings, ring):
    with MachineModel(settings, "ring", live=errors(ring)) as broken:
        proposal = correct_orbit(broken, "x")
        before, after = proposal.predicted["rms"]
        assert before > 1.0 and after < before / 50  # mm
        assert proposal.ok and proposal.operation == "orbit_correction" and proposal.references
        assert all(c.limits and c.limits[0] <= c.after <= c.limits[1] for c in proposal.changes)
        # What the proposal predicts is what the model then does.
        corrected = broken.orbit(proposal.values())["x"]
        assert np.sqrt(np.mean(corrected**2)) * 1e3 == pytest.approx(after, rel=1e-6)
        assert Proposal.model_validate_json(proposal.model_dump_json()) == proposal  # a tool's result


def test_micado_finds_the_correctors_that_explain_the_errors(settings, ring):
    wrong = errors(ring)
    with MachineModel(settings, "ring", live=wrong) as broken:
        proposal = correct_orbit(broken, "x", method="micado", count=2)
    assert set(proposal.parameters["correctors"]) == set(wrong)
    assert proposal.predicted["rms"][1] < 0.01  # mm: the two errors undone
    for change in proposal.changes:
        assert change.after == pytest.approx(ring.value(change.channel), abs=1e-3)  # back where they were


def test_a_measured_orbit_with_missing_bpms_is_corrected_where_it_was_read(settings, ring):
    with MachineModel(settings, "ring", live=errors(ring)) as broken:
        orbit = dict(zip((m.name for m in broken.bpms), broken.orbit()["x"], strict=True))
        del orbit[broken.bpms[3].name]
        proposal = correct_orbit(broken, "x", orbit)
    assert proposal.predicted["used"].count(False) == 1 and proposal.ok
    with pytest.raises(ValueError, match="not BPMs"):
        correct_orbit(ring, "x", {"nowhere": 0.001})


def test_a_closed_three_corrector_bump_reaches_its_amplitude_and_leaks_nowhere(ring):
    bpm = ring.bpms[4]
    proposal = local_bump(ring, "x", bpm.name, 1e-3)
    assert proposal.ok and len(proposal.changes) == 3
    change = ring.orbit(proposal.values())["x"] - ring.orbit()["x"]
    inside = np.array(proposal.predicted["inside"])
    assert change[4] == pytest.approx(1e-3, abs=1e-8)
    assert np.max(np.abs(change[~inside])) < 1e-8 and (~inside).sum() >= 6  # closed: unchanged outside
    knobs = {k.channel: k for k in ring.correctors["x"]}
    positions = sorted(knobs[c.channel].s for c in proposal.changes)
    assert positions[0] < bpm.s < positions[-1]  # the correctors bracket the point


def test_a_four_corrector_bump_sets_position_and_angle(ring):
    proposal = local_bump(ring, "y", ring.bpms[6].name, 0.5e-3, angle=0.1e-3)
    assert proposal.ok and len(proposal.changes) == 4
    assert proposal.predicted["amplitude"] == pytest.approx(0.5, abs=1e-6)
    assert proposal.predicted["angle_mrad"] == pytest.approx(0.1, abs=1e-6)
    assert proposal.predicted["leak"] < 1e-6


def test_a_bump_asked_of_the_wrong_correctors_or_place_says_why(ring):
    with pytest.raises(ValueError, match="no BPM or element"):
        local_bump(ring, "x", "nowhere", 1e-3)
    with pytest.raises(ValueError, match="takes 3 correctors"):
        local_bump(
            ring, "x", ring.bpms[4].name, 1e-3, correctors=[k.channel for k in ring.correctors["x"][:2]]
        )


def test_a_tune_move_reaches_the_tunes_with_the_least_beta_beating(ring):
    proposal = move_tunes(ring, 0.01, -0.005)
    (qx0, qy0), (qx1, qy1) = proposal.predicted["tunes"]
    assert (qx1 - qx0, qy1 - qy0) == pytest.approx((0.01, -0.005), abs=1e-6)
    assert proposal.ok
    optics = ring.optics(proposal.values())
    assert (optics.tune_x, optics.tune_y) == pytest.approx((qx1, qy1), abs=1e-9)
    # Two quadrupoles alone reach the same tunes, beating the optics more than all of them together.
    pair = [proposal.parameters["quadrupoles"][0], proposal.parameters["quadrupoles"][1]]
    two = move_tunes(ring, 0.01, -0.005, quadrupoles=pair)
    assert two.predicted["tunes"][1] == pytest.approx([qx1, qy1], abs=1e-4)
    assert max(two.predicted["beta_beating"]) > 2 * max(proposal.predicted["beta_beating"])


@pytest.mark.skipif(not (EIN / "machine.yaml").exists(), reason="EIN machine folder not found")
def test_a_transfer_lines_trajectory_is_steered_and_bumped():
    from eicvibe.store import load_machine

    machine = load_machine(EIN)
    settings = MachineSettings(machine.lattice, machine.registry, {})
    with MachineModel(settings, "ltr") as line:
        assert not line.periodic and len(line.bpms) == 15
        corrector = line.correctors["x"][2]
        with MachineModel(settings, "ltr", live={corrector.channel: 1.0}) as kicked:
            proposal = correct_orbit(kicked, "x")
        assert proposal.ok and proposal.predicted["rms"][1] < proposal.predicted["rms"][0] / 20
        bump = local_bump(line, "x", line.bpms[7].name, 1e-3)
        assert bump.ok and bump.predicted["leak"] < 1e-6
        with pytest.raises(ValueError, match="a line"):
            move_tunes(line, 0.01, 0.0)


def test_a_horizontal_bump_in_a_ring_with_rf_keeps_the_energy(settings, ring):
    """With RF holding the frequency (6D), a bump that lengthens the path changes the beam's energy and
    the orbit everywhere through the dispersion; one more corrector keeps the energy."""
    from eicvibe.engines.contracts import Effect, Fidelity, Options

    with MachineModel(
        settings,
        "ring",
        options=Options(fidelity=Fidelity.STANDARD, effects=frozenset({Effect.LONGITUDINAL})),
    ) as rf:
        bump = local_bump(rf, "x", rf.bpms[4].name, 1e-3)
        assert bump.ok and len(bump.changes) == 4 and "energy unchanged" in bump.method
        flat = local_bump(ring, "x", ring.bpms[4].name, 1e-3)  # three correctors, closed only in 4D
        change = rf.orbit(flat.values())["x"] - rf.orbit()["x"]
        outside = ~np.array(flat.predicted["inside"])
        assert np.max(np.abs(change[outside])) > 1e-4  # it leaks a tenth of a millimetre and more
    vertical = local_bump(ring, "y", ring.bpms[4].name, 1e-3)
    assert len(vertical.changes) == 3  # no dispersion vertically: three correctors close it
