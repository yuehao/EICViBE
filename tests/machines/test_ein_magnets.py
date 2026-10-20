"""The EIN's magnets' transfer functions and supplies' limits, from its parameter databases (ADR 0014)."""

import os
import shutil
from pathlib import Path

import pytest

from eicvibe.importers.magnet_tables import (
    MU0,
    read_preinjector_magnets,
    read_rcs_magnets,
    read_transfer_line_magnets,
)
from eicvibe.machines.ein_magnets import DATABASES, apply_ein_magnets, ein_magnets
from eicvibe.store import check_machine, load_machine

FOLDER = Path(os.environ.get("EICVIBE_EIN_FOLDER", "/Users/haoyue/Research/EIC/virtual_accelerator/EIN"))
MACHINE = FOLDER / "machine"
pytestmark = pytest.mark.skipif(
    not all((FOLDER / p).is_file() for p in DATABASES.values()) or not (MACHINE / "machine.yaml").is_file(),
    reason="no EIN folder with its parameter databases and machine",
)


def test_the_databases_give_each_magnet_its_field_per_ampere():
    lines = {m.name: m for m in read_transfer_line_magnets(FOLDER / DATABASES["lines"])}
    bates = lines["LTR_QUA01"]  # 64 turns a pole, 6.5 cm full gap
    assert bates.per_ampere == pytest.approx(2 * MU0 * 64 / 0.0325**2)
    # The sheet's own currents, with its mu0 of 1.257 G cm / A: 3e-4 from ours.
    assert bates.field / bates.per_ampere == pytest.approx(bates.current, rel=4e-4)
    assert lines["QDH1"].per_ampere == pytest.approx(2 * MU0 * 64 / 0.02**2)
    assert lines["LTR_DIP01"].per_ampere == pytest.approx(2 * MU0 * 32 / 0.04)
    assert max(m.maximum or 0 for m in lines.values()) == pytest.approx(110.86, abs=0.01)  # MIT Bates
    pre = read_preinjector_magnets(FOLDER / DATABASES["preinjector"])
    quads = [m for m in pre if m.kind == "quadrupole"]
    assert len(quads) == 27 and {m.maximum for m in quads} == {150.0}  # 16 V, 150 A supplies
    assert [round(m.per_ampere, 6) for m in quads[:4]] == [0.010053, 0.010053, 0.010053, 0.062832]
    solenoids = [m for m in pre if m.kind == "solenoid" and m.maximum]
    assert len(solenoids) == 16 and solenoids[0].per_ampere == pytest.approx(MU0 * 40 / 0.1)
    rcs = {m.name: m for m in read_rcs_magnets(FOLDER / DATABASES["rcs"])}
    assert rcs["RCS dipole"].maximum == pytest.approx(3444, abs=1)  # 0.649 T at 9 GeV
    assert rcs["RCS sextupole"].per_ampere == pytest.approx(6 * MU0 * 4 / 0.02**3)
    assert rcs["RCS corrector"].per_ampere == pytest.approx(0.013 / 25.205)


def test_the_designers_currents_are_ours_where_the_designs_are_the_same():
    plan = ein_magnets(load_machine(MACHINE), FOLDER)
    checks = {c.magnet: c for c in plan.checks}
    # The same magnets at the same strengths: the sheets' rounded mu0 (and 1/0.3 for c) apart.
    for magnet in ("ltr/ltr_qua01", "ltr/ltr_dip04", "rte/qfh1", "rte/qf_bates_1", "prj/prj-q6093"):
        assert checks[magnet].difference < 1.2e-3, checks[magnet]
    assert checks["rcs/b0_1"].ours == pytest.approx(287.0, abs=0.2)  # 0.054 T at injection
    assert checks["ltr/ltr_qua13"].difference > 0.5  # rematched without a BAR mode
    assert {a.transfer for a in plan.assignments} >= {
        "mit-bates-quadrupole",
        "rte-quadrupole",
        "rcs-dipole",
        "rcs-quadrupole",
        "rcs-sextupole",
        "rcs-corrector",
        "preinjector-quadrupole-2cm",
        "preinjector-solenoid-8cm",
        "preinjector-gun-dipole",
    }


def test_the_machine_folder_takes_them_in_its_power_file_only(tmp_path):
    folder = tmp_path / "machine"
    shutil.copytree(MACHINE, folder, ignore=shutil.ignore_patterns("build", "measurements"))
    before = {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    path, plan = apply_ein_magnets(folder, FOLDER)
    after = {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    assert [p for p in after if after[p] != before.get(p)] == [path]
    machine = load_machine(folder)
    supplies = {s.key: s for s in machine.description.supplies}
    [dipoles] = supplies["rcs.b0"].circuits
    assert dipoles.limits == (0.0, 3450.0) and dipoles.magnets[0].transfer.source.value == "design"
    assert supplies["rcs/cor_1"].circuits[0].limits == (-25.3, 25.3)
    # The July 2025 magnet table asks more of eight linac quadrupoles than their September 2024 supplies'
    # 150 A: the check says so.
    beyond = [p for p in check_machine(folder) if "beyond its supply's limits" in p]
    assert len(beyond) == 8 and all("[0, 150]" in p for p in beyond)
    assert any("prj-q6093 needs 227.6 A" in p for p in beyond)
    assert len(plan.assignments) == 669


def test_a_design_beyond_its_supply_is_marked_where_people_see_it(tmp_path):
    from eicvibe.api.views import RevisionViews
    from eicvibe.va.settings import MachineSettings

    folder = tmp_path / "machine"
    shutil.copytree(MACHINE, folder, ignore=shutil.ignore_patterns("build", "measurements"))
    apply_ein_magnets(folder, FOLDER)
    machine = load_machine(folder)
    settings = MachineSettings(machine.lattice, machine.registry, {}).settings
    beyond = sorted(s.name for s in settings.values() if s.beyond)
    assert len(beyond) == 8 and "prj-q6093-ps:sp" in beyond
    sources = {s.source for s in settings.values()}
    assert sources == {"design", "placeholder", None}  # None: RF and other parameter channels
    quad = machine.lattice.placement("prj-q6093", branch="prj").id
    [power] = [
        p for d in RevisionViews(machine.lattice, machine.registry).occurrence(quad).devices for p in d.power
    ]
    assert (
        power.beyond
        and power.limits == [0.0, 150.0]
        and power.design_current == pytest.approx(227.6, abs=0.1)
    )
