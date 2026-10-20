"""Ramp profiles in a machine folder (ADR 0013): ramps/<name>.yaml and its RF table, read back exactly."""

import pytest

from eicvibe.store import FileError, check_machine, load_machine, read_ramps, write_machine, write_ramp
from tests.domain.machine.test_ramps import profile, ring


def ramp_problems(folder) -> list[str]:
    return [p for p in check_machine(folder) if p.startswith("ramps:")]  # a folder without naming rules


def test_a_ramp_profile_reads_back_exactly_and_is_checked_with_the_machine(tmp_path):
    folder = tmp_path / "machine"
    write_machine(folder, ring())
    written = write_ramp(folder, profile())
    assert [p.relative_to(folder).as_posix() for p in written] == ["ramps/rcs.yaml", "ramps/rcs.csv"]
    assert read_ramps(folder) == {"rcs": profile()}
    machine = load_machine(folder)
    assert machine.ramps["rcs"].energy.extraction == 9e9 and ramp_problems(folder) == []

    write_ramp(folder, profile(name="other", branch="line"))
    assert ramp_problems(folder) == ["ramps: ramp other: line is not a ring"]
    write_ramp(folder, profile(name="other", branch="nowhere"))
    with pytest.raises(FileError, match="ramps of no branch of the lattice: nowhere"):
        load_machine(folder)


def test_a_ramp_file_without_its_table_says_so(tmp_path):
    folder = tmp_path / "machine"
    write_machine(folder, ring())
    write_ramp(folder, profile())
    (folder / "ramps" / "rcs.csv").write_text("time,voltage\n0,1\n")
    with pytest.raises(FileError, match="needs the columns phase"):
        read_ramps(folder)
