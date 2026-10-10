"""Bmad lattices through Bmad's own MAD-X writer, with what it loses put back."""

import re

import pytest

from eicvibe.importers import import_madx
from eicvibe.importers.bmad import (
    BmadElement,
    BmadUnavailable,
    _pitch_signs,
    _readouts,
    bmad_to_madx,
    bmad_tools,
    tao_table,
)

WRITTEN = """LTR_PAT01__t: translation, dx =  3.09000000000000004E-002;
LTR_PAT01__y: yrotation, angle =  4.39130560846672999E-003;
D1: drift, l =  1.0;
LTR_BPM05_1: drift, l =  5.0E-002;
MARK: marker;
LTR_BPM05_2: drift, l =  5.0E-002;
BPM: drift, l =  2.0E-002;
line_1: line = (LTR_PAT01__t, LTR_PAT01__y, D1, LTR_BPM05_1, MARK, LTR_BPM05_2, BPM, D1, BPM);
"""


def test_monitors_come_back_and_the_patch_pitch_its_sign():
    elements = [
        BmadElement("D1", "drift", 1.0, 1.0),
        BmadElement("LTR_BPM05#1", "monitor", 1.05, 0.05),
        BmadElement("MARK", "marker", 1.05, 0.0),
        BmadElement("LTR_BPM05#2", "monitor", 1.1, 0.05),
        BmadElement("BPM", "monitor", 1.12, 0.02),
        BmadElement("D1", "drift", 2.12, 1.0),
        BmadElement("BPM", "monitor", 2.14, 0.02),
    ]
    notes = []
    text = _pitch_signs(_readouts(WRITTEN, elements, notes), notes)
    assert "LTR_PAT01__y: yrotation, angle = -0.00439130560846673;" in text  # YROTATION = x_pitch
    assert "LTR_PAT01__t: translation, dx =  3.09000000000000004E-002;" in text  # offsets were right
    # A BPM split by a marker keeps its name on its first piece; one placed twice is defined once.
    assert "LTR_BPM05: monitor, l =  5.0E-002;" in text and "LTR_BPM05_2: drift, l =" in text
    assert "line = (LTR_PAT01__t, LTR_PAT01__y, D1, LTR_BPM05, MARK, LTR_BPM05_2, BPM, D1, BPM);" in text
    assert re.findall(r"(?m)^BPM: \w+", text) == ["BPM: monitor"]
    assert [n.code for n in notes] == ["bmad-readout", "bmad-split-readout", "bmad-patch-pitch"]


LINE = """beginning[e_tot] = 0.75e9
beginning[beta_a] = 10
beginning[beta_b] = 10
parameter[particle] = electron
parameter[geometry] = open
p1: pipe, l = 3
q1: quadrupole, l = 0.3, k1 = 1.5, &
    superimpose, ref = p1, offset = 0.5, ref_origin = beginning, ele_origin = beginning
m1: monitor, l = 0.1, superimpose, ref = p1, offset = 1.95, ref_origin = beginning, ele_origin = beginning
mid: marker, superimpose, ref = p1, offset = 2.0, ref_origin = beginning
pat: patch, x_offset = 0.01, x_pitch = 0.002
d2: drift, l = 1
line1: line = (p1, pat, d2)
use, line1
"""


@pytest.fixture(scope="module")
def tools():
    try:
        return bmad_tools()
    except BmadUnavailable as error:
        pytest.skip(str(error))


def test_a_converted_line_keeps_its_names_and_bpms_and_goes_where_bmad_goes(tools, tmp_path):
    madx_module = pytest.importorskip("cpymad.madx")
    source = tmp_path / "design" / "line.bmad"
    source.parent.mkdir()
    source.write_text(LINE)
    conversion = bmad_to_madx(source, tmp_path / "line.madx", tools=tools)
    assert [p.name for p in source.parent.iterdir()] == ["line.bmad"]  # Bmad ran on a copy
    assert {n.code for n in conversion.notes} == {
        "bmad-pipe-as-drift",
        "bmad-readout",
        "bmad-split-readout",
        "bmad-patch-pitch",
    }
    imported = import_madx(conversion.madx, kind="linac", name="line", workdir=tmp_path).lattice
    kinds = {p.name: p.kind.value for p in imported.layout(imported.branches[0].name).placements}
    assert kinds["q1"] == "Quadrupole" and kinds["m1"] == "Instrument" and kinds["mid"] == "Marker"

    # Orbit and floor at the end, after the patch, as Bmad has them.
    columns = ["orbit.vec.1", "orbit.vec.2", "ele.x_position", "ele.z_position", "ele.theta_position"]
    end = tao_table(source, columns, tools)[-1]
    with madx_module.Madx(stdout=False, command_log=None) as madx:
        madx.chdir(str(tmp_path))
        madx.call(str(conversion.madx))
        madx.use(sequence="lat")
        twiss, survey = madx.twiss(betx=10, bety=10), madx.survey()
        ours = [twiss["x"][-1], twiss["px"][-1], survey["x"][-1], survey["z"][-1], survey["theta"][-1]]
    assert end["orbit.vec.2"] == pytest.approx(-0.002)  # x_pitch = +2 mrad: the beam leaves at -2 mrad
    # The codes' drifts part by 4e-9 m after a metre at 2 mrad (third order in the angle).
    assert ours == pytest.approx([end[c] for c in columns], abs=1e-8)
