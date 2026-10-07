import pytest

from eicvibe.domain.lattice import LatticeBuilder, SequentialIds


def fodo_builder() -> LatticeBuilder:
    """Periodic FODO cell: QF D QD D, 5 m long."""
    b = LatticeBuilder("fodo", ids=SequentialIds())
    b.define("QF", "Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": 1.2}, tags=["QUAD"])
    b.define("QD", "Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": -1.2}, tags=["QUAD"])
    b.define("D", "Drift", length=2.0)
    b.add_branch("cell", "periodic", ["QF", "D", "QD", "D"])
    return b


def complex_builder() -> LatticeBuilder:
    """Ring with an injection line sharing a septum and a plain extraction line."""
    b = LatticeBuilder("complex", ids=SequentialIds())
    b.define("Q", "Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": 0.8})
    b.define("D", "Drift", length=1.0)
    b.define("SEPT_RING", "Drift", length=1.0)
    b.define("SEPT_INJ", "Bend", length=1.0, BendP={"angle_ref": 0.05})
    b.define("EXT", "Marker")
    b.define(
        "BEG",
        "BeginningEle",
        ReferenceP={"species_ref": "electron", "pc_ref": 8999999985.0, "E_tot_ref": 9e9},
        TwissP={"beta_a": 10.0, "beta_b": 5.0},
    )
    b.add_branch("ring", "periodic", ["Q", "D", "SEPT_RING", "Q", "EXT", "D"])
    b.add_branch("injection", "linac", ["BEG", "D", "SEPT_INJ"])
    b.add_branch("extraction", "linac", ["D", "D"])
    b.connect_injection("injection", "ring", via=("SEPT_INJ_1", "SEPT_RING_1"), component="septum")
    b.connect_extraction("ring", "extraction", at="EXT_1")
    return b


@pytest.fixture
def fodo():
    return fodo_builder().build()


@pytest.fixture
def machine():
    return complex_builder().build()
