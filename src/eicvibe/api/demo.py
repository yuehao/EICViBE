"""A small made-up machine for trying the viewer and for browser tests: ``eicvibe-web --demo``.

An injection line feeds a ring through a shared septum; an extraction line
leaves the ring. The ring has FODO cells with dipoles, sextupoles, BPMs,
correctors and an RF cavity, and a generated machine registry, so every
part of the viewer has something to show.
"""

import math
from pathlib import Path

from eicvibe.domain.lattice import Lattice, LatticeBuilder, SequentialIds
from eicvibe.domain.machine import NamingArea, generate_machine_registry

CELLS = 12


def demo_lattice() -> Lattice:
    b = LatticeBuilder("demo", ids=SequentialIds())
    reference = {"species_ref": "electron", "pc_ref": 2e9}
    b.define("RING_START", "BeginningEle", ReferenceP=reference)
    b.define("QF", "Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": 0.8})
    b.define("QD", "Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": -0.8})
    b.define("B", "Bend", length=2.0, BendP={"angle_ref": math.pi / CELLS})
    b.define("SF", "Sextupole", length=0.2, MagneticMultipoleP={"Kn2": 4.0})
    b.define("SD", "Sextupole", length=0.2, MagneticMultipoleP={"Kn2": -6.0})
    b.define("BPM", "Instrument")
    b.define("COR", "Kicker", length=0.1)
    b.define("D", "Drift", length=0.6)
    b.define("DL", "Drift", length=2.0)
    b.define("CAV", "RFCavity", length=0.5, RFP={"voltage": 0.5e6, "harmon": 20})
    b.define("SEPTUM", "Bend", length=1.0, BendP={"angle_ref": 0.0})
    cell = ["QF", "D", "SF", "BPM", "D", "B", "D", "COR", "QD", "D", "SD", "D", "B", "D"]
    ring = ["RING_START", "SEPTUM", "DL"] + cell * (CELLS // 2 - 1) + ["CAV", "DL"] + cell * (CELLS // 2)
    b.add_branch("ring", "periodic", ring)

    b.define("LINE_START", "BeginningEle", ReferenceP=reference)
    b.define("Q1", "Quadrupole", length=0.4, MagneticMultipoleP={"Kn1": 0.6})
    b.define("Q2", "Quadrupole", length=0.4, MagneticMultipoleP={"Kn1": -0.6})
    b.define("BI", "Bend", length=1.5, BendP={"angle_ref": -0.12})
    b.define("DI", "Drift", length=3.0)
    b.add_branch("injection", "linac", ["LINE_START", "DI", "Q1", "DI", "BPM", "Q2", "DI", "BI", "DI", "Q1"])

    b.define("BX", "Bend", length=1.5, BendP={"angle_ref": 0.15})
    b.define("X_START", "BeginningEle", ReferenceP=reference)
    b.add_branch("extraction", "linac", ["X_START", "DI", "Q2", "DI", "BX", "DI", "Q1", "DI", "BPM"])

    b.connect_injection("injection", "ring", at="SEPTUM_1")
    b.add_fork("ring", "extraction", after="BPM_7")
    return b.build()


def write_demo(folder: Path) -> Path:
    """Write ``demo_lattice.json`` and ``demo_registry.json`` into ``folder``; returns the lattice path."""
    lattice = demo_lattice()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "demo_lattice.json"
    path.write_text(lattice.to_json())
    areas = {
        "ring": (NamingArea(prefix="dr", start=0.0, end=lattice.layout("ring").length + 1.0),),
        "injection": None,
        "extraction": (NamingArea(prefix="dx", start=0.0, end=lattice.layout("extraction").length + 1.0),),
    }
    registry = generate_machine_registry(lattice, areas)
    (folder / "demo_registry.json").write_text(registry.model_dump_json(indent=1))
    return path
