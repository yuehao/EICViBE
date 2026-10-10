"""A small made-up machine for trying the viewer and for browser tests: ``eicvibe-web --demo``.

An injection line feeds a ring through a shared septum; an extraction line
leaves the ring. The ring has FODO cells with dipoles, sextupoles, BPMs,
correctors and an RF cavity, and a generated machine registry, so every
part of the viewer has something to show. The ring is stable (tunes 2.21,
1.69) with its chromaticity corrected to about +1. Two routes join the
branches: ``transfer`` (injection, ring, extraction) and ``fill`` (injection,
once round the ring).
"""

import math
from pathlib import Path

from eicvibe.domain.lattice import ElementKind, Lattice, LatticeBuilder, SequentialIds
from eicvibe.domain.machine import NamingArea, generate_machine_registry
from eicvibe.store.machine import write_routes_file

CELLS = 12
ROUTES = {"transfer": ["injection", "ring", "extraction"], "fill": ["injection", "ring"]}


def demo_lattice() -> Lattice:
    b = LatticeBuilder("demo", ids=SequentialIds())
    reference = {"species_ref": "electron", "pc_ref": 2e9}
    b.define("RING_START", "BeginningEle", ReferenceP=reference)
    b.define("QF", "Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": 0.4})
    b.define("QD", "Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": -0.4})
    b.define("B", "Bend", length=2.0, BendP={"angle_ref": math.pi / CELLS})
    b.define("SF", "Sextupole", length=0.2, MagneticMultipoleP={"Kn2": 0.6})
    b.define("SD", "Sextupole", length=0.2, MagneticMultipoleP={"Kn2": -0.8})
    b.define("BPM", "Instrument")
    b.define("COR", "Kicker", length=0.1)
    b.define("D", "Drift", length=0.6)
    b.define("DL", "Drift", length=2.0)
    # 0 is the crest (PALS's ACCELERATING); a stored beam above transition wants pi/2 further on.
    b.define("CAV", "RFCavity", length=0.5, RFP={"voltage": 0.5e6, "harmon": 20, "phase": math.pi / 2})
    b.define("SEPTUM", "Bend", length=1.0, BendP={"angle_ref": 0.0})
    cell = ["QF", "D", "SF", "BPM", "D", "B", "D", "COR", "QD", "D", "SD", "D", "B", "D"]
    ring = ["RING_START", "SEPTUM", "DL"] + cell * (CELLS // 2 - 1) + ["CAV", "DL"] + cell * (CELLS // 2)
    b.add_branch("ring", "periodic", ring)

    b.define("LINE_START", "BeginningEle", ReferenceP=reference, TwissP={"beta_a": 8.0, "beta_b": 4.0})
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
    """Write the demo's lattice, registry and routes into ``folder``; returns the lattice file."""
    lattice = demo_lattice()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "demo_lattice.json"
    path.write_text(lattice.to_json())
    registry = _demo_registry(lattice)
    (folder / "demo_registry.json").write_text(registry.model_dump_json(indent=1))
    write_routes_file(folder / "demo_routes.yaml", ROUTES)
    return path


def _demo_registry(lattice: Lattice):
    areas = {
        "ring": (NamingArea(prefix="dr", start=0.0, end=lattice.layout("ring").length + 1.0),),
        "injection": None,
        "extraction": (NamingArea(prefix="dx", start=0.0, end=lattice.layout("extraction").length + 1.0),),
    }
    # The ring's dipoles and quadrupole families are strings; the lines' magnets have their own supplies.
    return generate_machine_registry(
        lattice, areas, series={"ring": (ElementKind.BEND, ElementKind.QUADRUPOLE)}
    )


def start_demo_runs(folder: Path, *, rate: float = 2000.0, kick_every: int = 2048, shots: float = 4.0):
    """The demo ring tracked turn after turn (ADR 0007), its beam kicked now and then, and shots along
    the ``transfer`` route, both with Xsuite.

    The ring is paced at ``rate`` turns per second and the route at ``shots`` per second, so
    they show live orbits without taking a core each. Returns the runs: none where no
    engine is installed.
    """
    import threading

    from eicvibe.domain.lattice import route_through
    from eicvibe.engines import engine
    from eicvibe.engines.contracts import Fidelity
    from eicvibe.virtual import Beam, RingRun, RouteRun

    if not engine("xsuite").available():
        return []
    lattice = demo_lattice()
    registry = _demo_registry(lattice)
    run = RingRun(
        lattice,
        "ring",
        folder / "demo-ring",
        engine="xsuite",
        fidelity=Fidelity.FAST,
        beam=Beam(particles=20),
        chunk=256,
        registry=registry,
        retention_bytes=64 << 20,
        max_rate=rate,
    ).start()

    def kicks() -> None:
        sign = 1.0
        while run.running:
            run.kick(x=sign * 1.5e-4, y=7e-5)
            sign = -sign
            target = run.turn + kick_every
            while run.running and run.turn < target:
                threading.Event().wait(0.1)

    threading.Thread(target=kicks, name="demo kicks", daemon=True).start()
    transfer = RouteRun(
        lattice,
        route_through(lattice, "transfer", ROUTES["transfer"]),
        folder / "demo-transfer",
        engine="xsuite",
        fidelity=Fidelity.FAST,
        beam=Beam(particles=20),
        rate=shots,
        chunk=2,
        registry=registry,
        retention_bytes=16 << 20,
    ).start()
    return [run, transfer]
