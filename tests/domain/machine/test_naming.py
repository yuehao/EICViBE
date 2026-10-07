from dataclasses import dataclass

import pytest

from eicvibe.domain.lattice import ElementDefinition
from eicvibe.domain.machine import (
    NamingArea,
    default_repository,
    lattice_name,
    non_lattice_name,
    parse_name,
    ring_sectors,
    slot_names,
)

# Rows of the ESR naming table (NameOps resources, "es swn lattice.csv"): official name, element, length,
# s at the exit. The table writes append numbers ADO-style (dt42.1); Rev 02 writes them dt42:1. Its
# correctors are "ks"; the abbreviation database maps HKICKER to "th", which is what EICViBE uses.
ESR_EXCERPT = [
    ("es6-q7", "Quadrupole", 1.8, 7.1),
    ("es6-dh22", "Bend", 5.500091668, 22.25009),
    ("es6-dt42:1", "Drift", 19.8146, 42.06469),
    ("es6-q42", "Quadrupole", 0.25, 42.31469),
    ("es6-dt42:2", "Drift", 0.15, 42.46469),
    ("es6-q43", "Quadrupole", 0.6, 43.06469),
    ("es6-q0", "Quadrupole", 0.5, 320.1873),  # past the nominal 320 m, so the clock wraps to 0
    ("es7-dt320", "Drift", 0.156, 320.4683),  # odd sectors count back from the next IP (IP8 at 640 m)
    ("es7-sx319:1", "Sextupole", 0.24, 320.7083),
    ("es7-dt319", "Drift", 0.104, 320.8123),
    ("es7-sx319:2", "Sextupole", 0.24, 321.0523),
    ("es7-q7", "Quadrupole", 0.8, 633.3339),
    ("es9-sx1", "Sextupole", 0.24, 959.21671),  # sector 9 starts at 959.0 m, before the nominal 960 m
    ("es9-sx0", "Sextupole", 0.24, 959.56071),
    ("es9-q320", "Quadrupole", 0.5, 960.0917),
    ("es10-dt318", "Drift", 0.15, 1277.88326),  # sector 10 starts before IP10, so the clock wraps back
    ("es5-q5", "Quadrupole", 0.5, 3515.4828),  # the last sector counts back from the nominal 3840 m
    ("es5-th4", "Kicker", 0.2, 3515.87178),
    ("es5-q12", "Quadrupole", 1.2, 3828.2042),
]


@dataclass
class Placed:
    id: str
    name: str
    element: ElementDefinition
    s_enter: float
    s_exit: float


def element(kind: str, length: float) -> ElementDefinition:
    groups = {
        "Bend": {"BendP": {"angle_ref": 0.01}},
        "Kicker": {"tags": ["HCOR"]},
        "Quadrupole": {"MagneticMultipoleP": {"Kn1": 0.1}},
        "Sextupole": {"MagneticMultipoleP": {"Kn2": 0.1}},
    }.get(kind, {})
    return ElementDefinition(id="ele_1", name="E", kind=kind, length=length, **groups)


def test_slot_names_reproduce_the_official_esr_names():
    placed = [
        Placed(f"occ_{i}", f"e{i}", element(kind, length), s - length, s)
        for i, (_, kind, length, s) in enumerate(ESR_EXCERPT)
    ]
    areas = ring_sectors("es", boundaries={7: 320.3, 9: 959.0, 10: 1277.75, 5: 3515.0})
    names = slot_names(placed, areas)
    assert [names[p.id] for p in placed] == [official for official, *_ in ESR_EXCERPT]


def test_sectors_of_the_esr():
    sectors = {a.prefix: a for a in ring_sectors("es")}
    assert list(sectors)[:3] == ["es6", "es7", "es8"] and len(sectors) == 12
    assert (sectors["es6"].origin, sectors["es6"].direction) == (0.0, 1)
    assert (sectors["es7"].origin, sectors["es7"].direction) == (640.0, -1)
    assert (sectors["es5"].start, sectors["es5"].end, sectors["es5"].origin) == (3520.0, 3840.0, 3840.0)
    assert NamingArea(prefix="rcb1", start=0.0, end=100.0).position(39.5) == 40  # rounded half up


def test_parse_lattice_and_non_lattice_names():
    supply = parse_name("es6-qf50:2-ps:sp")
    assert (supply.form, supply.area, supply.device, supply.position, supply.append) == (
        "lattice",
        "es6",
        "qf",
        50,
        2,
    )
    assert (supply.controller, supply.signals) == ("ps", ("sp",))
    assert parse_name("es9-cav2:sp:volt").signals == ("sp", "volt")
    tune = parse_name("es-btun-rdx")
    assert (tune.form, tune.device, tune.position, tune.signals) == ("non-lattice", "btun", None, ("rdx",))
    for old in ("QF:I", "es6-qf50_2", "ES6-Q42"):  # provisional, Rev 01 and upper case
        with pytest.raises(ValueError, match="not an EIC name"):
            parse_name(old)


def test_builders_and_repository():
    assert lattice_name("es6", "q", 42, 2, "ps", ("sp",)) == "es6-q42:2-ps:sp"
    assert non_lattice_name("es", "btun", signals=("rdx",)) == "es-btun-rdx"
    repository = default_repository()
    assert repository.problems("es6-q42-ps:sp") == []
    assert repository.problems("es9-cav2:sp:volt") == ["signal 'volt'"]
    assert repository.problems("zz6-xyz42") == ["area 'zz6'", "device 'xyz'"]
