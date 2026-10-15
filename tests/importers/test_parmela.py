"""PARMELA decks as EICViBE reads them (ADR 0012)."""

import pytest

from eicvibe.importers.parmela import read_deck, read_field_maps

DECK = """\
RUN  2000 1 2856  -0.0188 0.4e-6  1
! ****** gun-e-field ****** -1 means 350 keV
poisson  -7.44 -0.914  0  0  0  0  1  1
GUN350_22.T7
CATHODE 0 1.26 1 0 0.0
drift  30.7  3  1 ! the gun's drift
cfield 2
197M.T7
cell  80  2.5  1  -30.0  0.2359  0  0  0  197.055 ; the buncher
"""


def test_a_decks_beamline_and_the_field_maps_it_reads(tmp_path):
    deck = tmp_path / "deck.inp"
    deck.write_text(DECK)
    drift, cell = read_deck(deck)
    assert (drift.kind, drift.start, drift.end) == ("drift", 0.0, pytest.approx(0.307))
    assert (cell.kind, cell.start, cell.length) == ("cell", pytest.approx(0.307), pytest.approx(0.8))
    gun, buncher = read_field_maps(deck)
    assert (gun.command, gun.values[:2], gun.file, gun.line) == (
        "poisson",
        (-7.44, -0.914),
        "GUN350_22.T7",
        3,
    )
    assert (buncher.command, buncher.values, buncher.file) == ("cfield", (2.0,), "197M.T7")
