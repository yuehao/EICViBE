"""What engines promise, independent of any engine being installed."""

import pytest

from eicvibe.domain.lattice import ElementKind, LatticeBuilder, SequentialIds
from eicvibe.engines import ENGINES, engine, select
from eicvibe.engines.contracts import (
    Capabilities,
    Computation,
    Effect,
    Fidelity,
    Options,
    Support,
    TimeScale,
    UnsupportedRequest,
)
from eicvibe.engines.model import SETTINGS, Program, quasi_static
from tests.domain.lattice.conftest import complex_builder

CAPABILITIES = Capabilities(
    engine="toy",
    computations=frozenset({Computation.OPTICS, Computation.ORBIT}),
    fidelities=frozenset({Fidelity.STANDARD}),
    effects={Effect.LONGITUDINAL: Support.NATIVE, Effect.SPACE_CHARGE: Support.APPROXIMATE},
    computation_gaps={Computation.OPTICS: frozenset({Effect.LONGITUDINAL})},
    time_scales={TimeScale.EQUILIBRIUM: Support.NATIVE, TimeScale.CYCLE: Support.APPROXIMATE},
    kinds=frozenset({ElementKind.DRIFT, ElementKind.QUADRUPOLE}),
    updatable=SETTINGS,
    notes={"optics.longitudinal": "4D optics", Effect.WAKEFIELDS.value: "not yet"},
)


def test_capabilities_name_every_gap_of_a_request():
    asked = Options(fidelity=Fidelity.PRECISE, effects=frozenset({Effect.WAKEFIELDS, Effect.SPACE_CHARGE}))
    reasons = CAPABILITIES.gaps(asked, Computation.TRACKING, TimeScale.TURNS, frozenset({ElementKind.BEND}))
    assert reasons == [
        "no tracking computation",
        "no precise fidelity",
        "effect space_charge: approximate",
        "effect wakefields: none (not yet)",
        "time scale turns: none",
        "element kinds Bend",
    ]


def test_approximations_are_accepted_only_when_allowed():
    approximate = Options(effects=frozenset({Effect.SPACE_CHARGE}), allow_approximate=True)
    assert CAPABILITIES.gaps(approximate, Computation.ORBIT, TimeScale.CYCLE) == []
    assert CAPABILITIES.gaps(Options(), Computation.ORBIT, TimeScale.CYCLE) == [
        "time scale cycle: approximate"
    ]


def test_an_effect_can_be_missing_from_one_computation():
    six = Options(effects=frozenset({Effect.LONGITUDINAL}))
    assert CAPABILITIES.gaps(six, Computation.ORBIT, TimeScale.EQUILIBRIUM) == []
    assert CAPABILITIES.gaps(six, Computation.OPTICS, TimeScale.EQUILIBRIUM) == [
        "effect longitudinal in optics: none (4D optics)"
    ]
    assert CAPABILITIES.gaps(six, None, TimeScale.EQUILIBRIUM) == []  # what opening a session checks


def test_the_engines_declare_capabilities_without_starting():
    for name in ENGINES:
        capabilities = engine(name).capabilities()
        assert capabilities.engine == name
        assert Computation.OPTICS in capabilities.computations
        assert capabilities.updatable == SETTINGS


def test_select_picks_the_first_engine_that_can():
    lattice = complex_builder().build()
    six = Options(effects=frozenset({Effect.LONGITUDINAL}))
    available = [name for name in ENGINES if engine(name).available()]
    if "xsuite" not in available:
        pytest.skip("Xsuite is not installed")
    # TrackPad's optics are 4D, so 6D optics go to Xsuite even when TrackPad is preferred.
    assert select(lattice, "ring", six, Computation.OPTICS, prefer=["trackpad", "xsuite"]).name == "xsuite"
    if "trackpad" in available:
        assert (
            select(lattice, "ring", six, Computation.ORBIT, prefer=["trackpad", "xsuite"]).name == "trackpad"
        )
    beam_beam = Options(effects=frozenset({Effect.BEAM_BEAM}))
    with pytest.raises(UnsupportedRequest, match="every engine cannot do this: .*beam_beam") as refused:
        select(lattice, "ring", beam_beam)
    assert len(refused.value.reasons) == len(ENGINES)


def test_select_refuses_kinds_no_engine_models():
    b = LatticeBuilder("crab", ids=SequentialIds())
    b.define("C", "CrabCavity", length=1.0)
    b.add_branch("line", "linac", ["C"])
    with pytest.raises(UnsupportedRequest, match="element kinds CrabCavity"):
        select(b.build(), "line")


def test_a_program_interpolates_and_holds_its_ends():
    key = ("occ_q", "MagneticMultipoleP.Kn1")
    program = Program({key: [(1.0, 0.5), (0.0, 0.3)]})  # given out of order
    assert program.at(-1.0) == {key: 0.3}
    assert program.at(0.25) == {key: pytest.approx(0.35)}
    assert program.at(2.0) == {key: 0.5}
    with pytest.raises(ValueError, match="no points"):
        Program({key: []})


def test_quasi_static_applies_each_snapshot_before_computing():
    class Recorder:
        def __init__(self):
            self.values = {}

        def update(self, values):
            self.values.update(values)

    key = ("occ_q", "MagneticMultipoleP.Kn1")
    session = Recorder()
    results = quasi_static(
        session, Program({key: [(0.0, 1.0), (1.0, 2.0)]}), (0.0, 0.5, 1.0), lambda s: s.values[key]
    )
    assert results == [(0.0, 1.0), (0.5, 1.5), (1.0, 2.0)]


def test_an_engine_must_be_an_adapter_with_every_computation(monkeypatch):
    """A new engine gets the contract's checks by subclassing; leaving a computation out, or not
    subclassing, is refused before it can run (ADR 0006)."""
    import sys
    import types

    from eicvibe.engines import EngineAdapter, EngineSession

    class Partial(EngineSession):  # no _orbit, _track, _track_beam
        engine = "partial"

        def _update(self, changes): ...

        def _optics(self, request, occurrences, partial): ...

        def close(self): ...

    with pytest.raises(TypeError, match="abstract"):
        Partial(None, None, None, None, {})

    class Bare:  # the shape of an engine, but none of its checks
        name = "bare"

        def available(self):
            return True

    module = types.ModuleType("bare_engine")
    module.Bare = Bare
    monkeypatch.setitem(sys.modules, "bare_engine", module)
    monkeypatch.setitem(ENGINES, "bare", "bare_engine:Bare")
    with pytest.raises(TypeError, match="not an EngineAdapter"):
        engine("bare")
    assert issubclass(type(engine("xsuite")), EngineAdapter)
