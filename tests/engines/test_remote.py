"""An engine on an engine host (ADR 0015): what crosses, and a server run here as a host would run it."""

import json
import os
import subprocess
import sys
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pytest

from eicvibe.api.demo import demo_lattice
from eicvibe.domain.lattice import ElementKind
from eicvibe.engines.contracts import (
    BeamTrack,
    Effect,
    Fidelity,
    Options,
    Ramp,
    TrackRequest,
    TrackResult,
    UnsupportedRequest,
)
from eicvibe.engines.remote import (
    _FRAME,
    _MAGIC,
    EngineHost,
    RemoteEngineError,
    _Encoder,
    _read,
    _Undecodable,
    _write,
)
from eicvibe.engines.worker import EngineWorker

SERVER = [sys.executable, "-m", "eicvibe.engines.remote", "serve"]


def crossed(value):
    """``value`` written and read back, as a message crosses."""
    reading, writing = _pipe()
    _write(writing, value)
    writing.close()
    return _read(reading)


def _pipe():
    r, w = os.pipe()
    return os.fdopen(r, "rb"), os.fdopen(w, "wb")


def test_the_contract_crosses_as_it_is():
    options = Options(fidelity=Fidelity.PRECISE, effects=frozenset({Effect.LONGITUDINAL}))
    ramp = Ramp(
        {("q1", "MagneticMultipoleP.Kn1L"): 0.1},
        {("q1", "MagneticMultipoleP.Kn1L"): 0.2},
        np.linspace(0, 1, 5),
    )
    track = BeamTrack(
        observe=("bpm_1",),
        turns=3,
        x=np.array([[1e-3, np.nan, 2e-3]]),
        y=np.zeros((1, 3)),
        count=np.array([[2, 0, 1]]),
        final=np.arange(12.0).reshape(2, 6),
        lost=np.array([False, True]),
    )
    result = TrackResult(
        engine="xsuite",
        fidelity=Fidelity.FAST,
        effects=frozenset(),
        branch="ring",
        observe=("bpm_1",),
        turns=1,
        x=(((1.0, float("nan")),),),
        px=(((0.0, 0.0),),),
        y=(((0.0, 0.0),),),
        py=(((0.0, 0.0),),),
        lost=(False, True),
    )
    nothing = np.zeros((0, 4))  # no BPMs observed
    sent = [options, ramp, track, result, ("a", 1), {("q1", "x"): 2.0}, Effect.SPIN, nothing]
    o, r, t, res, pair, keyed, effect, empty = crossed({"id": 3, "args": sent})["args"]
    assert empty.shape == (0, 4) and empty.dtype == nothing.dtype
    assert o == options and pair == ("a", 1) and keyed == {("q1", "x"): 2.0} and effect is Effect.SPIN
    assert r.start == ramp.start and r.end == ramp.end and np.array_equal(r.fraction, ramp.fraction)
    assert t.observe == ("bpm_1",) and t.lost.dtype == bool and t.count.dtype == track.count.dtype
    np.testing.assert_array_equal(t.x, track.x)  # NaN where no particle is left
    np.testing.assert_array_equal(t.final, track.final)
    assert res.lost == (False, True) and np.isnan(res.x[0][0][1]) and res.x[0][0][0] == 1.0


def test_only_the_contracts_types_cross():
    with pytest.raises(TypeError, match="cannot cross"):
        _Encoder()(object())
    with pytest.raises(TypeError, match="cannot cross"):
        _Encoder()(np.array([object()]))
    # A message naming anything else (here, to run a command) is refused where it is read.
    for named in (
        {"$c": "Popen", "v": {"args": "true"}},
        {"$m": "Ramp", "v": {}},
        {"$e": "Options", "v": "x"},
    ):
        header = json.dumps({"id": 7, "args": [named]}).encode()
        reading, writing = _pipe()
        writing.write(_FRAME.pack(_MAGIC, len(header), 0) + header)
        writing.close()
        with pytest.raises(_Undecodable) as refused:
            _read(reading)
        assert refused.value.key == 7 and "not one of the engine contract's types" in str(refused.value)


@pytest.fixture(scope="module")
def ring():
    pytest.importorskip("xtrack")
    lattice = demo_lattice()
    bpms = [p.id for p in lattice.layout("ring").placements if p.element.kind is ElementKind.INSTRUMENT]
    return lattice, bpms


def test_a_host_tracks_as_a_local_worker_does(ring):
    lattice, bpms = ring
    options = Options(effects=frozenset({Effect.LONGITUDINAL}))
    bunch = np.random.default_rng(3).normal(0, 1, (16, 6)) * [1e-4, 1e-5, 1e-4, 1e-5, 1e-3, 1e-4]
    here, there = EngineWorker("xsuite"), EngineWorker("xsuite", EngineHost("here", SERVER))
    try:
        tracks = [
            w.open_session(lattice, "ring", options).track_beam(bunch, 50, bpms, options)
            for w in (here, there)
        ]
        local, remote = tracks
        for name in ("count", "final", "lost"):  # each particle tracks alone: the same, bit for bit
            np.testing.assert_array_equal(getattr(remote, name), getattr(local, name))
        for name in ("x", "y"):  # threads and GPUs add up the centroids in their own order
            np.testing.assert_allclose(getattr(remote, name), getattr(local, name), rtol=1e-12, atol=1e-18)
        assert there._pool.info["contract"] and there._pool.info["xsuite_context"].startswith(("cpu", "cuda"))
        # The contract's errors cross as they are.
        session = there.open_session(lattice, "ring", options)
        with pytest.raises(UnsupportedRequest) as refused:
            session.track(
                TrackRequest(
                    options=Options(effects=frozenset({Effect.SPACE_CHARGE})), particles=((0,) * 6,), turns=1
                )
            )
        assert refused.value.engine == "xsuite" and refused.value.reasons
    finally:
        here.close()
        there.close()


def test_a_lost_host_fails_what_waits_and_the_next_call_starts_it_again(ring):
    lattice, bpms = ring
    worker = EngineWorker("xsuite", EngineHost("here", SERVER))
    try:
        session = worker.open_session(lattice, "ring")
        first = worker._pool
        first._process.kill()  # the connection is lost
        first._process.wait()
        with pytest.raises((BrokenProcessPool, KeyError)):
            session.track_beam(np.zeros((2, 6)), 5, bpms, Options())
        again = worker.open_session(lattice, "ring")  # a new connection
        assert worker._pool is not first
        track = again.track_beam(np.zeros((2, 6)), 5, bpms, Options())
        assert track.count.shape == (len(bpms), 5)
    finally:
        worker.close()


def test_text_a_shell_prints_first_is_skipped(ring):
    lattice, _ = ring
    banner = (
        "import sys, runpy; print('Welcome to the cluster'); sys.stdout.flush(); sys.argv = ['x', 'serve']; "
    )
    banner += "runpy.run_module('eicvibe.engines.remote', run_name='__main__')"
    worker = EngineWorker("xsuite", EngineHost("bannered", [sys.executable, "-c", banner]))
    try:
        assert worker.open_session(lattice, "ring").branch
    finally:
        worker.close()


def test_a_host_with_another_contract_is_refused():
    fake = (
        "import sys, time; from eicvibe.engines.remote import _read, _write; "
        "m = _read(sys.stdin.buffer); "
        "_write(sys.stdout.buffer, {'id': m['id'], 'result': {'contract': '0000'}}); "
        "time.sleep(30)"
    )
    worker = EngineWorker("xsuite", EngineHost("old", [sys.executable, "-c", fake]))
    try:
        with pytest.raises(RemoteEngineError, match="another EICViBE"):
            worker.open_session(demo_lattice(), "ring")
    finally:
        worker.close()


def test_the_command_runs_as_a_console_script():
    done = subprocess.run(
        [sys.executable, "-m", "eicvibe.engines.remote", "serve", "--help"], capture_output=True, text=True
    )
    assert done.returncode == 0 and "--xsuite-context" in done.stdout
