"""A bunch shared out among Xsuite contexts (several GPUs, ADR 0015) tracks as one bunch does.

Shares run here on the CPU context, each on a line of its own in a thread of its own, as they do on
GPUs: the combined BPM readings and final coordinates must be one bunch's.
"""

import math

import numpy as np
import pytest

pytest.importorskip("xtrack")

import eicvibe.engines.xsuite as xsuite  # noqa: E402
from eicvibe.api.demo import demo_lattice  # noqa: E402
from eicvibe.domain.lattice import ElementKind  # noqa: E402
from eicvibe.engines import engine  # noqa: E402
from eicvibe.engines.contracts import Effect, Options  # noqa: E402
from eicvibe.engines.convert import reference  # noqa: E402


def test_a_bunch_shared_out_tracks_as_one_bunch(monkeypatch):
    lattice = demo_lattice()
    bpms = [p.id for p in lattice.layout("ring").placements if p.element.kind is ElementKind.INSTRUMENT]
    options = Options(effects=frozenset({Effect.LONGITUDINAL}))
    bunch = np.random.default_rng(5).normal(0, 1, (300, 6)) * [1e-4, 1e-5, 1e-4, 1e-5, 1e-3, 1e-4]
    bunch[7] = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]  # lost at once: losses are combined too
    pc = reference("test", lattice, "ring")["pc"]
    momentum = pc * (1.0 + 1e-5 * np.arange(40))
    whole = engine("xsuite").open(lattice, "ring", options)
    shared = engine("xsuite").open(lattice, "ring", options)
    whole._contexts, shared._contexts = (None,), (None, None, None)  # the CPU's, wherever this runs
    monkeypatch.setattr(xsuite, "PER_GPU", 1)
    try:
        for kwargs in ({}, {"momentum": momentum}):
            one = whole.track_beam(bunch, 40, bpms, options, **kwargs)
            three = shared.track_beam(bunch, 40, bpms, options, **kwargs)
            for name in ("final", "lost", "count"):  # each particle tracks alone
                np.testing.assert_array_equal(getattr(three, name), getattr(one, name))
            for name in ("x", "y"):  # the centroids, combined by count
                np.testing.assert_allclose(getattr(three, name), getattr(one, name), rtol=1e-12, atol=1e-18)
            assert three.lost[7] and math.isfinite(float(np.nansum(three.x)))
        assert len({k[2] for k in shared._kept}) == 3  # a line kept for each share
    finally:
        whole.close()
        shared.close()


def test_a_small_bunch_stays_on_one_device(monkeypatch):
    lattice = demo_lattice()
    options = Options(effects=frozenset({Effect.LONGITUDINAL}))
    session = engine("xsuite").open(lattice, "ring", options)
    session._contexts = (None, None)
    try:
        session.track_beam(np.zeros((xsuite.PER_GPU - 1, 6)), 2, [], options)
        assert {k[2] for k in session._kept} == {0}
    finally:
        session.close()


def test_a_context_spec_names_its_gpus():
    with pytest.raises(ValueError, match="named twice"):
        xsuite._contexts("cuda:1,1")
    with pytest.raises(ValueError, match="expected cpu"):
        xsuite.compute_contexts("tpu")
    assert xsuite.compute_contexts("cpu") == (None,)
