"""Optics in the viewer API: computed once per revision by engine workers, refused with reasons."""

from importlib.util import find_spec

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("xtrack")

from fastapi.testclient import TestClient  # noqa: E402

from eicvibe.api import Catalog, create_app  # noqa: E402
from eicvibe.domain.machine import generate_registry  # noqa: E402

from .test_app import build, digest_of  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    folder = tmp_path_factory.mktemp("optics")
    lattice = build()
    (folder / "demo_lattice.json").write_text(lattice.to_json())
    (folder / "demo_registry.json").write_text(generate_registry(lattice, "ring").model_dump_json())
    with TestClient(create_app(Catalog([folder], refresh_interval=0.0), static_dir=None)) as client:
        yield client  # leaving stops the engine workers


def optics(client, branch: str, **query):
    return client.get(f"/api/v1/lattices/{digest_of(client)}/branches/{branch}/optics", params=query)


def test_the_engines_are_listed_with_what_they_offer(client):
    engines = {e["name"]: e for e in client.get("/api/v1/engines").json()}
    assert engines["xsuite"]["available"] and "standard" in engines["xsuite"]["fidelities"]
    assert "linear" not in engines["xsuite"]["fidelities"]
    assert "TPSA" in engines["trackpad"]["method"]


def test_a_ring_has_its_periodic_optics(client):
    response = optics(client, "ring")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] and body["periodic"] and body["origin"] == "periodic solution"
    assert body["engine"] == "xsuite" and body["fidelity"] == "standard" and body["version"]
    assert body["profile"] is None
    n = len(body["occurrences"])
    assert len(body["s"]) == n + 1 and body["s"][0] == 0.0
    assert all(len(values) == n + 1 for values in body["values"].values())
    assert {c["key"] for c in body["columns"]} == set(body["values"])
    beta = body["values"]["beta_x"]
    assert beta[0] == pytest.approx(beta[-1], rel=1e-6)  # periodic
    assert 0 < body["tune_x"] and body["chroma_x"] is not None and body["momentum_compaction"] > 0
    assert max(abs(v) for v in body["values"]["y"]) > 0  # the skew kicker's vertical closed orbit
    again = optics(client, "ring", engine="xsuite", fidelity="standard")
    assert again.headers["etag"] == response.headers["etag"]
    cached = client.get(again.url, headers={"If-None-Match": again.headers["etag"]})
    assert cached.status_code == 304


def test_a_line_without_a_start_says_why(client):
    body = optics(client, "line").json()
    assert not body["available"] and body["s"] == [] and body["tune_x"] is None
    assert "no initial conditions" in body["origin"]


def test_requests_that_cannot_be_served_are_refused(client):
    assert optics(client, "ring", engine="nope").status_code == 404
    assert optics(client, "nowhere").status_code == 404
    refused = optics(client, "ring", fidelity="linear")
    assert refused.status_code == 422 and "no linear fidelity" in refused.json()["detail"]
    assert optics(client, "ring", fidelity="sloppy").status_code == 422


@pytest.mark.skipif(find_spec("juliacall") is None, reason="juliacall is not installed")
def test_trackpad_computes_with_tpsa_and_agrees_with_xsuite(client):
    ours = optics(client, "ring", engine="trackpad", fidelity="precise").json()
    theirs = optics(client, "ring", engine="xsuite", fidelity="precise").json()
    assert ours["method"] == "TPSA" and ours["version"]
    assert theirs["profile"] is None  # Xsuite: values at element exits only
    profile = ours["profile"]  # TrackPad: also at every integrator step inside magnets
    assert len(profile["s"]) == len(profile["index"]) + 1 > len(ours["s"])
    assert set(profile["values"]) == set(ours["values"]) - {"x", "y"}
    assert (
        profile["index"] == sorted(profile["index"]) and profile["index"][-1] == len(ours["occurrences"]) - 1
    )
    assert (ours["tune_x"], ours["tune_y"]) == pytest.approx((theirs["tune_x"], theirs["tune_y"]), abs=1e-6)
    assert ours["values"]["beta_x"] == pytest.approx(theirs["values"]["beta_x"], rel=1e-5)
    assert (ours["chroma_x"], ours["chroma_y"]) == pytest.approx(
        (theirs["chroma_x"], theirs["chroma_y"]), rel=1e-2
    )


def test_a_route_beside_a_lattice_file_has_its_optics(tmp_path):
    from eicvibe.api.demo import write_demo

    write_demo(tmp_path)
    with TestClient(create_app(Catalog([tmp_path]), static_dir=None)) as client:
        digest = digest_of(client)
        routes = {r["name"]: r for r in client.get(f"/api/v1/lattices/{digest}").json()["routes"]}
        assert set(routes) == {"transfer", "fill"}
        transfer = routes["transfer"]
        assert transfer["id"] == "route:transfer" and transfer["branches"] == [
            "injection",
            "ring",
            "extraction",
        ]
        assert len(transfer["segments"]) == 3 and transfer["problems"] == []
        url = f"/api/v1/lattices/{digest}/branches/route:transfer/optics"
        body = client.get(url, params={"engine": "xsuite"}).json()
        assert body["s"][-1] == pytest.approx(transfer["length"]) and not body["periodic"]
        assert client.get(f"/api/v1/lattices/{digest}/branches/route:nowhere/optics").status_code == 404


def test_optics_in_an_operating_mode_of_a_machine_folder(tmp_path):
    from eicvibe.api.demo import demo_lattice
    from eicvibe.domain.lattice import ElementKind
    from eicvibe.domain.machine import describe_machine
    from eicvibe.store import Setting, write_machine, write_settings

    lattice = demo_lattice()
    folder = tmp_path / "demo"
    description = describe_machine(lattice, {"ring": None, "injection": None, "extraction": None})
    write_machine(folder, lattice, description, source={"importer": "test"})
    quads = [p for p in lattice.layout("ring").placements if p.kind is ElementKind.QUADRUPOLE]
    stronger = [
        Setting(p.id, "MagneticMultipoleP.Kn1", 1.02 * p.element.MagneticMultipoleP.Kn1) for p in quads
    ]
    write_settings(folder, "stronger", stronger, lattice, "every ring quadrupole 2 % stronger")
    http = TestClient(create_app(Catalog([folder], refresh_interval=0.0), static_dir=None))
    digest = http.get("/api/v1/lattices").json()[0]["digest"]
    summary = http.get(f"/api/v1/lattices/{digest}").json()
    assert summary["modes"] == ["stronger"]
    ring = next(b["id"] for b in summary["branches"] if b["name"] == "ring")  # the folder's own ids
    path = f"/api/v1/lattices/{digest}/branches/{ring}/optics"
    design = http.get(path).json()
    mode = http.get(path, params={"mode": "stronger"}).json()
    assert (design.get("mode"), mode["mode"]) == (None, "stronger")
    assert abs(mode["tune_x"] - design["tune_x"]) > 1e-3  # the mode's settings, not the lattice's
    assert http.get(path, params={"mode": "nope"}).status_code == 404
    # A mode's file edited, then removed, with the lattice as it was: served at once.
    stronger = [Setting(s.occurrence, s.parameter, 1.02 * s.value) for s in stronger]
    write_settings(folder, "stronger", stronger, lattice, "every ring quadrupole 4 % stronger")
    edited = http.get(path, params={"mode": "stronger"}).json()
    assert abs(edited["tune_x"] - mode["tune_x"]) > 1e-3
    (folder / "settings" / "stronger.yaml").unlink()
    assert http.get(f"/api/v1/lattices/{digest}").json()["modes"] == []
    assert http.get(path, params={"mode": "stronger"}).status_code == 404
