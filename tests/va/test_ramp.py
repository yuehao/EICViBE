"""A ring with a ramp profile in the virtual accelerator (ADR 0013): ramped on command by an operator."""

import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("xtrack")

from fastapi.testclient import TestClient  # noqa: E402

from eicvibe.api import Catalog, create_app  # noqa: E402
from eicvibe.api.demo import write_demo  # noqa: E402
from eicvibe.api.runs import RunCatalog  # noqa: E402
from eicvibe.va import VirtualAccelerator, load_source  # noqa: E402
from tests.va.test_va import login, serving  # noqa: E402
from tests.virtual.test_ring import demo_ramp  # noqa: E402


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    folder = tmp_path_factory.mktemp("va-ramp")
    source = load_source(write_demo(folder / "files"))
    source.ramps["ring"] = demo_ramp(source.lattice)  # as a machine folder's ramps/ would give it
    va = VirtualAccelerator(source, folder / "state")
    password = va.first_admin()
    app = create_app(Catalog([folder / "files"]), static_dir=None, runs=RunCatalog([va.runs]), va=va)
    with TestClient(app) as client:
        admin = login(TestClient(client.app), "admin", password)
        people = {"admin": admin}
        for name, role in (("olga", "operator"), ("vic", "viewer")):
            body = admin.post("/api/v1/va/users", json={"name": name, "role": role}).json()
            people[name] = login(TestClient(client.app), name, body["password"])
        yield client, people


def machine(client, ident: str) -> dict:
    return client.get(f"/api/v1/va/machines/{ident}").json()


def test_an_operator_ramps_a_ring_and_it_comes_back_to_injection(server):
    client, people = server
    body = {"name": "Synchrotron", "kind": "ring", "target": "ring", "fidelity": "fast", "particles": 8}
    made = people["admin"].post("/api/v1/va/machines", json=body | {"rate": 20000}).json()
    assert made["ramp"] == "demo" and made["engine"] is None  # Xsuite, for its radiation
    people["olga"].post(f"/api/v1/va/machines/{made['id']}/start", json={"reason": "to ramp"})
    for _ in range(300):
        state = machine(client, made["id"])
        if state["state"] in ("running", "failed") and state["position"]:
            break
        time.sleep(0.1)
    assert state["state"] == "running", state["error"]
    assert state["cycle"]["state"] == "injection"

    ramp = {"action": "ramp", "reason": "to 2.01 GeV"}
    refused = people["vic"].post(f"/api/v1/va/machines/{made['id']}/beam", json=ramp)
    assert refused.status_code == 403
    assert people["olga"].post(f"/api/v1/va/machines/{made['id']}/beam", json=ramp).status_code == 204
    seen = set()
    for _ in range(600):  # ramping, flattop, then injection again with a fresh bunch
        seen.add(machine(client, made["id"])["cycle"]["state"])
        if (
            "injection" in seen
            and len(seen) > 1
            and machine(client, made["id"])["cycle"]["state"] == "injection"
        ):
            break
        time.sleep(0.05)
    assert {"ramping", "injection"} <= seen
    view = client.get(f"/api/v1/va/machines/{made['id']}/ramp").json()
    assert (view["ramp"], view["flattop_turn"], view["extraction_turn"], view["refusal"]) == (
        "demo",
        250,
        300,
        None,
    )
    program = view["program"]
    assert len(program["time"]) == 241 and program["energy"][0] == pytest.approx(2.0, abs=1e-6)
    assert program["energy"][-1] == pytest.approx(2.01, abs=1e-6) and max(
        program["voltage"]
    ) == pytest.approx(0.5)
    [done] = view["ramps"]
    assert (done["extracted"] - done["turn"], done["particles"], done["injected"]) == (300, 8, 8)
    assert done["capture"].endswith("-ramp") and done["capture_turns"] == 300  # every turn of it, its BPMs'
    actions = people["admin"].get("/api/v1/va/actions", params={"machine": made["id"]}).json()
    assert any(a["action"] == "ramp" and a["user"] == "olga" for a in actions)
    people["olga"].post(f"/api/v1/va/machines/{made['id']}/stop", json={"reason": "done"})


def test_a_route_or_a_ring_without_a_ramp_cannot_ramp(server):
    client, people = server
    route = (
        people["admin"]
        .post(
            "/api/v1/va/machines",
            json={"name": "Transfer", "kind": "route", "target": "transfer", "fidelity": "fast"},
        )
        .json()
    )
    assert route["ramp"] is None
    people["olga"].post(f"/api/v1/va/machines/{route['id']}/start", json={"reason": "to try"})
    for _ in range(300):
        if machine(client, route["id"])["state"] in ("running", "failed"):
            break
        time.sleep(0.1)
    refused = people["olga"].post(f"/api/v1/va/machines/{route['id']}/beam", json={"action": "ramp"})
    assert refused.status_code == 422 and "for rings" in refused.json()["detail"]
    assert client.get(f"/api/v1/va/machines/{route['id']}/ramp").status_code == 404
    people["olga"].post(f"/api/v1/va/machines/{route['id']}/stop", json={"reason": "done"})


def test_a_notebook_ramps_a_ring_and_reads_its_ramp_and_tunes(server):
    from eicvibe.va.client import Client, VaError

    client, people = server
    body = {"name": "From a notebook", "kind": "ring", "target": "ring", "fidelity": "fast", "particles": 8}
    made = people["admin"].post("/api/v1/va/machines", json=body | {"rate": 20000}).json()
    token = people["olga"].post("/api/v1/va/me/tokens", json={"label": "notebook"}).json()["token"]
    with serving(client.app, 8798) as url:
        notebook = Client(url, token=token)
        assert notebook.ramp_view(made["id"])["refusal"].endswith("start it to ramp")
        notebook.start(made["id"], reason="from a notebook")
        for _ in range(300):
            if machine(client, made["id"])["state"] in ("running", "failed"):
                break
            time.sleep(0.1)
        notebook.ramp(made["id"], reason="from a notebook")
        for _ in range(600):
            ramps = notebook.ramp_view(made["id"])["ramps"]
            if ramps and ramps[0]["extracted"] is not None:
                break
            time.sleep(0.05)
        assert ramps[0]["particles"] == 8
        for _ in range(300):
            try:
                tunes = notebook.tunes(made["id"])
                break
            except VaError as error:
                assert error.status == 404
                time.sleep(0.05)
        assert 0.0 <= tunes["x"] <= 0.5 and tunes["turns"] == 1024
        notebook.stop(made["id"], reason="done")
