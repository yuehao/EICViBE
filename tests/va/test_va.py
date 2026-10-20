"""The virtual accelerator service: who may change what, what the log keeps, and the machine files kept."""

import hashlib
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("xtrack")

from fastapi.testclient import TestClient  # noqa: E402

from eicvibe.api import Catalog, create_app  # noqa: E402
from eicvibe.api.demo import write_demo  # noqa: E402
from eicvibe.api.runs import RunCatalog  # noqa: E402
from eicvibe.va import MachineSettings, Role, VirtualAccelerator, load_source  # noqa: E402
from eicvibe.virtual import BpmHistory  # noqa: E402


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    folder = tmp_path_factory.mktemp("va")
    lattice_file = write_demo(folder / "files")
    va = VirtualAccelerator(load_source(lattice_file), folder / "state", sandbox_quota=2)
    password = va.first_admin()
    app = create_app(Catalog([folder / "files"]), static_dir=None, runs=RunCatalog([va.runs]), va=va)
    with TestClient(app) as client:
        yield client, va, password, lattice_file


def login(client, name, password) -> TestClient:
    response = client.post("/api/v1/va/login", json={"name": name, "password": password})
    assert response.status_code == 200, response.text
    return client


def as_user(server, name, password):
    client, va, *_ = server
    other = TestClient(client.app)
    return login(other, name, password)


@pytest.fixture(scope="module")
def people(server):
    client, va, password, _ = server
    admin = as_user(server, "admin", password)
    made = {}
    for name, role in (("ann", "physicist"), ("olga", "operator"), ("vic", "viewer")):
        body = admin.post("/api/v1/va/users", json={"name": name, "role": role}).json()
        made[name] = as_user(server, name, body["password"])
    made["admin"] = admin
    return made


@pytest.fixture(scope="module")
def ring(server, people):
    client, va, *_ = server
    body = {"name": "Main ring", "kind": "ring", "target": "ring", "engine": "xsuite", "fidelity": "fast"}
    machine = people["admin"].post("/api/v1/va/machines", json=body | {"particles": 8, "rate": 20000}).json()
    people["olga"].post(f"/api/v1/va/machines/{machine['id']}/start", json={"reason": "for the tests"})
    for _ in range(200):
        state = client.get(f"/api/v1/va/machines/{machine['id']}").json()
        if state["state"] in ("running", "failed"):
            break
        time.sleep(0.1)
    assert state["state"] == "running", state["error"]
    return state


def corrector(client, machine) -> dict:
    rows = client.get(f"/api/v1/va/machines/{machine}/settings").json()
    return next(r for r in rows if r["channel"].endswith(":sp:h"))


def test_anyone_reads_but_changing_needs_a_login_and_a_role(server, people, ring):
    client, *_ = server
    assert client.get("/api/v1/va/status").json()["enabled"]
    assert client.get("/api/v1/va/me").json() == {
        "name": "anonymous",
        "role": "viewer",
        "logged_in": False,
        "agent": False,
    }
    assert client.post("/api/v1/va/login", json={"name": "ann", "password": "nope"}).status_code == 401
    assert client.get("/api/v1/va/machines").json()[0]["id"] == "main-ring"
    assert (
        client.get("/api/v1/va/machines").json()[0]["warnings"] == []
    )  # the demo cavity's phase holds a bunch
    channel = corrector(client, ring["id"])["channel"]
    write = {"changes": [{"channel": channel, "value": 1.0}], "reason": "try"}
    assert client.post(f"/api/v1/va/machines/{ring['id']}/settings", json=write).status_code == 401
    assert people["vic"].post(f"/api/v1/va/machines/{ring['id']}/settings", json=write).status_code == 403
    machine = {"name": "x", "kind": "ring", "target": "ring"}
    assert people["ann"].post("/api/v1/va/machines", json=machine).status_code == 403  # shared: admins only
    assert people["olga"].get("/api/v1/va/users").status_code == 403
    assert client.get("/api/v1/va/log").status_code == 401  # the write log: for users who are logged in
    assert people["vic"].get("/api/v1/va/log").status_code == 200
    assert people["ann"].get("/api/v1/va/actions").status_code == 403  # the action log: operators, admins
    actions = [a["action"] for a in people["olga"].get("/api/v1/va/actions").json()]
    assert "login refused" in actions and "start" in actions


def test_a_write_is_logged_with_the_old_value_and_reaches_the_running_engine(server, people, ring):
    client, va, *_ = server
    row = corrector(client, ring["id"])
    write = {"changes": [{"channel": row["channel"], "value": 2.0}], "reason": "bump at the septum"}
    result = people["ann"].post(f"/api/v1/va/machines/{ring['id']}/settings", json=write).json()
    assert result["applied"] and result["writes"][0]["old"] == row["original"] == 0.0
    # The demo's supplies have placeholder transfer functions, and their designs are within their limits.
    assert (row["source"], row["beyond"]) == ("placeholder", False)
    (logged,) = people["vic"].get("/api/v1/va/log", params={"correlation": result["correlation"]}).json()
    assert (logged["user"], logged["new"], logged["reason"], logged["applied"]) == (
        "ann",
        2.0,
        write["reason"],
        True,
    )
    changed = client.get(f"/api/v1/va/machines/{ring['id']}/settings", params={"changed": True}).json()
    assert [(c["channel"], c["value"], c["by"]) for c in changed] == [(row["channel"], 2.0, "ann")]
    history = BpmHistory.open(va.runs / ring["run"])
    for _ in range(100):  # applied between chunks, and logged in the run's own events
        events = [e for e in history.events() if e["kind"] == "settings"]
        if events:
            break
        time.sleep(0.1)
    assert events[-1]["by"] == "ann" and events[-1]["correlation"] == result["correlation"]


def test_a_refused_write_refuses_the_whole_change_and_is_logged(server, people, ring):
    client, *_ = server
    row = corrector(client, ring["id"])
    write = {
        "changes": [{"channel": row["channel"], "value": 500.0}, {"channel": "dr-q4-ps:sp", "value": 1.0}]
    }
    result = people["ann"].post(f"/api/v1/va/machines/{ring['id']}/settings", json=write).json()
    assert not result["applied"]
    assert "outside the limits" in result["writes"][0]["refusal"]
    assert result["writes"][1]["refusal"] == "another write of this change was refused"
    logged = people["vic"].get("/api/v1/va/log", params={"correlation": result["correlation"]}).json()
    assert len(logged) == 2 and not any(w["applied"] for w in logged)


def test_reset_and_snapshots_bring_settings_back_and_are_logged_too(server, people, ring):
    client, *_ = server
    url = f"/api/v1/va/machines/{ring['id']}"
    row = corrector(client, ring["id"])
    people["ann"].post(f"{url}/settings", json={"changes": [{"channel": row["channel"], "value": 3.0}]})
    snapshot = people["ann"].post(f"{url}/snapshots", json={"name": "3 A bump"}).json()
    assert snapshot["changes"] == 1
    assert (
        people["ann"].post(f"{url}/reset", json={}).status_code == 403
    )  # all of a shared machine: operators
    reset = people["olga"].post(f"{url}/reset", json={"reason": "back to the design"}).json()
    assert reset["writes"] == [
        {"channel": row["channel"], "unit": "A", "old": 3.0, "new": 0.0, "applied": True, "refusal": None}
    ]
    assert client.get(f"{url}/settings", params={"changed": True}).json() == []
    restored = people["olga"].post(f"{url}/snapshots/{snapshot['id']}/restore", json={}).json()
    assert restored["writes"][0]["new"] == 3.0
    reasons = [
        w["reason"] for w in people["vic"].get("/api/v1/va/log", params={"machine": ring["id"]}).json()
    ]
    assert reasons[:2] == ["restore snapshot 3 A bump", "back to the design"]


def test_a_sandbox_is_a_users_own_copy_within_a_quota(server, people, ring):
    client, *_ = server
    url = f"/api/v1/va/machines/{ring['id']}"
    assert people["vic"].post(f"{url}/fork", json={}).status_code == 403
    sandbox = people["ann"].post(f"{url}/fork", json={"name": "ann's tests"}).json()
    assert (sandbox["shared"], sandbox["owner"], sandbox["parent"]) == (False, "ann", ring["id"])
    assert sandbox["changes"] == client.get(url).json()["changes"]  # the settings as they were
    row = corrector(client, sandbox["id"])
    write = {"changes": [{"channel": row["channel"], "value": -1.0}]}
    assert people["olga"].post(f"/api/v1/va/machines/{sandbox['id']}/settings", json=write).status_code == 403
    assert people["ann"].post(f"/api/v1/va/machines/{sandbox['id']}/settings", json=write).json()["applied"]
    people["ann"].post(f"{url}/fork", json={})
    assert people["ann"].post(f"{url}/fork", json={}).status_code == 403  # the quota
    assert people["ann"].delete(f"/api/v1/va/machines/{sandbox['id']}").status_code == 204


def test_a_notebook_writes_with_its_token(server, people, ring):
    client, *_ = server
    token = people["ann"].post("/api/v1/va/me/tokens", json={"label": "orbit notebook"}).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/v1/va/me", headers=headers).json()["name"] == "ann"
    row = corrector(client, ring["id"])
    write = {"changes": [{"channel": row["channel"], "value": 0.5}], "reason": "from a notebook"}
    response = client.post(f"/api/v1/va/machines/{ring['id']}/settings", json=write, headers=headers)
    assert response.json()["applied"]


def test_the_log_cannot_be_changed_and_the_machine_files_are_never_written(server, people, ring):
    client, va, _, lattice_file = server
    before = hashlib.sha256(lattice_file.read_bytes()).hexdigest()
    row = corrector(client, ring["id"])
    people["ann"].post(
        f"/api/v1/va/machines/{ring['id']}/settings",
        json={"changes": [{"channel": row["channel"], "value": 4.0}]},
    )
    assert hashlib.sha256(lattice_file.read_bytes()).hexdigest() == before
    with pytest.raises(Exception, match="append-only"):
        va.store._run("UPDATE writes SET new = 0")
    with pytest.raises(Exception, match="append-only"):
        va.store._run("DELETE FROM actions")
    runs = client.get("/api/v1/runs").json()
    assert any(r["machine"] == ring["id"] for r in runs)  # its BPM history, for the Orbit tab


def test_originals_are_the_machine_files_values():
    from eicvibe.api.demo import _demo_registry, demo_lattice

    lattice = demo_lattice()
    registry = _demo_registry(lattice)
    quad = next(p for p in lattice.layout("ring").placements if p.name == "QF_1")
    mode = {(quad.id, "MagneticMultipoleP.Kn1"): 0.44}
    plain, moded = MachineSettings(lattice, registry, {}), MachineSettings(lattice, registry, mode)
    channel = next(n for n, s in plain.settings.items() if (quad.id, "MagneticMultipoleP.Kn1") in s.keys)
    assert moded.settings[channel].original == pytest.approx(plain.settings[channel].original * 1.1)
    # at its original current the string gives every magnet its value back; nothing changed keeps it exactly
    values = plain.lattice_values({channel: plain.settings[channel].original}, [channel])
    assert values[(quad.id, "MagneticMultipoleP.Kn1")] == pytest.approx(0.4, rel=1e-9)
    assert plain.lattice_values({}, [channel])[(quad.id, "MagneticMultipoleP.Kn1")] == 0.4
    rf = next(n for n in plain.settings if n.endswith(":sp:phase"))
    assert plain.settings[rf].original == pytest.approx(90.0)  # degrees, from the lattice's pi/2


def test_a_server_without_a_virtual_accelerator_says_so(tmp_path):
    client = TestClient(create_app(Catalog([]), static_dir=None))
    assert client.get("/api/v1/va/status").json() == {"enabled": False, "source": None}
    assert client.get("/api/v1/va/machines").status_code == 404


def test_roles_go_up():
    assert Role.ADMIN > Role.OPERATOR > Role.PHYSICIST > Role.VIEWER
    assert Role.named("Operator") is Role.OPERATOR
    with pytest.raises(ValueError, match="roles: viewer"):
        Role.named("boss")


def test_the_notebook_client_writes_and_reads_the_log(server, people, ring):
    import threading

    import uvicorn

    from eicvibe.va.client import Client, Refused

    client, va, *_ = server
    token = people["ann"].post("/api/v1/va/me/tokens", json={"label": "client test"}).json()["token"]
    config = uvicorn.Config(client.app, host="127.0.0.1", port=8797, log_level="warning", lifespan="off")
    web = uvicorn.Server(config)
    thread = threading.Thread(target=web.run, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if web.started:
                break
            time.sleep(0.05)
        notebook = Client("http://127.0.0.1:8797", token=token)
        assert notebook.me()["name"] == "ann"
        channel = next(c for c in notebook.settings(ring["id"]) if c.endswith(":sp:h"))
        result = notebook.set(ring["id"], {channel: 1.25}, reason="client")
        assert notebook.get(ring["id"], [channel]) == {channel: 1.25}
        assert notebook.log(correlation=result["correlation"])[0]["user"] == "ann"
        with pytest.raises(Refused, match="outside the limits"):
            notebook.set(ring["id"], {channel: 1e6})
        assert notebook.orbits(ring["id"])["bpms"]
    finally:
        web.should_exit = True
        thread.join(5)


def test_the_console_manages_accounts_and_prints_the_log(server, capsys):
    from eicvibe.va.__main__ import main

    _, va, *_ = server
    state = str(va.state)
    main(["add-user", state, "carl", "--role", "operator"])
    password = capsys.readouterr().out.strip()
    assert va.store.check_password("carl", password).role is Role.OPERATOR
    main(["log", state, "--limit", "3"])
    assert "->" in capsys.readouterr().out
