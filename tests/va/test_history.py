"""What the virtual accelerator keeps besides its logs, which an administrator may delete: the runs' BPM
histories on disk (never a running machine's) and the proposals (finished ones, or waiting ones too)."""

import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("xtrack")

from fastapi.testclient import TestClient  # noqa: E402

from eicvibe.api import Catalog, create_app  # noqa: E402
from eicvibe.api.demo import write_demo  # noqa: E402
from eicvibe.api.runs import RunCatalog  # noqa: E402
from eicvibe.va import VirtualAccelerator, load_source  # noqa: E402

VA = "/api/v1/va"


def wait(condition, what: str, timeout: float = 120.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if found := condition():
            return found
        time.sleep(0.2)
    raise TimeoutError(f"no {what} after {timeout} s")


def test_an_administrator_deletes_past_runs_and_proposals_and_the_log_keeps_it(tmp_path):
    va = VirtualAccelerator(load_source(write_demo(tmp_path / "files")), tmp_path / "state")
    password = va.first_admin()
    app = create_app(Catalog([tmp_path / "files"]), static_dir=None, runs=RunCatalog([va.runs]), va=va)
    with TestClient(app) as admin:
        assert admin.post(f"{VA}/login", json={"name": "admin", "password": password}).status_code == 200
        pat = TestClient(app)
        made = admin.post(f"{VA}/users", json={"name": "pat", "role": "physicist"}).json()
        assert pat.post(f"{VA}/login", json={"name": "pat", "password": made["password"]}).status_code == 200
        token = admin.post(f"{VA}/me/tokens", json={"label": "harness", "kind": "agent"}).json()["token"]
        agent = TestClient(app, headers={"Authorization": f"Bearer {token}"})

        body = {"name": "Ring", "kind": "ring", "target": "ring", "engine": "xsuite", "fidelity": "fast"}
        ring = admin.post(f"{VA}/machines", json=body | {"particles": 10, "rate": 20000}).json()["id"]

        def state() -> str:
            return admin.get(f"{VA}/machines/{ring}").json()["state"]

        # One run, stopped: past; a second, running: in use.
        admin.post(f"{VA}/machines/{ring}/start", json={"reason": "first"})
        wait(lambda: state() == "running", "running ring")
        admin.post(f"{VA}/machines/{ring}/stop", json={"reason": "done"})
        wait(lambda: state() == "stopped", "stopped ring")
        time.sleep(1.1)  # run ids carry the second they start
        admin.post(f"{VA}/machines/{ring}/start", json={"reason": "second"})
        wait(lambda: state() == "running", "running ring again")
        current = admin.get(f"{VA}/machines/{ring}").json()["run"]

        history = admin.get(f"{VA}/history").json()
        assert [(r["id"] == current, r["in_use"]) for r in history["runs"]] == [(True, True), (False, False)]
        past = history["runs"][1]["id"]
        assert all(r["machine"] == ring and r["bytes"] > 0 for r in history["runs"])

        # Only a person who administers deletes, and never a running machine's run.
        assert pat.get(f"{VA}/history").status_code == 403
        assert agent.delete(f"{VA}/runs").status_code == 403
        assert admin.delete(f"{VA}/runs/{current}").status_code == 409
        assert admin.delete(f"{VA}/runs/..%2Fva.sqlite").status_code == 404  # only the runs' own folders
        deleted = admin.delete(f"{VA}/runs", params={"reason": "disk"}).json()
        assert deleted["deleted"] == [past] and deleted["bytes"] > 0
        assert not (va.runs / past).exists() and (va.runs / current).is_dir()
        assert [r["id"] for r in admin.get("/api/v1/runs").json()] == [current]  # the viewer's runs follow

        # Proposals: the finished go; those waiting for a person stay unless asked.
        request = {"operation": "tune_move", "dqx": 0.001, "dqy": 0.0}
        done = admin.post(f"{VA}/machines/{ring}/proposals", json=request).json()
        admin.post(f"{VA}/proposals/{done['id']}/discard", json={})
        pending = admin.post(f"{VA}/machines/{ring}/proposals", json=request).json()
        assert admin.get(f"{VA}/history").json()["proposals"] == {"discarded": 1, "proposed": 1}
        assert admin.delete(f"{VA}/proposals").json() == {"deleted": 1}
        assert admin.get(f"{VA}/proposals/{pending['id']}").status_code == 200
        assert admin.delete(f"{VA}/proposals", params={"waiting": True}).json() == {"deleted": 1}
        assert admin.get(f"{VA}/history").json()["proposals"] == {}

        actions = {a["action"]: a for a in admin.get(f"{VA}/actions").json()}
        assert (
            actions["delete runs"]["detail"]["runs"] == [past] and actions["delete runs"]["reason"] == "disk"
        )
        assert actions["delete proposals"]["detail"]["count"] == 1
        admin.post(f"{VA}/machines/{ring}/stop", json={"reason": "done"})
        wait(lambda: state() == "stopped", "stopped ring")
