"""A machine's errors in the virtual accelerator (ADR 0011): chosen by whoever may configure it,
drawn from its seed whenever it starts, hidden from everyone else and from the log."""

import json
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("xtrack")

from fastapi.testclient import TestClient  # noqa: E402

from eicvibe.api import Catalog, create_app  # noqa: E402
from eicvibe.api.demo import write_demo  # noqa: E402
from eicvibe.api.runs import RunCatalog  # noqa: E402
from eicvibe.va import VirtualAccelerator, load_source  # noqa: E402

ERRORS = {
    "magnets": [
        {
            "select": {"kinds": ["Quadrupole"]},
            "x_offset": 1e-4,
            "y_offset": 1e-4,
            "roll": 2e-4,
            "field": 5e-4,
        },
        {"select": {"kinds": ["Bend"], "names": ["b_*"]}, "field": 2e-4, "pitch": 1e-4, "yaw": 1e-4},
    ],
    "supplies": [{"select": {"kinds": ["Bend"], "names": ["b_*"]}, "rms": 1e-5, "bandwidth": 1000.0}],
    "bpms": {
        "offset_rms": 0.1,
        "gain_rms": 0.01,
        "roll_rms": 1e-3,
        "resolution": 0.01,
        "orbit_resolution": 3e-4,
    },
}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    folder = tmp_path_factory.mktemp("va-errors")
    lattice_file = write_demo(folder / "files")
    va = VirtualAccelerator(load_source(lattice_file), folder / "state")
    password = va.first_admin()
    app = create_app(Catalog([folder / "files"]), static_dir=None, runs=RunCatalog([va.runs]), va=va)
    with TestClient(app) as client:
        admin = TestClient(app)
        assert admin.post("/api/v1/va/login", json={"name": "admin", "password": password}).status_code == 200
        made = admin.post("/api/v1/va/users", json={"name": "phil", "role": "physicist"}).json()
        phil = TestClient(app)
        assert (
            phil.post("/api/v1/va/login", json={"name": "phil", "password": made["password"]}).status_code
            == 200
        )
        yield client, admin, phil, va


def wait(client, machine: str, state: str) -> dict:
    for _ in range(300):
        entry = client.get(f"/api/v1/va/machines/{machine}").json()
        if entry["state"] in (state, "failed"):
            return entry
        time.sleep(0.1)
    return entry


def test_a_machines_errors_are_its_configurers_to_see_and_change(server):
    client, admin, phil, va = server
    body = {"name": "Errors ring", "kind": "ring", "target": "ring", "engine": "xsuite", "fidelity": "fast"}
    machine = admin.post("/api/v1/va/machines", json=body | {"particles": 4, "errors": ERRORS}).json()["id"]
    assert "errors" not in client.get(f"/api/v1/va/machines/{machine}").json()  # not in what everyone sees
    assert client.get(f"/api/v1/va/machines/{machine}/errors").status_code == 401
    assert phil.get(f"/api/v1/va/machines/{machine}/errors").status_code == 403  # a shared machine's: admins
    entry = admin.get(f"/api/v1/va/machines/{machine}/errors").json()
    spec = entry["errors"]
    seed = spec["seed"]
    assert entry["chosen"] == {"magnets": [22, 22], "supplies": [1]} and entry["problems"] == []
    assert isinstance(seed, int)  # drawn when it was made: the same errors every start
    assert spec["magnets"][0]["roll"] == 2e-4 and spec["bpms"]["roll_rms"] == 1e-3

    # Changed while stopped, keeping the seed when none is given; never while it runs.
    smaller = ERRORS | {"magnets": ERRORS["magnets"][:1]}
    changed = admin.put(
        f"/api/v1/va/machines/{machine}/errors", json={"errors": smaller, "reason": "one family"}
    )
    assert changed.status_code == 200
    assert changed.json()["errors"]["seed"] == seed and len(changed.json()["errors"]["magnets"]) == 1
    # A row that chooses nothing is a mistake, not no errors: said while editing, refused when saved.
    nothing = smaller | {
        "magnets": [{"select": {"kinds": ["Quadrupole"], "names": ["any"]}, "x_offset": 1e-3}]
    }
    checked = admin.post(f"/api/v1/va/machines/{machine}/errors/check", json={"errors": nothing}).json()
    assert checked["chosen"]["magnets"] == [0] and "no Quadrupole named any here" in checked["problems"][0]
    refused = admin.put(f"/api/v1/va/machines/{machine}/errors", json={"errors": nothing})
    assert refused.status_code == 422 and "nothing would be drawn for it" in refused.json()["detail"]
    admin.post(f"/api/v1/va/machines/{machine}/start", json={"reason": "errors"})
    assert wait(client, machine, "running")["state"] == "running"
    refused = admin.put(f"/api/v1/va/machines/{machine}/errors", json={"errors": ERRORS})
    assert refused.status_code == 409 and "draws its errors when it starts" in refused.json()["detail"]

    # The run drew them: hidden.json has them and the BPMs' rolls; the run's settings do not.
    run = admin.get(f"/api/v1/va/machines/{machine}").json()["run"]
    hidden = json.loads((va.runs / run / "hidden.json").read_text())
    assert hidden["errors"]["seed"] == seed and hidden["errors"]["magnets"]
    assert any(abs(r) > 0 for r in hidden["rolls_rad"])
    info = json.loads((va.runs / run / "run.json").read_text())
    assert not any("BodyShiftP" in k for k in info["values"]) and info["orbit_resolution"] == 3e-4
    admin.post(f"/api/v1/va/machines/{machine}/stop", json={"reason": "done"})
    assert wait(client, machine, "stopped")["state"] == "stopped"

    # The log says what changed and how large, never the seed.
    actions = admin.get("/api/v1/va/actions", params={"machine": machine}).json()
    logged = [a for a in actions if a["action"] in ("create", "errors")]
    assert {a["action"] for a in logged} == {"create", "errors"}
    assert all(str(seed) not in json.dumps(a) for a in logged)

    # A sandbox is a copy of the same machine: the same errors, its owner's to change.
    sandbox = phil.post(f"/api/v1/va/machines/{machine}/fork", json={}).json()["id"]
    own = phil.get(f"/api/v1/va/machines/{sandbox}/errors").json()["errors"]
    assert own["seed"] == seed and own["magnets"] == changed.json()["errors"]["magnets"]
    renewed = phil.put(f"/api/v1/va/machines/{sandbox}/errors", json={"errors": own | {"seed": seed + 1}})
    assert renewed.status_code == 200 and renewed.json()["errors"]["seed"] == seed + 1
    original = admin.get(f"/api/v1/va/machines/{machine}/errors").json()["errors"]
    assert original["seed"] == seed  # the original keeps its own


def test_a_machine_from_before_errors_keeps_its_bpms(server):
    client, admin, phil, va = server
    from eicvibe.va.service import MachineSpec

    old = MachineSpec(id="old", name="Old", kind="ring", target="ring", owner="admin")
    assert old.errors.seed is None and old.errors.bpm_model().seed == 7  # the BPMs it always had
