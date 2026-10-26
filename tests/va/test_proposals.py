"""Proposals in the virtual accelerator (ADR 0010): computed on the machine as set, applied by a person on a
shared machine (an agent's token only proposes there), verified against what the BPMs and the tune monitor
then show."""

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


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    folder = tmp_path_factory.mktemp("proposals")
    va = VirtualAccelerator(load_source(write_demo(folder / "files")), folder / "state", sandbox_quota=2)
    password = va.first_admin()
    app = create_app(Catalog([folder / "files"]), static_dir=None, runs=RunCatalog([va.runs]), va=va)
    with TestClient(app) as client:
        admin = TestClient(app)
        assert admin.post(f"{VA}/login", json={"name": "admin", "password": password}).status_code == 200
        ann = TestClient(app)
        made = admin.post(f"{VA}/users", json={"name": "ann", "role": "physicist"}).json()
        assert ann.post(f"{VA}/login", json={"name": "ann", "password": made["password"]}).status_code == 200
        token = ann.post(f"{VA}/me/tokens", json={"label": "opencode", "kind": "agent"}).json()["token"]
        agent = TestClient(app, headers={"Authorization": f"Bearer {token}"})
        body = {"name": "Ring", "kind": "ring", "target": "ring", "engine": "xsuite", "fidelity": "fast"}
        machine = admin.post(
            f"{VA}/machines", json=body | {"particles": 50, "rate": 20000, "ramp_turns": 256}
        )
        ring = machine.json()["id"]
        admin.post(f"{VA}/machines/{ring}/start", json={"reason": "for the tests"})
        wait(lambda: client.get(f"{VA}/machines/{ring}/tunes").status_code == 200, "a tune reading")
        yield {"client": client, "admin": admin, "ann": ann, "agent": agent, "ring": ring, "va": va}


def wait(condition, what: str, timeout: float = 120.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if found := condition():
            return found
        time.sleep(0.2)
    raise TimeoutError(f"no {what} after {timeout} s")


def verified(person: TestClient, ident: str) -> dict:
    """Verify once the machine has measured after the change."""

    def done():
        body = person.post(f"{VA}/proposals/{ident}/verify").json()
        return body if body["verification"] and body["verification"].get("ready") else None

    return wait(done, f"verification of {ident}")


def channel(client: TestClient, ring: str, suffix: str, index: int) -> str:
    rows = client.get(f"{VA}/machines/{ring}/settings").json()
    return [r["channel"] for r in rows if r["channel"].endswith(suffix)][index]


def bpm_names(s: dict) -> tuple[str, ...]:
    """The ring's BPMs in beam order, as its run records them."""
    return s["va"]._orbit_samples(s["va"]._instance(s["ring"])).bpms


def settled_orbit(s: dict) -> None:
    """Wait until the slow orbit has a sample taken after the last change finished ramping."""
    va, ring = s["va"], s["ring"]
    instance = va._instance(ring)
    target = instance.position + 256 + 64
    wait(
        lambda: len(va._orbit_samples(instance).turn) and va._orbit_samples(instance).turn[-1] >= target,
        "orbit",
    )


def test_an_agent_proposes_on_a_shared_machine_and_a_person_applies(server):
    s = server
    ring, agent, ann = s["ring"], s["agent"], s["ann"]
    assert s["client"].get(f"{VA}/me", headers=agent.headers).json()["agent"]
    # Two correctors set wrong: an orbit error to correct.
    errors = [
        {"channel": channel(ann, ring, ":sp:h", 2), "value": 3.0},
        {"channel": channel(ann, ring, ":sp:h", 7), "value": -2.0},
    ]
    assert ann.post(f"{VA}/machines/{ring}/settings", json={"changes": errors, "reason": "errors"}).json()[
        "applied"
    ]
    assert agent.post(f"{VA}/machines/{ring}/settings", json={"changes": errors}).status_code == 403
    settled_orbit(s)
    made = agent.post(f"{VA}/machines/{ring}/proposals", json={"operation": "orbit_correction", "plane": "x"})
    assert made.status_code == 201, made.text
    body = made.json()
    assert body["agent"] and body["state"] == "proposed" and body["proposal"]["context"]["orbit_before"]
    before, after = body["proposal"]["predicted"]["rms"]
    assert after < before / 5 and all(c["ok"] for c in body["proposal"]["checks"])
    refused = agent.post(f"{VA}/proposals/{body['id']}/apply", json={"reason": "agent"})
    assert refused.status_code == 403 and "a person applies" in refused.json()["detail"]
    applied = ann.post(f"{VA}/proposals/{body['id']}/apply", json={"reason": "looks right"}).json()
    assert applied["state"] == "applied" and applied["decided_by"] == "ann"
    log = ann.get(f"{VA}/log", params={"correlation": applied["correlation"]}).json()
    assert {w["channel"] for w in log} == {c["channel"] for c in body["proposal"]["changes"]}
    check = verified(agent, body["id"])["verification"]
    assert check["ok"], check  # the BPMs moved as predicted (their offsets cancel)


def test_a_bump_is_verified_by_the_bpms_and_a_stale_proposal_is_refused(server):
    s = server
    ring, ann = s["ring"], s["ann"]
    request = {"operation": "local_bump", "plane": "y", "at": bpm_names(s)[4], "amplitude_mm": 0.5}
    stale = ann.post(f"{VA}/machines/{ring}/proposals", json=request).json()
    moved = stale["proposal"]["changes"][0]["channel"]
    ann.post(f"{VA}/machines/{ring}/settings", json={"changes": [{"channel": moved, "value": 0.25}]})
    refused = ann.post(f"{VA}/proposals/{stale['id']}/apply", json={})
    assert refused.status_code == 409 and "propose again" in refused.json()["detail"]
    assert ann.get(f"{VA}/proposals/{stale['id']}").json()["state"] == "stale"
    settled_orbit(s)
    bump = ann.post(f"{VA}/machines/{ring}/proposals", json=request).json()
    assert bump["proposal"]["predicted"]["amplitude"] == pytest.approx(0.5, abs=1e-6)
    ann.post(f"{VA}/proposals/{bump['id']}/apply", json={})
    check = verified(ann, bump["id"])["verification"]
    assert check["ok"] and check["rms_predicted_change_mm"] > 0.1, check


def test_the_operations_and_their_parameters_come_from_the_catalog(server):
    s = server
    ring, ann = s["ring"], s["ann"]
    listed = s["client"].get(f"{VA}/operations").json()
    assert [o["name"] for o in listed] == ["orbit_correction", "local_bump", "tune_move"]
    bump = listed[1]
    assert bump["verified_by"] == "orbit" and bump["machines"] == ["ring"]
    assert set(bump["parameters"]["required"]) == {"at", "amplitude_mm"}
    assert bump["parameters"]["properties"]["amplitude_mm"]["x-form"] == {"unit": "mm"}
    # A request is checked against its operation's parameters before anything is computed.
    missing = ann.post(f"{VA}/machines/{ring}/proposals", json={"operation": "local_bump", "plane": "x"})
    assert missing.status_code == 422 and "amplitude_mm" in missing.text
    unknown = ann.post(f"{VA}/machines/{ring}/proposals", json={"operation": "chromaticity_move"})
    assert unknown.status_code == 422 and "tune_move" in unknown.text


def test_a_tune_move_is_verified_by_the_tune_monitor(server):
    s = server
    ring, ann = s["ring"], s["ann"]
    reading = s["client"].get(f"{VA}/machines/{ring}/tunes").json()
    assert 0 < reading["x"] < 0.5 and reading["turns"] == 1024
    move = ann.post(
        f"{VA}/machines/{ring}/proposals", json={"operation": "tune_move", "dqx": 0.01, "dqy": 0.005}
    )
    body = move.json()
    assert move.status_code == 201 and body["proposal"]["context"]["tunes_before"]
    ann.post(f"{VA}/proposals/{body['id']}/apply", json={})
    check = verified(ann, body["id"])["verification"]
    assert check["ok"], check
    assert check["planes"]["x"]["measured_change"] == pytest.approx(0.01, abs=5e-4)


def test_an_agent_applies_in_its_own_sandbox_and_proposals_can_be_discarded(server):
    s = server
    ring, agent, ann = s["ring"], s["agent"], s["ann"]
    sandbox = agent.post(f"{VA}/machines/{ring}/fork", json={"name": "agent's sandbox"}).json()
    request = {"operation": "local_bump", "plane": "x", "at": bpm_names(s)[4], "amplitude_mm": 1.0}
    made = agent.post(f"{VA}/machines/{sandbox['id']}/proposals", json=request)
    assert made.status_code == 201, made.text
    made = made.json()
    applied = agent.post(f"{VA}/proposals/{made['id']}/apply", json={"reason": "trying it"})
    assert applied.status_code == 200 and applied.json()["state"] == "applied"
    other = ann.post(f"{VA}/machines/{ring}/proposals", json=request).json()
    waiting = {p["id"] for p in ann.get(f"{VA}/machines/{ring}/proposals", params={"waiting": True}).json()}
    assert other["id"] in waiting
    assert agent.post(f"{VA}/proposals/{other['id']}/discard", json={}).json()["state"] == "discarded"
    assert ann.post(f"{VA}/proposals/{other['id']}/apply", json={}).status_code == 409
    # What waits for a person is only what is proposed or applied; what was done stays to be read.
    waiting = ann.get(f"{VA}/machines/{ring}/proposals", params={"waiting": True}).json()
    assert other["id"] not in {p["id"] for p in waiting}
    assert all(p["state"] in ("proposed", "applied") for p in waiting)
    everything = ann.get(f"{VA}/machines/{ring}/proposals").json()
    assert other["id"] in {p["id"] for p in everything} and {"verified", "discarded"} <= {
        p["state"] for p in everything
    }
    assert agent.post(f"{VA}/users", json={"name": "x", "role": "viewer"}).status_code == 403
    assert agent.post(f"{VA}/me/tokens", json={"label": "more"}).status_code == 403
    assert ann.delete(f"{VA}/machines/{sandbox['id']}").status_code == 204


def test_an_operation_the_engine_cannot_compute_is_refused_with_the_engines_reason(server, monkeypatch):
    from dataclasses import replace

    from eicvibe.operations.catalog import OPERATIONS

    def lost(model, parameters, measured):
        raise RuntimeError("Particle lost, p.state = -1")  # as Xsuite's closed orbit search says it

    monkeypatch.setitem(OPERATIONS, "tune_move", replace(OPERATIONS["tune_move"], propose=lost))
    made = server["ann"].post(
        f"{VA}/machines/{server['ring']}/proposals", json={"operation": "tune_move", "dqx": 0.01}
    )
    assert made.status_code == 422, made.text
    assert (
        "Tune move could not be computed in the model: RuntimeError: Particle lost" in made.json()["detail"]
    )
