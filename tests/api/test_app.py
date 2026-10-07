"""The lattice viewer API: content-addressed, cached, and the same for every client."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from eicvibe.api import Catalog, create_app  # noqa: E402
from eicvibe.api.views import RevisionViews  # noqa: E402
from eicvibe.domain.lattice import Lattice, LatticeBuilder, SequentialIds  # noqa: E402
from eicvibe.domain.machine import generate_registry  # noqa: E402

ROOT = Path(__file__).parents[2]


def build(kn1: float = 0.3) -> Lattice:
    b = LatticeBuilder("demo", ids=SequentialIds())
    b.define("BEG", "BeginningEle", ReferenceP={"species_ref": "electron", "pc_ref": 1e9})
    b.define("QF", "Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": kn1})
    b.define("QD", "Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": -kn1})
    b.define("B", "Bend", length=2.0, BendP={"angle_ref": 3.141592653589793 / 4})
    b.define("SX", "Sextupole", length=0.2, MagneticMultipoleP={"Kn2L": 0.1})
    b.define("K", "Kicker", MagneticMultipoleP={"Ks0L": 1e-4})
    b.define("BPM", "Instrument")
    b.define("CAV", "RFCavity", length=0.3, RFP={"voltage": 1e6, "harmon": 10})
    b.define("D", "Drift", length=1.0)
    b.add_branch("ring", "periodic", ["BEG"] + ["QF", "D", "B", "SX", "QD", "BPM", "K", "D", "B", "CAV"] * 4)
    b.define("LBEG", "BeginningEle", ReferenceP={"species_ref": "electron", "pc_ref": 1e9})
    b.add_branch("line", "linac", ["LBEG", "D", "QF", "D"])
    b.add_fork("line", "ring", to="QF_1")
    return b.build()


@pytest.fixture
def folder(tmp_path):
    lattice = build()
    (tmp_path / "demo_lattice.json").write_text(lattice.to_json())
    (tmp_path / "demo_registry.json").write_text(generate_registry(lattice, "ring").model_dump_json())
    return tmp_path


@pytest.fixture
def client(folder):
    return TestClient(create_app(Catalog([folder], refresh_interval=0.0), static_dir=None))


def digest_of(client) -> str:
    [entry] = client.get("/api/v1/lattices").json()
    return entry["digest"]


def test_lists_lattices_of_a_folder_with_their_registry(client):
    [entry] = client.get("/api/v1/lattices").json()
    assert entry["source"] == "demo" and entry["registry"] and entry["branches"] == 2
    assert entry["digest"] == build().digest.removeprefix("sha256:")


def test_summary_places_the_line_through_its_fork(client):
    summary = client.get(f"/api/v1/lattices/{digest_of(client)}").json()
    branches = {b["name"]: b for b in summary["branches"]}
    [fork] = summary["forks"]
    assert branches["ring"]["placed_by"] is None and branches["line"]["placed_by"] == fork["id"]
    assert branches["ring"]["reference"] == {"species": "electron", "pc": 1e9, "E_tot": None}
    assert summary["kinds"]["Quadrupole"] == 9 and summary["devices"] > 0
    assert summary["closures"] == []
    assert fork["s_from"] == pytest.approx(2.5) and fork["to_side"] == "entrance"


def test_branch_columns_line_up(client):
    data = client.get(f"/api/v1/lattices/{digest_of(client)}/branches/ring").json()
    table, floor = data["table"], data["floor"]
    n = len(table["id"])
    assert n == 41 and all(len(table[k]) == n for k in ("name", "kind", "element", "s", "length", "device"))
    assert all(len(column) == n for column in table["values"].values())
    assert {c["key"] for c in table["columns"]} == {
        "angle",
        "Kn1",
        "Kn1L",
        "Kn2",
        "Kn2L",
        "Kn0L",
        "Ks0L",
        "voltage",
    }
    first_quad = table["kind"].index("Quadrupole")
    assert table["values"]["Kn1"][first_quad] == 0.3 and table["values"]["Kn1L"][first_quad] == 0.15
    assert table["values"]["Kn1"][0] is None and table["device"][first_quad].endswith("-ps")
    sextupole = table["kind"].index("Sextupole")
    assert table["values"]["Kn2"][sextupole] == pytest.approx(0.5)
    assert len(floor["offsets"]) == n + 1 and floor["offsets"][-1] == len(floor["x"]) == len(floor["z"])
    bend = table["kind"].index("Bend")
    assert (
        floor["offsets"][bend + 1] - floor["offsets"][bend] == 80
    )  # pi/4 in 0.01 rad steps, plus the entrance
    end = floor["offsets"][-1] - 1
    start = floor["offsets"][0]
    assert (floor["x"][end], floor["z"][end]) == pytest.approx(
        (floor["x"][start], floor["z"][start]), abs=1e-6
    )


def test_occurrence_detail_with_devices_and_overrides(client):
    digest = digest_of(client)
    table = client.get(f"/api/v1/lattices/{digest}/branches/ring").json()["table"]
    quad = table["id"][table["kind"].index("Quadrupole")]
    detail = client.get(f"/api/v1/lattices/{digest}/occurrences/{quad}").json()
    [group] = detail["groups"]
    assert group["group"] == "MagneticMultipoleP"
    assert group["parameters"] == [
        {
            "name": "Kn1",
            "value": 0.3,
            "unit": "1/m^2",
            "description": "Normal quadrupole strength",
            "overridden": False,
        }
    ]
    [device] = detail["devices"]
    setting = next(c for c in device["channels"] if c["role"] == "setting")
    assert setting["parameter"] == "MagneticMultipoleP.Kn1" and setting["name"].endswith(":sp")
    assert detail["entrance"]["z"] < detail["exit"]["z"]
    assert client.get(f"/api/v1/lattices/{digest}/occurrences/nope").status_code == 404
    assert client.get(f"/api/v1/lattices/{digest}/branches/nope").status_code == 404
    assert client.get("/api/v1/lattices/0123/branches/ring").status_code == 404


def test_responses_are_gzipped_and_revalidated(client):
    url = f"/api/v1/lattices/{digest_of(client)}/branches/ring"
    first = client.get(url)
    assert first.headers["content-encoding"] == "gzip" and first.headers["cache-control"] == "no-cache"
    again = client.get(url, headers={"If-None-Match": first.headers["etag"]})
    assert again.status_code == 304 and again.content == b""


def test_download_round_trips(client):
    digest = digest_of(client)
    response = client.get(f"/api/v1/lattices/{digest}/lattice.json")
    assert Lattice.from_json(response.text) == build()
    assert "attachment" in response.headers["content-disposition"]


def test_a_changed_file_is_a_new_revision_and_the_old_one_stays(client, folder):
    old = digest_of(client)
    (folder / "demo_lattice.json").write_text(build(kn1=0.4).to_json())
    new = digest_of(client)
    assert new != old
    assert client.get(f"/api/v1/lattices/{old}").status_code == 200
    # The registry still names the same devices, so it stays paired with the new revision.
    assert client.get(f"/api/v1/lattices/{new}").json()["devices"] > 0


def test_concurrent_first_requests_build_a_response_once(folder, monkeypatch):
    calls, gate = [], threading.Event()
    original = RevisionViews.branch

    def slow_branch(self, branch_id):
        calls.append(branch_id)
        gate.wait(5)
        return original(self, branch_id)

    monkeypatch.setattr(RevisionViews, "branch", slow_branch)
    client = TestClient(create_app(Catalog([folder]), static_dir=None))
    url = f"/api/v1/lattices/{digest_of(client)}/branches/ring"
    with ThreadPoolExecutor(8) as pool:
        futures = [pool.submit(client.get, url) for _ in range(8)]
        gate.set()
        bodies = {f.result().content for f in futures}
    assert len(calls) == 1 and len(bodies) == 1


def test_a_registry_for_another_lattice_is_ignored(tmp_path):
    lattice = build()
    (tmp_path / "demo_lattice.json").write_text(lattice.to_json())
    other = LatticeBuilder("other", ids=SequentialIds())
    other.define("BEG", "BeginningEle", ReferenceP={"species_ref": "electron", "pc_ref": 1e9})
    other.define("Q", "Quadrupole", length=0.5)
    other.add_branch("ring", "periodic", ["BEG", "Q"])
    (tmp_path / "demo_registry.json").write_text(generate_registry(other.build()).model_dump_json())
    [revision] = Catalog([tmp_path]).current()
    assert revision.views.registry is None


def test_the_web_client_types_come_from_the_current_api():
    committed = ROOT / "web/src/api/openapi.json"
    if not committed.exists():
        pytest.skip("web client not present")
    current = create_app(Catalog([]), static_dir=None).openapi()
    assert json.loads(committed.read_text()) == current, "run `npm --prefix web run generate-types`"


def test_the_demo_machine_is_complete(tmp_path):
    from eicvibe.api.demo import write_demo

    [revision] = Catalog([write_demo(tmp_path)]).current()
    summary = revision.views.summary()
    assert {b.name for b in summary.branches} == {"ring", "injection", "extraction"}
    assert all(b.placed_by for b in summary.branches if b.name != "ring")
    assert summary.closures == [] and summary.diagnostics == [] and summary.devices > 0
    assert revision.views.registry.check_against(revision.lattice) == []
