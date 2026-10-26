"""The example notebooks (ADR 0016): their code compiles; with EICVIBE_RUN_NOTEBOOKS=1 those that need
no server run from start to end (minutes)."""

import ast
import json
import os
from pathlib import Path

import pytest

CHECKOUT = Path(__file__).parents[2]
NOTEBOOKS = sorted((CHECKOUT / "notebooks").glob("*.ipynb"))
MACHINE = Path(os.environ.get("EICVIBE_MACHINE", CHECKOUT.parent / "virtual_accelerator/EIN/machine"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_a_notebooks_code_compiles(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert code, f"{path.name} has no code"
    for k, cell in enumerate(code):
        ast.parse("".join(cell["source"]), filename=f"{path.name}, code cell {k + 1}")


@pytest.mark.skipif(not os.environ.get("EICVIBE_RUN_NOTEBOOKS"), reason="EICVIBE_RUN_NOTEBOOKS=1 runs them")
@pytest.mark.parametrize("name", ["01_rcs_ramp.ipynb", "03_bpm_data.ipynb"])
def test_a_notebook_runs_from_start_to_end(name, tmp_path, monkeypatch):
    nbclient = pytest.importorskip("nbclient")
    import nbformat

    if not MACHINE.is_dir():
        pytest.skip(f"no EIN machine folder at {MACHINE} (EICVIBE_MACHINE)")
    monkeypatch.setenv("EICVIBE_MACHINE", str(MACHINE))
    monkeypatch.setenv("EICVIBE_RUNS", str(tmp_path / "runs"))
    notebook = nbformat.read(CHECKOUT / "notebooks" / name, as_version=4)
    nbclient.NotebookClient(notebook, timeout=1800, resources={"metadata": {"path": str(tmp_path)}}).execute()
