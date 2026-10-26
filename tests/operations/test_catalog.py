"""The operations' catalog (ADR 0017): each declared once, light enough for the API to be made from it."""

import subprocess
import sys

import pytest
from pydantic import ValidationError

from eicvibe.operations.catalog import OPERATIONS, LocalBump, operation


def test_the_catalog_imports_no_engine():
    loaded = "any(m.startswith('eicvibe.engines') for m in sys.modules)"
    code = f"import sys, eicvibe.operations.catalog; print({loaded})"
    found = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert found.stdout.strip() == "False"


def test_every_operation_says_how_its_parameters_are_shown():
    assert list(OPERATIONS) == ["orbit_correction", "local_bump", "tune_move"]
    for declared in OPERATIONS.values():
        assert declared.verified_by in ("orbit", "tunes") and declared.machines == {"ring"}
        for name, field in declared.parameters.model_json_schema()["properties"].items():
            assert "x-form" in field, f"{declared.name}'s {name} has no x-form hints"
    assert operation("local_bump") is OPERATIONS["local_bump"]
    with pytest.raises(ValueError, match="no operation 'nope'; operations: orbit_correction, local_bump"):
        operation("nope")
    with pytest.raises(ValidationError, match="amplitude_mm"):
        LocalBump.model_validate({"plane": "x", "at": "bpm1"})  # how large is required
