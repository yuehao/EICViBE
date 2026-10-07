"""The domain layer stays free of GUI, server and engine dependencies (roadmap sprint item 2)."""

import json
import subprocess
import sys

FORBIDDEN = {
    "PyQt5",
    "PyQt6",
    "PySide6",
    "matplotlib",
    "fastapi",
    "starlette",
    "sqlalchemy",
    "xtrack",
    "xpart",
    "xobjects",
    "cpymad",
    "juliacall",
    "julia",
    "numpy",
    "yaml",
}


def test_domain_imports_only_stdlib_and_pydantic():
    code = (
        "import json, sys, eicvibe.domain.lattice\n"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))"
    )
    loaded = set(json.loads(subprocess.check_output([sys.executable, "-c", code], text=True)))
    assert not loaded & FORBIDDEN, loaded & FORBIDDEN
    third_party = {
        m
        for m in loaded - set(sys.stdlib_module_names)
        if not m.startswith("_") and m not in {"eicvibe", "sitecustomize"}
    }
    assert third_party <= {
        "pydantic",
        "pydantic_core",
        "typing_extensions",
        "typing_inspection",
        "annotated_types",
    }
