"""Serve the lattice viewer: ``eicvibe-web LATTICE_OR_FOLDER... [--host H] [--port P] [--workers N]``.

For example, the EIN assembled by ``python -m eicvibe.machines.ein``::

    eicvibe-web /path/to/EIN/EICVIBE_lattice --open

``eicvibe-web --demo`` serves a small made-up machine instead.

The server listens on 127.0.0.1 by default. ``--host 0.0.0.0`` lets others
on the network use it; the API only reads lattices, but it has no
authentication yet, so do that only on a trusted network.
"""

import argparse
import json
import os
import sys
import tempfile
import webbrowser
from pathlib import Path

from .app import LATTICES_VARIABLE, create_app
from .catalog import Catalog, discover


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="eicvibe-web", description="Serve the EICViBE lattice viewer.")
    parser.add_argument("paths", nargs="*", help="lattice JSON files, or folders of *_lattice.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1, help="server processes; each holds its own cache")
    parser.add_argument("--open", action="store_true", help="open the viewer in a browser")
    parser.add_argument("--demo", action="store_true", help="also serve a small made-up machine")
    parser.add_argument("--openapi", action="store_true", help="print the OpenAPI document and exit")
    args = parser.parse_args(argv)

    if args.openapi:
        json.dump(create_app(Catalog([]), static_dir=None).openapi(), sys.stdout, indent=1, sort_keys=True)
        sys.stdout.write("\n")
        return
    if args.demo:
        from .demo import write_demo

        args.paths.append(str(write_demo(Path(tempfile.mkdtemp(prefix="eicvibe-demo-")))))
    if not args.paths:
        parser.error("give at least one lattice file or folder, or --demo")
    sources = discover(args.paths)  # fail early on a wrong path
    if not sources:
        parser.error(f"no *_lattice.json in {', '.join(args.paths)}")

    import uvicorn

    os.environ[LATTICES_VARIABLE] = os.pathsep.join(str(s.lattice) for s in sources)
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}/"
    print(f"Serving {', '.join(s.name for s in sources)} at {url}")
    if args.open:
        webbrowser.open(url)
    uvicorn.run(
        "eicvibe.api.app:app_from_environment",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
