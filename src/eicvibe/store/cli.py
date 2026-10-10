"""``eicvibe check FOLDER`` and ``eicvibe build FOLDER`` for machine folders."""

import argparse
import sys
from pathlib import Path

from .machine import build_machine, check_machine, load_machine, update_names
from .yamlio import FileError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eicvibe", description="Work with an EICViBE machine folder.")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="load the folder and report problems")
    check.add_argument("folder", type=Path)
    build = commands.add_parser("build", help="check, then write build/ (lattice, registry, manifest)")
    build.add_argument("folder", type=Path)
    args = parser.parse_args(argv)
    try:
        problems = check_machine(args.folder)
        machine = load_machine(args.folder)
    except FileError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    lattice, registry = machine.lattice, machine.registry
    occurrences = sum(len(b.occurrences) for b in lattice.branches)
    strings = sum(c.series for c in registry.circuits)
    print(
        f"{lattice.name}: {len(lattice.branches)} branches, {occurrences} elements, "
        f"{len(registry.devices)} devices, {len(registry.circuits)} power circuits ({strings} strings)"
    )
    for problem in problems:
        print(f"  {problem}")
    if args.command == "build":
        changes = update_names(args.folder)
        names = [c for c in changes if c.startswith("eic ")]
        for change in names[:20]:
            print(f"  {change}")
        if len(names) > 20:
            print(f"  … and {len(names) - 20} more EIC names")
        for change in changes:
            if change.startswith("ids "):
                print(f"  {change}")
        for path in build_machine(args.folder).values():
            print(f"wrote {path}")
        return 1 if [p for p in problems if not p.startswith("names: ")] else 0
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
