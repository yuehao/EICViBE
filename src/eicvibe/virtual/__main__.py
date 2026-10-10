"""Run a virtual ring or route and look at its BPM data: ``eicvibe-virtual``.

    eicvibe-virtual ring MACHINE --branch bar --out runs/bar1 [--turns N] [--kick-x RAD] [--kick-every N]
                    [--set rfc:RFP.phase=3.14159] [--engine trackpad] [--fidelity standard] [--particles 100]
    eicvibe-virtual route MACHINE --route bar --mode bar_mode --out runs/bar-route [--shots N] [--rate 2]
    eicvibe-virtual route LATTICE --through ltr,rcs --out runs/direct
    eicvibe-virtual show runs/bar1 [--turns 1024]
    eicvibe-virtual export runs/bar1 acquisition.sdds [--start TURN] [--turns 1024]

``ring`` tracks until N turns (or Ctrl-C), recording every BPM every turn into
the run folder; ``route`` sends shots along a route of the machine's
``routes.yaml`` (or through the branches given), recording every BPM every
shot; ``show`` reads a run's folder, also while the run goes on; ``export``
writes turns (or shots) in RHIC's turn-by-turn file layout, which tools made
for RHIC's files read.
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

from eicvibe.diagnostics import amplitudes, closed_orbit, tune, write_rhic
from eicvibe.engines.contracts import Fidelity

from .history import BpmHistory
from .ring import Beam, RingRun
from .route import RouteRun


def _settings(lattice, branches: list[str], items: list[str]) -> dict[tuple[str, str], float]:
    values = {}
    for item in items:
        target, _, value = item.partition("=")
        name, _, parameter = target.partition(":")
        if not value or not parameter:
            raise SystemExit(f"--set wants OCCURRENCE:Group.parameter=value, not {item!r}")
        for branch in branches:
            try:
                values[(lattice.placement(name, branch=branch).id, parameter)] = float(value)
                break
            except KeyError:
                continue
        else:
            raise SystemExit(f"--set: no {name} in {', '.join(branches)}")
    return values


def _values(args: argparse.Namespace, lattice, branches: list[str]) -> dict[tuple[str, str], float]:
    """The settings a run starts with: an operating mode's (settings/<mode>.yaml), then each --set."""
    values = {}
    if args.mode:
        from eicvibe.store import is_machine, read_settings

        if not is_machine(args.machine):
            raise SystemExit("--mode needs a machine folder, whose settings/ holds the modes")
        values = {
            (s.occurrence, s.parameter): s.value for s in read_settings(args.machine, args.mode, lattice)
        }
    return values | _settings(lattice, branches, args.set)


def _load(path: Path):
    """A machine folder's, or a lattice file's, lattice, registry and routes.

    A lattice file ``<stem>_lattice.json`` takes the ``<stem>_registry.json`` and ``<stem>_routes.yaml``
    beside it, when they are there.
    """
    from eicvibe.domain.lattice import Lattice
    from eicvibe.domain.machine import MachineRegistry
    from eicvibe.store import is_machine, load_machine
    from eicvibe.store.machine import read_routes_file

    if is_machine(path):
        machine = load_machine(path)
        return machine.lattice, machine.registry, list(machine.routes)
    if not path.is_file():
        raise SystemExit(f"{path} is neither a machine folder nor a lattice file")
    lattice = Lattice.from_json(path.read_text())
    stem = path.name.removesuffix("_lattice.json")
    registry_file = path.with_name(f"{stem}_registry.json")
    registry = (
        MachineRegistry.model_validate_json(registry_file.read_text()) if registry_file.is_file() else None
    )
    return lattice, registry, list(read_routes_file(path.with_name(f"{stem}_routes.yaml"), lattice))


def ring(args: argparse.Namespace) -> None:
    lattice, registry, _ = _load(args.machine)
    run = RingRun(
        lattice,
        args.branch,
        args.out,
        engine=args.engine,
        fidelity=Fidelity(args.fidelity),
        beam=Beam(particles=args.particles),
        chunk=args.chunk,
        ramp_turns=args.ramp_turns,
        buffer_turns=args.buffer,
        slow_period=args.slow,
        values=_values(args, lattice, [args.branch]),
        registry=registry,
        turns=args.turns,
    )
    stopping = False

    def interrupt(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, interrupt)
    print(f"starting {args.branch} with {run.engine} in {args.out} ...", flush=True)
    kicking = bool(args.kick_x or args.kick_y)
    if kicking:
        run.kick(args.kick_x, args.kick_y)  # queued: it comes before the first turn
    run.start()
    matched = run.history.info.beam["matched"]
    print(f"bunch matched to {matched['to']}: {_matched_text(matched)}", flush=True)
    for warning in run.warnings:
        print(f"warning: {warning}", flush=True)
    started, shown = time.monotonic(), 0
    next_kick = args.kick_every if kicking and args.kick_every else None
    try:
        while not stopping and not run.done:
            if run.error is not None:
                raise SystemExit(f"the run stopped: {run.error}")
            if next_kick is not None and run.turn >= next_kick:
                run.kick(args.kick_x, args.kick_y)
                next_kick += args.kick_every
            if run.turn != shown:
                shown = run.turn
                rate = shown / max(time.monotonic() - started, 1e-9)
                print(f"turn {shown:>10}  ({rate:,.0f} turns/s)", flush=True)
            time.sleep(0.2)
    finally:
        run.stop()
    print(f"stopped at turn {run.turn}; read it with: eicvibe-virtual show {args.out}")


def route(args: argparse.Namespace) -> None:
    from eicvibe.domain.lattice import route_through

    lattice, registry, routes = _load(args.machine)
    if args.through:
        chosen = route_through(lattice, args.route or "route", args.through.split(","))
    else:
        named = {r.name: r for r in routes}
        if args.route not in named:
            known = ", ".join(named) or "none; give --through BRANCH,BRANCH,..."
            raise SystemExit(f"no route {args.route!r} in {args.machine} (routes: {known})")
        chosen = named[args.route]
    run = RouteRun(
        lattice,
        chosen,
        args.out,
        engine=args.engine,
        fidelity=Fidelity(args.fidelity),
        beam=Beam(particles=args.particles),
        rate=args.rate,
        values=_values(args, lattice, [s.branch for s in chosen.segments]),
        registry=registry,
        shots=args.shots,
    )
    stopping = False

    def interrupt(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, interrupt)
    print(f"starting route {chosen.name} ({', '.join(chosen.branches)}) in {args.out} ...", flush=True)
    run.start()
    matched = run.history.info.beam["matched"]
    print(f"bunch matched to {matched['to']}: {_matched_text(matched)}", flush=True)
    shown = 0
    try:
        while not stopping and not run.done:
            if run.error is not None:
                raise SystemExit(f"the run stopped: {run.error}")
            if run.shot != shown:
                shown = run.shot
                print(f"shot {shown:>8}", flush=True)
            time.sleep(0.2)
    finally:
        run.stop()
    print(f"stopped at shot {run.shot}; read it with: eicvibe-virtual show {args.out}")


def show(args: argparse.Namespace) -> None:
    history = BpmHistory.open(args.run)
    info = history.info
    first, end = history.span()
    print(
        f"run {info.id}: {info.lattice}/{info.branch}, {info.engine} ({info.fidelity}), "
        f"started {info.start:%Y-%m-%d %H:%M:%S} UTC"
    )
    unit = info.unit
    if info.kind == "route":
        print(f"{len(info.bpms)} BPMs; shots {first} to {end} kept; {1 / info.revolution_period:.4g} shots/s")
    else:
        kept = (
            f"the buffer holds turns {first} to {end}"
            if info.buffer_turns
            else f"turns {first} to {end} kept"
        )
        print(f"{len(info.bpms)} BPMs; {kept}; revolution {info.revolution_period * 1e6:.4f} us")
        slow = history.orbit()
        if len(slow.turn):
            every = f"every {info.slow_period:g} s" if info.slow_period else "one per chunk"
            print(f"slow orbit: {len(slow.turn)} samples, {every}")
        for capture in history.captures():
            partial = ", partial" if capture.get("partial") else ""
            print(
                f"capture {capture['id']}: {capture['kind']} at turn {capture['trigger']}, "
                f"turns {capture['first']} to {capture['first'] + capture['turns']}{partial}"
            )
    if matched := info.beam.get("matched"):
        print(f"bunch matched to {matched['to']}: {_matched_text(matched)}")
    for event in history.events():
        detail = {k: v for k, v in event.items() if k not in ("turn", "kind", "wall")}
        print(f"  {unit} {event['turn']:>10}  {event['kind']}  {detail if detail else ''}")
    if end == first:
        return
    tbt = history.turn_by_turn(turns=args.turns)
    if info.kind == "route":
        x, y = tbt.x.position, tbt.y.position
        print(f"latest {tbt.turns} shots from shot {tbt.first_turn}:")
        print(
            f"  trajectory rms x {np.nanstd(np.nanmean(x, 1)):.4f} mm  y {np.nanstd(np.nanmean(y, 1)):.4f} mm"
        )
        print(
            f"  shot-to-shot jitter x {np.nanmedian(np.nanstd(x, 1)):.4f} mm  "
            f"y {np.nanmedian(np.nanstd(y, 1)):.4f} mm"
        )
        print(f"  readings with no beam {np.isnan(x).mean():.1%}")
        return
    x, y = closed_orbit(tbt)
    print(f"latest {tbt.turns} turns from turn {tbt.first_turn}:")
    print(f"  tunes x {tune(tbt, 'x'):.5f}  y {tune(tbt, 'y'):.5f}")
    print(f"  orbit rms x {np.nanstd(x):.4f} mm  y {np.nanstd(y):.4f} mm")
    print(
        f"  oscillation amplitude x {np.nanmedian(amplitudes(tbt.x)):.4f} mm  "
        f"y {np.nanmedian(amplitudes(tbt.y)):.4f} mm"
    )


def _matched_text(m: dict) -> str:
    def g(v: float, digits: int = 4) -> str:
        return f"{v + 0.0:.{digits}g}" if abs(v) > 1e-12 else "0"  # no "-0"

    parts = [
        f"beta {g(m['beta_x'], 5)}, {g(m['beta_y'], 5)} m",
        f"alpha {g(m['alpha_x'])}, {g(m['alpha_y'])}",
        f"eta_x {g(m['eta_x'])} m",
    ]
    if m.get("x") or m.get("px") or m.get("y") or m.get("py"):
        parts.append(f"orbit x {m['x'] * 1e3:.4g} mm, px {m['px'] * 1e3:.4g} mrad, y {m['y'] * 1e3:.4g} mm")
    if "synchrotron_tune" in m:
        parts.append(f"Qs {m['synchrotron_tune']:.5f}, sigma_zeta {m['sigma_zeta'] * 1e3:.4g} mm")
    return "; ".join(parts)


def export(args: argparse.Namespace) -> None:
    history = BpmHistory.open(args.run)
    tbt = history.capture(args.capture) if args.capture else history.turn_by_turn(args.start, args.turns)
    write_rhic(tbt, args.output)
    print(
        f"wrote turns {tbt.first_turn} to {tbt.first_turn + tbt.turns} "
        f"of {len(tbt.x.bpms)} BPMs to {args.output}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="eicvibe-virtual", description="Virtual rings and routes and their BPM data."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    r = commands.add_parser("ring", help="track a ring turn after turn, recording its BPMs")
    r.add_argument("machine", type=Path, help="machine folder or lattice file")
    r.add_argument("--branch", required=True)
    r.add_argument("--out", type=Path, required=True, help="the run's folder (new)")
    r.add_argument("--engine", choices=["xsuite", "trackpad"], default=None)
    r.add_argument("--fidelity", choices=["fast", "standard", "precise"], default="standard")
    r.add_argument("--particles", type=int, default=100)
    r.add_argument(
        "--chunk", type=int, default=1024, help="most turns an engine call tracks (each is sized to ~0.25 s)"
    )
    r.add_argument(
        "--ramp-turns",
        type=int,
        default=1024,
        help="turns a setting change ramps over (half cosine); 0 steps",
    )
    r.add_argument(
        "--buffer", type=int, default=4096, help="turns the BPMs' buffer keeps (and a capture holds)"
    )
    r.add_argument("--slow", type=float, default=1.0, help="the slow orbit's sample, s of wall clock")
    r.add_argument(
        "--turns", type=int, default=None, help="stop after this many turns; default: until Ctrl-C"
    )
    r.add_argument("--kick-x", type=float, default=0.0, help="kick at the start, rad")
    r.add_argument("--kick-y", type=float, default=0.0)
    r.add_argument("--kick-every", type=int, default=0, help="repeat the kick every N turns")
    r.add_argument("--mode", default=None, help="an operating mode's settings, settings/<mode>.yaml")
    r.add_argument("--set", action="append", default=[], help="OCCURRENCE:Group.parameter=value")
    r.set_defaults(handler=ring)
    o = commands.add_parser("route", help="send shots along a route, recording its BPMs")
    o.add_argument("machine", type=Path, help="machine folder or lattice file")
    o.add_argument("--route", default=None, help="a route of the machine's routes.yaml")
    o.add_argument("--through", default=None, help="or the branches it passes, in beam order: ltr,rcs")
    o.add_argument("--out", type=Path, required=True, help="the run's folder (new)")
    o.add_argument("--engine", choices=["xsuite", "trackpad"], default=None)
    o.add_argument("--fidelity", choices=["fast", "standard", "precise"], default="standard")
    o.add_argument("--particles", type=int, default=100)
    o.add_argument("--rate", type=float, default=2.0, help="shots per second, at most")
    o.add_argument(
        "--shots", type=int, default=None, help="stop after this many shots; default: until Ctrl-C"
    )
    o.add_argument("--mode", default=None, help="an operating mode's settings, e.g. bar_mode")
    o.add_argument("--set", action="append", default=[], help="OCCURRENCE:Group.parameter=value")
    o.set_defaults(handler=route)
    s = commands.add_parser("show", help="what a run has recorded")
    s.add_argument("run", type=Path)
    s.add_argument("--turns", type=int, default=1024)
    s.set_defaults(handler=show)
    e = commands.add_parser("export", help="turns of a run as a RHIC turn-by-turn file (SDDS)")
    e.add_argument("run", type=Path)
    e.add_argument("output", type=Path)
    e.add_argument("--start", type=int, default=None, help="first turn; default: the latest turns")
    e.add_argument("--turns", type=int, default=1024)
    e.add_argument("--capture", default=None, help="a capture's id (eicvibe-virtual show lists them)")
    e.set_defaults(handler=export)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
