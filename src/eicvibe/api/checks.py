"""A ring run's tracking checked against the model: what its BPMs measured, what the optics give.

The measurement is the one made on a real ring after a pinger's kick (ADR 0007):
the tune from the spectrum of the turn-by-turn readings, the phase advance between
BPMs from the phase of the tune line, beta from its amplitude, the closed orbit from
the mean reading, and how tune and amplitude drift as the bunch decoheres. The
model is the engine's optics of the run's lattice with the settings the run had at
the start of the stretch looked at: those it started with, then its settings events.
Where they disagree, the tracking, the matching or the BPMs are wrong, or the beam
is not where linear optics hold: a large kick moves the tune with the amplitude,
which the detuning curve (single particles tracked by the same engine) shows.
"""

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from eicvibe.diagnostics import (
    PlaneData,
    TurnByTurn,
    amplitudes,
    beta_beating,
    closed_orbit,
    harmonics,
    phase_errors,
    spectrum,
    tune,
    tune_drift,
    tune_of,
)
from eicvibe.engines.contracts import Effect, Fidelity, OpticsRequest, Options, OrbitRequest, TrackRequest
from eicvibe.engines.worker import EngineWorker
from eicvibe.virtual import BpmHistory

from . import schemas
from .catalog import Catalog
from .runs import RunCatalog

#: Turns of single-particle tracking for the detuning curve, and its particles.
DETUNING_TURNS = 512
DETUNING_PARTICLES = 8
#: Below this many times the BPM resolution, a plane's oscillation is noise: no phase or beta from it.
SIGNAL = 5.0


@dataclass(frozen=True)
class _Model:
    tune: dict[str, float]
    beta: dict[str, np.ndarray]
    mu: dict[str, np.ndarray]
    orbit: dict[str, np.ndarray]
    """mm"""
    start: dict[str, float]
    """Beta at the ring's start, m."""
    detuning: schemas.Detuning | None


def _round(values, digits: int = 6) -> list[float | None]:
    return [None if v is None or not np.isfinite(v) else float(f"{v:.{digits}g}") for v in values]


def _value(v: float | None, digits: int = 7) -> float | None:
    return None if v is None or not np.isfinite(v) else float(f"{v:.{digits}g}")


class RunChecks:
    """Checks of ring runs, computed with the engine workers and kept for reuse."""

    def __init__(
        self, runs: RunCatalog, catalog: Catalog, worker: Callable[[str], EngineWorker], *, keep: int = 16
    ) -> None:
        self._runs, self._catalog, self._worker = runs, catalog, worker
        self._keep = keep
        self._lock = threading.Lock()
        self._models: OrderedDict[tuple, _Model] = OrderedDict()

    def check(
        self,
        run_id: str,
        source: str = "kick",
        first: int | None = None,
        turns: int = 1024,
        engine: str | None = None,
        detuning: bool = False,
    ) -> schemas.RunCheck:
        """The run's readings over ``turns`` turns from ``first`` against the model of ``engine``.

        ``source`` is ``kick`` (the latest kick's capture, from the kick on), ``buffer`` (the
        latest turns kept) or ``capture:<id>``. Raises KeyError for an unknown run, ValueError
        for data that cannot be checked.
        """
        history = self._runs.history(run_id)
        info = history.info
        if info.kind != "ring":
            raise ValueError(
                "a check against the optics needs a ring's turns; a route's run records single passes"
            )
        asked = source
        tbt, source, start, trigger = _stretch(history, source, first, turns)
        if tbt.turns < 64:
            raise ValueError(f"{tbt.turns} turns are too few to measure a tune; 64 at least")
        notes: list[str] = []
        if asked == "kick" and source == "buffer":
            notes.append(
                "No kick has been captured yet, so these are the latest turns. Kick the beam, and check "
                "again once its capture is saved (an eighth of the buffer before the kick, the rest after)."
            )
        values, settings = _settings(history, start, start + tbt.turns, notes)
        revision = self._catalog.get(info.lattice_digest)
        if (
            revision is None
        ):  # the lattice was imported again since: its current revision, if the ring is there
            revision = next(
                (
                    r
                    for r in self._catalog.current()
                    if r.lattice.name == info.lattice and info.branch_id in {b.id for b in r.lattice.branches}
                ),
                None,
            )
            if revision is None:
                raise ValueError(
                    f"the run tracked a revision of {info.lattice} that this server does not have"
                )
            notes.append(
                f"The run tracked an earlier revision of {info.lattice}; the model has the current one "
                f"({revision.digest[:19]}), whose {info.branch} may differ."
            )
        engine = engine or info.engine
        model = self._model(revision.lattice, info, engine, values, tbt if detuning else None)

        window = min(256, max(64, tbt.turns // 8))
        drift_turns, _, _ = tune_drift(tbt.x, window, window // 2)
        frequencies, _ = spectrum(tbt.x)
        planes = {}
        for name in ("x", "y"):
            plane: PlaneData = tbt.plane(name)
            line = tune(tbt, name)
            amplitude = float(np.nanmedian(amplitudes(plane)))
            _, power = spectrum(plane)
            _, q, a = tune_drift(plane, window, window // 2)
            measured_tune = phase = beat = None
            if amplitude > SIGNAL * max(info.resolution, 1e-6):
                size, angle = harmonics(plane, line)
                errors, mirrored = phase_errors(angle, model.mu[name])
                measured_tune = 1.0 - line if mirrored else line
                phase = errors * 360.0
                beat = beta_beating(size, model.beta[name])
            else:
                notes.append(
                    f"The beam hardly oscillates in {name} ({amplitude:.3g} mm, BPM resolution "
                    f"{info.resolution:g} mm): no tune, phase or beta from it. A kick in {name} gives them."
                )
            full = None
            if measured_tune is not None:
                integer = np.floor(model.tune[name])
                full = min(
                    (integer + k + measured_tune for k in (-1, 0, 1)), key=lambda v: abs(v - model.tune[name])
                )
            matched = (info.beam.get("matched") or {}).get("tunes") or [None, None]
            planes[name] = schemas.PlaneCheck(
                measured_tune=_value(measured_tune),
                measured_full=_value(full),
                model_tune=_value(model.tune[name]),
                map_tune=_value(matched[0 if name == "x" else 1]),
                amplitude=_value(amplitude, 5),
                initial_amplitude=_value(float(np.nanmedian(amplitudes(plane.window(0, 32)))), 5),
                spectrum=_round(power, 4),
                phase_error=[] if phase is None else _round(phase, 4),
                phase_rms=None if phase is None else _value(float(np.sqrt(np.nanmean(phase**2))), 4),
                beta_beating=[] if beat is None else _round(beat, 4),
                beta_beating_rms=None if beat is None else _value(float(np.sqrt(np.nanmean(beat**2))), 4),
                orbit=_round(closed_orbit(tbt)[0 if name == "x" else 1], 5),
                model_orbit=_round(model.orbit[name], 5),
                drift_tune=_round(q, 6),
                drift_amplitude=_round(a, 4),
            )
        if engine != info.engine:
            notes.append(f"The model is {engine}'s; the run tracked with {info.engine}.")
        return schemas.RunCheck(
            run=run_id,
            engine=engine,
            fidelity=info.fidelity,
            source=source,
            first_turn=start,
            turns=tbt.turns,
            trigger=trigger,
            settings=settings,
            bpms=list(tbt.x.bpms),
            s=_round(tbt.x.s, 9),
            frequencies=_round(frequencies, 6),
            drift_turns=[int(start + t) for t in drift_turns],
            drift_window=window,
            synchrotron_tune=_value((info.beam.get("matched") or {}).get("synchrotron_tune")),
            x=planes["x"],
            y=planes["y"],
            detuning=model.detuning,
            notes=notes,
        )

    def _model(self, lattice, info, engine: str, values: dict, tbt: TurnByTurn | None) -> _Model:
        """The model's optics and closed orbit at the run's BPMs; with ``tbt``, its detuning too."""
        key = (
            lattice.digest,
            info.branch_id,
            engine,
            info.fidelity,
            tuple(info.effects),
            tuple(sorted(values.items())),
        )
        with self._lock:
            found = self._models.get(key)
        if found is not None and (tbt is None or found.detuning is not None):
            return found
        flat = Options(fidelity=Fidelity(info.fidelity))
        run = Options(fidelity=Fidelity(info.fidelity), effects=frozenset(Effect(e) for e in info.effects))
        session = self._worker(engine).open_session(lattice, info.branch_id, run, values)
        try:
            optics = session.optics(OpticsRequest(options=flat))
            orbit = session.orbit(OrbitRequest(options=run))
            where = {o: i for i, o in enumerate(optics.occurrences)}
            at = [where[o] for o in info.occurrences]
            closed = {o: i for i, o in enumerate(orbit.occurrences)}
            model = _Model(
                tune={"x": optics.tune_x, "y": optics.tune_y},
                beta={k: np.asarray(getattr(optics, f"beta_{k}"))[at] for k in "xy"},
                mu={k: np.asarray(getattr(optics, f"mu_{k}"))[at] for k in "xy"},
                orbit={
                    k: 1e3 * np.asarray(getattr(orbit, k))[[closed[o] for o in info.occurrences]]
                    for k in "xy"
                },
                start={"x": optics.start.beta_x, "y": optics.start.beta_y},
                detuning=None,
            )
            if tbt is not None:
                model = replace(model, detuning=_detuning(session, info, model, orbit, run, tbt))
        finally:
            session.close()
        with self._lock:
            self._models[key] = model
            self._models.move_to_end(key)
            while len(self._models) > self._keep:
                self._models.popitem(last=False)
        return model


def _stretch(
    history: BpmHistory, source: str, first: int | None, turns: int
) -> tuple[TurnByTurn, str, int, int | None]:
    """The readings to check, what they are, their first turn and a capture's trigger."""
    captures = history.captures()
    if source == "kick":
        kicks = [c for c in captures if c["kind"] == "kick"]
        if not kicks:
            source = "buffer"
        else:
            source = f"capture:{kicks[-1]['id']}"
    if source == "buffer":
        first_kept, end = history.span()
        count = min(turns, end - first_kept)
        start = end - count if first is None else max(first, first_kept)
        count = min(count, end - start)
        return history.turn_by_turn(start, count), source, start, None
    entry = next((c for c in captures if c["id"] == source.removeprefix("capture:")), None)
    if entry is None:
        raise ValueError(f"no capture {source.removeprefix('capture:')}")
    begin = entry["trigger"] if entry["kind"] == "kick" else entry["first"]
    start = begin if first is None else first
    offset = start - entry["first"]
    if not 0 <= offset < entry["turns"]:
        raise ValueError(f"the capture holds turns {entry['first']} to {entry['first'] + entry['turns']}")
    count = min(turns, entry["turns"] - offset)
    return history.capture(entry["id"]).window(offset, count), source, start, entry["trigger"]


def _settings(history: BpmHistory, start: int, end: int, notes: list[str]) -> tuple[dict, str]:
    """The run's settings at turn ``start``, and what they are, in words."""
    info = history.info
    values = {tuple(k.split(" ", 1)): v for k, v in info.values.items()}
    recorded = "values" in info.model_fields_set
    changes = 0
    for event in history.events():
        if event["kind"] != "settings":
            continue
        update = {tuple(k.split(" ", 1)): v for k, v in event.get("values", {}).items()}
        if event["turn"] <= start:
            values |= update
            changes += 1
            if event["turn"] + event.get("ramp_turns", 0) > start:
                notes.append(
                    f"A setting changed at turn {event['turn']} was still ramping when the stretch began; "
                    "the model has its final value."
                )
        elif event["turn"] < end:
            notes.append(
                f"Settings changed at turn {event['turn']}, inside the stretch: the model has those before."
            )
    for event in history.events():
        if event["kind"] == "kick" and start < event["turn"] < end:
            notes.append(f"Another kick at turn {event['turn']}, inside the stretch.")
    if not recorded:
        notes.append(
            "This run does not record the settings it started with (runs before 2026-10-10): the model "
            "has the lattice's values, with the run's later changes."
        )
    words = (
        f"the run's at turn {start}: {len(values)} beyond the lattice's"
        + (f", {changes} change{'s' if changes != 1 else ''} during the run" if changes else "")
        if recorded
        else "the lattice's (the run does not record its own)"
    )
    return values, words


def _detuning(session, info, model: _Model, orbit, options: Options, tbt: TurnByTurn) -> schemas.Detuning:
    """Single particles from the closed orbit, kicked in the plane the beam oscillates in most."""
    action = {k: float(np.nanmedian(amplitudes(tbt.plane(k)) ** 2 / model.beta[k])) for k in "xy"}
    plane = max(action, key=action.get)
    # The largest kick reaches a quarter more than the largest amplitude measured (1 mm at least):
    # a kick a at the start gives a sqrt(beta_start beta) at a BPM.
    measured = float(np.nanmax(amplitudes(tbt.plane(plane).window(0, 64))))
    beta_max = float(np.nanmax(model.beta[plane]))
    largest = 1.25e-3 * max(measured, 1.0) / np.sqrt(model.start[plane] * beta_max)
    start = [orbit.x[0], orbit.px[0], orbit.y[0], orbit.py[0], orbit.zeta[0], orbit.delta[0]]
    other = 1 if plane == "y" else 3
    particles = []
    for k in range(1, DETUNING_PARTICLES + 1):
        p = list(start)
        p[1 if plane == "x" else 3] += largest * k / DETUNING_PARTICLES
        p[other] += 1e-7  # just enough to see the other plane's tune
        particles.append(tuple(float(v) for v in p))
    result = session.track(
        TrackRequest(
            options=options, particles=tuple(particles), turns=DETUNING_TURNS, observe=tuple(info.occurrences)
        )
    )
    xs, ys = np.asarray(result.x, dtype=float), np.asarray(result.y, dtype=float)  # [point, turn, particle]
    amplitude, qx, qy = [], [], []
    for i in range(len(particles)):
        x, y = xs[:, :, i], ys[:, :, i]
        if np.isnan(x).any():
            break  # lost: larger kicks are too
        signal = x if plane == "x" else y
        amplitude.append(float(np.nanmedian(np.sqrt(2) * np.std(signal, axis=1)) * 1e3))
        qx.append(float(np.median([tune_of(row) for row in x])))
        qy.append(float(np.median([tune_of(row) for row in y])))
    return schemas.Detuning(
        plane=plane,
        amplitude=_round(amplitude, 5),
        tune_x=_round(qx, 6),
        tune_y=_round(qy, 6),
        turns=DETUNING_TURNS,
    )
