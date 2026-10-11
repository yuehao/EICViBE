"""The viewer's AI features (ADR 0009): what the local model is told, and one task at a time.

The model only puts these facts into words. Every number in them comes from
EICViBE (the check, the optics, the lattice), and the comparisons that matter are
made here, in code, so the model never has to calculate: the local 27B model got
an amplitude wrong by a factor of six when it did (ADR 0009). The instructions
come first and stay the same, so the model server's prompt cache keeps them.
"""

import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator

from eicvibe.ai import AiUnavailable, LocalModel, Message, Model
from eicvibe.domain.lattice import Lattice

from . import schemas

SYSTEM = (
    "You help accelerator physicists and engineers who use EICViBE, a virtual accelerator and lattice "
    "viewer. Answer from the facts you are given: EICViBE computed every number in them, and they are "
    "right. Do not calculate new numbers beyond comparing the ones given, and do not invent elements, "
    "settings or values. If the facts do not answer the question, say so and say where in EICViBE to "
    "look. Do not judge whether a value is good, bad or corrected unless the facts give the criterion; "
    "otherwise give the value and say what it would be compared with. You cannot change anything; never "
    "say you did. Write plain sentences: no headings, no lists, no formulas in LaTeX."
)


def _n(value, digits: int = 5) -> str:
    if value is None:
        return "unknown"
    return f"{value:.{digits}g}"


def check_messages(check: schemas.RunCheck) -> list[Message]:
    """Ask for a short reading of a ring run's check against its model."""
    lines = [
        f"Run {check.run}: a ring tracked with the {check.engine} engine at {check.fidelity} fidelity; "
        f"the model is that engine's optics with {check.settings}.",
        f"Data: turns {check.first_turn} to {check.first_turn + check.turns}"
        + (f", from the capture of the kick at turn {check.trigger}" if check.trigger is not None else "")
        + ".",
    ]
    if check.kick:
        lines.append(
            "The kick: " + ", ".join(f"{k} {1e3 * v:g} mrad" for k, v in check.kick.items() if v) + "."
        )
    for name in ("x", "y"):
        p: schemas.PlaneCheck = getattr(check, name)
        if p.measured_full is None:
            lines.append(
                f"Plane {name}: the beam hardly oscillates ({_n(p.amplitude, 3)} mm), so no tune, phase or "
                f"beta was measured; the model's tune is {_n(p.model_tune, 7)}."
            )
            continue
        difference = p.measured_full - p.model_tune
        lines.append(
            f"Plane {name}: measured tune {p.measured_full:.4f}, model {p.model_tune:.4f}, difference "
            f"{difference:+.4f} ({'within' if abs(difference) < 1e-3 else 'outside'} 0.001); the run's own "
            f"one-turn map gave {_n(p.map_tune, 5)}. Oscillation amplitude {_n(p.initial_amplitude, 3)} mm "
            f"right after the kick"
            + (
                f" ({_n(p.kick_sigma, 3)} beam sizes; the beam size is {_n(p.beam_size, 3)} mm)"
                if p.kick_sigma
                else ""
            )
            + f", {_n(p.amplitude, 3)} mm over the turns analysed. Phase advance error "
            f"{_n(p.phase_rms, 3)} degrees rms; beta from the amplitudes "
            f"{_n(100 * (p.beta_beating_rms or 0), 2)} % rms (BPM gain errors alone give about 1 to 2 %)."
        )
    d = check.detuning
    if d is not None and d.amplitude:
        points = "; ".join(
            f"{a:.3g} mm: x {qx:.4f}, y {qy:.4f}"
            for a, qx, qy in zip(d.amplitude, d.tune_x, d.tune_y, strict=True)
        )
        lines.append(f"The model's tune against amplitude (single particles kicked in {d.plane}): {points}.")
        if d.expected_x is not None:
            measured = getattr(check, d.plane)
            lines.append(
                f"At the amplitude measured after the kick the model gives x {d.expected_x:.4f} and y "
                f"{d.expected_y:.4f}; the beam's measured {d.plane} tune is {_n(measured.measured_tune, 5)}."
            )
    else:
        lines.append("The model's tune against amplitude was not computed (the 'tune vs amplitude' choice).")
    if check.synchrotron_tune:
        lines.append(f"Synchrotron tune of the run's one-turn map: {check.synchrotron_tune:.4f}.")
    lines += [f"Note: {note}" for note in check.notes]
    findings, steps = _verdict(check)
    lines.append("EICViBE's reading of these numbers: " + " ".join(findings))
    lines.append(
        "Next steps that EICViBE offers: " + " ".join(f"({i}) {step}" for i, step in enumerate(steps, 1))
    )
    question = (
        "In at most 90 words, explain EICViBE's reading to the person in plain words, then recommend one of "
        "the next steps offered (no other)."
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n".join(lines) + "\n\n" + question},
    ]


def _line(q: float) -> float:
    """Where a tune shows in a spectrum: q and 1 - q look alike."""
    q %= 1.0
    return min(q, 1.0 - q)


def _verdict(check: schemas.RunCheck) -> tuple[list[str], list[str]]:
    """What the check's numbers say, decided here in code, and the next steps EICViBE has for it."""
    findings, steps = [], []
    d = check.detuning
    large = [n for n in ("x", "y") if (getattr(check, n).kick_sigma or 0) > 10]
    for name in ("x", "y"):
        p: schemas.PlaneCheck = getattr(check, name)
        if p.measured_full is None or p.model_tune is None:
            continue
        difference = p.measured_full - p.model_tune
        expected = getattr(d, f"expected_{name}") if d is not None else None
        if abs(difference) < 1e-3:
            findings.append(
                f"In {name} the measured tune agrees with the model's: the tracking is consistent."
            )
        elif expected is not None and abs(_line(p.measured_tune) - expected) < 1e-3:
            findings.append(
                f"In {name} the measured tune differs from the model's linear tune by {difference:+.4f} "
                f"but equals the model's tune at the amplitude measured ({expected:.4f}): the difference "
                "is amplitude-dependent detuning from a large kick, so the tracking agrees with the model. "
                "The phase-advance error follows from the different tune, not from the BPMs."
            )
        elif large and d is None:
            findings.append(
                f"In {name} the measured tune differs from the model's linear tune by {difference:+.4f}. "
                "The kick is large, so amplitude-dependent detuning may explain it, which the model's tune "
                "against amplitude would show."
            )
            step = "Tick 'tune vs amplitude' on the Check tab and check again."
            if step not in steps:
                steps.append(step)
        else:
            findings.append(
                f"In {name} the measured tune differs from the model's by {difference:+.4f}, which detuning "
                "does not explain: the model may not have the run's settings, or the engines differ."
            )
            steps.append("Choose the other engine as the model, to see whether the engines differ.")
    for name in large:
        p = getattr(check, name)
        if check.kick and check.kick.get(name) and p.kick_sigma:
            small = abs(check.kick[name]) * 3.0 / p.kick_sigma
            steps.append(
                f"Kick again in {name} by about {1e3 * small:.2g} mrad (three beam sizes instead of "
                f"{p.kick_sigma:.3g}) from the administration page, and check that kick: it measures the "
                "linear tune."
            )
    if any("does not record the settings" in note for note in check.notes):
        steps.append("Start a new run of the machine: new runs record their settings, so the model has them.")
    if not findings:
        findings.append("The beam hardly oscillates, so there is nothing to compare.")
        steps.append(
            "Kick the beam by a few beam sizes from the administration page, then check its capture."
        )
    steps.append(
        "Read the Check tab's charts: the spectrum, the tune over the turns, and phase and beta per BPM."
    )
    return findings, steps


def view_facts(
    lattice: Lattice,
    branch: str,
    *,
    route: schemas.RouteSummary | None = None,
    tab: str | None = None,
    occurrence: schemas.OccurrenceDetail | None = None,
    optics: schemas.BranchOptics | None = None,
    run: schemas.RunEntry | None = None,
    machine: dict | None = None,
) -> str:
    """What the viewer shows, as facts: lattice, branch, selected element, optics, run and machine."""
    lines = [
        f"Lattice {lattice.name}, revision {lattice.digest.removeprefix('sha256:')[:12]}, branches "
        + ", ".join(f"{b.name} ({b.kind.value})" for b in lattice.branches)
        + "."
    ]
    if route is not None:
        lines.append(
            f"Shown: route {route.name}, a single pass through "
            + ", ".join(route.branches)
            + f", {route.length:.6g} m."
            + (f" The {tab} tab is open." if tab else "")
        )
    else:
        layout = lattice.layout(branch)
        kinds = Counter(p.kind.value for p in layout.placements)
        lines.append(
            f"Shown: branch {layout.branch.name}, {'a ring' if layout.periodic else 'a line'}, "
            f"{layout.length:.6g} m, {len(layout.placements)} occurrences: "
            + ", ".join(f"{n} {k}" for k, n in kinds.most_common())
            + "."
            + (f" The {tab} tab is open." if tab else "")
        )
        reference = layout.placements[0].element.ReferenceP if layout.placements else None
        if reference is not None and reference.pc_ref:
            lines.append(f"Reference: {reference.species_ref}, momentum {reference.pc_ref / 1e9:.6g} GeV/c.")
    if optics is not None and optics.available:
        line = f"Optics ({optics.engine}, {optics.fidelity} fidelity, design settings)"
        if optics.tune_x is not None:
            line += (
                f": tunes {optics.tune_x:.4f} and {optics.tune_y:.4f}, chromaticities "
                f"{_n(optics.chroma_x, 4)} and {_n(optics.chroma_y, 4)}, momentum compaction "
                f"{_n(optics.momentum_compaction, 4)}"
            )
        beta = optics.values.get("beta_x"), optics.values.get("beta_y")
        if beta[0] and beta[1]:
            line += f"; beta x up to {max(beta[0]):.4g} m, beta y up to {max(beta[1]):.4g} m"
        lines.append(line + ".")
    elif optics is not None:
        lines.append(f"Optics: none for this branch ({optics.origin}).")
    else:
        lines.append("Optics: not computed yet (the Optics tab computes them).")
    if occurrence is not None:
        parameters = [
            f"{g.group}.{p.name} = {_n(p.value, 6) if isinstance(p.value, float) else p.value}"
            f"{' ' + p.unit if p.unit else ''}"
            for g in occurrence.groups
            for p in g.parameters
            if p.value not in (None, 0, 0.0, False, "", [])
        ]
        lines.append(
            f"Selected: {occurrence.name}"
            + (f" (EIC name {occurrence.eic_name})" if occurrence.eic_name else "")
            + f", a {occurrence.kind} from s = {occurrence.s_enter:.6g} to {occurrence.s_exit:.6g} m, length "
            f"{occurrence.length:.6g} m"
            + (f"; parameters: {'; '.join(parameters[:24])}" if parameters else "")
            + (f"; devices: {', '.join(d.name for d in occurrence.devices)}" if occurrence.devices else "")
            + "."
        )
    else:
        lines.append("No element is selected.")
    if machine is not None:
        lines.append(
            f"Virtual accelerator machine {machine.get('name')}: {machine.get('state')}, tracking the "
            f"{machine.get('kind')} {machine.get('target')} with "
            f"{machine.get('engine') or 'the default engine'}, "
            f"{machine.get('changes', 0)} settings changed from their original values"
            + (f"; warnings: {' '.join(machine['warnings'])}" if machine.get("warnings") else "")
            + "."
        )
    if run is not None:
        lines.append(
            f"Run {run.id} ({run.kind}, {run.engine}): {run.state}, {run.end_turn} {run.unit}s recorded"
            + (f", {run.captures} BPM captures" if getattr(run, "captures", 0) else "")
            + "."
        )
    return "\n".join(lines)


def view_messages(question: str, facts: str) -> list[Message]:
    """Ask the person's question about what the viewer shows."""
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"{facts}\n\nQuestion: {question.strip()}\nAnswer in at most 80 words.",
        },
    ]


class AiOff(LookupError):
    """The AI features are turned off."""


class Busy(RuntimeError):
    """The model is answering as many as it can."""


class Assistant:
    """The model behind the viewer's AI features, as configured now: short tasks, a few at a time.

    ``model`` gives the model to ask (None: the features are off); it is called for
    every request, so a change on the administration page takes effect at once. A
    local model answers one request at a time, a provider's several; one more waits
    a little, then is told the model is busy rather than queued for minutes. Every
    answer is capped in tokens and in time.
    """

    def __init__(self, model: Callable[[], Model | None], *, wait: float = 20.0, limit: float = 60.0) -> None:
        self.model = model
        self._active = 0
        self._free = threading.Condition()
        self._wait, self._limit = wait, limit

    def status(self) -> schemas.AiStatus:
        model = self.model()
        if model is None:
            return schemas.AiStatus(
                enabled=False, available=False, provider=None, model=None, where=None, reason="turned off"
            )
        available, reason = model.status()
        name = model.resolved() if available and isinstance(model, LocalModel) else model.model
        return schemas.AiStatus(
            enabled=True,
            available=available,
            provider=model.provider,
            model=name or None,
            where=model.where,
            reason=reason,
        )

    def stream(self, messages: list[Message], max_tokens: int) -> Iterator[str]:
        """The answer, piece by piece. Raises AiOff, AiUnavailable or Busy before the first piece."""
        model = self.model()
        if model is None:
            raise AiOff("the AI features are turned off")
        available, reason = model.status()
        if not available:
            raise AiUnavailable(reason)
        with self._free:
            if self._active >= model.concurrency:
                raise Busy(
                    f"{model.model or 'the model'} is answering other questions; try again in a few seconds"
                )
        free, wait, limit = self._free, self._wait, self._limit

        def answer() -> Iterator[str]:
            with free:
                if not free.wait_for(lambda: self._active < model.concurrency, timeout=wait):
                    yield "[the model is busy; try again in a few seconds]"
                    return
                self._active += 1
            try:
                deadline = time.monotonic() + limit
                for piece in model.stream(messages, max_tokens=max_tokens):
                    yield piece
                    if time.monotonic() > deadline:
                        yield f" [stopped after {limit:g} s]"
                        return
            except AiUnavailable as error:
                yield f" [the model stopped: {error}]"
            finally:
                with free:
                    self._active -= 1
                    free.notify()

        return answer()
