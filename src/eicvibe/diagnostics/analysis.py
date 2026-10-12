"""Analysis of BPM data, the same for a real machine and a virtual one.

Every function takes a :class:`~.bpm.TurnByTurn` (or an orbit series), never
an engine result, so a notebook that works on a RHIC file works on a virtual
ring's history and back.
"""

import numpy as np

from .bpm import PlaneData, TurnByTurn


def closed_orbit(tbt: TurnByTurn) -> tuple[np.ndarray, np.ndarray]:
    """The mean reading of every BPM over the acquisition, x and y, mm."""
    return np.nanmean(tbt.x.position, axis=1), np.nanmean(tbt.y.position, axis=1)


def _prepared(signal: np.ndarray) -> np.ndarray | None:
    """Readings without their straight-line trend (a slowly drifting orbit), NaN as zero; None if none."""
    signal = np.asarray(signal, dtype=float)
    valid = np.isfinite(signal)
    if not valid.any():
        return None
    turns = np.arange(len(signal))
    trend = np.polyfit(turns[valid], signal[valid], 1) if valid.sum() > 2 else (0.0, np.nanmean(signal))
    return np.where(valid, signal - np.polyval(trend, turns), 0.0)


def tune_of(signal: np.ndarray, low: float = 0.0) -> float:
    """The fractional tune of one BPM's readings: the strongest line of a Hann-windowed spectrum.

    The readings' straight-line trend is removed first (a slowly drifting orbit), and lines
    below ``low`` are ignored (e.g. a synchrotron sideband). The peak is refined between bins
    (quadratic fit of the log magnitude, about 1e-4 of a turn at a thousand turns). NaN
    readings count as zero.
    """
    values = _prepared(signal)
    if values is None:
        return float("nan")
    n = len(values)
    spectrum = np.abs(np.fft.rfft(values * np.hanning(n)))
    spectrum[: max(1, int(np.ceil(low * n)))] = 0.0
    k = int(np.argmax(spectrum))
    if 0 < k < len(spectrum) - 1:
        a, b, c = np.log(spectrum[k - 1 : k + 2] + 1e-300)
        k += 0.5 * (a - c) / (a - 2 * b + c)
    return float(k / n)


def averaged_tune(signals: np.ndarray, low: float = 0.0) -> float:
    """The fractional tune of many pickups' readings ``[pickup, turn]`` together: the strongest line
    of their power spectra summed, as a tune monitor averages to see a weak signal over noise and
    sidebands. Otherwise as :func:`tune_of`."""
    rows = [v for row in np.asarray(signals, dtype=float) if (v := _prepared(row)) is not None]
    if not rows:
        return float("nan")
    n = len(rows[0])
    window = np.hanning(n)
    power = np.sum([np.abs(np.fft.rfft(v * window)) ** 2 for v in rows], axis=0)
    power[: max(1, int(np.ceil(low * n)))] = 0.0
    k = int(np.argmax(power))
    if 0 < k < len(power) - 1:
        a, b, c = 0.5 * np.log(power[k - 1 : k + 2] + 1e-300)  # log magnitude, as tune_of fits
        k += 0.5 * (a - c) / (a - 2 * b + c)
    return float(k / n)


def tunes(plane: PlaneData, low: float = 0.0) -> np.ndarray:
    """The fractional tune every BPM of a plane sees."""
    return np.array([tune_of(row, low) for row in plane.position])


def tune(tbt: TurnByTurn, plane: str = "x", low: float = 0.0) -> float:
    """A plane's fractional tune, the median over its BPMs.

    Between 0 and 0.5: one BPM's readings cannot tell q from 1 - q.
    """
    return float(np.nanmedian(tunes(tbt.plane(plane), low)))


def amplitudes(plane: PlaneData) -> np.ndarray:
    """Every BPM's oscillation amplitude about its mean, mm: sqrt(2) times the rms."""
    centred = plane.position - np.nanmean(plane.position, axis=1, keepdims=True)
    return np.sqrt(2.0) * np.sqrt(np.nanmean(centred**2, axis=1))


def spectrum(plane: PlaneData) -> tuple[np.ndarray, np.ndarray]:
    """The plane's spectrum: frequencies (fractions of the revolution, 0 to 0.5) and the mean over
    its BPMs of each line's amplitude, mm (a pure oscillation of amplitude A shows a line near A)."""
    n = plane.turns
    window = np.hanning(n)
    lines = [np.abs(np.fft.rfft(v * window)) for row in plane.position if (v := _prepared(row)) is not None]
    mean = np.mean(lines, axis=0) if lines else np.zeros(n // 2 + 1)
    return np.fft.rfftfreq(n), 2.0 * mean / window.sum()


def harmonics(plane: PlaneData, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Every BPM's line at frequency ``q``: its amplitude (mm) and phase (rad).

    Readings A cos(2 pi q n + phi) give A and phi; along the ring the phase grows by 2 pi times
    the phase advance, or falls by it when the tune is 1 - q.
    """
    n = plane.turns
    window = np.hanning(n)
    carrier = window * np.exp(-2j * np.pi * q * np.arange(n))
    lines = np.array(
        [np.sum(v * carrier) if (v := _prepared(row)) is not None else np.nan for row in plane.position]
    )
    return 2.0 * np.abs(lines) / window.sum(), np.angle(lines)


def phase_errors(phase: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, bool]:
    """Measured phase advances from the first BPM against the model's ``mu`` (turns), in turns.

    The readings' phase grows with the phase advance when the line seen is the tune, and falls
    when it is 1 - tune; the orientation that fits the model better is taken, and returned as
    whether the line is the tune's mirror image.
    """
    measured = (phase - phase[0]) / (2 * np.pi)
    model = mu - mu[0]

    def wrapped(sign: int) -> np.ndarray:
        return (sign * measured - model + 0.5) % 1.0 - 0.5

    rising, falling = wrapped(1), wrapped(-1)
    mirrored = bool(np.nanmean(falling**2) < np.nanmean(rising**2))
    return (falling if mirrored else rising), mirrored


def beta_beating(amplitude: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Relative beta error at every BPM from oscillation amplitudes: A^2 / beta should be the same
    everywhere (twice the action); its spread about the median is the beta-beating, BPM gain errors
    included."""
    ratio = amplitude**2 / beta
    return ratio / np.nanmedian(ratio) - 1.0


def tune_drift(plane: PlaneData, window: int, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tune and oscillation amplitude over the acquisition: the first turn of each ``window``-turn
    stretch, ``step`` turns apart, its fractional tune (median over the BPMs) and amplitude (median,
    mm). Shows decoherence, and a tune that moves with the amplitude."""
    starts = np.arange(0, max(plane.turns - window, 0) + 1, max(step, 1))
    q, a = [], []
    for start in starts:
        part = plane.window(int(start), window)
        q.append(float(np.nanmedian(tunes(part))))
        a.append(float(np.nanmedian(amplitudes(part))))
    return starts, np.array(q), np.array(a)
