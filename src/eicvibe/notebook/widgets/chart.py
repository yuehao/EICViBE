"""The viewer's chart, for any x-y data (ADR 0016): lines and points, markers, a hover readout."""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import anywidget
from traitlets import Bool, Dict, Float, Int, List, Unicode

from ._base import MODULE, STYLES

#: Series colours, light and dark, in this order: the viewer's (web/src/model/optics.ts).
PALETTE = (("#1f5fbf", "#6ea8ff"), ("#c7362f", "#ff7b72"), ("#2b8a3e", "#69db7c"), ("#8a5bd6", "#c4a7ff"))
_GREY = ("#6b7280", "#9ca3af")


class Chart(anywidget.AnyWidget):
    """The viewer's chart: lines and points, markers, a hover readout of the nearest point.

    Make one with :func:`chart`.
    """

    _esm = MODULE
    _css = STYLES

    view = Unicode("chart").tag(sync=True)
    title = Unicode().tag(sync=True)
    x_label = Unicode().tag(sync=True)
    y_label = Unicode().tag(sync=True)
    series = List(Dict()).tag(sync=True)
    markers = List(Dict()).tag(sync=True)
    log = List(Float(), default_value=None, allow_none=True).tag(sync=True)
    """A logarithmic y axis over these decades, ``[low, high]`` (e.g. ``[-4, 0]``); None: linear."""
    x_domain = List(Float(), default_value=None, allow_none=True).tag(sync=True)
    y_zero = Bool(True).tag(sync=True)
    """Whether the y axis reaches zero."""
    names = List(Unicode()).tag(sync=True)
    """Names of the first series' points (BPMs, say), for the readout."""
    height = Int(260).tag(sync=True)


def chart(
    series: Mapping[str, tuple[Any, Any]] | Sequence[Mapping[str, Any]],
    *,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    style: str = "line",
    markers: Iterable[Mapping[str, Any]] = (),
    log: tuple[float, float] | None = None,
    x_domain: tuple[float, float] | None = None,
    y_zero: bool = True,
    names: Iterable[str] = (),
    height: int = 260,
) -> Chart:
    """The viewer's chart of ``series``: ``{label: (x, y)}``, or dicts with ``label``, ``x``, ``y`` and
    perhaps ``style`` (line, points or both), ``dash`` and ``color``.

    Series take the viewer's colours in order; a fifth and later take them again, dashed.
    ``markers`` are vertical (``x``) or horizontal (``y``) lines, perhaps with a ``label``.
    """
    rows = (
        [{"label": str(label), "x": xy[0], "y": xy[1]} for label, xy in series.items()]
        if isinstance(series, Mapping)
        else [dict(row) for row in series]
    )
    out = []
    for k, row in enumerate(rows):
        light, dark = PALETTE[k % len(PALETTE)]
        given = row.get("color")
        dash = row.get("dash") or ([6, 3] if k >= len(PALETTE) else None)
        out.append(
            {
                "label": str(row["label"]),
                "x": [float(v) for v in row["x"]],
                "y": [None if v is None or not math.isfinite(float(v)) else float(v) for v in row["y"]],
                "style": row.get("style", style),
                "color": given or light,
                "darkColor": given or dark,
                **({"dash": list(dash)} if dash else {}),
            }
        )
    lines = [
        {
            **{k: float(m[k]) for k in ("x", "y") if m.get(k) is not None},
            **({"label": str(m["label"])} if m.get("label") else {}),
            "color": m.get("color", _GREY[0]),
            "darkColor": m.get("color", _GREY[1]),
            **({"dash": list(m["dash"])} if m.get("dash") else {}),
        }
        for m in markers
    ]
    return Chart(
        title=title,
        x_label=x_label,
        y_label=y_label,
        series=out,
        markers=lines,
        log=list(log) if log is not None else None,
        x_domain=list(x_domain) if x_domain is not None else None,
        y_zero=y_zero,
        names=[str(n) for n in names],
        height=height,
    )
