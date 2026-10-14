"""What an operation proposes: setting changes, what they should do, and the checks made in the model.

A proposal is data, written to nothing. It says which channels change and by how
much (in their units, within their limits), what the model of the machine as
it is set predicts they do, which checks that prediction passed, and by which
method, with references. A person, or an agent in its own sandbox, applies it
(ADR 0010).
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChannelChange(BaseModel):
    channel: str
    device: str
    unit: str
    before: float
    after: float
    limits: tuple[float, float] | None = None

    @property
    def change(self) -> float:
        return self.after - self.before


class Check(BaseModel):
    """One check of the prediction, made in the model with the changes applied."""

    name: str
    ok: bool
    value: float | None = None
    limit: float | None = None
    note: str = ""


class Proposal(BaseModel):
    operation: Literal["orbit_correction", "local_bump", "tune_move"]
    machine: str = Field(description="The branch the operation is for")
    summary: str = Field(description="What it does, in one sentence with the numbers")
    changes: list[ChannelChange]
    predicted: dict[str, Any] = Field(description="What the model predicts: before and after, with units")
    checks: list[Check]
    method: str = Field(description="How the changes were found")
    parameters: dict[str, Any] = Field(default_factory=dict, description="The method's choices")
    engine: str
    references: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def values(self) -> dict[str, float]:
        """The channels' new values, to write."""
        return {c.channel: c.after for c in self.changes}


def headroom(changes: list[ChannelChange]) -> Check:
    """Every new value within its channel's limits: the smallest margin to a limit, as a fraction of
    the channel's range (0 at a limit, negative beyond it)."""
    margin, which = 1.0, ""
    for c in changes:
        if not c.limits:
            continue
        low, high = c.limits
        span = high - low
        here = min(c.after - low, high - c.after) / span if span > 0 else 0.0
        if here < margin:
            margin, which = here, c.channel
    return Check(
        name="within limits",
        ok=margin >= 0.0,
        value=margin,
        limit=0.0,
        note=f"the smallest margin is {which}'s" if which else "no channel has limits",
    )
