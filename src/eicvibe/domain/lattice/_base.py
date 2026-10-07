"""Shared building blocks for the immutable lattice description.

Every published model derives from :class:`DomainModel`: frozen, closed to
unknown fields, and never updated in place. Numeric fields use :data:`Real`,
which rejects booleans, strings, NaN and infinities and folds ``-0.0`` into
``0.0``, so equal physics always serializes to equal bytes.
"""

import math
import numbers
from collections.abc import Mapping
from typing import Annotated, Any, Self

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field


def _require_real(value: Any) -> Any:
    # bool is an Integral; a True that silently becomes 1.0 m is a unit bug.
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"expected a real number, got {type(value).__name__}")
    return value


def _require_finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("must be finite")
    return value + 0.0  # -0.0 -> 0.0


def _require_integer(value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("expected an integer, got bool")
    return value


#: A finite real number.
Real = Annotated[float, BeforeValidator(_require_real), AfterValidator(_require_finite)]
NonNegativeReal = Annotated[Real, Field(ge=0.0)]
PositiveReal = Annotated[Real, Field(gt=0.0)]
PositiveInt = Annotated[int, BeforeValidator(_require_integer), Field(gt=0)]

#: Human-readable name of an element, branch, occurrence or component. Names
#: are labels and lookup aliases; they are not identities.
Label = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.\-]{1,128}$")]

#: Opaque stable identifier. Generated once and carried across revisions.
Id = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:\-]{0,127}$")]


def _sorted_unique(tags: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(set(tags)))


#: Free-form grouping labels (families, sectors, planes), kept sorted and unique.
Tags = Annotated[tuple[Label, ...], AfterValidator(_sorted_unique)]


class DomainModel(BaseModel):
    """Frozen base model for published lattice data."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=False)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        # pydantic applies ``update`` without validation, which would let an
        # invalid object pose as a validated revision. Edits go through
        # LatticeBuilder instead.
        if update:
            raise TypeError(f"{type(self).__name__} is immutable; build a changed copy with LatticeBuilder")
        return super().model_copy(deep=deep)

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        return self  # immutable, like a tuple

    def __getstate__(self) -> dict[Any, Any]:
        # Derived caches (functools.cached_property) are rebuilt on demand.
        state = super().__getstate__()
        fields = type(self).model_fields
        state["__dict__"] = {k: v for k, v in state["__dict__"].items() if k in fields}
        return state


def resolve_ref(ref: str, by_id: Mapping[str, Any], by_name: Mapping[str, str], what: str) -> str:
    """Resolve ``ref`` as an id or a name. A string that is an id of one entity
    and the name of another is rejected rather than guessed."""
    named = by_name.get(ref)
    if ref in by_id:
        if named is not None and named != ref:
            raise KeyError(f"{ref!r} is both the id of one {what} and the name of another")
        return ref
    if named is not None:
        return named
    raise KeyError(f"no {what} with id or name {ref!r}")
