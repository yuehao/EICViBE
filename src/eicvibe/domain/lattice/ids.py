"""Identifier generation.

Ids are opaque and never reused for a different entity. Random ids are the
default: two collaborators editing the same base revision can both add
elements without colliding, which sequential counters cannot guarantee.

An :data:`IdFactory` receives a category prefix (``ele``, ``br``, ``occ``,
``cmp``) and returns a new id. Importers use :func:`derived_id` so that
importing the same source twice yields the same ids and the same digest.
"""

import hashlib
import secrets
from collections.abc import Callable

type IdFactory = Callable[[str], str]

ELEMENT, BRANCH, OCCURRENCE, COMPONENT = "ele", "br", "occ", "cmp"


def random_id(prefix: str) -> str:
    """``<prefix>_`` followed by 48 random bits in hex."""
    return f"{prefix}_{secrets.token_hex(6)}"


def derived_id(prefix: str, *parts: str) -> str:
    """Deterministic id from a namespace and key, e.g. a source checksum and a position."""
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:16]}"


class SequentialIds:
    """Deterministic ``<prefix>_<n>`` ids for tests and hand-written fixtures."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}

    def __call__(self, prefix: str) -> str:
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}_{self._counters[prefix]}"
