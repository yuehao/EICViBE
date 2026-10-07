"""Canonical JSON encoding and content digests.

The encoding sorts keys, uses no insignificant whitespace, writes floats with
Python's shortest round-trip ``repr`` and refuses NaN and infinities. Models
fold ``-0.0`` into ``0.0`` and omit defaults before encoding, so the digest
changes only when content changes, and adding a defaulted field in a later
version leaves existing digests intact.
"""

import hashlib
import json
from typing import Any


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def content_digest(data: Any) -> str:
    """``sha256:<hex>`` of the canonical encoding of ``data``."""
    return "sha256:" + hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()
