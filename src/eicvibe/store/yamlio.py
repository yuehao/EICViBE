"""YAML and CSV files as people read and edit them.

Round-trip YAML (ruamel.yaml) keeps comments and key order, never wraps long
lines, and writes floats so that they read back to the same value. Short
mappings and lists are written on one line (flow style), long ones one item
per line, which keeps diffs of edited files small.
"""

import csv
import io
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq


def _yaml() -> YAML:
    yaml = YAML()
    yaml.width = 1 << 16  # one item per line, however long
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.default_flow_style = False
    return yaml


class FileError(ValueError):
    """A problem in a machine file, with where it is."""

    def __init__(self, path: Path | str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = Path(path)


def read_yaml(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return _yaml().load(stream)
    except OSError as error:
        raise FileError(path, error.strerror or str(error)) from None
    except Exception as error:  # ruamel's parser errors carry the line and column
        raise FileError(path, f"not valid YAML: {error}") from None


def write_yaml(path: Path, data: Any, header: str | None = None) -> Path:
    """Write ``data`` with ``header`` as a comment block at the top."""
    if header:
        if not isinstance(data, CommentedMap):
            data = CommentedMap(data)
        data.yaml_set_start_comment(header)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    _yaml().dump(data, buffer)
    path.write_text(buffer.getvalue(), encoding="utf-8")
    return path


def flow(value: Any) -> Any:
    """``value`` (a mapping or list, recursively) written on one line."""
    if isinstance(value, Mapping):
        out = CommentedMap((k, flow(v)) for k, v in value.items())
        out.fa.set_flow_style()
        return out
    if isinstance(value, list | tuple):
        out = CommentedSeq(flow(v) for v in value)
        out.fa.set_flow_style()
        return out
    return value


def block(items: Iterable[Any]) -> CommentedSeq:
    """A list written one item per line."""
    out = CommentedSeq(items)
    out.fa.set_block_style()
    return out


def mapping(items: Iterable[tuple[str, Any]] | Mapping[str, Any] = ()) -> CommentedMap:
    """A mapping written one key per line."""
    out = CommentedMap(items)
    out.fa.set_block_style()
    return out


def comment(container: CommentedMap, key: str, text: str) -> None:
    """A comment at the end of ``key``'s line."""
    container.yaml_add_eol_comment(text, key)


def read_table(path: Path) -> tuple[list[str], list[tuple[float, ...]]]:
    """A CSV table of numbers: its header and its rows. Lines starting with # are comments."""
    try:
        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#")
        ]
    except OSError as error:
        raise FileError(path, error.strerror or str(error)) from None
    rows = [row for row in csv.reader(lines) if any(cell.strip() for cell in row)]
    if len(rows) < 2:
        raise FileError(path, "a table needs a header line and at least one row")
    header, values = [cell.strip() for cell in rows[0]], []
    for number, row in enumerate(rows[1:], start=2):
        try:
            values.append(tuple(float(cell) for cell in row))
        except ValueError:
            raise FileError(path, f"row {number} is not all numbers: {row}") from None
        if len(values[-1]) != len(header):
            raise FileError(path, f"row {number} has {len(values[-1])} values for {len(header)} columns")
    return header, values


def write_table(
    path: Path, header: list[str], rows: Iterable[Iterable[float]], note: str | None = None
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    if note:
        buffer.writelines(f"# {line}\n" for line in note.splitlines())
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows([repr(float(v)) for v in row] for row in rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")
    return path
