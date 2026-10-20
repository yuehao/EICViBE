"""An engine on another computer, reached through a command such as ssh (ADR 0015).

A host is a command that starts ``eicvibe-engine serve`` somewhere: ``ssh aphpc ...``, or through a
cluster's gateway and its scheduler (``srun``). The command's standard input and output carry the
engine worker's calls (:mod:`.worker`) and their results; its standard error is logged. Nothing opens
a port: the connection is the command's, with the keys, tickets and gateways it already uses.

Messages are a JSON header and the raw bytes of the arrays it names, never pickles: the header holds
only plain data, tuples, sets, dicts, numpy arrays and the engine contract's own types
(:mod:`.contracts`), so reading a message builds nothing else. Both ends say first which contract
they speak (a digest of the modules that define it); a server whose copy of EICViBE differs is
refused.

The server answers one request at a time, in order, as a local worker does. A lost connection fails
what was waiting with ``BrokenProcessPool``, as a local worker's crash does, and the next request runs
the command again: the virtual ring then goes on from the same beam.
"""

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import platform
import shlex
import struct
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from enum import Enum
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import IO, Any

import numpy as np
from pydantic import BaseModel

from . import contracts, worker
from .contracts import EngineUnavailable, ParticleLost, UnsupportedRequest

log = logging.getLogger(__name__)

_MAGIC = b"EVBE"
_FRAME = struct.Struct("<4sII")  # magic, header bytes, buffers
_LENGTH = struct.Struct("<Q")
#: Text a host's shell may print before the server starts, skipped (and logged) at most.
_PREAMBLE = 1 << 20

#: The worker's functions a server runs, by the name a request gives.
_CALLS: dict[str, Callable[..., Any]] = {
    "open_session": worker._open_session,
    "call": worker._call,
    "close_session": worker._close_session,
    "machine_optics": worker._machine_optics,
    "route_optics": worker._route_optics,
    "route_start": worker._route_start,
}
_NAMES = {function.__name__: name for name, function in _CALLS.items()}


class RemoteEngineError(RuntimeError):
    """The engine host cannot be used: its command failed, or it runs another EICViBE."""


# -- what crosses ---------------------------------------------------------------------------------


@cache
def _types() -> dict[str, type]:
    """The contract's types, by name: the only ones a message may name."""
    return {
        name: value
        for name, value in vars(contracts).items()
        if isinstance(value, type)
        and value.__module__ == contracts.__name__
        and (issubclass(value, (BaseModel, Enum)) or dataclasses.is_dataclass(value))
    }


@cache
def contract() -> str:
    """A digest of the modules that define what crosses: both ends must have the same."""
    digest = hashlib.sha256()
    for path in (contracts.__file__, worker.__file__, __file__):
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()[:16]


class _Encoder:
    """Values as JSON, with arrays set aside as buffers."""

    def __init__(self) -> None:
        self.buffers: list[memoryview] = []

    def __call__(self, value: Any) -> Any:
        if isinstance(value, Enum):
            return {"$e": self._named(value), "v": value.value}
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("an array of objects cannot cross to an engine host")
            array = np.ascontiguousarray(value)
            self.buffers.append(memoryview(array.reshape(-1).view(np.uint8)))  # empty ones too
            return {"$a": [len(self.buffers) - 1, array.dtype.str, list(array.shape)]}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, BaseModel):
            return {"$m": self._named(value), "v": self(value.model_dump(mode="python"))}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            fields = {f.name: self(getattr(value, f.name)) for f in dataclasses.fields(value)}
            return {"$c": self._named(value), "v": fields}
        if isinstance(value, tuple):
            return {"$t": [self(v) for v in value]}
        if isinstance(value, list):
            return [self(v) for v in value]
        if isinstance(value, (set, frozenset)):
            return {"$s": [self(v) for v in value]}
        if isinstance(value, dict):
            if all(isinstance(k, str) and not isinstance(k, Enum) and not k.startswith("$") for k in value):
                return {k: self(v) for k, v in value.items()}
            return {"$d": [[self(k), self(v)] for k, v in value.items()]}
        raise TypeError(f"a {type(value).__name__} cannot cross to an engine host")

    @staticmethod
    def _named(value: Any) -> str:
        name = type(value).__name__
        if _types().get(name) is not type(value):
            raise TypeError(f"{name} is not one of the engine contract's types")
        return name


_TAGS: tuple[tuple[str, Callable[[type], bool]], ...] = (
    ("$e", lambda kind: issubclass(kind, Enum)),
    ("$m", lambda kind: issubclass(kind, BaseModel)),
    ("$c", dataclasses.is_dataclass),
)


class _Undecodable(Exception):
    """A message read whole that names what may not cross: the call it answers or asks fails."""

    def __init__(self, key: Any, error: Exception) -> None:
        super().__init__(str(error))
        self.key = key
        self.error = error


def _decode(value: Any, buffers: Sequence[bytes]) -> Any:
    if isinstance(value, list):
        return [_decode(v, buffers) for v in value]
    if not isinstance(value, dict):
        return value
    if "$a" in value:
        index, dtype, shape = value["$a"]
        kind = np.dtype(dtype)
        if kind.hasobject:
            raise TypeError("an array of objects cannot cross from an engine host")
        return np.frombuffer(buffers[index], dtype=kind).reshape(shape).copy()
    if "$t" in value:
        return tuple(_decode(v, buffers) for v in value["$t"])
    if "$s" in value:
        return frozenset(_decode(v, buffers) for v in value["$s"])
    if "$d" in value:
        return {_decode(k, buffers): _decode(v, buffers) for k, v in value["$d"]}
    for tag, fits in _TAGS:
        if tag in value:
            kind = _types().get(value[tag])
            if kind is None or not fits(kind):
                raise TypeError(f"{value[tag]!r} is not one of the engine contract's types")
            inner = _decode(value["v"], buffers)
            if tag == "$e":
                return kind(inner)
            if tag == "$m":
                return kind.model_validate(inner)
            return kind(**inner)
    return {k: _decode(v, buffers) for k, v in value.items()}


def _write(stream: IO[bytes], message: Any) -> None:
    encoder = _Encoder()
    header = json.dumps(encoder(message), separators=(",", ":")).encode()
    stream.write(_FRAME.pack(_MAGIC, len(header), len(encoder.buffers)) + header)
    for buffer in encoder.buffers:
        stream.write(_LENGTH.pack(buffer.nbytes))
        stream.write(buffer)
    stream.flush()


def _exactly(stream: IO[bytes], n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            raise EOFError("the connection ended")
        data += chunk
    return bytes(data)


def _read(stream: IO[bytes], start: bytes = b"") -> Any:
    head = start + _exactly(stream, _FRAME.size - len(start))
    magic, size, count = _FRAME.unpack(head)
    if magic != _MAGIC:
        raise EOFError(f"not an EICViBE engine message: {head!r}")
    header = json.loads(_exactly(stream, size))
    buffers = [_exactly(stream, _LENGTH.unpack(_exactly(stream, _LENGTH.size))[0]) for _ in range(count)]
    try:
        return _decode(header, buffers)
    except (TypeError, ValueError, KeyError) as error:
        raise _Undecodable(header.get("id") if isinstance(header, dict) else None, error) from None


def _first(stream: IO[bytes], note: Callable[[str], None]) -> Any:
    """The first message, after any text a shell printed first (a login banner), which ``note`` gets."""
    seen = bytearray()
    while not seen.endswith(_MAGIC):
        byte = stream.read(1)
        if not byte:
            raise EOFError("the connection ended before the engine started")
        seen += byte
        if len(seen) > _PREAMBLE:
            raise EOFError("no engine answered: the command printed only text")
    if len(seen) > len(_MAGIC):
        note(seen[: -len(_MAGIC)].decode(errors="replace"))
    return _read(stream, _MAGIC)


def _error(error: BaseException) -> dict[str, Any]:
    """An error as it crosses back: the contract's own, ValueError and KeyError as they are."""
    if isinstance(error, UnsupportedRequest):
        return {"type": "UnsupportedRequest", "args": [error.engine, list(error.reasons)]}
    if isinstance(error, ParticleLost):
        return {"type": "ParticleLost", "args": [error.engine, error.occurrence]}
    for kind in (EngineUnavailable, KeyError, ValueError):
        if isinstance(error, kind):
            return {"type": kind.__name__, "args": [error.args[0] if error.args else str(error)]}
    return {"type": "RuntimeError", "args": [str(error) or type(error).__name__]}


_ERRORS: dict[str, Callable[..., BaseException]] = {
    "UnsupportedRequest": UnsupportedRequest,
    "ParticleLost": ParticleLost,
    "EngineUnavailable": EngineUnavailable,
    "KeyError": KeyError,
    "ValueError": ValueError,
    "RuntimeError": RuntimeError,
}


# -- the client ------------------------------------------------------------------------------------


class RemoteEngine:
    """An engine host's command, running: an executor for :class:`.worker.EngineWorker`, which sends it
    the worker's calls in order and resolves their futures as the answers come back."""

    def __init__(self, command: Sequence[str], label: str | None = None) -> None:
        self.command = list(command)
        self.label = label or self.command[0]
        self.info: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._pending: dict[int, Future] = {}
        self._next = 0
        self._broken: str | None = None
        self._refused = False
        self._tail: deque[str] = deque(maxlen=20)
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # a Ctrl-C at the terminal is the server's, not the host's
        )
        threading.Thread(target=self._errors, name=f"engine host {self.label} log", daemon=True).start()
        threading.Thread(target=self._answers, name=f"engine host {self.label}", daemon=True).start()
        self._send("hello", [{"contract": contract(), "eicvibe": _version("eicvibe")}])  # id 0

    def submit(self, function: Callable[..., Any], *args: Any) -> Future:
        """Send one of the worker's calls; its future resolves when the host answers."""
        name = _NAMES.get(getattr(function, "__name__", ""))
        if name is None:
            raise ValueError(f"{function!r} is not an engine worker's call")
        return self._send(name, list(args))

    def _send(self, call: str, args: list[Any]) -> Future:
        with self._lock:
            if self._broken:
                raise (RemoteEngineError if self._refused else BrokenProcessPool)(self._broken)
            key, self._next = self._next, self._next + 1
            future: Future = Future()
            self._pending[key] = future
            try:
                _write(self._process.stdin, {"id": key, "call": call, "args": args})
            except (OSError, ValueError) as error:  # the command has ended: its pipe is closed
                self._pending.pop(key, None)
                self._fail(f"cannot send to it ({error})", locked=True)
                raise BrokenProcessPool(self._broken) from None
        return future

    def _answers(self) -> None:
        stream = self._process.stdout
        try:
            message = _first(
                stream, lambda text: log.info("engine host %s said: %s", self.label, text.strip())
            )
            while True:
                if message is not None:
                    self._answer(message)
                try:
                    message = _read(stream)
                except _Undecodable as bad:  # this answer cannot be read; the others can
                    message = {"id": bad.key, "error": _error(RemoteEngineError(f"its answer: {bad.error}"))}
        except Exception as error:  # noqa: BLE001 - the connection is gone, however it went
            self._fail(str(error) if not isinstance(error, EOFError) else "the connection ended")

    def _answer(self, message: dict[str, Any]) -> None:
        with self._lock:
            future = self._pending.pop(message.get("id"), None)
        if future is not None and not future.cancelled():
            if "error" in message:
                error = message["error"]
                future.set_exception(_ERRORS.get(error["type"], RuntimeError)(*error["args"]))
            else:
                future.set_result(message["result"])
        if message.get("id") == 0:  # the hello, answered first
            if "error" in message:
                self._fail(f"it cannot start: {message['error']['args'][0]}", refused=True)
            else:
                self._check(message.get("result"))

    def _check(self, info: dict[str, Any] | None) -> None:
        """The host's hello: refused unless it speaks this contract."""
        if not info or info.get("contract") != contract():
            there = (info or {}).get("contract", "none")
            self._fail(
                f"it runs another EICViBE (contract {there}, here {contract()}): update its copy",
                refused=True,
            )
            return
        self.info = info
        log.info(
            "engine host %s: %s, Python %s, %s",
            self.label,
            info.get("host"),
            info.get("python"),
            ", ".join(f"{k} {v}" for k, v in sorted(info.get("packages", {}).items())),
        )

    def _errors(self) -> None:
        for raw in self._process.stderr:
            line = raw.decode(errors="replace").rstrip()
            if line:
                self._tail.append(line)
                log.debug("engine host %s: %s", self.label, line)

    def _fail(self, reason: str, *, locked: bool = False, refused: bool = False) -> None:
        """The host is gone: what waits fails, and the next call starts its command again."""
        if not locked:
            with self._lock:
                return self._fail(reason, locked=True, refused=refused)
        if self._broken:
            return None
        self._process.poll()
        tail = " | ".join(list(self._tail)[-5:])
        self._broken = f"engine host {self.label}: {reason}" + (f" ({tail})" if tail else "")
        self._refused = refused
        log.warning("%s", self._broken)
        error = RemoteEngineError(self._broken) if refused else BrokenProcessPool(self._broken)
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()
        self._stop()
        return None

    def _stop(self) -> None:
        try:
            self._process.stdin.close()  # the server ends when its input does
        except OSError:
            pass
        if self._process.poll() is None:
            threading.Timer(10.0, self._kill).start()

    def _kill(self) -> None:
        if self._process.poll() is None:
            self._process.kill()

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        with self._lock:
            if cancel_futures:
                for future in self._pending.values():
                    future.cancel()
            self._broken = self._broken or f"engine host {self.label}: closed"
            self._stop()
        if wait:
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._kill()


class EngineHost:
    """Where an engine runs when not here: a name and the command that starts its server."""

    def __init__(self, name: str, command: str | Sequence[str]) -> None:
        self.name = name
        self.command = shlex.split(command) if isinstance(command, str) else list(command)

    def connect(self) -> RemoteEngine:
        return RemoteEngine(self.command, self.name)

    def __repr__(self) -> str:
        return f"EngineHost({self.name!r}, {shlex.join(self.command)!r})"


# -- the server ------------------------------------------------------------------------------------


def _hello(client: dict[str, Any]) -> dict[str, Any]:
    """What this server is: its contract, computer and engines' packages (none loaded: Julia would start)."""
    packages = {name: _version(name) for name in ("xtrack", "juliacall", "cupy-cuda12x")}
    info = {
        "contract": contract(),
        "eicvibe": _version("eicvibe"),
        "host": platform.node(),
        "python": platform.python_version(),
        "packages": {name: v for name, v in packages.items() if v},
    }
    if packages["xtrack"]:
        from .xsuite import compute_contexts, describe_context  # noqa: PLC0415 - Xsuite loads in the server

        info["xsuite_context"] = describe_context(compute_contexts())
    if client.get("contract") != info["contract"]:
        log.warning("the client speaks contract %s, this server %s", client.get("contract"), info["contract"])
    return info


def serve(stream_in: IO[bytes], stream_out: IO[bytes]) -> None:
    """Answer the worker's calls read from ``stream_in`` on ``stream_out``, until the input ends."""
    calls = {"hello": _hello, **_CALLS}
    try:
        while True:
            try:
                message = _read(stream_in)
            except EOFError:
                return
            except _Undecodable as bad:
                _write(stream_out, {"id": bad.key, "error": _error(ValueError(f"the request: {bad.error}"))})
                continue
            key = message.get("id")
            try:
                call = calls.get(message.get("call"))
                if call is None:
                    raise ValueError(f"no call {message.get('call')!r} on an engine host")
                reply = {"id": key, "result": call(*message.get("args", []))}
            except Exception as error:  # noqa: BLE001 - it goes back to the caller
                reply = {"id": key, "error": _error(error)}
            try:
                _write(stream_out, reply)
            except TypeError as error:  # a result that cannot cross
                _write(stream_out, {"id": key, "error": _error(RuntimeError(str(error)))})
    finally:
        for key in list(worker._SESSIONS):
            worker._close_session(key)


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="eicvibe-engine", description="EICViBE's engines, for a server elsewhere (ADR 0015)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    serve_parser = commands.add_parser(
        "serve",
        help="answer an engine worker's calls on standard input and output: an engine host's command runs it",
    )
    serve_parser.add_argument(
        "--xsuite-context",
        help="where Xsuite tracks bunches: cpu (one thread, the default), cpu:N or cpu:auto (OpenMP "
        "threads), cuda or cuda:N (a GPU, with CuPy)",
    )
    args = parser.parse_args(argv)
    if args.xsuite_context:
        os.environ["EICVIBE_XSUITE_CONTEXT"] = args.xsuite_context
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="eicvibe-engine: %(message)s")
    # Standard output carries only messages: what else is printed (Xsuite's compiler, Julia) goes to
    # standard error, which the client logs.
    stream_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    serve(sys.stdin.buffer, stream_out)


if __name__ == "__main__":
    main()
