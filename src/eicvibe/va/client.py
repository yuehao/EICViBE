"""The virtual accelerator from a notebook (ADR 0008): read and write a machine's settings by channel name.

    from eicvibe.va.client import Client

    va = Client("http://localhost:8000", token="...")   # or EICVIBE_URL and EICVIBE_TOKEN
    va.settings("main-ring", changed=True)
    result = va.set("main-ring", {"dr-t8-ps:sp:h": 2.0, "dr-t20-ps:sp:h": -1.5}, reason="orbit correction")
    va.log(machine="main-ring", limit=20)

A token comes from the admin page ("API tokens") or ``eicvibe-va token STATE NAME``.
Every write is checked, applied and logged by the service as the page's are; a
change refused in part is refused whole (``atomic``) and :class:`Refused` says why.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping
from typing import Any


class VaError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class Refused(VaError):
    """A change the service logged but did not apply."""

    def __init__(self, result: dict[str, Any]) -> None:
        reasons = "; ".join(f"{w['channel']}: {w['refusal']}" for w in result["writes"] if w["refusal"])
        super().__init__(422, reasons)
        self.result = result


class Client:
    """A virtual accelerator service, over HTTP."""

    def __init__(self, url: str | None = None, token: str | None = None, timeout: float = 30.0) -> None:
        self.url = (url or os.environ.get("EICVIBE_URL", "http://localhost:8000")).rstrip("/")
        self.token = token or os.environ.get("EICVIBE_TOKEN")
        self.timeout = timeout

    def _call(self, method: str, path: str, body: Any = None, query: Mapping[str, Any] | None = None) -> Any:
        url = f"{self.url}/api/v1{path}"
        if query:
            url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Accept", "application/json")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                text = response.read()
        except urllib.error.HTTPError as error:
            try:
                detail = json.loads(error.read()).get("detail", error.reason)
            except ValueError:
                detail = error.reason
            raise VaError(error.code, str(detail)) from None
        return json.loads(text) if text else None

    # -- reading --------------------------------------------------------------------------------

    def me(self) -> dict[str, Any]:
        return self._call("GET", "/va/me")

    def machines(self) -> list[dict[str, Any]]:
        return self._call("GET", "/va/machines")

    def machine(self, machine: str) -> dict[str, Any]:
        return self._call("GET", f"/va/machines/{machine}")

    def settings(self, machine: str, changed: bool = False) -> dict[str, dict[str, Any]]:
        """Every setting channel of a machine, by name: original, value now, unit, limits, who changed it."""
        rows = self._call("GET", f"/va/machines/{machine}/settings", query={"changed": str(changed).lower()})
        return {r["channel"]: r for r in rows}

    def get(self, machine: str, channels: Iterable[str]) -> dict[str, float]:
        """The values of ``channels`` now."""
        rows = self.settings(machine)
        return {c: rows[c]["value"] for c in channels}

    def log(self, **filters: Any) -> list[dict[str, Any]]:
        """The write log, newest first; filters: machine, user, channel, correlation, before, limit."""
        return self._call("GET", "/va/log", query=filters)

    def orbits(self, machine: str, every: int = 1, count: int = 1) -> dict[str, Any]:
        """The latest BPM orbits of a running machine (mm), as the Orbit tab reads them."""
        run = self.machine(machine)["run"]
        if run is None:
            raise VaError(409, f"{machine} is not running")
        return self._call("GET", f"/runs/{run}/orbits", query={"every": every, "count": count})

    # -- changing ---------------------------------------------------------------------------------

    def set(
        self, machine: str, values: Mapping[str, float], reason: str = "", atomic: bool = True
    ) -> dict[str, Any]:
        """Write setting channels as one logged change; raises :class:`Refused` if it was not applied."""
        body = {
            "changes": [{"channel": c, "value": float(v)} for c, v in values.items()],
            "reason": reason,
            "atomic": atomic,
        }
        result = self._call("POST", f"/va/machines/{machine}/settings", body)
        if not result["applied"]:
            raise Refused(result)
        return result

    def reset(self, machine: str, channels: Iterable[str] | None = None, reason: str = "") -> dict[str, Any]:
        """Channels (all changed ones by default) back to the machine files' values."""
        body = {"channels": list(channels) if channels is not None else None, "reason": reason}
        return self._call("POST", f"/va/machines/{machine}/reset", body)

    def snapshot(self, machine: str, name: str) -> dict[str, Any]:
        return self._call("POST", f"/va/machines/{machine}/snapshots", {"name": name})

    def snapshots(self, machine: str) -> list[dict[str, Any]]:
        return self._call("GET", f"/va/machines/{machine}/snapshots")

    def restore(self, machine: str, snapshot: int, reason: str = "") -> dict[str, Any]:
        return self._call("POST", f"/va/machines/{machine}/snapshots/{snapshot}/restore", {"reason": reason})

    def kick(self, machine: str, x: float = 0.0, y: float = 0.0, reason: str = "") -> None:
        self._call(
            "POST", f"/va/machines/{machine}/beam", {"action": "kick", "x": x, "y": y, "reason": reason}
        )

    def inject(self, machine: str, reason: str = "") -> None:
        self._call("POST", f"/va/machines/{machine}/beam", {"action": "inject", "reason": reason})

    def acquire(self, machine: str, reason: str = "") -> None:
        """Capture a ring's BPM buffer now (``captures`` lists it once saved)."""
        self._call("POST", f"/va/machines/{machine}/acquire", {"reason": reason})

    def captures(self, machine: str) -> list[dict[str, Any]]:
        """The captures of a machine's run: its BPM buffer frozen by kicks, injections, losses, requests."""
        run = self.machine(machine)["run"]
        if run is None:
            raise VaError(409, f"{machine} is not running")
        return self._call("GET", f"/runs/{run}/captures")

    def fork(self, machine: str, name: str | None = None, reason: str = "") -> dict[str, Any]:
        """A sandbox of your own, with the machine's settings as they are now."""
        return self._call("POST", f"/va/machines/{machine}/fork", {"name": name, "reason": reason})

    def start(self, machine: str, reason: str = "") -> dict[str, Any]:
        return self._call("POST", f"/va/machines/{machine}/start", {"reason": reason})

    def stop(self, machine: str, reason: str = "") -> dict[str, Any]:
        return self._call("POST", f"/va/machines/{machine}/stop", {"reason": reason})
