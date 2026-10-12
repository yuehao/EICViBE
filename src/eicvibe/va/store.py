"""What the virtual accelerator keeps (ADR 0008): accounts, machines, live settings and the log.

One SQLite file in the service's state folder. The write log and the action log
are append-only (triggers refuse updates and deletions), so what was changed,
by whom and why, is never lost; the live settings of each machine are the
current state the log leads to. The machine files (ADR 0005) are never touched:
what people changed lives here, on top of them.
"""

import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    name TEXT PRIMARY KEY, role TEXT NOT NULL, password TEXT NOT NULL,
    created TEXT NOT NULL, disabled INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS tokens (
    hash TEXT PRIMARY KEY, user TEXT NOT NULL, kind TEXT NOT NULL, label TEXT NOT NULL,
    created TEXT NOT NULL, expires TEXT
);
CREATE TABLE IF NOT EXISTS machines (
    id TEXT PRIMARY KEY, spec TEXT NOT NULL, created TEXT NOT NULL, created_by TEXT NOT NULL, removed TEXT
);
CREATE TABLE IF NOT EXISTS live (
    machine TEXT NOT NULL, channel TEXT NOT NULL, value REAL NOT NULL, user TEXT NOT NULL, time TEXT NOT NULL,
    PRIMARY KEY (machine, channel)
);
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT, machine TEXT NOT NULL, name TEXT NOT NULL,
    time TEXT NOT NULL, user TEXT NOT NULL, live TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS writes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT NOT NULL, user TEXT NOT NULL, machine TEXT NOT NULL,
    channel TEXT NOT NULL, unit TEXT NOT NULL, old REAL, new REAL NOT NULL, applied INTEGER NOT NULL,
    refusal TEXT, correlation TEXT NOT NULL, reason TEXT NOT NULL, run TEXT, turn INTEGER
);
CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT NOT NULL, user TEXT NOT NULL, machine TEXT,
    action TEXT NOT NULL, detail TEXT NOT NULL, reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS proposals (
    id TEXT PRIMARY KEY, machine TEXT NOT NULL, created TEXT NOT NULL, user TEXT NOT NULL,
    agent INTEGER NOT NULL, operation TEXT NOT NULL, request TEXT NOT NULL, proposal TEXT NOT NULL,
    state TEXT NOT NULL, turn INTEGER, decided TEXT, decided_by TEXT, applied_turn INTEGER,
    correlation TEXT, verification TEXT
);
CREATE INDEX IF NOT EXISTS proposals_by_machine ON proposals (machine, created);
CREATE INDEX IF NOT EXISTS writes_by_machine ON writes (machine, id);
CREATE INDEX IF NOT EXISTS writes_by_correlation ON writes (correlation);
CREATE TRIGGER IF NOT EXISTS writes_no_update BEFORE UPDATE ON writes
    BEGIN SELECT RAISE(ABORT, 'the write log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS writes_no_delete BEFORE DELETE ON writes
    BEGIN SELECT RAISE(ABORT, 'the write log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS actions_no_update BEFORE UPDATE ON actions
    BEGIN SELECT RAISE(ABORT, 'the action log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS actions_no_delete BEFORE DELETE ON actions
    BEGIN SELECT RAISE(ABORT, 'the action log is append-only'); END;
"""

#: How long a browser login lasts.
SESSION_HOURS = 12


class Role(IntEnum):
    """What a user may do, each role everything the one before may (ADR 0003)."""

    VIEWER = 0
    """Reads machines, settings and the log."""
    PHYSICIST = 1
    """Changes settings of shared machines, within limits; runs sandboxes of their own."""
    OPERATOR = 2
    """Also starts and stops shared machines, kicks and injects, resets and restores them."""
    ADMIN = 3
    """Also creates and removes shared machines and manages users."""

    @classmethod
    def named(cls, name: str) -> Role:
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"no role {name!r}; roles: {', '.join(r.name.lower() for r in cls)}") from None

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class User:
    name: str
    role: Role
    agent: bool = False
    """Calling with an agent's token: it may propose on shared machines, and act only in its sandboxes."""

    def can(self, role: Role) -> bool:
        return self.role >= role


ANONYMOUS = User("anonymous", Role.VIEWER)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${digest.hex()}"


def _password_matches(stored: str, password: str) -> bool:
    _, salt, _ = stored.split("$")
    return hmac.compare_digest(hash_password(password, bytes.fromhex(salt)), stored)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class Store:
    """The service's SQLite file; safe to use from the service's threads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _rows(self, sql: str, args: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, tuple(args)).fetchall()

    def _run(self, sql: str, args: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._db.execute(sql, tuple(args))

    # -- users and tokens ---------------------------------------------------------------

    def add_user(self, name: str, role: Role, password: str | None = None) -> str:
        """Add a user; returns the password (a new random one unless given)."""
        if not name or not name.replace("-", "").replace("_", "").replace(".", "").isalnum():
            raise ValueError("a user name is letters, digits, '-', '_' and '.'")
        password = password or secrets.token_urlsafe(12)
        try:
            self._run(
                "INSERT INTO users (name, role, password, created) VALUES (?, ?, ?, ?)",
                (name, role.label, hash_password(password), now()),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"user {name} exists already") from None
        return password

    def users(self) -> list[dict[str, Any]]:
        return [
            {"name": r["name"], "role": r["role"], "created": r["created"], "disabled": bool(r["disabled"])}
            for r in self._rows("SELECT * FROM users ORDER BY name")
        ]

    def user(self, name: str) -> User | None:
        rows = self._rows("SELECT * FROM users WHERE name = ? AND disabled = 0", (name,))
        return User(rows[0]["name"], Role.named(rows[0]["role"])) if rows else None

    def has_users(self) -> bool:
        return bool(self._rows("SELECT 1 FROM users LIMIT 1"))

    def set_role(self, name: str, role: Role) -> None:
        if not self._run("UPDATE users SET role = ? WHERE name = ?", (role.label, name)).rowcount:
            raise KeyError(name)

    def set_disabled(self, name: str, disabled: bool) -> None:
        if not self._run("UPDATE users SET disabled = ? WHERE name = ?", (int(disabled), name)).rowcount:
            raise KeyError(name)
        if disabled:
            self._run("DELETE FROM tokens WHERE user = ?", (name,))

    def set_password(self, name: str, password: str | None = None) -> str:
        password = password or secrets.token_urlsafe(12)
        if not self._run(
            "UPDATE users SET password = ? WHERE name = ?", (hash_password(password), name)
        ).rowcount:
            raise KeyError(name)
        self._run("DELETE FROM tokens WHERE user = ? AND kind = 'session'", (name,))
        return password

    def check_password(self, name: str, password: str) -> User | None:
        rows = self._rows("SELECT * FROM users WHERE name = ? AND disabled = 0", (name,))
        if rows and _password_matches(rows[0]["password"], password):
            return User(rows[0]["name"], Role.named(rows[0]["role"]))
        return None

    def issue_token(self, user: str, kind: str, label: str = "", hours: float | None = None) -> str:
        """A new token: ``session`` (a browser's login), ``api`` (a notebook's, until revoked) or
        ``agent`` (a model's harness: its user's rights, but on shared machines it only proposes)."""
        token = secrets.token_urlsafe(32)
        expires = (datetime.now(UTC) + timedelta(hours=hours)).isoformat() if hours else None
        self._run(
            "INSERT INTO tokens (hash, user, kind, label, created, expires) VALUES (?, ?, ?, ?, ?, ?)",
            (_token_hash(token), user, kind, label, now(), expires),
        )
        return token

    def resolve_token(self, token: str) -> User | None:
        rows = self._rows("SELECT * FROM tokens WHERE hash = ?", (_token_hash(token),))
        if not rows:
            return None
        if rows[0]["expires"] and datetime.fromisoformat(rows[0]["expires"]) < datetime.now(UTC):
            self.revoke_token(token)
            return None
        found = self.user(rows[0]["user"])
        return replace(found, agent=True) if found and rows[0]["kind"] == "agent" else found

    def revoke_token(self, token: str) -> None:
        self._run("DELETE FROM tokens WHERE hash = ?", (_token_hash(token),))

    def tokens(self, user: str) -> list[dict[str, Any]]:
        return [
            {"label": r["label"], "created": r["created"], "kind": r["kind"]}
            for r in self._rows(
                "SELECT * FROM tokens WHERE user = ? AND kind IN ('api', 'agent') ORDER BY created", (user,)
            )
        ]

    # -- machines, live settings, snapshots ----------------------------------------------------

    def add_machine(self, machine_id: str, spec: Mapping[str, Any], user: str) -> None:
        try:
            self._run(
                "INSERT INTO machines (id, spec, created, created_by) VALUES (?, ?, ?, ?)",
                (machine_id, json.dumps(dict(spec)), now(), user),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"machine {machine_id} exists already") from None

    def update_machine(self, machine_id: str, spec: Mapping[str, Any]) -> None:
        self._run("UPDATE machines SET spec = ? WHERE id = ?", (json.dumps(dict(spec)), machine_id))

    def remove_machine(self, machine_id: str) -> None:
        self._run("UPDATE machines SET removed = ? WHERE id = ?", (now(), machine_id))

    def machines(self) -> list[dict[str, Any]]:
        return [
            json.loads(r["spec"]) | {"created": r["created"], "created_by": r["created_by"]}
            for r in self._rows("SELECT * FROM machines WHERE removed IS NULL ORDER BY created")
        ]

    def machine_ids(self) -> set[str]:
        """Every id ever used, removed machines' too: an id names one machine in the log, for good."""
        return {r["id"] for r in self._rows("SELECT id FROM machines")}

    def live(self, machine: str) -> dict[str, dict[str, Any]]:
        """The machine's live settings: value, and who set it when, by channel."""
        return {
            r["channel"]: {"value": r["value"], "user": r["user"], "time": r["time"]}
            for r in self._rows("SELECT * FROM live WHERE machine = ?", (machine,))
        }

    def record_writes(
        self,
        machine: str,
        user: str,
        entries: list[dict[str, Any]],
        correlation: str,
        reason: str,
        run: str | None = None,
        turn: int | None = None,
    ) -> None:
        """Append writes to the log and apply the applied ones to the live settings, in one transaction.

        An entry has ``channel``, ``unit``, ``old``, ``new``, ``applied``, ``refusal`` and ``original``:
        a value back at its original leaves the live settings.
        """
        time = now()
        with self._lock:
            self._db.execute("BEGIN")
            try:
                for e in entries:
                    self._db.execute(
                        "INSERT INTO writes (time, user, machine, channel, unit, old, new, applied, refusal, "
                        "correlation, reason, run, turn) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            time,
                            user,
                            machine,
                            e["channel"],
                            e["unit"],
                            e["old"],
                            e["new"],
                            int(e["applied"]),
                            e.get("refusal"),
                            correlation,
                            reason,
                            run,
                            turn,
                        ),
                    )
                    if not e["applied"]:
                        continue
                    if e.get("original"):
                        self._db.execute(
                            "DELETE FROM live WHERE machine = ? AND channel = ?", (machine, e["channel"])
                        )
                    else:
                        self._db.execute(
                            "INSERT INTO live (machine, channel, value, user, time) VALUES (?, ?, ?, ?, ?) "
                            "ON CONFLICT (machine, channel) DO UPDATE SET value = excluded.value, "
                            "user = excluded.user, time = excluded.time",
                            (machine, e["channel"], e["new"], user, time),
                        )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def copy_live(self, source: str, target: str, user: str) -> int:
        time = now()
        rows = self._rows("SELECT channel, value FROM live WHERE machine = ?", (source,))
        with self._lock:
            for r in rows:
                self._db.execute(
                    "INSERT OR REPLACE INTO live (machine, channel, value, user, time) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (target, r["channel"], r["value"], user, time),
                )
        return len(rows)

    def add_snapshot(self, machine: str, name: str, user: str, live: Mapping[str, float]) -> int:
        cursor = self._run(
            "INSERT INTO snapshots (machine, name, time, user, live) VALUES (?, ?, ?, ?, ?)",
            (machine, name, now(), user, json.dumps(dict(live))),
        )
        return int(cursor.lastrowid)

    def snapshots(self, machine: str) -> list[dict[str, Any]]:
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "time": r["time"],
                "user": r["user"],
                "changes": len(json.loads(r["live"])),
            }
            for r in self._rows("SELECT * FROM snapshots WHERE machine = ? ORDER BY id DESC", (machine,))
        ]

    def snapshot(self, snapshot_id: int) -> dict[str, Any]:
        rows = self._rows("SELECT * FROM snapshots WHERE id = ?", (snapshot_id,))
        if not rows:
            raise KeyError(snapshot_id)
        r = rows[0]
        return {"id": r["id"], "machine": r["machine"], "name": r["name"], "live": json.loads(r["live"])}

    # -- the logs ------------------------------------------------------------------------

    # -- proposals ------------------------------------------------------------------------------

    def add_proposal(
        self, ident: str, machine: str, user: User, operation: str, request: Mapping[str, Any],
        proposal: Mapping[str, Any], turn: int | None,
    ) -> None:  # fmt: skip
        self._run(
            "INSERT INTO proposals (id, machine, created, user, agent, operation, request, proposal, state,"
            " turn) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'proposed', ?)",
            (ident, machine, now(), user.name, int(user.agent), operation, json.dumps(request),
             json.dumps(proposal), turn),
        )  # fmt: skip

    def update_proposal(self, ident: str, **fields: Any) -> None:
        names = ("state", "decided", "decided_by", "applied_turn", "correlation", "verification")
        sets = [(k, json.dumps(v) if k == "verification" else v) for k, v in fields.items() if k in names]
        self._run(
            f"UPDATE proposals SET {', '.join(f'{k} = ?' for k, _ in sets)} WHERE id = ?",
            [v for _, v in sets] + [ident],
        )

    def proposal(self, ident: str) -> dict[str, Any]:
        rows = self._rows("SELECT * FROM proposals WHERE id = ?", (ident,))
        if not rows:
            raise KeyError(f"no proposal {ident}")
        return _proposal_row(rows[0])

    def proposals(self, machine: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._rows(
            "SELECT * FROM proposals WHERE machine = ? ORDER BY created DESC LIMIT ?", (machine, limit)
        )
        return [_proposal_row(r) for r in rows]

    def act(
        self, user: str, action: str, machine: str | None = None, reason: str = "", **detail: Any
    ) -> None:
        """Append an action (start, stop, fork, reset, kick, a new user, ...) to the action log."""
        self._run(
            "INSERT INTO actions (time, user, machine, action, detail, reason) VALUES (?, ?, ?, ?, ?, ?)",
            (now(), user, machine, action, json.dumps(detail, default=str), reason),
        )

    def writes(
        self,
        *,
        machine: str | None = None,
        user: str | None = None,
        channel: str | None = None,
        correlation: str | None = None,
        before: int | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """The write log, newest first; ``before`` an id pages back."""
        where, args = self._filter(machine=machine, user=user, channel=channel, correlation=correlation)
        if before is not None:
            where.append("id < ?")
            args.append(before)
        sql = (
            f"SELECT * FROM writes {'WHERE ' + ' AND '.join(where) if where else ''} ORDER BY id DESC LIMIT ?"
        )
        return [dict(r) | {"applied": bool(r["applied"])} for r in self._rows(sql, [*args, limit])]

    def actions(
        self,
        *,
        machine: str | None = None,
        user: str | None = None,
        before: int | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        where, args = self._filter(machine=machine, user=user)
        if before is not None:
            where.append("id < ?")
            args.append(before)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        sql = f"SELECT * FROM actions {clause} ORDER BY id DESC LIMIT ?"
        return [dict(r) | {"detail": json.loads(r["detail"])} for r in self._rows(sql, [*args, limit])]

    @staticmethod
    def _filter(**fields: Any) -> tuple[list[str], list[Any]]:
        where, args = [], []
        for key, value in fields.items():
            if value is not None:
                where.append(f"{key} = ?")
                args.append(value)
        return where, args


def _proposal_row(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    out["agent"] = bool(out["agent"])
    for key in ("request", "proposal", "verification"):
        out[key] = json.loads(out[key]) if out[key] else None
    return out
