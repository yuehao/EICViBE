"""Accounts and the log of a virtual accelerator's state folder: ``eicvibe-va``.

    eicvibe-va users STATE
    eicvibe-va add-user STATE NAME --role physicist      prints the new password
    eicvibe-va password STATE NAME                       prints a new password
    eicvibe-va role STATE NAME operator
    eicvibe-va token STATE NAME --label "orbit notebook" prints a token for the notebook client
    eicvibe-va log STATE [--machine M] [--user U] [--limit 50]
    eicvibe-va actions STATE [--machine M] [--limit 50]

The service itself runs in the web server: ``eicvibe-web MACHINE --va STATE``
(ADR 0008); these commands work on the same folder while it runs. Every one of
them is logged as an action of the user "console".
"""

import argparse
import sys
from pathlib import Path

from .store import Role, Store

CONSOLE = "console"


def _store(args: argparse.Namespace) -> Store:
    path = Path(args.state) / "va.sqlite"
    if not path.is_file() and args.command not in ("add-user",):
        raise SystemExit(f"no virtual accelerator state in {args.state}")
    return Store(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="eicvibe-va", description="Accounts and the log of a virtual accelerator."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("users", help="list the users")
    c.add_argument("state")
    c = commands.add_parser("add-user", help="add a user; prints the password")
    c.add_argument("state")
    c.add_argument("name")
    c.add_argument("--role", default="physicist", help="viewer, physicist, operator or admin")
    c = commands.add_parser("password", help="give a user a new password; prints it")
    c.add_argument("state")
    c.add_argument("name")
    c = commands.add_parser("role", help="change a user's role")
    c.add_argument("state")
    c.add_argument("name")
    c.add_argument("role")
    c = commands.add_parser("token", help="a personal token for the notebook client; prints it")
    c.add_argument("state")
    c.add_argument("name")
    c.add_argument("--label", default="notebook")
    for name in ("log", "actions"):
        c = commands.add_parser(name, help=f"the {'write' if name == 'log' else 'action'} log, newest first")
        c.add_argument("state")
        c.add_argument("--machine")
        c.add_argument("--user")
        c.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(argv)
    store = _store(args)
    try:
        if args.command == "users":
            for u in store.users():
                print(f"{u['name']:<20} {u['role']:<10} {'disabled' if u['disabled'] else ''}")
        elif args.command == "add-user":
            role = Role.named(args.role)
            password = store.add_user(args.name, role)
            store.act(CONSOLE, "add user", name=args.name, role=role.label)
            print(password)
        elif args.command == "password":
            print(store.set_password(args.name))
            store.act(CONSOLE, "reset password", name=args.name)
        elif args.command == "role":
            role = Role.named(args.role)
            store.set_role(args.name, role)
            store.act(CONSOLE, "change role", name=args.name, role=role.label)
        elif args.command == "token":
            if store.user(args.name) is None:
                raise SystemExit(f"no user {args.name}")
            print(store.issue_token(args.name, "api", args.label))
            store.act(CONSOLE, "issue token", name=args.name, label=args.label)
        elif args.command == "log":
            for w in reversed(store.writes(machine=args.machine, user=args.user, limit=args.limit)):
                outcome = "" if w["applied"] else f"  REFUSED: {w['refusal']}"
                print(
                    f"{w['time'][:19]}  {w['user']:<10} {w['machine']:<14} {w['channel']:<22} "
                    f"{w['old']!s:>12} -> {w['new']:<12} {w['unit']:<4} "
                    f"[{w['correlation']}] {w['reason']}{outcome}"
                )
        elif args.command == "actions":
            for a in reversed(store.actions(machine=args.machine, user=args.user, limit=args.limit)):
                print(
                    f"{a['time'][:19]}  {a['user']:<10} {a['machine'] or '':<14} {a['action']:<14} "
                    f"{a['detail']} {a['reason']}"
                )
    except (KeyError, ValueError) as error:
        raise SystemExit(str(error)) from None
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
