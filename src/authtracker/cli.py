"""authtracker config show | doctor | sync | export | import"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from .backends import get_backend
from .settings import get_settings


def _transfer(args, settings) -> int:
    from . import registry
    from .errors import AuthTrackerError
    if args.tracking_uri:
        settings = dataclasses.replace(settings, tracking_uri=args.tracking_uri, registry_uri=args.tracking_uri)
    if args.cmd == "export" and bool(args.model) == args.all:
        print("FAIL: give one MODEL to export, or --all")
        return 1
    if args.cmd == "import" and args.all and args.name:
        print("FAIL: --name goes with one model; it cannot be used with --all")
        return 1
    try:
        if args.cmd == "export" and args.all:
            r = registry.export_all(args.out, settings=settings)
            for name, one in r["models"].items():
                print(f"exported {name}: {one['runs']} runs, {one['versions']} versions")
            print(f"{r['count']} models -> {r['path']}")
        elif args.cmd == "import" and args.all:
            r = registry.import_all(args.folder, settings=settings)
            for name, one in r["imported"].items():
                print(f"imported {name}: {one['runs']} runs, versions {one['versions'] or 'none'}")
            for name, reason in r["failed"].items():
                print(f"FAILED {name}: {reason}")
            print(f"{r['count']} imported, {len(r['failed'])} failed")
            return 1 if r["failed"] else 0
        elif args.cmd == "export":
            r = registry.export_model(args.model, args.out, settings=settings)
            print(f"exported {r['model']}: {r['runs']} runs, {r['metrics']} metric points, "
                  f"{r['models']} model folders, {r['versions']} versions -> {r['path']}")
        else:
            r = registry.import_model(args.folder, args.name, settings=settings)
            print(f"imported {r['model']}: {r['runs']} runs, {r['metrics']} metric points, {r['models']} model folders")
            if r["versions"]:
                print("versions (old -> new): " + ", ".join(f"{o}->{n}" for o, n in r["versions"].items()))
            if r["aliases"]:
                print("aliases: " + ", ".join(f"{a}=v{v}" for a, v in r["aliases"].items()))
    except AuthTrackerError as e:
        print(f"FAIL: {e}")
        return 1
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="authtracker")
    sub = p.add_subparsers(dest="cmd", required=True)
    cfg = sub.add_parser("config", help="show resolved settings")
    cfg.add_argument("action", choices=["show"])
    sub.add_parser("doctor", help="check settings and backend connectivity")
    sub.add_parser("sync", help="upload operations queued while the server was unreachable")
    ta = sub.add_parser("test-alert", help="send one test message to AUTHTRACKER_ALERT_WEBHOOK")
    ta.add_argument("message", nargs="?", default="authtracker test alert")
    exp = sub.add_parser("export", help="write a model (runs, files, versions, aliases) to a folder")
    exp.add_argument("model", nargs="?", help="the model to export (or use --all)")
    exp.add_argument("--all", action="store_true", help="export every model that has runs")
    exp.add_argument("--out", required=True, help="an empty or new folder")
    exp.add_argument("--tracking-uri", help="the MLflow server or store to read from (default: your settings)")
    imp = sub.add_parser("import", help="read a folder written by `authtracker export` into a store")
    imp.add_argument("folder")
    imp.add_argument("--all", action="store_true", help="the folder is an export of all models (from export --all)")
    imp.add_argument("--name", help="import under this model name (default: the exported name)")
    imp.add_argument("--tracking-uri", help="the MLflow server or store to write to (default: your settings)")
    args = p.parse_args(argv)

    s = get_settings()
    if args.cmd in ("export", "import"):
        return _transfer(args, s)
    if args.cmd == "test-alert":
        from . import alerts
        if not s.alert_webhook:
            print("FAIL: AUTHTRACKER_ALERT_WEBHOOK is not set")
            return 1
        sent = alerts.send(s, "custom", args.message, "Sent by `authtracker test-alert`.", "info", force=True)
        print("sent" if sent else "FAIL: the webhook did not accept the message (see the warning above)")
        return 0 if sent else 1
    if args.cmd == "config":
        print(json.dumps(s.masked(), indent=2))
        return 0

    if args.cmd == "sync":
        b = get_backend(s)
        if not hasattr(b, "flush"):
            print("Offline queue is disabled (AUTHTRACKER_OFFLINE_QUEUE=false)")
            return 0
        before = b.pending()
        done = b.flush(force=True, all_tokens=True)
        print(f"uploaded {done} of {before} queued operations; {b.pending()} remain")
        return 0 if b.pending() == 0 else 1

    print(f"backend      : {s.backend}")
    print(f"tracking_uri : {s.masked()['tracking_uri']}")
    print(f"alerts       : {s.masked()['alert_webhook'] or 'off (AUTHTRACKER_ALERT_WEBHOOK is not set)'}")
    try:
        get_backend(s).ping()
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        return 1
    print("OK: backend reachable")
    b = get_backend(s)
    if hasattr(b, "pending"):
        print(f"queued ops   : {b.pending()}  (authtracker sync uploads them)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
