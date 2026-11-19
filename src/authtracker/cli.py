"""authtracker config show | doctor"""
from __future__ import annotations

import argparse
import json
import sys

from .backends import get_backend
from .settings import get_settings


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="authtracker")
    sub = p.add_subparsers(dest="cmd", required=True)
    cfg = sub.add_parser("config", help="show resolved settings")
    cfg.add_argument("action", choices=["show"])
    sub.add_parser("doctor", help="check settings and backend connectivity")
    sub.add_parser("sync", help="upload operations queued while the server was unreachable")
    args = p.parse_args(argv)

    s = get_settings()
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
