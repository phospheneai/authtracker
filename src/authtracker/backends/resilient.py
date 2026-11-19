"""Retries and an on-disk offline queue around any backend.

A tracking outage must not kill a training job:

* Every call is retried with exponential backoff when the failure looks transient.
* Write operations (start/end run, tags, params, metrics, artifacts) that still fail are
  appended to a queue directory and replayed, in order, once the server is back.
* Operations that need an answer or a live model (log_model, register, search, load...)
  cannot be queued; they raise BackendError after the retries.

Queued files are replayed automatically by the process that wrote them (when the server
answers again, and at interpreter exit). Leftovers from crashed processes are uploaded
with ``authtracker sync``.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..errors import BackendError
from ..settings import Settings
from .base import Backend

log = logging.getLogger("authtracker")

OFFLINE = "offline-"  # prefix of run ids handed out while the server is unreachable


class _Unreachable(Exception):
    """Transient failures persisted through all retries."""


class _Unresolved(Exception):
    """A queued op refers to an offline run id that was never created."""


class ResilientBackend(Backend):
    def __init__(self, inner: Backend, settings: Settings, *,
                 sleep: Callable[[float], None] = time.sleep, cooldown: float = 30.0):
        self.inner = inner
        self.retries = settings.retries
        self.wait = settings.retry_wait
        self.dir = Path(settings.queue_dir
                        or Path.home() / ".authtracker" / "queue").expanduser()
        (self.dir / "files").mkdir(parents=True, exist_ok=True)
        (self.dir / "failed").mkdir(exist_ok=True)
        self.token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._sleep = sleep
        self._cooldown = cooldown
        self._down_until = 0.0
        self._warned = False
        self._atexit_registered = False
        self._idmap_file = self.dir / "idmap.json"
        self._idmap: Dict[str, str] = self._read_idmap()

    @property
    def captures_console(self) -> bool:  # type: ignore[override]
        return bool(getattr(self.inner, "captures_console", False))

    def __getattr__(self, name: str) -> Any:  # delegate backend-specific attributes
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    # ------------------------------------------------------------------ helpers
    def _transient(self, exc: Exception) -> bool:
        return self.inner.is_transient(exc)

    def _retry(self, fn: Callable[[], Any]) -> Any:
        attempt = 0
        while True:
            try:
                return fn()
            except Exception as e:
                if not self._transient(e):
                    raise
                if attempt >= self.retries:
                    raise _Unreachable(e) from e
                self._sleep(self.wait * (2 ** attempt))
                attempt += 1

    def _mark_down(self) -> None:
        self._down_until = time.monotonic() + self._cooldown

    def _read_idmap(self) -> Dict[str, str]:
        try:
            return json.loads(self._idmap_file.read_text())
        except (FileNotFoundError, ValueError):
            return {}

    def _save_idmap(self) -> None:
        self._idmap_file.write_text(json.dumps(self._idmap))

    def _res(self, run_id: Optional[str]) -> Optional[str]:
        return self._idmap.get(run_id, run_id) if run_id else run_id

    def _unresolved(self, run_id: Optional[str]) -> bool:
        r = self._res(run_id)
        return bool(r) and r.startswith(OFFLINE)

    # ------------------------------------------------------------------ queue
    def _entries(self, all_tokens: bool) -> List[Path]:
        files = sorted(self.dir.glob("*.json"))
        files = [p for p in files if p.name not in ("idmap.json",)]
        if not all_tokens:
            files = [p for p in files if p.name.endswith(f"-{self.token}.json")]
        return files

    def pending(self) -> int:
        """Number of queued operations (from every process) waiting to be uploaded."""
        return len(self._entries(all_tokens=True))

    def _enqueue(self, op: str, args: Dict[str, Any]) -> None:
        path = self.dir / f"{time.time_ns():020d}-{self.token}.json"
        path.write_text(json.dumps({"op": op, "args": args}, default=str))
        if not self._warned:
            self._warned = True
            log.warning("Tracking server unreachable: queuing data in %s "
                        "(it is uploaded automatically, or run `authtracker sync`)", self.dir)
        if not self._atexit_registered:
            self._atexit_registered = True
            atexit.register(self._flush_at_exit)

    def _flush_at_exit(self) -> None:
        try:
            self.flush(force=True)
            left = len(self._entries(all_tokens=False))
            if left:
                log.warning("%d tracking operations remain queued in %s; "
                            "run `authtracker sync` later", left, self.dir)
        except Exception:
            pass

    def _fail(self, path: Path, exc: Exception) -> None:
        log.warning("Dropping queued op %s: %s", path.name, exc)
        shutil.move(str(path), str(self.dir / "failed" / path.name))

    def _apply(self, entry: Dict[str, Any]) -> None:
        op, a = entry["op"], entry["args"]
        rid = self._res(a.get("run_id"))
        if rid is not None and rid.startswith(OFFLINE):
            raise _Unresolved(rid)
        if op == "start_run":
            parent = self._res(a["parent_id"])
            if parent is not None and parent.startswith(OFFLINE):
                raise _Unresolved(parent)
            real = self.inner.start_run(a["experiment"], a["run_name"], a["tags"], parent)
            self._idmap[a["placeholder"]] = real
            self._save_idmap()
        elif op == "end_run":
            self.inner.end_run(rid, a["status"])
        elif op == "set_tags":
            self.inner.set_tags(rid, a["tags"])
        elif op == "log_params":
            self.inner.log_params(rid, a["params"])
        elif op == "log_metrics":
            self.inner.log_metrics(rid, a["metrics"], a["step"])
        elif op == "log_artifact":
            self.inner.log_artifact(rid, a["path"])
        else:
            raise BackendError(f"Unknown queued op {op!r}")

    def flush(self, *, force: bool = False, all_tokens: bool = False) -> int:
        """Replay queued operations in order. Returns how many were uploaded."""
        files = self._entries(all_tokens)
        if not files:
            return 0
        if not force and time.monotonic() < self._down_until:
            return 0
        done = 0
        for path in files:
            entry = json.loads(path.read_text())
            try:
                self._apply(entry)
            except _Unresolved as e:
                self._fail(path, e)
                continue
            except Exception as e:
                if self._transient(e):
                    self._mark_down()
                    break
                self._fail(path, e)
                continue
            if entry["op"] == "log_artifact":
                shutil.rmtree(Path(entry["args"]["path"]).parent, ignore_errors=True)
            path.unlink()
            done += 1
        if done:
            log.info("Uploaded %d queued tracking operations", done)
        return done

    # ------------------------------------------------------------------ queued writes
    def _write(self, op: str, args: Dict[str, Any], live: Callable[[], Any],
               run_id: Optional[str]) -> None:
        self.flush()
        if self._entries(False) or self._unresolved(run_id):
            self._enqueue(op, args)
            return
        try:
            self._retry(live)
        except _Unreachable:
            self._mark_down()
            self._enqueue(op, args)

    def start_run(self, experiment, run_name, tags, parent_id):
        self.flush()

        def enqueue() -> str:
            placeholder = OFFLINE + uuid.uuid4().hex
            self._enqueue("start_run", {
                "experiment": experiment, "run_name": run_name, "tags": tags,
                "parent_id": parent_id, "placeholder": placeholder})
            return placeholder

        if self._entries(False) or self._unresolved(parent_id):
            return enqueue()
        try:
            return self._retry(lambda: self.inner.start_run(
                experiment, run_name, tags, self._res(parent_id)))
        except _Unreachable:
            self._mark_down()
            return enqueue()

    def end_run(self, run_id, status):
        self._write("end_run", {"run_id": run_id, "status": status},
                    lambda: self.inner.end_run(self._res(run_id), status), run_id)

    def set_tags(self, run_id, tags):
        self._write("set_tags", {"run_id": run_id, "tags": tags},
                    lambda: self.inner.set_tags(self._res(run_id), tags), run_id)

    def log_params(self, run_id, params):
        self._write("log_params", {"run_id": run_id, "params": params},
                    lambda: self.inner.log_params(self._res(run_id), params), run_id)

    def log_metrics(self, run_id, metrics, step):
        self._write("log_metrics", {"run_id": run_id, "metrics": metrics, "step": step},
                    lambda: self.inner.log_metrics(self._res(run_id), metrics, step), run_id)

    def log_artifact(self, run_id, path):
        # Copy now: the caller may delete the file (e.g. a temp directory) before replay.
        def queue_copy() -> None:
            folder = self.dir / "files" / str(time.time_ns())  # keeps the file's real name
            folder.mkdir(parents=True)
            dest = folder / Path(path).name
            shutil.copy(path, dest)
            self._enqueue("log_artifact", {"run_id": run_id, "path": str(dest)})

        self.flush()
        if self._entries(False) or self._unresolved(run_id):
            return queue_copy()
        try:
            self._retry(lambda: self.inner.log_artifact(self._res(run_id), path))
        except _Unreachable:
            self._mark_down()
            queue_copy()

    # ------------------------------------------------------------------ not queueable
    def _read(self, what: str, fn: Callable[[], Any], run_id: Optional[str] = None) -> Any:
        self.flush()
        if self._unresolved(run_id):
            raise BackendError(f"{what}: run {run_id} is not on the server yet "
                               "(still queued offline)")
        try:
            return self._retry(fn)
        except _Unreachable as e:
            raise BackendError(
                f"{what}: tracking server unreachable after {self.retries} retries "
                f"({e.__cause__})") from e.__cause__

    def ping(self) -> None:
        self.inner.ping()

    def log_model(self, run_id, model, flavor, **kwargs):
        return self._read("log_model",
                          lambda: self.inner.log_model(self._res(run_id), model, flavor, **kwargs),
                          run_id)

    def log_model_dir(self, run_id, local_dir):
        return self._read("log_model_dir",
                          lambda: self.inner.log_model_dir(self._res(run_id), local_dir), run_id)

    def download_bundle(self, name, alias, dst=None):
        return self._read("download_bundle", lambda: self.inner.download_bundle(name, alias, dst))

    def download_artifact(self, run_id, name, folder):
        return self._read("download_artifact", lambda: self.inner.download_artifact(self._res(run_id), name, folder),
                          run_id)

    def find_run(self, experiment, name, tags, parent_id):
        return self._read("find_run", lambda: self.inner.find_run(experiment, name, tags, self._res(parent_id)),
                          parent_id)

    def resume_run(self, run_id, parent_id):
        return self._read("resume_run", lambda: self.inner.resume_run(self._res(run_id), self._res(parent_id)),
                          run_id)

    def last_step(self, run_id):
        return self._read("last_step", lambda: self.inner.last_step(self._res(run_id)), run_id)

    def get_run(self, run_id):
        return self._read("get_run", lambda: self.inner.get_run(self._res(run_id)), run_id)

    def search_runs(self, experiment, tags, metric, mode, limit):
        return self._read("search_runs",
                          lambda: self.inner.search_runs(experiment, tags, metric, mode, limit))

    def register(self, run_id, name, model_uri):
        return self._read("register",
                          lambda: self.inner.register(self._res(run_id), name, model_uri), run_id)

    def set_alias(self, name, version, alias):
        return self._read("set_alias", lambda: self.inner.set_alias(name, version, alias))

    def get_alias(self, name, alias):
        return self._read("get_alias", lambda: self.inner.get_alias(name, alias))

    def version_run_id(self, name, version):
        return self._read("version_run_id", lambda: self.inner.version_run_id(name, version))

    def load(self, name, alias):
        return self._read("load", lambda: self.inner.load(name, alias))
