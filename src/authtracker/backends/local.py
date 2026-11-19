"""JSON-file backend: no server, no dependencies. Good for tests and offline use."""
from __future__ import annotations

import json
import pickle
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..errors import BackendError
from ..settings import Settings
from .base import Backend, TAG_MODEL_URI


class LocalBackend(Backend):
    def __init__(self, settings: Settings):
        self.root = Path(settings.tracking_uri or ".authtracker").expanduser()
        (self.root / "runs").mkdir(parents=True, exist_ok=True)
        self._registry_file = self.root / "registry.json"

    def ping(self) -> None:
        if not self.root.is_dir():
            raise BackendError(f"{self.root} is not a directory")

    # ---- storage helpers
    def _run_path(self, run_id: str) -> Path:
        return self.root / "runs" / f"{run_id}.json"

    def _load(self, run_id: str) -> dict:
        try:
            return json.loads(self._run_path(run_id).read_text())
        except FileNotFoundError:
            raise BackendError(f"Run {run_id} not found") from None

    def _save(self, run: dict) -> None:
        self._run_path(run["run_id"]).write_text(json.dumps(run))

    def _artifacts(self, run_id: str) -> Path:
        p = self.root / "artifacts" / run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _registry(self) -> dict:
        if self._registry_file.exists():
            return json.loads(self._registry_file.read_text())
        return {}

    def _save_registry(self, reg: dict) -> None:
        self._registry_file.write_text(json.dumps(reg))

    # ---- runs
    def start_run(self, experiment, run_name, tags, parent_id):
        run_id = uuid.uuid4().hex
        self._save({
            "run_id": run_id, "experiment": experiment, "name": run_name,
            "tags": dict(tags), "params": {}, "metrics": {}, "status": "RUNNING",
            "parent": parent_id, "start": time.time(),
        })
        return run_id

    def end_run(self, run_id, status):
        run = self._load(run_id)
        run["status"] = status
        self._save(run)

    def set_tags(self, run_id, tags):
        run = self._load(run_id)
        run["tags"].update({k: str(v) for k, v in tags.items()})
        self._save(run)

    def log_params(self, run_id, params):
        run = self._load(run_id)
        for k, v in params.items():
            if k in run["params"] and run["params"][k] != str(v):
                raise BackendError(f"Param {k!r} already logged with a different value")
            run["params"][k] = str(v)
        self._save(run)

    def log_metrics(self, run_id, metrics, step):
        run = self._load(run_id)
        for k, v in metrics.items():
            run["metrics"].setdefault(k, []).append([step or 0, float(v)])
        self._save(run)

    def log_artifact(self, run_id, path):
        shutil.copy(path, self._artifacts(run_id) / Path(path).name)

    def log_model(self, run_id, model, flavor, **kwargs):
        target = self._artifacts(run_id) / "model.pkl"
        with open(target, "wb") as f:
            pickle.dump(model, f)
        uri = str(target)
        self.set_tags(run_id, {TAG_MODEL_URI: uri})
        return uri

    def log_model_dir(self, run_id, local_dir):
        target = self._artifacts(run_id) / "model_bundle"
        shutil.copytree(local_dir, target, dirs_exist_ok=True)
        uri = str(target)
        self.set_tags(run_id, {TAG_MODEL_URI: uri})
        return uri

    def download_bundle(self, name, alias, dst=None):
        import tempfile
        version = self.get_alias(name, alias)
        if version is None:
            return None
        source = Path(self._registry()[name]["versions"][version]["uri"])
        if not (source / "bundle.json").is_file():
            return None
        target = Path(dst or tempfile.mkdtemp(prefix="authtracker-load-"))
        shutil.copytree(source, target, dirs_exist_ok=True)
        return str(target)

    def find_run(self, experiment, name, tags, parent_id):
        best = None
        for p in (self.root / "runs").glob("*.json"):
            run = json.loads(p.read_text())
            if run["experiment"] != experiment or run["name"] != name or run.get("parent") != parent_id:
                continue
            if any(run["tags"].get(k) != v for k, v in tags.items()):
                continue
            if best is None or run["start"] > best["start"]:
                best = run
        return best["run_id"] if best else None

    def resume_run(self, run_id, parent_id):
        run = self._load(run_id)
        run["status"] = "RUNNING"
        self._save(run)
        return run_id

    def last_step(self, run_id):
        steps = [step for points in self._load(run_id)["metrics"].values() for step, _ in points]
        return max(steps) if steps else None

    def download_artifact(self, run_id, name, folder):
        source = self._artifacts(run_id) / name
        if not source.is_file():
            return None
        return shutil.copy(source, Path(folder) / name)

    def _summary(self, run: dict) -> Dict[str, Any]:
        return {
            "run_id": run["run_id"], "name": run["name"], "tags": run["tags"],
            "params": run["params"],
            "metrics": {k: v[-1][1] for k, v in run["metrics"].items() if v},
        }

    def get_run(self, run_id):
        return self._summary(self._load(run_id))

    def search_runs(self, experiment, tags, metric, mode, limit):
        found = []
        for p in (self.root / "runs").glob("*.json"):
            run = json.loads(p.read_text())
            if run["experiment"] != experiment:
                continue
            if any(run["tags"].get(k) != v for k, v in tags.items()):
                continue
            s = self._summary(run)
            if metric in s["metrics"]:
                found.append(s)
        found.sort(key=lambda r: r["metrics"][metric], reverse=(mode == "max"))
        return found[:limit]

    # ---- registry
    def register(self, run_id, name, model_uri):
        reg = self._registry()
        entry = reg.setdefault(name, {"versions": {}, "aliases": {}})
        version = str(len(entry["versions"]) + 1)
        entry["versions"][version] = {"run_id": run_id, "uri": model_uri}
        self._save_registry(reg)
        return version

    def set_alias(self, name, version, alias):
        reg = self._registry()
        if name not in reg or version not in reg[name]["versions"]:
            raise BackendError(f"Model {name!r} version {version} does not exist")
        reg[name]["aliases"][alias] = version
        self._save_registry(reg)

    def get_alias(self, name, alias):
        return self._registry().get(name, {}).get("aliases", {}).get(alias)

    def version_run_id(self, name, version):
        try:
            return self._registry()[name]["versions"][str(version)]["run_id"]
        except KeyError:
            raise BackendError(f"Model {name!r} version {version} does not exist") from None

    def load(self, name, alias):
        version = self.get_alias(name, alias)
        if version is None:
            raise BackendError(f"Model {name!r} has no alias {alias!r}")
        uri = self._registry()[name]["versions"][version]["uri"]
        with open(uri, "rb") as f:
            return pickle.load(f)
