"""Weights & Biases backend. Requires: pip install wandb

Mapping (same three levels as every backend):

    model        -> W&B project
    experiment   -> run group; the parent run (job type "experiment")
    run          -> a run (job type "run") in that group

Settings: sign in with WANDB_API_KEY, optional WANDB_ENTITY (team/user), WANDB_MODE=offline to
write locally and `wandb sync` later. AUTHTRACKER_TRACKING_URI is used as the W&B base URL for a
self-hosted server. Run ids look like "<project>/<wandb run id>".

Supported: runs, params, metrics, tags, files, best_run, and models saved as artifacts.
Not supported yet: the model registry (register / promote / rollback / load). Those raise a clear
BackendError.
"""
from __future__ import annotations

import os
import pickle
import shutil
from typing import Any, Dict, List, Optional

from ..errors import BackendError
from ..settings import Settings
from .base import TAG_LEVEL, TAG_MODEL_URI, Backend

_TAGS_KEY = "authtracker_tags"          # our key/value tags, stored in the run config
_PARENT_KEY = "authtracker_parent"
_INTERNAL = (_TAGS_KEY, _PARENT_KEY, "_wandb")


# Runs of this process. Shared by every WandbBackend instance, because authtracker may create a new
# backend object per call (for example for best_run) and it must still see runs started earlier.
_PROCESS_RECORDS: Dict[str, Dict[str, Any]] = {}
_PROCESS_RUNS: Dict[str, Any] = {}


def _reset_process_state() -> None:
    """Forget this process's runs (used by tests)."""
    _PROCESS_RECORDS.clear()
    _PROCESS_RUNS.clear()


def _wb_tags(tags: Dict[str, str]) -> List[str]:
    """W&B tags are plain strings (max 64 chars), so key/value pairs become 'key=value'."""
    return [f"{k}={v}"[:64] for k, v in tags.items()]


class WandbBackend(Backend):
    buffers_offline = True   # W&B has its own offline mode, so no extra authtracker queue
    captures_console = True  # W&B records print/logging output itself (the run's Logs tab)

    def __init__(self, settings: Settings, api: Any = None):
        import wandb  # noqa: F401  (ImportError is turned into a helpful message by the factory)
        self._wandb = wandb
        self._settings = settings
        self._api_obj = api
        self._records = _PROCESS_RECORDS    # latest values of this process's runs
        self._runs = _PROCESS_RUNS          # run id -> open wandb run

    # ------------------------------------------------------------------ helpers
    @property
    def _offline(self) -> bool:
        return os.environ.get("WANDB_MODE", "").lower() in ("offline", "disabled", "dryrun")

    def _api(self):
        if self._api_obj is None:
            overrides = {"base_url": self._settings.tracking_uri} if self._settings.tracking_uri else {}
            self._api_obj = self._wandb.Api(overrides=overrides)
        return self._api_obj

    def _entity(self) -> str:
        return os.environ.get("WANDB_ENTITY") or self._api().default_entity

    @staticmethod
    def _split(run_id: str):
        project, _, wb_id = run_id.partition("/")
        return project, wb_id

    def _open(self, run_id: str):
        try:
            return self._runs[run_id]
        except KeyError:
            raise BackendError(f"Run {run_id} is not open in this process") from None

    @staticmethod
    def _public(rec: Dict[str, Any]) -> Dict[str, Any]:
        return {"run_id": rec["run_id"], "name": rec["name"], "metrics": dict(rec["metrics"]),
                "params": dict(rec["params"]), "tags": dict(rec["tags"])}

    @staticmethod
    def _from_remote(r: Any, project: str) -> Dict[str, Any]:
        config = dict(r.config)
        summary = dict(r.summary)
        return {
            "run_id": f"{project}/{r.id}",
            "name": r.name,
            "tags": {k: str(v) for k, v in (config.get(_TAGS_KEY) or {}).items()},
            "params": {k: str(v) for k, v in config.items() if k not in _INTERNAL},
            "metrics": {k: float(v) for k, v in summary.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                        and not k.startswith("_")},
        }

    # ------------------------------------------------------------------ required
    def ping(self) -> None:
        if self._offline:
            return
        self._api().viewer  # noqa: B018  (raises when the key or server is wrong)

    def start_run(self, experiment, run_name, tags, parent_id):
        config: Dict[str, Any] = {_TAGS_KEY: dict(tags)}
        if parent_id:
            config[_PARENT_KEY] = parent_id
        wb_settings = self._wandb.Settings(
            silent=True, **({"base_url": self._settings.tracking_uri}
                            if self._settings.tracking_uri else {}))
        # The SDK reads WANDB_DIR / WANDB_MODE once per process; pass them on every call instead.
        env_args = {k: os.environ[v] for k, v in (("dir", "WANDB_DIR"), ("mode", "WANDB_MODE"))
                    if os.environ.get(v)}
        try:
            run = self._wandb.init(
                project=experiment, name=run_name, group=tags.get("experiment"),
                job_type=tags.get(TAG_LEVEL), tags=_wb_tags(tags), config=config,
                reinit="create_new", settings=wb_settings, **env_args)
        except TypeError as e:
            raise BackendError("This backend needs a newer wandb (several runs open at once): "
                               "pip install -U wandb") from e
        run_id = f"{experiment}/{run.id}"
        self._runs[run_id] = run
        self._records[run_id] = {"run_id": run_id, "name": run_name, "project": experiment,
                                 "tags": {k: str(v) for k, v in tags.items()},
                                 "params": {}, "metrics": {}}
        return run_id

    def end_run(self, run_id, status):
        run = self._runs.pop(run_id, None)
        if run is not None:
            run.finish(exit_code=0 if status == "FINISHED" else 1)

    def set_tags(self, run_id, tags):
        run, rec = self._open(run_id), self._records[run_id]
        rec["tags"].update({k: str(v) for k, v in tags.items()})
        run.config.update({_TAGS_KEY: dict(rec["tags"])}, allow_val_change=True)
        run.tags = tuple(_wb_tags(rec["tags"]))

    def log_params(self, run_id, params):
        run, rec = self._open(run_id), self._records[run_id]
        for k, v in params.items():
            run.config.update({k: v})            # W&B raises if a value is changed
            rec["params"][k] = str(v)

    def log_metrics(self, run_id, metrics, step):
        run, rec = self._open(run_id), self._records[run_id]
        values = {k: float(v) for k, v in metrics.items()}
        run.log(values) if step is None else run.log(values, step=step)
        rec["metrics"].update(values)

    def log_artifact(self, run_id, path):
        run = self._open(run_id)
        dest = os.path.join(run.dir, os.path.basename(path))   # copy: the caller may delete it
        shutil.copy(path, dest)
        run.save(dest, base_path=run.dir, policy="now")

    def get_run(self, run_id):
        if run_id in self._records:
            return self._public(self._records[run_id])
        project, wb_id = self._split(run_id)
        return self._from_remote(self._api().run(f"{self._entity()}/{project}/{wb_id}"), project)

    def search_runs(self, experiment, tags, metric, mode, limit):
        found: Dict[str, Dict[str, Any]] = {}
        if not self._offline:
            try:
                remote = list(self._api().runs(f"{self._entity()}/{experiment}"))
            except Exception as e:
                if "not find" not in str(e).lower() and "not found" not in str(e).lower():
                    raise
                remote = []
            for r in remote:
                s = self._from_remote(r, experiment)
                found[s["run_id"]] = s
        for rec in self._records.values():        # this process's runs win over the server's copy
            if rec["project"] == experiment:
                found[rec["run_id"]] = self._public(rec)
        hits = [s for s in found.values() if metric in s["metrics"]
                and all(s["tags"].get(k) == v for k, v in tags.items())]
        hits.sort(key=lambda s: s["metrics"][metric], reverse=(mode == "max"))
        return hits[:limit]

    # ------------------------------------------------------------------ optional: models
    def log_model(self, run_id, model, flavor, **kwargs):
        """Save a model as a W&B artifact (type 'model').

        `model` is a file or folder path, a torch module (saved with torch.save) or any picklable
        object. Options: code_paths=[files or folders] (stored under code/) and
        pip_requirements=[...]. W&B has no model flavors, so `flavor` is ignored.
        """
        run, rec = self._open(run_id), self._records[run_id]
        code_paths = kwargs.pop("code_paths", None) or []
        requirements = kwargs.pop("pip_requirements", None)
        if kwargs:
            raise BackendError(f"The wandb backend does not support log_model options {sorted(kwargs)}"
                               " (supported: code_paths, pip_requirements)")
        name = f"model-{run_id.split('/', 1)[1]}"
        art = self._wandb.Artifact(name, type="model", metadata={"run": run_id, **{
            k: v for k, v in rec["tags"].items() if k in ("model", "experiment")}})
        if isinstance(model, (str, os.PathLike)) and os.path.exists(model):
            path = os.fspath(model)
            art.add_dir(path, name="model") if os.path.isdir(path) else art.add_file(path, name=os.path.basename(path))
        elif type(model).__module__.split(".")[0] == "torch" and hasattr(model, "state_dict"):
            import torch
            with art.new_file("model.pt", "wb") as f:
                torch.save(model, f)
        else:
            try:
                import cloudpickle as pickler
            except ImportError:
                pickler = pickle
            with art.new_file("model.pkl", "wb") as f:
                pickler.dump(model, f)
        for p in code_paths:
            base = os.path.basename(os.fspath(p).rstrip("/"))
            if os.path.isdir(p):
                art.add_dir(os.fspath(p), name=f"code/{base}")
            else:
                art.add_file(os.fspath(p), name=f"code/{base}")
        if requirements:
            with art.new_file("requirements.txt") as f:
                f.write("\n".join(requirements) + "\n")
        run.log_artifact(art)
        uri = f"wandb-artifact:{run_id.split('/', 1)[0]}/{name}"
        self.set_tags(run_id, {TAG_MODEL_URI: uri})
        return uri
