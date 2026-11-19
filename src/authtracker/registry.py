"""Versioning: register, promote (with gates), rollback, load, best_run."""
from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Optional, Union

from . import bundle
from .backends import Backend, get_backend
from .backends.base import TAG_LEVEL, TAG_MODEL_URI
from .errors import PromotionError, ValidationError
from .run import Run
from .settings import Settings, get_settings

RunRef = Union[Run, str, Dict[str, Any]]


def _ctx(settings: Optional[Settings], backend: Optional[Backend]):
    s = settings or get_settings()
    return s, backend or get_backend(s)


def _run_id(ref: RunRef) -> str:
    if isinstance(ref, Run):
        return ref.id
    if isinstance(ref, dict):
        return ref["run_id"]
    return ref


def best_run(model: str, experiment: str, metric: str, mode: str = "max",
             *, settings=None, backend=None) -> Optional[Dict[str, Any]]:
    """Best child run of `experiment` under `model` by `metric` (None if no runs)."""
    if mode not in ("max", "min"):
        raise ValueError("mode must be 'max' or 'min'")
    _, b = _ctx(settings, backend)
    runs = b.search_runs(model, {"model": model, "experiment": experiment, TAG_LEVEL: "run"},
                         metric, mode, 1)
    return runs[0] if runs else None


def register(run: Optional[RunRef] = None, name: Optional[str] = None, *,
             settings=None, backend=None) -> str:
    """Create a new model version from the model logged in `run` (default: the current run)."""
    _, b = _ctx(settings, backend)
    if run is None:
        from .simple import current_run
        run = current_run()
    info = b.get_run(_run_id(run))
    uri = info["tags"].get(TAG_MODEL_URI)
    if not uri:
        raise ValidationError("This run has no logged model; call run.log_model(...) first")
    return b.register(info["run_id"], name or info["tags"]["model"], uri)


def _metric(b: Backend, name: str, version: str, metric: str) -> Optional[float]:
    return b.get_run(b.version_run_id(name, version))["metrics"].get(metric)


def _check_gate(s: Settings, b: Backend, name: str, version: str, to: str) -> None:
    if s.promote_metric is None:
        return
    value = _metric(b, name, version, s.promote_metric)
    if value is None:
        raise PromotionError(f"Version {version} has no metric {s.promote_metric!r}")
    if s.promote_min is not None and value < s.promote_min:
        raise PromotionError(
            f"{s.promote_metric}={value} is below the minimum {s.promote_min} for {to!r}")
    if s.promote_must_beat_prod:
        current = b.get_alias(name, to)
        if current is not None and current != version:
            cur_value = _metric(b, name, current, s.promote_metric)
            if cur_value is not None and value <= cur_value:
                raise PromotionError(
                    f"{s.promote_metric}={value} does not beat current {to!r} "
                    f"(v{current}: {cur_value})")


def promote(name: str, version: Union[str, int], to: str, *, settings=None, backend=None) -> None:
    """Point alias `to` at `version`. Gated aliases must pass the configured checks."""
    s, b = _ctx(settings, backend)
    version = str(version)
    b.version_run_id(name, version)  # raises if it does not exist
    if to in s.gated_aliases:
        _check_gate(s, b, name, version, to)
    previous = b.get_alias(name, to)
    if previous is not None and previous != version:
        b.set_alias(name, previous, f"previous-{to}")
    b.set_alias(name, version, to)


def rollback(name: str, alias: str = "production", *, settings=None, backend=None) -> str:
    """Move `alias` back to the version it pointed at before the last promote."""
    _, b = _ctx(settings, backend)
    previous = b.get_alias(name, f"previous-{alias}")
    if previous is None:
        raise PromotionError(f"No previous version recorded for alias {alias!r}")
    current = b.get_alias(name, alias)
    b.set_alias(name, previous, alias)
    if current is not None:
        b.set_alias(name, current, f"previous-{alias}")
    return previous


def load(name: str, alias: str = "production", *, device: str = "cpu", pickle: bool = False,
         settings=None, backend=None) -> Any:
    """Load a model by registry alias (never by run id).

    A PyTorch bundle comes back as the real torch.nn.Module (on `device`, in eval mode). A model that was stored as
    files (`at.log_model("weights.pth")`) comes back as the path of a new folder holding them: it is yours to use and
    delete. Any other model comes back as an MLflow model with predict().
    """
    _, b = _ctx(settings, backend)
    folder = b.download_bundle(name, alias)
    if folder is not None:
        try:
            return bundle.load_dir(folder, device=device, pickle=pickle)
        finally:
            shutil.rmtree(folder, ignore_errors=True)
    files = b.download_model_files(name, alias)
    if files is not None:
        return files
    return b.load(name, alias)


def export_model(name: str, dest: Union[str, "os.PathLike[str]"], *, settings=None, backend=None) -> Dict[str, Any]:
    """Write model `name` (runs, files, model files, versions, aliases) into the empty folder `dest`.

    Returns a summary. Import the folder into another store with `import_model`. MLflow backend only.
    """
    _, b = _ctx(settings, backend)
    return b.export_model(name, os.fspath(dest))


def import_model(src: Union[str, "os.PathLike[str]"], name: Optional[str] = None, *,
                 settings=None, backend=None) -> Dict[str, Any]:
    """Read a folder written by `export_model` into the current store, as `name` (default: its own name).

    Refuses to add to a model that already has runs or versions. Run ids and version numbers are new;
    the result says how old versions map to new ones. MLflow backend only.
    """
    _, b = _ctx(settings, backend)
    return b.import_model(os.fspath(src), name)


def export_all(dest: Union[str, "os.PathLike[str]"], *, settings=None, backend=None) -> Dict[str, Any]:
    """`export_model` for every model that has runs, into one empty folder (one sub-folder per model).

    Returns {"path", "count", "models": {name: summary}}. Import it with `import_all`. MLflow backend only.
    """
    _, b = _ctx(settings, backend)
    return b.export_all(os.fspath(dest))


def import_all(src: Union[str, "os.PathLike[str]"], *, settings=None, backend=None) -> Dict[str, Any]:
    """Import every model of a folder written by `export_all`.

    A model that cannot be imported (for example it already exists here) is listed under "failed" and the
    others still go in. Returns {"imported": {name: summary}, "failed": {name: reason}, "count"}.
    """
    _, b = _ctx(settings, backend)
    return b.import_all(os.fspath(src))
