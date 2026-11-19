"""Export a model (its experiment, runs, files, model files, versions and aliases) to a folder, and import it.

The folder is plain files, so it can be copied, zipped or kept in git-lfs, and imported into any other
MLflow store (a different server, a SQLite file, a file store). Run ids change on import, so everything
that points at a run id or a model URI (the parent of a run, the `authtracker.model_uri` tag, the run a
model version came from) is rewritten to the new ids.

    <folder>/manifest.json
    <folder>/runs/<old run id>/artifacts/...      files saved on the run
    <folder>/models/<n>/...                       the files of one logged model
"""
from __future__ import annotations

import json
import shutil
from urllib.parse import quote
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.entities import LoggedModelOutput, Metric, Param, RunTag
from mlflow.exceptions import MlflowException

from ..errors import ValidationError
from .base import TAG_LEVEL, TAG_MODEL_URI, TAG_RUN_ID

FORMAT = "authtracker-export"
ALL_FORMAT = "authtracker-export-all"           # a folder holding one export per model
FORMAT_VERSION = 1
IMPORTED_FROM = "authtracker.imported_run_id"      # tag on every imported run: the id it had before
_ID_TAGS = {"mlflow.parentRunId", "mlflow.log-model.history", "mlflow.runName"}   # rebuilt or meaningless after import
_BATCH = 100                                        # params and tags per call (MLflow's limit)
_METRIC_BATCH = 1000


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _read_manifest(src: Path) -> Dict[str, Any]:
    path = src / "manifest.json"
    if not path.is_file():
        raise ValidationError(f"{src} is not an authtracker export: manifest.json is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") == ALL_FORMAT:
        raise ValidationError(f"{src} holds several models (an export of all). Import it with import_all / --all.")
    if manifest.get("format") != FORMAT:
        raise ValidationError(f"{src} is not an authtracker export (format {manifest.get('format')!r})")
    if manifest.get("format_version", 0) > FORMAT_VERSION:
        raise ValidationError(
            f"{src} was written by a newer authtracker (export format {manifest['format_version']}, "
            f"this one reads up to {FORMAT_VERSION}). Upgrade authtracker to import it.")
    return manifest


def _all_runs(client, experiment_id: str) -> List[Any]:
    runs, token = [], None
    while True:
        page = client.search_runs([experiment_id], max_results=1000, page_token=token,
                                  order_by=["attributes.start_time ASC"])
        runs.extend(page)
        token = page.token
        if not token:
            return runs


def _download_model(uri: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(dest))


# --------------------------------------------------------------------------------------------- export
def export_model(client, name: str, dest) -> Dict[str, Any]:
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise ValidationError(f"{dest} is not empty. Export into a new or empty folder.")
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        raise ValidationError(f"There is no model {name!r} to export")
    dest.mkdir(parents=True, exist_ok=True)

    model_uris: Dict[str, str] = {}            # old model URI -> folder under models/
    model_info: Dict[str, Dict[str, Any]] = {}

    def want_model(uri: Optional[str]) -> None:
        if not uri or uri in model_uris:
            return
        folder = f"models/{len(model_uris) + 1}"
        model_uris[uri] = folder
        info: Dict[str, Any] = {"uri": uri, "dir": folder, "model_type": None}
        if uri.startswith("models:/m-"):
            try:
                info["model_type"] = client.get_logged_model(uri.split("/", 1)[1]).model_type
            except MlflowException:
                pass
        model_info[uri] = info

    runs = []
    n_metrics = 0
    for r in _all_runs(client, experiment.experiment_id):
        tags = dict(r.data.tags)
        history = {}
        for key in r.data.metrics:
            points = client.get_metric_history(r.info.run_id, key)
            history[key] = [[m.step, m.value, m.timestamp] for m in points]
            n_metrics += len(points)
        has_files = bool(client.list_artifacts(r.info.run_id))
        if has_files:
            files = dest / "runs" / r.info.run_id / "artifacts"
            files.mkdir(parents=True)
            client.download_artifacts(r.info.run_id, "", str(files))
        want_model(tags.get(TAG_MODEL_URI))
        runs.append({
            "id": r.info.run_id, "parent_id": tags.get("mlflow.parentRunId"),
            "name": tags.get("mlflow.runName") or r.info.run_name, "level": tags.get(TAG_LEVEL),
            "status": r.info.status, "start_time": r.info.start_time, "end_time": r.info.end_time,
            "tags": {k: v for k, v in tags.items() if k not in _ID_TAGS},
            "params": dict(r.data.params), "metrics": history,
            "artifacts": f"runs/{r.info.run_id}/artifacts" if has_files else None,
        })

    versions, aliases = [], {}
    try:
        registered = client.get_registered_model(name)
        aliases = {a: str(v) for a, v in (registered.aliases or {}).items()}
        found = sorted(client.search_model_versions(f"name = {_quote(name)}"), key=lambda v: int(v.version))
        for v in found:
            uri = f"models:/{name}/{v.version}"
            source = v.source if v.source.startswith(("models:/m-", "runs:/")) else uri
            want_model(source)
            tags = dict(v.tags or {})
            versions.append({"version": str(v.version), "run_id": tags.get(TAG_RUN_ID) or v.run_id,
                             "model": source, "description": v.description or "",
                             "tags": {k: t for k, t in tags.items() if k != TAG_RUN_ID}})
    except MlflowException:
        pass                                    # no registered model of this name: only runs are exported

    for uri, info in model_info.items():
        # download through a version (or the run) when the source URI cannot be read directly
        _download_model(uri, dest / info["dir"])

    manifest = {
        "format": FORMAT, "format_version": FORMAT_VERSION, "model": name,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "authtracker_version": _version(),
        "runs": runs, "models": list(model_info.values()), "versions": versions, "aliases": aliases,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"model": name, "path": str(dest), "runs": len(runs), "metrics": n_metrics,
            "models": len(model_info), "versions": len(versions), "aliases": dict(aliases)}


def _version() -> str:
    from .. import __version__
    return __version__


# --------------------------------------------------------------------------------------------- import
def import_model(client, src, name: Optional[str] = None) -> Dict[str, Any]:
    src = Path(src)
    manifest = _read_manifest(src)
    target = name or manifest["model"]

    existing = client.get_experiment_by_name(target)
    if existing is not None and existing.lifecycle_stage == "active" and client.search_runs(
            [existing.experiment_id], max_results=1):
        raise ValidationError(
            f"Model {target!r} already has runs here. Import into a new name: import_model(src, name='...').")
    try:
        if client.search_model_versions(f"name = {_quote(target)}"):
            raise ValidationError(
                f"A registered model {target!r} already has versions here. Import into a new name.")
    except MlflowException:
        pass
    experiment_id = existing.experiment_id if existing is not None else client.create_experiment(target)
    renamed = target != manifest["model"]

    # ---- runs: parents first, so a child can point at its new parent
    new_ids: Dict[str, str] = {}
    pending = sorted(manifest["runs"], key=lambda r: (r["parent_id"] is not None, r["start_time"] or 0))
    n_metrics = 0
    for r in pending:
        tags = dict(r["tags"])
        if renamed and "model" in tags:
            tags["model"] = target
        tags[IMPORTED_FROM] = r["id"]
        if r.get("name"):
            tags["mlflow.runName"] = r["name"]
        if r["parent_id"]:
            if r["parent_id"] not in new_ids:
                raise ValidationError(f"Run {r['id']} refers to a parent {r['parent_id']} that is not in the export")
            tags["mlflow.parentRunId"] = new_ids[r["parent_id"]]
        new = client.create_run(experiment_id, start_time=r["start_time"], tags=tags).info.run_id
        new_ids[r["id"]] = new

        params = [Param(k, str(v)) for k, v in r["params"].items()]
        for i in range(0, len(params), _BATCH):
            client.log_batch(new, params=params[i:i + _BATCH])
        metrics = [Metric(key, value, ts, step) for key, points in r["metrics"].items()
                   for step, value, ts in points]
        for i in range(0, len(metrics), _METRIC_BATCH):
            client.log_batch(new, metrics=metrics[i:i + _METRIC_BATCH])
        n_metrics += len(metrics)
        if r["artifacts"]:
            client.log_artifacts(new, str(src / r["artifacts"]))
        status = r["status"] if r["status"] in ("FINISHED", "FAILED", "KILLED") else "KILLED"
        client.set_terminated(new, status, r["end_time"])

    # ---- model files, as logged models in the new store
    first_run_of: Dict[str, str] = {}
    for r in manifest["runs"]:
        uri = r["tags"].get(TAG_MODEL_URI)
        if uri:
            first_run_of.setdefault(uri, new_ids[r["id"]])
    for v in manifest["versions"]:
        if v["run_id"] in new_ids:
            first_run_of.setdefault(v["model"], new_ids[v["run_id"]])
    new_uri: Dict[str, str] = {}
    for m in manifest["models"]:
        source_run = first_run_of.get(m["uri"])
        logged = client.create_logged_model(experiment_id=experiment_id, name="model",
                                            source_run_id=source_run, model_type=m.get("model_type"))
        try:
            client.log_model_artifacts(logged.model_id, str(src / m["dir"]))
        except Exception:
            client.finalize_logged_model(logged.model_id, "FAILED")
            raise
        client.finalize_logged_model(logged.model_id, "READY")
        new_uri[m["uri"]] = f"models:/{logged.model_id}"
    for r in manifest["runs"]:
        uri = r["tags"].get(TAG_MODEL_URI)
        if uri in new_uri:
            run_id = new_ids[r["id"]]
            client.set_tag(run_id, TAG_MODEL_URI, new_uri[uri])
            # link the run to its model, as MLflow does when a run saves one, so the run page lists it
            client.log_outputs(run_id, [LoggedModelOutput(model_id=new_uri[uri].split("/", 1)[1], step=0)])

    # ---- registry: versions in order, then aliases
    version_map: Dict[str, str] = {}
    for v in manifest["versions"]:
        mv = mlflow.register_model(new_uri[v["model"]], target)
        version_map[v["version"]] = str(mv.version)
        if v["run_id"] in new_ids:
            client.set_model_version_tag(target, mv.version, TAG_RUN_ID, new_ids[v["run_id"]])
        for key, value in v["tags"].items():
            client.set_model_version_tag(target, mv.version, key, value)
        if v["description"]:
            client.update_model_version(target, mv.version, description=v["description"])
    for alias, old_version in manifest["aliases"].items():
        if old_version in version_map:
            client.set_registered_model_alias(target, alias, version_map[old_version])

    return {"model": target, "runs": len(new_ids), "metrics": n_metrics, "models": len(new_uri),
            "versions": version_map,
            "aliases": {a: version_map[v] for a, v in manifest["aliases"].items() if v in version_map}}


# --------------------------------------------------------------------------------------------- all models
def _models_with_runs(client) -> List[str]:
    """Every experiment that has at least one run (so not the empty built-in 'Default'), by name."""
    names, token = [], None
    while True:
        page = client.search_experiments(max_results=1000, page_token=token)
        for e in page:
            if client.search_runs([e.experiment_id], max_results=1):
                names.append(e.name)
        token = page.token
        if not token:
            return sorted(names)


def export_all(client, dest) -> Dict[str, Any]:
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise ValidationError(f"{dest} is not empty. Export into a new or empty folder.")
    names = _models_with_runs(client)
    dest.mkdir(parents=True, exist_ok=True)
    entries, summaries = [], {}
    for name in names:
        folder = quote(name, safe="")                       # a safe folder name for any model name
        summaries[name] = export_model(client, name, dest / folder)
        entries.append({"model": name, "dir": folder})
    manifest = {"format": ALL_FORMAT, "format_version": FORMAT_VERSION,
                "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "authtracker_version": _version(), "models": entries}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"path": str(dest), "count": len(entries), "models": summaries}


def import_all(client, src) -> Dict[str, Any]:
    """Import every model of an `export_all` folder. A model that cannot be imported (for example one that
    already exists here) is reported and skipped, and the others still go in."""
    src = Path(src)
    path = src / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    if manifest is None:
        raise ValidationError(f"{src} is not an authtracker export: manifest.json is missing")
    if manifest.get("format") == FORMAT:
        raise ValidationError(f"{src} holds one model. Import it with import_model / without --all.")
    if manifest.get("format") != ALL_FORMAT:
        raise ValidationError(f"{src} is not an authtracker export of all models (format {manifest.get('format')!r})")
    if manifest.get("format_version", 0) > FORMAT_VERSION:
        raise ValidationError(f"{src} was written by a newer authtracker. Upgrade authtracker to import it.")
    imported, failed = {}, {}
    for entry in manifest["models"]:
        try:
            imported[entry["model"]] = import_model(client, src / entry["dir"])
        except Exception as e:              # one bad model must not stop the rest
            failed[entry["model"]] = f"{type(e).__name__}: {e}"
    return {"imported": imported, "failed": failed, "count": len(imported)}
