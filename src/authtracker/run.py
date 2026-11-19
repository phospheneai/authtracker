"""Tracking: model -> experiment -> run hierarchy.

    with authtracker.run(model="model1", experiment="churn-v2") as r:   # parent run
        with r.child("lr0.01") as c:                                    # child run
            c.log_params({...}); c.log_eval(f1=0.9)
"""
from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import tempfile
import time
import traceback
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Union

from . import alerts as _alerts
from . import bundle, capture, utils
from .backends import Backend, get_backend
from .backends.base import TAG_LEVEL
from .errors import BackendError, UnsupportedError, ValidationError
from .settings import Settings, get_settings
from .utils import capture_env_tags, requirements_text


def _changed_params(existing: Mapping[str, Any], params: Mapping[str, Any]) -> Dict[str, tuple]:
    return {k: (existing[k], str(v)) for k, v in params.items() if k in existing and existing[k] != str(v)}


def _params_conflict(changed: Mapping[str, tuple]) -> ValidationError:
    details = ", ".join(f"{k} was {old} and is now {new}" for k, (old, new) in changed.items())
    return ValidationError(
        f"A resumed run cannot change parameters it already has: {details}. Keep the old value, or "
        "start a new run instead of resuming (resume=False).")


class Run:
    def __init__(self, backend: Backend, run_id: str, tags: Dict[str, str], settings: Settings,
                 resume_n: int = 0):
        self._backend = backend
        self.id = run_id
        self.tags = tags
        self._settings = settings
        self.resume_n = resume_n          # 0 for a new run; 1, 2, ... for the first, second ... resume

    def _named(self, name: str) -> str:
        """A file name for this part of the run. Resumed runs add `.resume1`, `.resume2` ... so the files of
        earlier parts (the log, the code) are kept instead of overwritten: `output.log` -> `output.resume1.log`."""
        if not self.resume_n:
            return name
        base, extension = os.path.splitext(name)
        return f"{base}.resume{self.resume_n}{extension}"

    # logging
    def set_tags(self, tags: Mapping[str, Any]) -> None:
        clean = {k: str(v) for k, v in tags.items()}
        self._backend.set_tags(self.id, clean)
        self.tags.update(clean)      # runs started from this one afterwards inherit them

    def log_params(self, params: Mapping[str, Any]) -> None:
        params = dict(params)
        if self.resume_n:                  # a resumed run keeps the parameters it already has
            existing = self._backend.get_run(self.id)["params"]
            changed = _changed_params(existing, params)
            if changed:
                raise _params_conflict(changed)
            params = {k: v for k, v in params.items() if k not in existing}
        if params:
            self._backend.log_params(self.id, params)

    def log_param(self, key: str, value: Any) -> None:
        self.log_params({key: value})

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        self._backend.log_metrics(self.id, dict(metrics), step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        self.log_metrics({key: value}, step)

    def check_metric_names(self, names: Iterable[str]) -> None:
        """Raise if AUTHTRACKER_METRIC_NAMES is set and `names` contains anything else."""
        allowed = self._settings.metric_names
        if allowed:
            bad = sorted(set(names) - set(allowed))
            if bad:
                raise ValidationError(f"Non-standard metric names {bad}; allowed: {list(allowed)}")

    def log_eval(self, **metrics: float) -> None:
        """Log final metrics, enforcing AUTHTRACKER_METRIC_NAMES when it is set."""
        self.check_metric_names(metrics)
        self.log_metrics(metrics)

    def log_artifact(self, path: str) -> None:
        self._backend.log_artifact(self.id, path)

    def log_text(self, text: str, name: str) -> None:
        """Save `text` as an artifact called `name` (e.g. a training log: 'train.log')."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, os.path.basename(name))
            with open(path, "w") as f:
                f.write(text)
            self.log_artifact(path)

    def log_artifacts(self, source: Union[str, Sequence[str]]) -> None:
        """Log a bundle: every file directly inside a directory, or a list of file paths.

        Files keep their names (no sub-folders), so names inside a bundle must be unique.
        """
        if isinstance(source, str):
            if not os.path.isdir(source):
                raise ValidationError(f"{source!r} is not a directory")
            paths = sorted(os.path.join(source, n) for n in os.listdir(source)
                           if os.path.isfile(os.path.join(source, n)))
        else:
            paths = list(source)
        names = [os.path.basename(p) for p in paths]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValidationError(f"Duplicate file names in bundle: {dupes}")
        for p in paths:
            self.log_artifact(p)

    def restore(self, name: str, folder: Optional[str] = None) -> Optional[str]:
        """Download the file `name` saved on this run (for example a checkpoint) and return its path, or None
        if there is no such file. `folder` is where to put it (default: a new temporary folder)."""
        folder = folder or tempfile.mkdtemp(prefix="authtracker-restore-")
        os.makedirs(folder, exist_ok=True)
        return self._backend.download_artifact(self.id, name, folder)

    def log_model(self, model: Any, flavor: Optional[str] = None, **kwargs: Any) -> str:
        """Log a model; call it inside the `with` block of the run that owns it.

        flavor="bundle" stores a PyTorch model as a plain folder (architecture file, weights, sample
        inference script, requirements): see authtracker.bundle.
        """
        if flavor in (bundle.FLAVOR, bundle.PICKLE_FLAVOR):
            builder = bundle.build if flavor == bundle.FLAVOR else bundle.build_pickle
            container = {k[len("container_"):]: v for k, v in self.tags.items() if k.startswith("container_")}
            with tempfile.TemporaryDirectory(prefix="authtracker-bundle-") as folder:
                builder(model, folder, context={"container": container or None}, **kwargs)
                return self._backend.log_model_dir(self.id, folder)
        return self._backend.log_model(self.id, model, flavor, **kwargs)

    def child(self, name: str, params: Optional[Mapping[str, Any]] = None, resume: bool = False):
        """Start a nested child run (the 'run' level). With resume=True, continue the latest earlier run
        of this name under this experiment run, or start a new one if there is none."""
        inherited = {k: v for k, v in self.tags.items() if k != TAG_LEVEL and not k.startswith("resume")}
        return _managed(self._backend, self._settings, self.tags["experiment"], name, inherited,
                        parent=self, level="run", params=params, resume=resume)


def _periodic_upload(backend: Backend, run_id: str):
    """Best-effort upload of the log while the run is going. It bypasses the offline queue on
    purpose: a queued copy of a growing log every minute would pile up, and the final upload at the
    end of the run is the one that always happens."""
    inner = getattr(backend, "inner", backend)
    resolve = getattr(backend, "_res", lambda rid: rid)

    def upload(path: str) -> None:
        rid = resolve(run_id)
        if rid is None or str(rid).startswith("offline-"):
            return                                     # the run is not on the server yet
        inner.log_artifact(rid, path)

    return upload


def _save_captured_log(run: Run, log_capture, log_dir) -> None:
    """Stop copying console output and save it to the run as output.log (best effort)."""
    if log_capture is None:
        return
    try:
        capture.stop(log_capture)
        if os.path.getsize(log_capture.path) > 0:
            run.log_artifact(log_capture.path)
    except Exception as e:
        logging.getLogger("authtracker").warning("Could not save output.log: %s", e)
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)


def _validate(settings: Settings, tags: Dict[str, str], model: str) -> None:
    if settings.allowed_models and model not in settings.allowed_models:
        raise ValidationError(
            f"Unknown model {model!r}; allowed: {list(settings.allowed_models)}")
    missing = [t for t in settings.required_tags if not tags.get(t)]
    if missing:
        raise ValidationError(f"Missing required tags: {missing}")


def _save_code(run: Run, snapshot: Mapping[str, Any]) -> None:
    """Store the uncommitted changes and the running script with the experiment run."""
    patch, script = snapshot.get("changes", {}), snapshot.get("script", {})
    try:
        if patch.get("patch"):
            run.log_text(patch["patch"], run._named(utils.PATCH_NAME))
        if script.get("text"):
            run.log_text(script["text"], run._named(f"script_{script['name']}"))
    except Exception as e:
        logging.getLogger("authtracker").warning("Could not save the code: %s", e)


def _code_snapshot() -> Dict[str, Any]:
    """What to store about the code, and tags that say what was (and was not) stored and why."""
    changes, script = utils.code_changes(), utils.script_snapshot()
    tags: Dict[str, str] = {"code_changes": changes["status"], "code_script": script["status"]}
    if script.get("repo_path"):
        tags["git_script"] = script["repo_path"].replace(os.sep, "/")
    if changes["untracked_skipped"]:
        tags["code_changes_untracked_skipped"] = str(changes["untracked_skipped"])
    for label, part in (("changes", changes), ("script", script)):
        if part["status"].startswith("skipped"):
            logging.getLogger("authtracker").warning(
                "The code %s was not stored (%s). Pass save_code=False to at.init(...) to stop this check.",
                label, part["status"].replace("skipped: ", ""))
    return {"changes": changes, "script": script, "tags": tags}


def _find_earlier(backend: Backend, run_name: str, tags: Dict[str, str], parent: Optional[Run]) -> Optional[str]:
    """The newest earlier run to continue, or None. Only an unsupported backend is an error: if the server
    cannot be reached to look, a new run is started (and queued) instead of stopping the training."""
    wanted = {k: tags[k] for k in ("model", "experiment", "owner", TAG_LEVEL) if k in tags}
    try:
        return backend.find_run(tags["model"], run_name, wanted, parent.id if parent else None)
    except UnsupportedError:
        raise
    except BackendError as e:
        logging.getLogger("authtracker").warning(
            "Could not look for an earlier run to resume (%s). Starting a new run.", e)
        return None


_ENVIRONMENT_TAGS = ("python_version", "platform", "hostname")


def _tag_resume(run: Run, old: Mapping[str, str], new: Mapping[str, str], n: int) -> None:
    """Mark a run as resumed, and keep what is different about this part (its commit, its image) next to the
    original instead of replacing it: `git_sha` stays, `resume1_git_sha` is added."""
    updates = {"resumed": "true", "resume_count": str(n)}
    for key, value in new.items():
        if key in ("resumed", "resume_count") or key.startswith("resume"):
            continue
        if key not in old:
            updates[key] = value
        elif old[key] != value and (key.startswith(("git_", "code_", "container_")) or key in _ENVIRONMENT_TAGS):
            updates[f"resume{n}_{key}"] = value
    run.set_tags(updates)


@contextmanager
def _managed(backend: Backend, settings: Settings, experiment_name: str, run_name: str,
             tags: Dict[str, str], parent: Optional[Run], level: str,
             params: Optional[Mapping[str, Any]] = None, code: Optional[Mapping[str, Any]] = None,
             resume: bool = False) -> Iterator[Run]:
    full_tags = {**tags, TAG_LEVEL: level}
    earlier = _find_earlier(backend, run_name, full_tags, parent) if resume else None
    if earlier:
        old = dict(backend.get_run(earlier)["tags"])
        changed = _changed_params(backend.get_run(earlier)["params"], params or {})
        if changed:                                    # refuse BEFORE touching the old run: it stays as it was
            raise _params_conflict(changed)
        backend.resume_run(earlier, parent.id if parent else None)
        run_id = earlier
        n = int(old.get("resume_count") or 0) + 1
        run = Run(backend, run_id, {**old, **full_tags}, settings, resume_n=n)
        _tag_resume(run, old, full_tags, n)
    else:
        run_id = backend.start_run(tags["model"], run_name, full_tags,
                                   parent.id if parent else None)
        run = Run(backend, run_id, full_tags, settings)
    status = "FINISHED"
    failure: Optional[BaseException] = None
    started = time.time()
    log_capture, log_dir = None, None
    if settings.capture_logs and not getattr(backend, "captures_console", False):
        try:
            log_dir = tempfile.mkdtemp(prefix="authtracker-")
            log_capture = capture.start(os.path.join(log_dir, run._named("output.log")),
                                        periodic=_periodic_upload(backend, run_id),
                                        interval=settings.log_upload_interval)
        except Exception:
            log_capture = None   # capturing is a convenience: never stop the run for it
    try:
        if level == "experiment" and settings.capture_env:
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, run._named("requirements.txt"))
                with open(path, "w") as f:
                    f.write(requirements_text())
                run.log_artifact(path)
        if level == "experiment" and code:
            _save_code(run, code)
        if params:
            run.log_params(params)
        yield run
    except BaseException as e:
        status = "KILLED" if isinstance(e, KeyboardInterrupt) else "FAILED"
        failure = e
        try:
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, run._named("error.txt"))
                with open(path, "w") as f:
                    f.write("".join(traceback.format_exception(type(e), e, e.__traceback__)))
                run.log_artifact(path)
        except Exception:
            pass  # never mask the original error
        raise
    finally:
        _save_captured_log(run, log_capture, log_dir)
        backend.end_run(run_id, status)
        if level == "experiment":                  # one alert per experiment, not one for every nested run
            _alerts.notify_run_end(settings, run.tags, run_id, status, failure, time.time() - started)


def run(model: str, experiment: str, *, owner: Optional[str] = None,
        dataset_version: Optional[str] = None, tags: Optional[Mapping[str, str]] = None,
        save_code: Optional[bool] = None, image: Optional[str] = None, system_metrics: Optional[bool] = None,
        resume: bool = False, params: Optional[Mapping[str, Any]] = None,
        settings: Optional[Settings] = None, backend: Optional[Backend] = None):
    """Start the parent run for `experiment` under `model`. Use as a context manager.

    resume:    continue the latest earlier experiment run of this model, experiment and owner, or start a
               new one if there is none. A resumed run keeps its parameters and continues its metric steps.

    save_code: store the uncommitted changes and the script with the run (default: on).
    image:     the container image this runs in, ideally with its digest ("registry/app@sha256:...").
    system_metrics: record CPU, memory and GPU use while the run is open (default: on).
    These override the defaults from the settings, for this run only.
    """
    s = settings or get_settings()
    overrides = {k: v for k, v in (("save_code", save_code), ("image", image),
                                                 ("system_metrics", system_metrics)) if v is not None}
    if overrides:
        s = dataclasses.replace(s, **overrides)
    all_tags: Dict[str, str] = {**(tags or {}), "model": model, "experiment": experiment,
                                "owner": owner or s.owner}
    if dataset_version:
        all_tags["dataset_version"] = dataset_version
    _validate(s, all_tags, model)
    if s.capture_env:
        for k, v in capture_env_tags().items():
            all_tags.setdefault(k, v)
        container = utils.container_info(s.image)
        if container:
            for key, value in container.items():
                all_tags.setdefault(f"container_{key}", value)
    code = _code_snapshot() if s.save_code else None
    if code:
        for k, v in code["tags"].items():
            all_tags.setdefault(k, v)
    return _managed(backend or get_backend(s), s, experiment, experiment, all_tags,
                    parent=None, level="experiment", params=params, code=code, resume=resume)
