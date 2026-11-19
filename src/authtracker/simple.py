"""The simple API: init -> start -> log -> save -> finish. No `with`, no run objects.

    at.init(model="fake-image", experiment="baseline", dataset_version="v1")   # level 2
    at.start("lr0.01", config={"lr": 0.01})                                    # level 3
    at.log({"loss": 0.4})
    at.save(model)
    at.finish()                                                                # optional

Level 1 (the model) is the MLflow experiment. Runs close themselves when the script ends;
if it ends with an unhandled exception they are marked FAILED with the traceback saved.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
from typing import Any, Dict, Mapping, Optional, Union

from . import configfile
from . import dataset as dataset_module
from .errors import AuthTrackerError, ValidationError
from .run import Run
from .run import run as _open_experiment
from .settings import Settings
from .watch import watch as _watch


class _State:
    def __init__(self, cm, parent: Run, resume: bool = False):
        self.parent_cm, self.parent = cm, parent
        self.resume = resume                 # the default for start(): continue earlier runs of the same name
        self.child_cm = None
        self.child: Optional[Run] = None
        self.count = 0                       # child runs started (for default names)
        self.steps: Dict[str, int] = {}      # run id -> last auto step


_state: Optional[_State] = None
_crash: Optional[tuple] = None   # (type, value, traceback) of an unhandled exception
_hooked = False


def _install_hooks() -> None:
    global _hooked
    if _hooked:
        return
    _hooked = True
    previous = sys.excepthook

    def hook(exc_type, exc, tb):
        global _crash
        _crash = (exc_type, exc, tb)
        previous(exc_type, exc, tb)

    sys.excepthook = hook
    atexit.register(finish)


def _need() -> _State:
    if _state is None:
        raise AuthTrackerError("No active run: call authtracker.init(model=..., experiment=...) first")
    return _state


def current_run() -> Run:
    """The run that log()/save() write to: the current child, else the experiment run."""
    s = _need()
    return s.child or s.parent


def _continue_steps(s: _State, run: Run) -> None:
    """After a resume, the automatic step carries on from the last step already logged."""
    if run.resume_n:
        last = run._backend.last_step(run.id)
        if last is not None:
            s.steps[run.id] = last


def _close_child(s: _State, exc=(None, None, None)) -> None:
    if s.child_cm is not None:
        cm, s.child_cm, s.child = s.child_cm, None, None
        cm.__exit__(*exc)


def init(model: Optional[str] = None, experiment: Optional[str] = None, *,
         config: Union[None, str, "os.PathLike[str]", Mapping[str, Any]] = None,
         dataset: Union[None, str, "os.PathLike[str]", Mapping[str, Any]] = None,
         dataset_version: Optional[str] = None, owner: Optional[str] = None,
         tags: Optional[Mapping[str, str]] = None, save_code: Optional[bool] = None,
         image: Optional[str] = None, system_metrics: Optional[bool] = None,
         resume: bool = False, settings: Optional[Settings] = None) -> Run:
    """Open the experiment run for `model`. Closes any run that is still open.

    config   a training YAML file (its path), or a dict of parameters. A YAML file is saved with the
             run, its settings are logged as parameters, and an optional `authtracker:` block in it
             can set model, experiment, dataset, tags and owner. Arguments given here win over it.
    dataset  the data this run uses: a path, or {"name", "version", "path", "hash"}. A fingerprint
             (hash) of it is stored with the run.
    save_code  store the uncommitted changes and your script with the run. Default True: pass
             save_code=False to keep them out (for example when the code holds secrets).
    image    the container image this runs in, ideally pinned with its digest:
             image="registry/app@sha256:...". Recorded with the run; a tag alone gets a warning.
    system_metrics  record CPU, memory and GPU use. Default True: pass system_metrics=False to switch it off
             (for example on the ranks of a multi-GPU job that are not tracking).
    resume   continue the latest earlier experiment run of this model and experiment (yours), if there is one,
             and start a new one if not. The runs you start() afterwards continue earlier runs of the same
             name by default; pass resume=False to start() to make a new one.
    """
    global _state
    finish()
    params: Dict[str, Any] = {}
    block: Dict[str, Any] = {}
    config_path: Optional[str] = None
    if isinstance(config, (str, os.PathLike)):
        params, block, config_path = configfile.load(config)
    elif config:
        params = configfile.flat_params(config)

    model = model or block.get("model")
    experiment = experiment or block.get("experiment")
    if not model or not experiment:
        raise ValidationError(
            "model and experiment are required: pass them to init(...), or set them in the "
            "'authtracker:' block of the config file")

    spec = dataset if dataset is not None else block.get("dataset")
    record = dataset_module.describe(spec) if spec is not None else None
    version = dataset_version or block.get("dataset_version") or (record or {}).get("version")
    all_tags: Dict[str, str] = {**(block.get("tags") or {}), **(tags or {}),
                                **(dataset_module.as_tags(record) if record else {})}

    cm = _open_experiment(model, experiment, owner=owner or block.get("owner"),
                          dataset_version=version, tags=all_tags, save_code=save_code,
                          image=image, system_metrics=system_metrics, resume=resume, params=params, settings=settings)
    parent = cm.__enter__()
    _state = _State(cm, parent, resume)
    _install_hooks()
    _continue_steps(_state, parent)
    if config_path:                                            # the YAML itself, under its own name
        with open(config_path, encoding="utf-8") as f:
            parent.log_text(f.read(), parent._named(os.path.basename(config_path)))
    if record:
        parent.log_text(json.dumps(record, indent=2), parent._named("dataset.json"))
    return parent


def start(name: Optional[str] = None, config: Optional[Mapping[str, Any]] = None,
          resume: Optional[bool] = None) -> Run:
    """Start a run (one trial) inside the experiment. Closes the previous run.

    resume=True continues the latest earlier run of this name under this experiment run, or starts a new one
    if there is none. Left out, it follows the `resume` you gave to init().
    """
    s = _need()
    _close_child(s)
    s.count += 1
    cm = s.parent.child(name or f"run-{s.count}", params=config, resume=s.resume if resume is None else resume)
    s.child = cm.__enter__()
    s.child_cm = cm
    _continue_steps(s, s.child)
    return s.child


def log(metrics: Mapping[str, float], step: Optional[int] = None) -> None:
    """Log metrics to the current run. `step` counts up by itself unless you pass it.

    Metric names are checked against AUTHTRACKER_METRIC_NAMES when that is set.
    """
    s = _need()
    r = s.child or s.parent
    r.check_metric_names(metrics)
    if step is None:
        step = s.steps.get(r.id, -1) + 1
    s.steps[r.id] = step
    r.log_metrics(metrics, step)


def restore(name: str, folder: Optional[str] = None) -> Optional[str]:
    """Download the file `name` that this run saved earlier (a checkpoint, say) and return its path.

    Returns None if the run has no such file, so one script serves both the first run and every restart:

        path = at.restore("ckpt.pt")
        start = torch.load(path)["epoch"] + 1 if path else 0
    """
    return current_run().restore(name, folder)


def watch(model: Any, every: int = 100) -> Any:
    """Log the gradient size of every layer of a PyTorch `model`, every `every` training steps.

    Call it once, after init() and before training. It logs `grad_norm/<layer>` and `grad_norm/total`
    (a step is one training forward pass). Returns a function that stops watching.
    """
    return _watch(model, every, current_run)


def set_tags(tags: Mapping[str, Any]) -> None:
    current_run().set_tags(tags)


def save(obj: Any, name: Optional[str] = None, **kwargs: Any) -> Optional[str]:
    """Save anything to the current run.

    * a file path           -> artifact
    * a folder path         -> every file in it, as a bundle
    * a list of file paths  -> a bundle
    * text (with name=...)  -> a text artifact such as a training log
    * anything else         -> a model (extra keyword arguments go to log_model)
    """
    r = current_run()
    is_path = lambda x: isinstance(x, (str, os.PathLike))  # noqa: E731
    if is_path(obj) and os.path.exists(obj):
        path = os.fspath(obj)
        r.log_artifacts(path) if os.path.isdir(path) else r.log_artifact(path)
        return None
    if isinstance(obj, (list, tuple)) and obj and all(is_path(x) and os.path.exists(x) for x in obj):
        r.log_artifacts([os.fspath(x) for x in obj])
        return None
    if isinstance(obj, str):
        if name is None:
            raise ValidationError(
                f"{obj!r} is not an existing file or folder. To save text, pass a file name: "
                "save(text, name='train.log')")
        r.log_text(obj, name)
        return None
    return r.log_model(obj, **kwargs)


def finish() -> None:
    """Close the open runs. Optional: this also happens when the script ends."""
    global _state
    s, _state = _state, None
    if s is None:
        return
    exc = _crash or (None, None, None)
    try:
        _close_child(s, exc)
    finally:
        s.parent_cm.__exit__(*exc)
