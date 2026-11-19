# authtracker API reference

For setup and examples, see the [README](../README.md), the [user guide](GUIDE.md) and the
[examples](../examples/README.md). Every environment variable is in [Configuration](CONFIGURATION.md).

```python
import authtracker as at
```

Public names: `init`, `start`, `log`, `save`, `restore`, `watch`, `finish`, `set_tags`, `current_run`, `alert`,
`best_run`, `register`, `promote`, `rollback`, `load`, `export_model`, `import_model`, `export_all`, `import_all`, `load_bundle`, `run`, `Run`, `Settings`,
`configure`, `get_settings`, and the errors `AuthTrackerError`, `ValidationError`, `PromotionError`,
`BackendError`, `UnsupportedError`, `ConfigError`.

- [Tracking](#tracking): `init`, `start`, `log`, `save`, `restore`, `watch`, `set_tags`, `finish`, `current_run`
- [Versioning](#versioning): `best_run`, `register`, `promote`, `rollback`, `load`
- [Moving a model](#moving-a-model-between-stores): `export_model`, `import_model`, `export_all`, `import_all`
- [Bundles](#bundles): `save(flavor="bundle")`, `load_bundle`
- [The `with` API](#the-with-api): `run`, `Run`
- [Settings](#settings): `Settings`, `configure`, `get_settings`
- [Errors](#errors)
- [Command line](#command-line)
- [Writing a backend](#writing-a-backend)
- [Conventions](#conventions)

---

## Tracking

Runs live in three levels: a **model** (an MLflow experiment, a W&B project), an **experiment** (a parent
run) and **runs** (child runs, one per trial). Only one run is open at a time per process.

### `at.init(...)`

Opens the experiment run for a model. Closes anything still open.

```python
at.init(model=None, experiment=None, *, config=None, dataset=None, dataset_version=None,
        owner=None, tags=None, save_code=None, image=None, system_metrics=None,
        resume=False, settings=None) -> Run
```

| Argument | Default | Description |
|---|---|---|
| `model`, `experiment` | none | Where the run belongs. Required, here or in the `authtracker:` block of `config`. |
| `config` | none | A path to a training YAML (saved with the run, logged as parameters), or a dict of parameters. |
| `dataset` | none | The data used: a path, or `{"name", "version", "path", "hash"}`. A fingerprint is stored. `hash` is `manifest` (default), `contents` or `none`. |
| `dataset_version` | none | A version label. Counts as the `dataset_version` tag. |
| `owner` | `$AUTHTRACKER_OWNER`, then `$USER` | Who ran it. |
| `tags` | none | Extra tags (strings). |
| `save_code` | `AUTHTRACKER_SAVE_CODE` | Save the uncommitted changes and the running script. |
| `image` | `AUTHTRACKER_IMAGE` | The container image, ideally `name@sha256:...`. |
| `system_metrics` | `AUTHTRACKER_SYSTEM_METRICS` | Record CPU, memory and GPU use. |
| `resume` | `False` | Continue the latest earlier experiment run of the same model, experiment and owner, if there is one. |
| `settings` | from the environment | A full `Settings` object. |

Arguments to `init` override the `authtracker:` block of a YAML. A misspelled key in that block is an error.
Returns the experiment `Run`. Raises `ValidationError` when `model` or `experiment` is missing or a rule is broken.

### `at.start(name=None, config=None, resume=None)`

Starts a run (one trial) inside the experiment and closes the previous one. `name` defaults to `run-<n>`.
`config` is a dict of parameters. `resume=True` continues the latest earlier run of the same name under this
experiment run; left out, it follows `init(resume=...)`. Returns the `Run`.

### `at.log(metrics, step=None)`

Logs a dict of numbers to the current run. `step` counts up by itself, per run, unless you pass it. After a
resume it carries on from the last logged step. Names are checked against `AUTHTRACKER_METRIC_NAMES`.

### `at.save(obj, name=None, **kwargs)`

Saves something to the current run. What it does depends on `obj`:

| `obj` | Result |
|---|---|
| a path to a file | an artifact |
| a path to a folder | every file in it, as a bundle of artifacts |
| a list of file paths | a bundle of artifacts |
| text, with `name=` | a text artifact (a path that does not exist and no `name` is an error) |
| anything else | a model. Extra keyword arguments go to the backend (MLflow's `log_model`, or the bundle options below). |

A model's keyword arguments include `flavor` (`"bundle"`, `"pickle"`, or an MLflow flavor), `input_example`,
`code_paths`, `pip_requirements`, `serialization_format`. Returns a model URI for models and `None` otherwise.

### `at.restore(name, folder=None)`

Downloads the file `name` the current run saved earlier (a checkpoint, say) and returns its local path, or
`None` if there is no such file. `folder` defaults to a new temporary folder.

### `at.watch(model, every=100)`

Logs the gradient norm of each layer of a PyTorch `model` every `every` training steps, as
`grad_norm/<layer>` and `grad_norm/total`. Call it once, after `init` and before the loop. Returns a function
that stops watching. Needs PyTorch 2.1+.

### `at.set_tags(tags)`

Adds tags (values become strings) to the current run. Runs started afterwards inherit them.

### `at.alert(title, text="", level="warn")`

Sends a message to the alert webhook (`AUTHTRACKER_ALERT_WEBHOOK`, see [Configuration](CONFIGURATION.md#alerts-optional)).
`level` is `"info"`, `"warn"` or `"error"` and labels the message. It is sent only if `custom` is among the events
chosen in `AUTHTRACKER_ALERT_ON` (it is by default). The model, experiment, owner and run id are added when a run is
open. Returns `True` if the webhook accepted it, and `False` if no webhook is set, `custom` is not chosen, or it could
not be reached. It never raises, so it is safe inside a training loop.

```python
if not math.isfinite(loss):
    at.alert("Loss is NaN", f"epoch {epoch}, step {step}", level="error")
```

### `at.finish()`

Closes the open runs. Optional: it also happens when the script ends, as `FAILED` (with the traceback saved
as `error.txt`) if the script ended with an unhandled error. In a notebook, call it yourself.

### `at.current_run()`

The run that `log` and `save` write to: the current run, else the experiment run. Raises `AuthTrackerError`
when `init` has not been called.

---

## Versioning

A **version** is an integer the registry assigns. An **alias** (`staging`, `production`) points at a version.

### `at.best_run(model, experiment, metric, mode="max")`

The best run of `experiment` under `model` by `metric`, or `None` if there are no runs. `mode="min"` for loss.
Returns a dict: `{"run_id", "name", "metrics", "params", "tags"}`.

### `at.register(run=None, name=None)`

Creates a model version from the model logged in `run`: a `Run`, a run id, or the dict from `best_run`.
Default is the current run. `name` defaults to the model name. Returns the version. Raises
`ValidationError` if the run has no logged model.

### `at.promote(name, version, to)`

Points the alias `to` at `version`. Aliases in `AUTHTRACKER_GATED_ALIASES` (default `production`) must pass
the promotion rules or `PromotionError` is raised. The previous holder is remembered as `previous-<alias>`.

### `at.rollback(name, alias="production")`

Moves `alias` back to the version it held before the last `promote`. Returns that version.
Raises `PromotionError` when there is nothing to go back to.

### `at.load(name, alias="production", *, device="cpu", pickle=False)`

Loads a model by alias, never by run id. A bundle comes back as your own `torch.nn.Module` (on `device`,
in eval mode). Any other model comes back as an MLflow model with `predict()`. `pickle=True` is needed to
load a model saved with `flavor="pickle"` (it runs code from the file).

---

## Moving a model between stores

Copy a whole model to another MLflow server or store, for example when you change servers. MLflow backend only
(other backends raise `UnsupportedError`).

### `at.export_model(name, dest)`

Writes the model `name` into the new or empty folder `dest`: every run of its experiment (the three levels), their
parameters, tags, **every metric point with its step and time**, the files saved on each run, the files of every
model, and the registered versions with their tags, descriptions and aliases. Returns a summary
`{"model", "path", "runs", "metrics", "models", "versions", "aliases"}`. Raises `ValidationError` if the model does
not exist or `dest` has files in it.

### `at.import_model(src, name=None)`

Reads a folder written by `export_model` into the current store, as `name` (default: the exported name). Returns
`{"model", "runs", "metrics", "models", "versions", "aliases"}`, where `versions` maps each old version number to
the new one.

- **Run ids are new.** Parents, the `authtracker.model_uri` tag and the run each version came from are rewritten
  to the new ids. Every imported run gets the tag `authtracker.imported_run_id` with the id it had before.
- **Version numbers and aliases are kept** when the target has no versions yet (it must not, see below).
- **It never adds to existing work.** It raises `ValidationError` if the model already has runs, or a registered
  model of that name already has versions. Import under another `name` instead.
- `name` also retags the runs: the `model` tag becomes the new name, so `best_run` finds them.
- The folder format is versioned. A folder from a newer authtracker is refused with a message to upgrade.

Not carried over: deleted runs, MLflow traces, logged datasets, and metrics logged on a model rather than a run.
System metrics (`system/...`) are copied like any other metric.

### `at.export_all(dest)` and `at.import_all(src)`

The same for **every model that has runs** (empty experiments are skipped), into one folder with a sub-folder per
model. `export_all` returns `{"path", "count", "models": {name: summary}}`.

`import_all` imports each model like `import_model` does. A model that cannot be imported, for example one that
already exists here, is listed under `"failed"` with the reason and the others still go in:
`{"imported": {name: summary}, "failed": {name: reason}, "count"}`. A folder from `export_model` is refused by
`import_all`, and the other way round, with a message that names the right call.

The same from the command line, one store at a time:

```
authtracker export churn --out churn-export --tracking-uri http://old-server:5000
authtracker import churn-export --tracking-uri http://new-server:5000 [--name churn-v2]

authtracker export --all --out everything --tracking-uri http://old-server:5000     # every model
authtracker import --all everything --tracking-uri http://new-server:5000
```

---

## Bundles

A PyTorch model saved as a folder: the architecture file, the weights and what is needed to run it.

```python
at.save(net, flavor="bundle", init_args={"hidden": 32})
```

| Option | Description |
|---|---|
| `init_args` | Plain values (numbers, text, lists, dicts) used to build the class again. |
| `factory` | A function that builds the model, for models that have no class of yours (torchvision, timm, `nn.Sequential`). |
| `code` | Extra files or folders to ship next to `architecture.py`. |
| `assets` | Files the constructor reads: a list of paths, or `{folder: [patterns]}`. The matching `init_args` value is stored as `$BUNDLE/assets/<name>`. |
| `extras` | Plain attributes put back on the model after loading (`class_names`, `threshold`). |
| `inference` | Your own sample inference script instead of the generated one. |
| `requirements` | The list for `requirements.txt` instead of the detected one. |
| `pickle` | Also store `model.pkl`, the exact object. Load it with `at.load(..., pickle=True)`. |
| `check` | Test-build the bundle before storing it (default `True`). |

`flavor="pickle"` stores only the pickled object. See the [user guide](GUIDE.md#pytorch-models-as-a-bundle-plain-files-the-real-class-back)
for the files in a bundle and which models work.

### `at.load_bundle(folder, *, device="cpu", eval=True, verify=True, pickle=False)`

Loads a downloaded bundle folder directly, with no server. `verify` checks every file against the hashes in
`bundle.json` before anything is imported. Loading runs the bundle's `architecture.py`, so only load bundles you trust.

---

## The `with` API

The simple calls are built on explicit run objects, which you can use directly.

```python
with at.run(model, experiment, *, owner=None, dataset_version=None, tags=None, save_code=None,
            image=None, system_metrics=None, resume=False, params=None, settings=None) as r:
    with r.child("lr0.01", params={"lr": 0.01}) as c:
        c.log_metrics({"loss": 0.4}, step=0)
```

### `Run`

| Member | Description |
|---|---|
| `id` | The run id. |
| `tags` | The run's tags (a dict). |
| `child(name, params=None, resume=False)` | Start a nested run (use as a context manager). |
| `log_params(dict)`, `log_param(key, value)` | Parameters. Cannot be changed once logged. |
| `log_metrics(dict, step=None)`, `log_metric(key, value, step=None)` | Metrics. |
| `log_eval(**metrics)` | Final metrics, name-checked. |
| `set_tags(dict)` | Tags. |
| `log_artifact(path)`, `log_artifacts(folder_or_list)` | Files. |
| `log_text(text, name)` | A text file. |
| `log_model(model, flavor=None, **kwargs)` | A model. |
| `restore(name, folder=None)` | Download a file this run saved earlier. |

---

## Settings

### `Settings`

A frozen dataclass of every setting. `Settings.from_env()` reads the `AUTHTRACKER_*` variables;
`settings.masked()` returns a dict with credentials in URIs hidden. The fields and their variables are in
[Configuration](CONFIGURATION.md).

### `at.configure(settings=None)`

Sets the settings for the process. `configure(None)` goes back to reading the environment.

### `at.get_settings()`

The settings now in force.

---

## Errors

All inherit from `AuthTrackerError`, except `ConfigError`, which is a plain `ValueError` raised while the settings are read.

| Error | Raised when |
|---|---|
| `ConfigError` (a `ValueError`) | An `AUTHTRACKER_*` variable has an invalid value. |
| `ValidationError` (a `ValueError`) | A run breaks a rule (model name, required tag, metric name), an argument is wrong, or a resumed run changes a parameter. |
| `PromotionError` | A promotion fails a rule, or there is nothing to roll back to. |
| `BackendError` | The backend is unknown, missing a dependency, or the server could not be reached for a call that cannot be queued. |
| `UnsupportedError` (a `BackendError`) | The backend does not support the feature (the model registry on W&B, resuming on a backend that cannot look runs up). |

---

## Command line

```
authtracker config show     # the resolved settings, credentials masked
authtracker doctor          # backend, tracking URI, reachability, queued operations
authtracker sync            # upload operations queued while the server was down
authtracker test-alert [MESSAGE]     # send one test message to AUTHTRACKER_ALERT_WEBHOOK
authtracker export (MODEL | --all) --out FOLDER [--tracking-uri URI]      # see Moving a model
authtracker import [--all] FOLDER [--name NAME] [--tracking-uri URI]
```

`doctor` exits 1 when the backend cannot be reached. `sync` exits 1 when something stays queued. `export` and
`import` exit 1 with a `FAIL:` message when they are refused, and `import --all` exits 1 if any model failed. `--tracking-uri` overrides the store for that one
command (both the tracking and the registry address).

---

## Writing a backend

Subclass `authtracker.backends.Backend`, register it with `register_backend(name, factory)` or the
`authtracker.backends` entry point, and check it with `authtracker.testing.check_backend`. See the
[developer guide](DEVELOPMENT.md#adding-a-backend).

---

## Conventions

- **Three levels everywhere.** Model, experiment, run.
- **Tags are strings.** Values you pass are converted with `str()`.
- **Metric steps** count up from 0 per run unless you pass `step=`.
- **Parameters are write-once.** Repeating the same value is fine; changing it raises `ValidationError`.
- **Environment variables set defaults; arguments win.**
- **Runs close themselves** when the script ends. In a notebook, call `at.finish()`.
- **Models are loaded by alias,** never by run id.
- **Pickles run code when loaded,** so they are opt-in on save and on load.
