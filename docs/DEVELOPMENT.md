# Developer guide

For people who change authtracker itself. Using it is covered in the [README](../README.md), the
[user guide](GUIDE.md) and the [API reference](API.md).

## Set up

```bash
git clone https://github.com/phospheneai/authtracker.git && cd authtracker
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # authtracker, MLflow, W&B, scikit-learn, pytest, ruff
pip install torch                # optional: the PyTorch bundle, watch and example 07 tests
pytest -q -n auto                # about 3 minutes on 4 CPUs (7+ minutes without -n)
```

Python 3.10 or newer. Without torch, the tests that need it are skipped, not failed.

## Layout

```
src/authtracker/
  __init__.py        public names and __version__ (the release workflow edits this one line)
  simple.py          the simple API: init, start, log, save, restore, finish (module-level state)
  run.py             Run and the with-API; validation, code snapshot, resume, captured log upload
  registry.py        best_run, register, promote (with rules), rollback, load
  settings.py        Settings and the AUTHTRACKER_* variables
  configfile.py      the training YAML: parameters and the `authtracker:` block
  dataset.py         dataset records and hashes
  bundle.py          PyTorch bundles and pickles: build, load, verify
  capture.py         console output saved as output.log
  watch.py           gradient norms
  utils.py           git info, environment capture, container image, secret checks
  cli.py             `authtracker config | doctor | sync`
  testing.py         check_backend, the contract every backend must pass
  errors.py          the error classes
  backends/
    base.py          the Backend interface (the only thing the rest of the code talks to)
    mlflow_backend.py, wandb_backend.py, local.py
    mlflow_transfer.py   export_model / import_model: the export folder format and its MLflow reader and writer
    resilient.py     retry and offline queue, wrapped around every backend
tests/               one file per area; see tests/TESTING.md
examples/            nine runnable, tested scripts; see examples/README.md
docs/                this folder
```

## Rules the code follows

- **Nothing outside a backend knows its platform.** Only `backends/mlflow_backend.py` imports MLflow and only
  `backends/wandb_backend.py` imports W&B. A test checks that the core never imports MLflow.
- **Settings come only from environment variables,** read in `settings.py`. Anything per run is also an argument,
  and the argument wins.
- **Tracking never stops training.** Calls are retried and queued (`backends/resilient.py`). Only calls that need
  an answer (`register`, `load`, `best_run`, saving a model) raise.
- **Fail early and say how to fix it.** Errors name the problem and the fix, and are one of the classes in
  `errors.py`.
- **Never copy a secret on purpose.** Remote URLs lose their token, and a patch or script that looks like it
  holds a key is not stored.

## Tests

```bash
pytest -q -n auto                  # everything, in parallel (drop -n to debug one test)
pytest tests/test_simple.py -q     # one file
pytest tests/test_examples.py -q   # the examples, as real scripts (about 3 minutes)
ruff check src tests               # lint
```

Add a test with every change. Backend behaviour is tested against `local`, `mlflow` and `wandb` (offline) with
the same test, using a `params=[...]` fixture, so a feature that only works on one is caught. What has and has not
been tried on real hardware is in [tests/TESTING.md](../tests/TESTING.md).

Tests run in parallel, so each one must use its own folder (`tmp_path`) and never rely on another test's leftovers.
A fixture that is expensive to build and only read (like the model store in `test_transfer.py`) is built once per
module with `scope="module"`. Tests that open a repo or the examples start from a temporary folder, because MLflow caches its database
connection per path. If you open a store from a test, use a new folder for each test and pass both
`tracking_uri` and `registry_uri` to `MlflowClient`.

## Examples

Every numbered script in `examples/` is run by `tests/test_examples.py`, which then opens the example's database
and checks what was recorded. A guard test fails if you add an example without a test. See
[examples/README.md](../examples/README.md).

## Releasing

Releases are automatic. A push to `main` that changes `src/authtracker/**` runs
`.github/workflows/release.yml`:

1. installs `.[dev]` and CPU PyTorch and runs `pytest -n auto`;
2. bumps the patch number in `src/authtracker/__init__.py` (`0.0.3` becomes `0.0.4`);
3. builds the wheel and sdist and checks them with `twine check --strict`;
4. commits `chore: release <version> [skip ci]`, creates the annotated tag `<version>` (no `v`) whose message
   lists the commit subjects since the last tag, and pushes both;
5. uploads to PyPI with `twine`;
6. creates a GitHub release with the same notes.

You do not edit the version or create tags by hand. The bump commit carries `[skip ci]`, so it does not trigger
another release. Changes outside `src/authtracker/**` (docs, tests, examples) do not release; start the workflow
by hand from the Actions tab (`workflow_dispatch`) if you need one.

The workflow only ever adds 1 to the patch number. To move to a new minor or major version, set `__version__`
in a normal commit (for example `0.2.0`); the next release is then `0.2.1`.

**One-time setup**

1. Create the repository secret `PYPI_API_TOKEN` (Settings, Secrets and variables, Actions) with a PyPI API token.
2. Allow Actions to push: Settings, Actions, General, Workflow permissions, **Read and write**. The workflow
   asks for `contents: write` itself.
3. If `main` is protected, allow the `github-actions[bot]` to push to it, since the bump goes straight to `main`.

Check a build locally without releasing:

```bash
pip install build twine
python -m build && twine check --strict dist/*
```

## Adding a backend

MLflow is the default, but nothing outside `backends/mlflow_backend.py` knows about MLflow, and
nothing outside `backends/wandb_backend.py` knows about W&B (a test checks that the core never
imports MLflow). Another platform is a new class, with no change to
the rest of authtracker.

**1. Implement `Backend`** (`authtracker/backends/base.py`). Nine methods are required:
`ping`, `start_run`, `end_run`, `set_tags`, `log_params`, `log_metrics`, `log_artifact`,
`get_run`, `search_runs`. That is enough for `init / start / log / save (files, text) / best_run`.
Model and registry support is optional: implement `log_model`, `register`, `set_alias`, `get_alias`,
`version_run_id` and `load` when the platform has them. Anything left out raises a clear
`BackendError` ("The X backend does not support the model registry") and nothing else breaks. The
W&B backend is a real example: it implements the required methods and `log_model` only.

If the platform already buffers writes while offline (W&B does), set `buffers_offline = True` on
your class, and authtracker will not add its own retry and queue layer. If it already records the
console output, set `captures_console = True`, and authtracker will not save `output.log`.

```python
from authtracker.backends import Backend

class MyBackend(Backend):
    def __init__(self, settings): ...
    def ping(self): ...
    def start_run(self, experiment, run_name, tags, parent_id): ...   # return a run id string
    # ... the other required methods
```

**2. Make it available**, in one of two ways:

```python
# a) in code
from authtracker.backends import register_backend
register_backend("mine", lambda settings: MyBackend(settings))
```

```toml
# b) from its own package: no change to authtracker at all (pyproject.toml of that package)
[project.entry-points."authtracker.backends"]
mine = "my_package:make_backend"        # callable(settings) -> Backend
```

Then `export AUTHTRACKER_BACKEND=mine`. Backend-specific values (URL, token) should be read from the
backend's own environment variables or from `settings.tracking_uri`.

**3. Check it** with the shared contract, the same one the built-in backends pass:

```python
from authtracker.testing import check_backend

def test_my_backend():
    check_backend(MyBackend(...))                     # required behaviour
    check_backend(MyBackend(...), model=my_model)     # also models + registry, if implemented
```

`tests/test_contract.py` contains a complete 9-method example backend (`MemoryBackend`).

`Backend.api_version` is checked when a backend loads. It changes only when the interface changes in
a way that breaks existing backends, and loading then fails with a clear message. The retry and
offline-queue layer wraps every backend automatically.
