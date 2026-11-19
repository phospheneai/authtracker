# authtracker

[![PyPI](https://img.shields.io/pypi/v/authtracker)](https://pypi.org/project/authtracker/)
[![License](https://img.shields.io/pypi/l/authtracker)](https://github.com/phospheneai/authtracker/blob/main/LICENSE)

Experiment tracking and model versioning with a handful of calls. The same code runs on **MLflow**
(the default) or **Weights & Biases**, and you switch with one environment variable. Every run also
records the code, data and environment that produced it, so you can reproduce it later.

Small by design: no server of its own, settings only from environment variables, and other platforms
plug in as a single class.

- [API reference](https://github.com/phospheneai/authtracker/blob/main/docs/API.md)
- [User guide](https://github.com/phospheneai/authtracker/blob/main/docs/GUIDE.md)
- [Configuration (environment variables)](https://github.com/phospheneai/authtracker/blob/main/docs/CONFIGURATION.md)
- [Developer guide](https://github.com/phospheneai/authtracker/blob/main/docs/DEVELOPMENT.md)
- [Examples](https://github.com/phospheneai/authtracker/blob/main/examples/README.md)

## Quickstart

```
pip install "authtracker[mlflow]"
```

```python
import authtracker as at
from sklearn.linear_model import LogisticRegression

X, y = [[0], [1], [2], [3]], [0, 0, 1, 1]

at.init(model="model1", experiment="demo", dataset_version="v1")

for C in (0.1, 1.0, 10.0):
    at.start(f"C{C}", config={"C": C})            # one run per trial
    clf = LogisticRegression(C=C).fit(X, y)
    at.log({"accuracy": clf.score(X, y)})
    at.save(clf)                                  # the model

best = at.best_run("model1", "demo", metric="accuracy")
version = at.register(best)
at.promote("model1", version, "production")
print(at.load("model1", "production").predict([[3]]))
```

With no setup, runs go to a `mlflow.db` file in the current folder. To browse them, run `mlflow ui`.
To use a server, set `AUTHTRACKER_TRACKING_URI=https://mlflow.example.com` first.

## How it works

Every run lives in three levels:

```
fake-image                  <- model        (an MLflow experiment, a W&B project)
└── baseline                <- experiment   (a parent run)
    ├── lr0.01              <- run          (a child run: metrics, files, model)
    └── lr0.001
```

| Call | What it does |
|---|---|
| `at.init(model, experiment, ...)` | Opens the experiment. Takes `config="train.yaml"` and `dataset={...}` too. |
| `at.start(name, config={...})` | Starts a run (one trial) and closes the previous one. |
| `at.log({...})` | Logs metrics. The step counts up by itself. |
| `at.save(thing)` | Saves a file, a folder, text (with `name=`) or a model. |
| `at.best_run(model, experiment, metric)` | The best run by a metric. |
| `at.register()` / `at.promote(model, version, alias)` / `at.rollback(model)` | Versions and aliases. |
| `at.load(model, alias)` | Loads a model by alias, never by run id. |

Runs close when the script ends, as `FAILED` with the traceback saved if it ended with an error.
In a notebook, call `at.finish()`.

## Install

| Install | Use it for |
|---|---|
| `pip install "authtracker[mlflow]"` | MLflow, the default. Can also run a server or a local SQLite store. |
| `pip install "authtracker[mlflow-skinny]"` | A light client that only sends runs to an existing MLflow server. |
| `pip install "authtracker[wandb]"` | Weights & Biases. |
| `pip install "authtracker[gpu]"` | Also chart NVIDIA GPU use. Combine with the ones above. |

A plain `pip install authtracker` installs only the core, with no tracking platform, so pick one of the above.
Needs Python 3.10+. From a checkout, use `pip install -e ".[dev]"`.

## What each run records

- **Code:** git commit, branch, remote (a token in it is removed), the uncommitted changes as a patch, and
  your script. Rebuild the exact code of any run from them.
- **Data:** name, version, size and a fingerprint (hash) of your dataset folder.
- **Settings:** a training YAML, saved with the run and logged as parameters.
- **Environment:** package versions, `requirements.txt`, and the container image.
- **Output:** everything the run prints, saved as `output.log`, plus CPU, memory and GPU charts.

## Models

`at.save(model)` detects sklearn, XGBoost, LightGBM, PyTorch and others. For plain PyTorch, save a
**bundle**: a folder with your architecture file, the weights and a sample inference script. Loading it
gives you your own class back.

```python
at.save(net, flavor="bundle", init_args={"hidden": 32})
net = at.load("model1", "production")       # your Classifier, with its weights
```

## Choose a backend

```
export AUTHTRACKER_BACKEND=mlflow      # the default
export AUTHTRACKER_BACKEND=wandb       # Weights & Biases
```

Your training code does not change. On W&B, tracking, files and model artifacts work. The model registry
(`register`, `promote`, `rollback`, `load`) does not yet and raises a clear error.

## Configuration

Everything is an environment variable, so the same code runs anywhere. A few common ones:

| Variable | What it does |
|---|---|
| `AUTHTRACKER_BACKEND` | `mlflow` (default), `wandb` or `local`. |
| `AUTHTRACKER_TRACKING_URI` | Your MLflow server or store. |
| `AUTHTRACKER_REQUIRED_TAGS` | Tags every run must have, for example `owner,dataset_version`. |
| `AUTHTRACKER_ALLOWED_MODELS` | Allowed model names. |
| `AUTHTRACKER_PROMOTE_METRIC` / `_MIN` / `_MUST_BEAT_PROD` | Rules that must pass before `production`. |

All of them are in the [configuration reference](https://github.com/phospheneai/authtracker/blob/main/docs/CONFIGURATION.md).

```
authtracker config show     # the resolved settings
authtracker doctor          # is the backend reachable?
authtracker sync            # upload operations queued while the server was down
```

## Good to know

- One run is open at a time per process. For parallel trials, use separate processes.
- With several GPUs or nodes, call authtracker from the main process (rank 0) only.
- A tracking outage never stops training. Writes are retried, queued on disk and uploaded later.
  Saving a model, `register`, `load` and `best_run` need the server and raise an error instead.
- `at.init(..., resume=True)` continues an earlier run. It restores the record, not your model:
  save a checkpoint with `at.save` and get it back with `at.restore`.
- Models saved as pickles can run code when loaded. Only load models you trust.

## Troubleshooting

| You see | Do this |
|---|---|
| `Backend 'mlflow' needs MLflow` | Install an extra: `pip install "authtracker[mlflow]"`. |
| `Unknown model ...` or `Missing required tags` | A rule is set. Check `authtracker config show`. |
| `BackendError` on `register` or `load` | The server could not be reached. Run `authtracker doctor`. |
| `... cannot change parameters it already has` | You resumed a run with a different value. Use a new run (`resume=False`). |
| Runs show in the wrong place | The UI and your script use different stores. Point both at the same `AUTHTRACKER_TRACKING_URI`. |

## Status

**Experimental.** Tested with 250+ automated tests and eight runnable examples, on macOS with MLflow
(SQLite) and W&B in offline mode. Not yet verified: a remote or login-protected MLflow server, W&B online
mode, NVIDIA GPUs, multi-GPU jobs, and Windows. Expect rough edges and report them as issues.

## Development

```
pip install -e ".[dev]"
pytest
```

See the [developer guide](https://github.com/phospheneai/authtracker/blob/main/docs/DEVELOPMENT.md). Releases are
automatic: a merge to `main` that changes `src/authtracker/` runs the tests, bumps the version, tags it and
publishes to PyPI.

## License

Apache-2.0. See [LICENSE](https://github.com/phospheneai/authtracker/blob/main/LICENSE).
