# authtracker

[![PyPI](https://img.shields.io/badge/PyPI-authtracker-blue)](https://pypi.org/project/authtracker/)
[![License](https://img.shields.io/pypi/l/authtracker?color=brightgreen)](https://github.com/phospheneai/authtracker/blob/main/LICENSE)

Experiment tracking and model versioning with a handful of calls. The same code runs on **MLflow**
(the default) or **Weights & Biases**. Every run also records the code, data and environment that
produced it, so you can reproduce it later.

- [API reference](https://github.com/phospheneai/authtracker/blob/main/docs/API.md)
- [User guide](https://github.com/phospheneai/authtracker/blob/main/docs/GUIDE.md)
- [Configuration](https://github.com/phospheneai/authtracker/blob/main/docs/CONFIGURATION.md)
- [Examples](https://github.com/phospheneai/authtracker/blob/main/examples/README.md)
- [Developer guide](https://github.com/phospheneai/authtracker/blob/main/docs/DEVELOPMENT.md)

## Quickstart

```
pip install "authtracker[mlflow]"
```

```python
import authtracker as at
from sklearn.linear_model import LogisticRegression

at.set_tracking_uri("sqlite:///mlflow.db")      # where runs go: a file here, or a server URL like "https://mlflow.example.com"

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

Runs go to the `mlflow.db` file in the current folder. Browse them with `mlflow ui`. To use a server, change the
URI in that line, or leave it out and set `AUTHTRACKER_TRACKING_URI=https://mlflow.example.com` instead.

## How it works

```
fake-image                  <- model        (an MLflow experiment, a W&B project)
└── baseline                <- experiment   (a parent run)
    ├── lr0.01              <- run          (a child run: metrics, files, model)
    └── lr0.001
```

| Call | What it does |
|---|---|
| `at.init(model, experiment)` | Opens the experiment. |
| `at.start(name, config={...})` | Starts a run (one trial). |
| `at.log({...})` | Logs metrics. |
| `at.save(thing)` | Saves a file, folder, text or model. |
| `at.best_run(...)`, `at.register()`, `at.promote(...)`, `at.load(...)` | Pick, version and load a model. |

Runs close when the script ends. In a notebook, call `at.finish()`.

## Install

| | |
|---|---|
| `pip install "authtracker[mlflow]"` | MLflow, the default. Also charts NVIDIA GPU use when a GPU is there. |
| `pip install "authtracker[mlflow-skinny]"` | A light client that only sends runs to an existing MLflow server. |
| `pip install "authtracker[wandb]"` | Weights & Biases. |

Python 3.10+. From a checkout: `pip install -e ".[dev]"`.

## What each run records

Git commit, branch and uncommitted changes · your script · dataset name, version and hash · the training YAML
as parameters · package versions and container image · everything printed (`output.log`) · CPU, memory and
GPU charts.

## More

- **PyTorch models:** `at.save(net, flavor="bundle", init_args={...})` stores your architecture file and weights;
  `at.load` gives your own class back. [Guide](https://github.com/phospheneai/authtracker/blob/main/docs/GUIDE.md#pytorch-models-as-a-bundle-plain-files-the-real-class-back)
- **Resume a run:** `at.init(..., resume=True)` continues an earlier run.
- **Move to another server:** `authtracker export churn --out dir`, then `authtracker import dir` against the
  new server. Add `--all` for every model. MLflow only.
- **Alerts:** set `AUTHTRACKER_ALERT_WEBHOOK` (optional) and `AUTHTRACKER_ALERT_ON=failed,finished` to POST the events
  you choose to a webhook. `at.alert(...)` sends your own. [Details](https://github.com/phospheneai/authtracker/blob/main/docs/CONFIGURATION.md#alerts-optional)
- **Weights & Biases:** `export AUTHTRACKER_BACKEND=wandb`. Tracking and files work; the model registry does not yet.
- **Rules and settings** (allowed models, required tags, promotion rules) are environment variables:
  see [Configuration](https://github.com/phospheneai/authtracker/blob/main/docs/CONFIGURATION.md).

```
authtracker config show     # the resolved settings
authtracker doctor          # is the backend reachable?
```

## Troubleshooting

| You see | Do this |
|---|---|
| `Backend 'mlflow' needs MLflow` | Install an extra: `pip install "authtracker[mlflow]"`. |
| `BackendError` on `register` or `load` | The server could not be reached. Run `authtracker doctor`. |
| `Unknown model ...` or `Missing required tags` | A rule is set. Check `authtracker config show`. |

## Status

Experimental. About 300 automated tests and nine runnable examples, run on macOS with MLflow (SQLite)
and W&B offline. Not yet verified: a remote MLflow server, W&B online, NVIDIA GPUs, multi-GPU, Windows.

## Development

```
pip install -e ".[dev]"
pytest -n auto
```

Releases are automatic: a merge to `main` that changes `src/authtracker/` runs the tests, bumps the version,
tags it and publishes to PyPI. See the [developer guide](https://github.com/phospheneai/authtracker/blob/main/docs/DEVELOPMENT.md).

## License

Apache-2.0. See [LICENSE](https://github.com/phospheneai/authtracker/blob/main/LICENSE).
