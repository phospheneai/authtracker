# authtracker

[![PyPI](https://img.shields.io/badge/PyPI-authtracker-blue)](https://pypi.org/project/authtracker/)
[![License](https://img.shields.io/pypi/l/authtracker?color=brightgreen)](https://github.com/phospheneai/authtracker/blob/main/LICENSE)

**Know exactly which run produced the model in production.**

authtracker is a thin layer over MLflow (or Weights & Biases) that makes every training run reproducible and every model
version traceable. The same few calls cover tracking, versioning, promotion and rollback, and the code, data, config
and environment behind each run are recorded without you asking.

- **One API, two backends.** Same script on MLflow or W&B; switch with an environment variable.
- **Lineage by default.** Git commit, uncommitted diff, script, dataset hash, config and packages are saved with the run.
- **Models you can version.** `log_model`, `register`, `promote`, `rollback`; load by alias, never by run id.
- **Hard to lose work.** A crash ends `FAILED` with the traceback, an outage queues writes instead of killing the job, and
  `resume=True` picks a run back up.

**Docs:** [User guide](https://github.com/phospheneai/authtracker/blob/main/docs/GUIDE.md) · [API reference](https://github.com/phospheneai/authtracker/blob/main/docs/API.md) · [Settings](https://github.com/phospheneai/authtracker/blob/main/docs/CONFIGURATION.md) · [Examples](https://github.com/phospheneai/authtracker/blob/main/examples/README.md) · [Developer guide](https://github.com/phospheneai/authtracker/blob/main/docs/DEVELOPMENT.md)

## Install

| | |
|---|---|
| `pip install "authtracker[mlflow]"` | MLflow, the default. Also charts NVIDIA GPU use when a GPU is present. |
| `pip install "authtracker[mlflow-skinny]"` | Light client that only sends runs to an existing MLflow server. |
| `pip install "authtracker[wandb]"` | Weights & Biases. |

Python 3.10+. A plain `pip install authtracker` installs the core only, with no tracking backend.

## Quickstart

```python
import authtracker as at
from sklearn.linear_model import LogisticRegression

at.set_tracking_uri("sqlite:///mlflow.db")      # a local file, or a server such as "https://mlflow.example.com"

X, y = [[0], [1], [2], [3]], [0, 0, 1, 1]

at.init(model="model1", experiment="demo", dataset_version="v1")

for C in (0.1, 1.0, 10.0):
    at.start(f"C{C}", config={"C": C})            # one run per trial
    clf = LogisticRegression(C=C).fit(X, y)
    at.log({"accuracy": clf.score(X, y)})
    at.log_model(clf)

best = at.best_run("model1", "demo", metric="accuracy")
version = at.register(best)
at.promote("model1", version, "production")
print(at.load("model1", "production").predict([[3]]))
```

Browse the runs with `mlflow ui`.

## Data model

Three levels, so a model name, an idea you are testing and a single attempt never get mixed up:

```
churn                       model        MLflow experiment / W&B project
└── baseline                experiment   parent run: lineage files shared by its runs
    ├── lr0.01              run          child run: metrics, params, files, model
    └── lr0.001
```

`at.save` writes files (artifacts). `at.log_model` writes models (MLflow logged models, linked to the run).
Both go to the current run; add `experiment=True` to write to the experiment instead.

## Tasks

One call per line. Pick what you need.

**Track a run**
```python
at.log({"loss": 0.4, "accuracy": 0.9})          # metrics; the step counts up by itself
at.log({"loss": 0.3}, step=10)                  # explicit step
at.set_tags({"stage": "finetune"})              # tags
at.save("report.html")                          # a file; also a folder, or text with name=
at.watch(net, every=100)                        # gradient norms per layer (PyTorch)
```

**Tie a run to its config and data**
```python
at.init("churn", "baseline", config="train.yaml")                                            # YAML -> params
at.init("churn", "baseline", dataset={"name": "photos", "version": "v3", "path": "data/"})   # dataset + hash
```

**Log a model** (it is linked to the run that made it)
```python
at.log_model(clf)                                              # sklearn, PyTorch, XGBoost, LightGBM, ...
at.log_model(net, flavor="bundle", init_args={"hidden": 32})   # PyTorch: architecture file + weights
at.log_model("weights.pth")                                    # a file or folder as a versioned model
net = at.load_bundle("downloaded-folder")                      # a bundle with no server
```

**Pick the best, ship it, undo it**
```python
best = at.best_run("churn", "baseline", metric="f1")           # mode="min" for losses
version = at.register(best)                                    # new model version
at.promote("churn", version, "production")                     # alias; gated by your rules
at.rollback("churn")                                           # production -> previous version
model = at.load("churn", "production")                         # by alias, never by run id
```

**Share across runs** (write to the experiment instead of one run)
```python
at.save("labels.json", experiment=True)         # a file shared by every run
at.log({"best_lr": 0.01}, experiment=True)      # a result about the whole experiment
at.set_tags({"owner": "vision"}, experiment=True)
```

**Pick up after a crash**
```python
at.init("churn", "baseline", resume=True)       # continue the earlier experiment
at.start("lr0.01")                              # ...and its earlier run of this name; steps carry on
path = at.restore("ckpt.pt")                    # that run's last saved checkpoint, or None
```

**Point at a server, or move to another one**
```python
at.set_tracking_uri("https://mlflow.example.com")     # like mlflow.set_tracking_uri
```
```
authtracker export churn --out churn-export                              # runs, metrics, files, versions, aliases
authtracker import churn-export --tracking-uri https://new.example.com   # --all for every model
authtracker doctor                                                       # is the backend reachable?
```

**Get paged and keep a team honest** (environment variables)
```
export AUTHTRACKER_ALERT_WEBHOOK='https://...'            # POST failures to a webhook; unset = nothing sent
export AUTHTRACKER_ALERT_ON=failed,finished               # which events
export AUTHTRACKER_ALLOWED_MODELS=churn,fraud             # naming rule
export AUTHTRACKER_PROMOTE_METRIC=f1 AUTHTRACKER_PROMOTE_MIN=0.8   # gate "production"
```

## What is recorded

You do not call anything for these.

- **Code:** git commit, branch, remote (token removed), dirty flag, the uncommitted patch, and the script.
- **Data:** dataset name, version, file count, size and a hash (manifest or contents).
- **Config:** the training YAML as an artifact and as flattened params.
- **Environment:** installed packages (`requirements.txt`), Python version, container image (pinned by digest when known). PyTorch bundles also record torch, CUDA and device.
- **Run output:** stdout/stderr/logging as `output.log`; CPU, memory, disk and GPU system metrics.
- **Failures:** a crashed run ends `FAILED` with the traceback saved.

## Backends

| | MLflow (default) | Weights & Biases |
|---|---|---|
| Tracking, files, models | yes | yes |
| Registry (`register`, `promote`, `load`) | yes | not yet |
| Resume, export / import | yes | no |

Select with `export AUTHTRACKER_BACKEND=wandb`. Writing your own backend is one class:
see the [developer guide](https://github.com/phospheneai/authtracker/blob/main/docs/DEVELOPMENT.md#adding-a-backend).

## Troubleshooting

| You see | Do this |
|---|---|
| `Backend 'mlflow' needs MLflow` | `pip install "authtracker[mlflow]"` |
| `BackendError` on `register` / `load` | The server is unreachable: `authtracker doctor`. |
| `Unknown model ...`, `Missing required tags` | A rule is set: `authtracker config show`. |
| `at.save` rejects a model | Models go through `at.log_model`; `at.save` is for files. |

## Status

Experimental. About 300 tests and nine runnable examples, run on macOS with MLflow (SQLite) and W&B offline. Not yet
verified: a remote MLflow server, W&B online, NVIDIA GPUs, multi-GPU, Windows.

## Development

```
pip install -e ".[dev]" && pytest -n auto
```

A merge to `main` that changes `src/authtracker/` is tested, versioned and published to PyPI automatically.
See the [developer guide](https://github.com/phospheneai/authtracker/blob/main/docs/DEVELOPMENT.md).

## License

Apache-2.0. See [LICENSE](https://github.com/phospheneai/authtracker/blob/main/LICENSE).
