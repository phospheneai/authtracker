# authtracker user guide

**Start here** is a cheat sheet: one call per line, grouped by what you are trying to do. Everything after it is the
detailed reference. For exact signatures see the [API reference](API.md); for every setting see
[Settings](CONFIGURATION.md).

## Start here

### The data model

Three levels, so a project name, an idea you are testing and a single attempt never get mixed up:

```
churn                       model        MLflow experiment / W&B project
└── baseline                experiment   parent run: lineage files shared by its runs
    ├── lr0.01              run          child run: metrics, params, files, model
    └── lr0.001
```

`at.init` opens the experiment, `at.start` opens a run under it. `at.log`, `at.save` and `at.log_model` write to the
current run, or to the experiment with `experiment=True`. A model is the name of the project, not an artifact.

### Common tasks

One call per line. Pick what you need.

**Track a run**
```python
at.init("churn", "baseline", dataset_version="v1")   # open the experiment
at.start("lr0.01", config={"lr": 0.01})              # open a run; its config is logged as params
at.log({"loss": 0.4, "accuracy": 0.9})               # metrics; the step counts up by itself
at.log({"loss": 0.3}, step=10)                       # explicit step
at.set_tags({"stage": "finetune"})                   # tags
```

**Save files** (artifacts, not models)
```python
at.save("confusion_matrix.png")                      # a file
at.save("reports/")                                  # the files directly in a folder
at.save("epoch 1 done\n", name="train.log")          # text
```

**Log a model** (it is linked to the run that made it)
```python
at.log_model(clf)                                              # sklearn, PyTorch, XGBoost, LightGBM, ...
at.log_model(net, flavor="bundle", init_args={"hidden": 32})   # PyTorch: your architecture file + weights
at.log_model("weights.pth")                                    # a file or folder as a model: no loader, load() returns a folder
```

**Pick the best, ship it, undo it**
```python
best = at.best_run("churn", "baseline", metric="f1")  # mode="min" for losses
version = at.register(best)                           # a new model version
at.promote("churn", version, "production")            # alias; gated by your rules, see Settings
model = at.load("churn", "production")                # by alias, never by run id
at.rollback("churn")                                  # production -> the previous version
```

**Share across runs** (write to the experiment instead of one run)
```python
at.save("labels.json", experiment=True)               # a file shared by every run of the experiment
at.log({"best_lr": 0.01}, experiment=True)            # a result about the whole experiment
at.set_tags({"owner": "vision"}, experiment=True)
```
A library model (sklearn, ...) can go to the experiment only before the first `at.start`. A bundle, a file or a folder can go any time.

**Tie a run to its config and data**
```python
at.init("churn", "baseline", config="train.yaml")                                            # YAML -> params; the file is kept
at.init("churn", "baseline", dataset={"name": "photos", "version": "v3", "path": "data/"})   # name, version, hash
```

**Pick up after a crash**
```python
at.init("churn", "baseline", resume=True)             # continue the earlier experiment
at.start("lr0.01")                                    # ...and its earlier run of this name; steps carry on
path = at.restore("ckpt.pt")                          # that run's last saved checkpoint, or None
```

**PyTorch**
```python
at.watch(net, every=100)                              # gradient norm per layer, as metrics
net = at.load("churn", "production")                  # a bundle comes back as your own class
net = at.load_bundle("downloaded-folder")             # a bundle with no server
```

**Point at a server, or move to another one**
```python
at.set_tracking_uri("https://mlflow.example.com")     # before at.init; like mlflow.set_tracking_uri
```
```
authtracker export churn --out churn-export                              # one model: runs, metrics, files, versions, aliases
authtracker import churn-export --tracking-uri https://new.example.com   # add --all to either for every model
authtracker doctor                                                       # is the backend reachable?
```

**Get paged** (a webhook; with none set, nothing is sent)
```
export AUTHTRACKER_ALERT_WEBHOOK='https://...'        # a secret: keep it out of code and git
export AUTHTRACKER_ALERT_ON=failed,finished           # failed, stopped, finished, custom
authtracker test-alert                                # one test message
```
```python
at.alert("Loss is NaN", "epoch 3", level="error")     # your own message
```

**Keep a team honest** (a violation is refused before anything is saved)
```
export AUTHTRACKER_ALLOWED_MODELS=churn,fraud
export AUTHTRACKER_REQUIRED_TAGS=owner,dataset_version
export AUTHTRACKER_PROMOTE_METRIC=f1 AUTHTRACKER_PROMOTE_MIN=0.8
```

**Backend**
```
export AUTHTRACKER_BACKEND=wandb                      # same code; tracking and files work, the registry does not yet
```

**Where the lineage is.** Each run carries tags such as `git_sha`, `git_dirty` and `dataset_hash`. The files that rebuild
a run exactly (the script, the uncommitted patch, the config, the package list) are on the **experiment's** Artifacts
tab, shared by its runs.

---

## Install

```bash
pip install "authtracker[mlflow]"          # recommended: MLflow, the default backend (about 490 MB with its libraries)
pip install "authtracker[mlflow-skinny]"   # light: only sends runs to an existing MLflow server (about 60 MB)
pip install "authtracker[wandb]"           # Weights & Biases instead of, or next to, MLflow
pip install "authtracker[mlflow,wandb]"     # extras can be combined (NVIDIA GPU charts are included with mlflow, except on a Mac)

# from a checkout of this repository, replace "authtracker" with "-e ./authtracker"
```

A plain `pip install authtracker` installs only the core, with no tracking platform: pick one of the
extras above. Which MLflow to choose:

| | `[mlflow]` | `[mlflow-skinny]` |
|---|---|---|
| Use it on | the machine that runs the MLflow server or UI, or when tracking to a local SQLite file | machines that only train and send runs to a server |
| Size with its libraries | about 490 MB | about 60 MB |
| Tracking to `http://server:5000` | yes | yes |
| Tracking to `sqlite:///...` or a database URI | yes | no (it says so clearly) |
| Models | the framework libraries you use are still needed in both | |

## Quick start

```python
import authtracker as at
from sklearn.linear_model import LogisticRegression

X, y = [[0], [1], [2], [3]], [0, 0, 1, 1]

at.init(model="model1", experiment="demo", dataset_version="v1")   # model + experiment

for C in (0.1, 1.0, 10.0):
    at.start(f"C{C}", config={"C": C})            # one run per trial; closes the previous one
    clf = LogisticRegression(C=C).fit(X, y)
    at.log({"accuracy": clf.score(X, y)})         # step counts up by itself
    at.save("fitted ok\n", name="train.log")      # text -> a log file
    at.log_model(clf)                                  # a model

at.register()                                     # a version of the current run's model
at.promote("model1", 1, "production")
print(at.load("model1", "production").predict([[3]]))
```

Point it at your server first (otherwise MLflow uses a local `mlflow.db` in the current folder):

```bash
export AUTHTRACKER_TRACKING_URI=https://mlflow.example.com     # your MLflow server
# or in code, before at.init: at.set_tracking_uri("https://mlflow.example.com")
mlflow ui                  # or open your server's address to browse the runs
```

| Call | What it does |
|---|---|
| `at.init(model, experiment, ...)` | Opens the experiment (the parent run) under the model. Closes anything still open. Takes `config="train.yaml"` and `dataset={...}` too (see below). |
| `at.start(name, config={...})` | Starts a run (one trial) and closes the previous one. |
| `at.log({...})` | Logs metrics to the current run. `step` counts up by itself; pass `step=` to control it. |
| `at.save(thing)` | Saves files as artifacts: a file, a folder or list of files, or text (with `name=`). |
| `at.log_model(model)` | Saves a model so it can be registered, promoted and loaded: a model object, or a file or folder as files. |
| `at.finish()` | Optional. Runs also close when the script ends, as `FAILED` if it ended with an error. |
| `at.set_tags({...})` | Adds tags to the current run. |
| `at.best_run(model, experiment, metric)` | The best run (higher is better; `mode="min"` for loss). |
| `at.register()` | Creates a model version from the current run (or `at.register(best)`). |
| `at.promote(model, version, alias)` | Points an alias such as `staging` or `production` at a version. |
| `at.rollback(model)` | Moves `production` back to the previous version. |
| `at.load(model, alias)` | Loads a model by alias, never by run id. |

Good to know:
- Runs end as `FINISHED`, or as `FAILED` with the traceback saved as `error.txt` when the script
  ends with an unhandled error. In a notebook there is no "end of script", so call `at.finish()`.
- One run is open at a time per process. For parallel trials use separate processes.
- With several GPUs, call `log` and `save` from the main process only.
- `at.current_run()` returns the open run, if you need its id.

## Training YAML and dataset

Keep the settings of a training run in one YAML file, and tell authtracker which data it used:

```yaml
# train.yaml
authtracker:                       # optional: where this run belongs
  model: fake-image
  experiment: baseline
  tags: {team: vision}
optimizer: {name: adam, lr: 0.001} # everything else is saved as parameters
epochs: 15
```

```python
at.init(config="train.yaml",
        dataset={"name": "faces", "version": "v3", "path": "/data/faces"})
```

- The file is saved with the run (the **Artifacts** tab), and each setting is logged as its own
  parameter: `optimizer.lr`, `epochs`. Nested keys are joined with dots and lists become text.
- The `authtracker:` block may set `model`, `experiment`, `dataset`, `dataset_version`, `tags` and
  `owner`. **Anything you pass to `at.init(...)` overrides the block**, so one YAML can serve many runs:
  `at.init("fake-image", "bigger-net", config="train.yaml")`. A misspelled key in the block is an error.
- `config=` also takes a plain dict: `config={"lr": 0.1}`.
- **The dataset record** stores the name, version, path, number of files and size, plus a **hash**: a
  fingerprint of the data. Two runs with the same hash used the same data, and if a folder called `v3`
  changes, its hash changes. Every run carries the tags `dataset_version`, `dataset_name`,
  `dataset_hash` (the first 16 characters), `dataset_files` and `dataset_bytes`, so you can search for
  them in the UI. The full record is saved as `dataset.json`.
- Hash modes, set with `"hash": ...` in the dataset dict:

| Mode | What it reads | Catches |
|---|---|---|
| `manifest` (default) | file names and sizes only | files added, removed, renamed or resized. Fast even for millions of files. |
| `contents` | every byte | also a file edited without changing its size. Slow on big data. |
| `none` | nothing | no hash |

- Only local folders and files can be hashed. For `s3://...` and other remote paths the path is
  recorded and no hash is computed.
- `dataset_version` is still accepted on its own (`at.init(..., dataset_version="v1")`), and a version
  inside `dataset={...}` counts as the required `dataset_version` tag.

## Choose a backend

```bash
export AUTHTRACKER_BACKEND=mlflow     # the default
export AUTHTRACKER_BACKEND=wandb      # Weights & Biases
```

Your training code does not change. These are the same calls on both:

| | MLflow | Weights & Biases |
|---|---|---|
| Model (`init(model=...)`) | an MLflow experiment | a **project** |
| Experiment (`init(experiment=...)`) | a parent run | a run **group** and a parent run (job type `experiment`) |
| Run (`start(...)`) | a child run | a run (job type `run`) in that group |
| `log`, `save` files and text, `set_tags`, `best_run`, `finish` | yes | yes |
| `save(model)` | MLflow model flavors (sklearn, PyTorch, `pyfunc`, ...) | a **model artifact**: pickle, `torch.save`, or a path; `code_paths=[...]` and `pip_requirements=[...]` are stored with it |
| `register`, `promote`, `rollback`, `load` | yes | **not yet**: they raise a clear `BackendError` |
| Offline | authtracker's retry and queue layer (`authtracker sync`) | W&B's own offline mode (`WANDB_MODE=offline`, then `wandb sync`) |

**Weights & Biases setup**

```bash
export AUTHTRACKER_BACKEND=wandb
export WANDB_API_KEY=...              # from your W&B settings; keep it out of files and git
export WANDB_ENTITY=my-team           # optional: team or user
# self-hosted W&B server: export AUTHTRACKER_TRACKING_URI=https://wandb.example.com
# no account yet, or offline: export WANDB_MODE=offline   (runs are written to ./wandb)
```

Things that differ on W&B, because W&B works differently:
- There are no nested runs. The experiment run and its runs are separate W&B runs tied together by
  the group name, so group by **Group** in the W&B table to get the three levels.
- Tags are plain strings there, so `key=value` pairs are stored as `key=value` tags and in the run
  config, and read back from the config.
- `at.best_run` also finds runs from other processes through the W&B API, which can lag a few seconds
  behind a run that just finished. Runs started in the same process are always found immediately.
- A parameter cannot be changed after it is logged (W&B enforces this).
- `at.log_model(model, serialization_format=...)` and other MLflow-only options raise an error that lists
  what the W&B backend supports.
- `register` and `load` are not implemented, so the "ship architecture and `infer_single.py`" example
  below (a `pyfunc` model) is MLflow-only. On W&B, save the weights with
  `at.log_model(model, code_paths=["arch", "infer_single.py"])` and use W&B's own artifact aliases and
  registry for versions.

Status: tracking, files and model artifacts on W&B are tested with the real W&B SDK in offline mode.
Reading runs back from the W&B server (`best_run` across processes) is tested only against a
stand-in for the W&B API, because no account was available. Try it on a real project before relying
on it.

### Resuming a run

By default every execution of your script makes new runs, even with the same names. To **continue** instead,
pass `resume=True`: if an earlier run is there it is reopened, and if not a new one is started.

```python
at.init("detector", "baseline", resume=True, config={"lr": 0.1})   # continue the earlier experiment run, if any
at.start("abc", config={"epochs": 10})                             # ... and the earlier run named "abc", if any
for epoch in range(10):
    at.log({"loss": loss})                                         # the steps carry on where they stopped
```

Run that script a second time and there is still **one** `baseline` run and **one** `abc` run, with its loss curve
continuing instead of restarting.


How it matches: the same **model, experiment and owner**, and for a run also the same **name** under that
experiment run. Someone else's runs are never continued. If several earlier runs match, the newest one is used.

What a resume does, and does not do:
- **Metric steps continue** from the last step logged, so the chart extends instead of overlapping.
- **Parameters are kept.** Repeating a parameter with the same value, or adding new ones, is fine. Changing one is
  refused (`lr was 0.1 and is now 0.01`), **before** the old run is touched, so it stays as it was. Use a new
  run (`resume=False`) if the setting really changed.
- **Earlier parts are not overwritten.** The resumed part saves `output.resume1.log`, `requirements.resume1.txt`,
  `code_changes.resume1.patch`, `script_train.resume1.py` (and `.resume2` the next time), next to the originals.
  Tags keep the original commit and image, and add `resume1_git_sha` and the like when the new part differs.
  `resume_count` and `resumed` tags tell you it was continued.
- **Your model and optimizer are not restored.** Resuming only reopens the *record*. Save a checkpoint with
  `at.save("ckpt.pt")` and get it back with `at.restore("ckpt.pt")`, which returns the downloaded path, or `None`
  when the run has no such file. The same script then works the first time and after every restart:

  ```python
  at.init("detector", "baseline", resume=True)
  at.start("abc")                                  # the run the checkpoint was saved on
  path = at.restore("ckpt.pt")
  start = torch.load(path)["epoch"] + 1 if path else 0
  for epoch in range(start, 100):
      ...
      torch.save({"epoch": epoch, ...}, "ckpt.pt"); at.save("ckpt.pt")
      at.log({"loss": loss}, step=epoch)
  ```
- **The server must be reachable to look for an earlier run.** If it is not, a new run is started with a warning,
  so training is never stopped for this.
- A backend that cannot look runs up (Weights & Biases for now) stops with a message instead of silently
  starting fresh. MLflow and the local backend support it.

### Options of `at.init`

Everything you normally want to change for one run is an argument, so it shows up in your editor's hints.
Environment variables (see [Configuration](CONFIGURATION.md)) only set *defaults* for a machine, a container or a CI job, and an argument always wins.

| Argument | Meaning | Default |
|---|---|---|
| `model`, `experiment` | Where the run belongs (or set them in the YAML) | required |
| `config` | A training YAML (its path) or a dict of settings, saved as parameters | none |
| `dataset`, `dataset_version` | The data the run uses; a path or `{"name", "version", "path"}`, stored with a hash | none |
| `owner`, `tags` | Who ran it, and extra tags | `$USER` |
| `save_code` | Store the uncommitted changes and your script (a file that looks like a secret is skipped) | `True` |
| `image` | The container image, ideally `name@sha256:...` | found from the platform, if possible |
| `system_metrics` | Record CPU, memory and GPU use while the run is open | `True` |
| `resume` | Continue the earlier experiment run and its runs with the same names, if there are any; otherwise start fresh | `False` |
| `settings` | A full `Settings(...)` object, for anything else | read from the environment |

## What to log

**Metrics.** `at.log({"train_loss": 0.4, "val_acc": 0.9})` once per epoch gives a curve in the UI.
Final numbers are logged the same way, for example `at.log({"accuracy": 0.93})`. If
`AUTHTRACKER_METRIC_NAMES` is set, any other name is rejected.

**Gradients.** `at.watch(model, every=100)` logs how large each layer's gradient is while you train, to
help you spot exploding or vanishing gradients. Call it once, after `at.init` and before the training loop:

```python
at.init("fake-image", "baseline")
at.watch(model, every=100)          # the metrics are grad_norm/<layer name> and grad_norm/total
for batch in loader:
    ...
```

A step is one training forward pass, so with gradient accumulation it counts every micro-batch. The values
of a step are written when the next training step starts. `at.watch` returns a function that stops it (and
writes what is waiting). These are ordinary metrics, drawn as curves, so you get one number per layer and
no histograms. `DataParallel` and `DistributedDataParallel` wrappers are looked through, so layer names have
no `module.` prefix. Needs PyTorch 2.1 or newer. With DDP, call it on rank 0 only, like the rest.
It is tested on the local and MLflow backends with a small CPU model; not on a GPU.

**Logs.** Everything a run prints is saved for you as `output.log`: `print`, Python `logging` that
goes to the console, and `stderr`. Nothing to call. The experiment run's log has the whole script, and
each run's log has only what was printed while it was the current run. The screen output is unchanged.
The log is uploaded every 60 seconds while the run is going (`AUTHTRACKER_LOG_UPLOAD_INTERVAL`, `0` =
only at the end), so a job that is killed keeps its log up to the last upload. The page does not update
by itself: reload the run's **Artifacts** tab to see new lines. These uploads are best effort and are
not queued during an outage; the final upload at the end of the run always is.
You can still save your own text with `at.save("epoch 1: loss 0.90\n", name="train.log")`, and use
`at.set_tags({"stage": "finetune", "gpu": "A100"})` for short labels you can filter on.

What `output.log` does not capture: output of subprocesses or native code that writes straight to the
terminal, and loggers that write only to files. It stops growing at 20 MB. It records everything that
is printed, so do not print secrets. On Weights & Biases the platform records this itself (the run's
**Logs** tab), so authtracker does not add a second copy. Switch it off with
`AUTHTRACKER_CAPTURE_LOGS=false`.

**Files and bundles.**

```python
at.save("confusion_matrix.png")                          # one file
at.save("reports/")                                      # a bundle: the files directly in a folder
at.save(["config.yaml", "vocab.json", "labels.csv"])     # a bundle: a list of files
```

By default these go to the trial you are in. Add `experiment=True` (for example `at.save("labels.json", experiment=True)`) to put a
file, folder or text on the experiment instead, where every trial of it can use it. `at.log` and `at.set_tags` take the
same argument, e.g. `at.log({"best_lr": 0.01}, experiment=True)`. So does `at.log_model`: a bundle, a pickle, a file or a
folder can be logged on the experiment at any time, and a library model (sklearn, ...) only before the first `at.start`.

Files keep their names, and **files inside sub-folders are not saved** (they are skipped without a warning), so names inside one
bundle must be unique. To keep a folder structure, store the folder as a model with `at.log_model("folder/")`.
A path that does not exist is an error, not silently logged as text.

**Models.** `at.save` is for files only. A model goes through `at.log_model(model)`, which stores it in MLflow's model
section, linked to the run, so you can register, promote and load it. It detects sklearn, PyTorch, XGBoost, LightGBM and similar models. Pass
`flavor="pytorch"` or other MLflow options as keyword arguments when needed.

## PyTorch models as a bundle (plain files, the real class back)

For plain PyTorch, the simplest and most reproducible way to keep a model is a **folder**: your
architecture file, the weights, a sample inference script and the environment. Loading it gives you
back the real `nn.Module`, built from your own code.

```python
# architecture.py        the class lives in its own file
class Classifier(nn.Module):
    def __init__(self, n_features=8, hidden=32, n_classes=2): ...

# train.py
from architecture import Classifier
net = Classifier(hidden=32)
...                                                       # train it, on any device
at.log_model(net, flavor="bundle", init_args={"hidden": 32})   # the folder is stored with the run
at.register()                                             # ... and versioned
net = at.load("name", "production")                       # the real Classifier, with its weights
net = at.load("name", "production", device="cuda")        # on a GPU, if you want
```

What is stored (see [`examples/07_pytorch_bundle.py`](../examples/07_pytorch_bundle.py) for a runnable, tested version):

| File | What it holds |
|---|---|
| `architecture.py` | The file that defines your class. `code=["architecture.py", "layers/"]` adds more files or folders. |
| `weights.pth` | The weights as a plain `state_dict`, always saved from CPU tensors. It is loaded with `weights_only=True`, so it cannot run code. |
| `inference.py` | A sample script that loads the model and predicts: `python inference.py input.npy`. It needs `torch` and `requirements.txt`, **not authtracker**. Pass `inference="mine.py"` to use your own. |
| `requirements.txt` | Pinned versions of `torch` and every installed package your code uses, including names that arrive through `from something import *`. Pass `requirements=[...]` to set it yourself. |
| `assets/` | Files your constructor reads, added with `assets=[...]` (see below). |
| `environment.txt` | Every package installed when it was saved. |
| `bundle.json` | What to import (`module`, `class`), the `init_args`, a SHA-256 hash of every file, and the Python / torch / CUDA versions and device it was trained on. |

How it works:
- **Loading** imports the class from the stored `architecture.py`, builds it with `init_args`, loads the
  weights, and returns it in eval mode. Nothing of the original project is needed on that machine, and
  it never mixes up with a module of the same name in your own process.
- **You pass `init_args`** because PyTorch cannot tell how a model was constructed. If you forget them,
  or they do not match the weights, saving stops and tells you, because the bundle is test-built
  (without allocating the weights) before it is stored.
- **Constructors that read files** (a Hugging Face config folder, a vocabulary, a JSON of settings):
  the path you pass in `init_args` is a path on *your* machine, so loading elsewhere would fail. Add
  the files with `assets=[...]` and they are copied into the bundle, and the matching `init_args` value
  is stored as `$BUNDLE/assets/<name>`, which becomes the real path at load time:

```python
at.log_model(net, flavor="bundle",
        init_args={"backbone_path": "/data/models/dinov3-huge", "feature_dim": 1280},
        assets={"/data/models/dinov3-huge": ["*.json"]})   # only the files the constructor reads
```

  `assets=[path, ...]` copies each path whole. The dict form `{folder: [patterns]}` copies only the
  files that match, which is what you want for a model folder that has multi-GB weights next to its
  `config.json`: the bundle already holds the trained weights, the constructor only needs the config.

  Keep assets small: they are stored with every version and hashed on every load.
- **Packages the class needs.** They are pinned in `requirements.txt` if they are installed
  packages. A package that is only a folder in your project (not installed) is not a requirement: either
  install it on the loading machine, or ship it inside the bundle with `code=["architecture.py", "mypackage"]`.
- **Size and loading time.** The bundle holds *all* the weights, frozen backbone included, in every
  version. Loading builds the class first (so a model whose constructor builds a large network from a
  config allocates it, then the weights are loaded over it). `bundle.build(..., check=False)` skips the
  test build at save time, which can save time for very large models.
- **The class can be in its own file (best) or in your training script.** If it is defined in the
  script you run, authtracker copies the class out into `architecture.py` together with what it needs:
  the imports it uses, helper functions and classes from the script, and plain constants. The training
  code is left out. It works in scripts and notebooks. If the class uses a global that is not a plain
  number, text, list or dict (a `torch.device`, a tensor), saving stops and names it. The most robust
  setup is still to keep the class in its own file and import it; `code="architecture.py"` names the
  file explicitly.
- **Fine-tuning works.** A loaded model remembers its code and arguments, so
  `net = at.load(...)`, train some more, `at.log_model(net, flavor="bundle")` makes the next version with
  nothing to pass.
- **Tampering is caught.** Loading checks every file against the hashes in `bundle.json`
  (`authtracker.load_bundle(folder, verify=False)` skips it).
- **Loading a bundle runs its `architecture.py`**, so only load bundles you trust. The weights alone
  are safe.
- **No folder, no server:** `at.load_bundle("path/to/folder")` loads a downloaded bundle directly.
- On MLflow the folder is kept exactly as it is inside an MLflow "logged model", so it can be
  registered and promoted like any other model. Weights & Biases does not support bundles yet.
- The name `"bundle"` is one constant, `FLAVOR` in `authtracker/bundle.py`.

### The exact object back, like fastai's `export.pkl` (opt-in)

The folder rebuilds the model from your code and the weights. That is portable and safe, but it is a *new*
object: anything you set on it afterwards is gone, and a model you changed after creating it needs a
builder function. If you want the object exactly as it was, there are two more options:

```python
net.class_names = ["real", "fake"]                       # set after creating it
net.threshold = 0.3

# 1. the folder (as before), plus model.pkl
at.log_model(net, flavor="bundle", init_args={"hidden": 16}, pickle=True)
# 2. only model.pkl, the requirements and the environment
at.log_model(net, flavor="pickle")

net = at.load("detector", "production")                  # rebuilt: architecture + weights (1 and 2: see below)
net = at.load("detector", "production", pickle=True)     # the exact object: class_names and threshold are back
```

| | Folder (default) | Folder + `pickle=True` | `flavor="pickle"` |
|---|---|---|---|
| Rebuilds a fresh model from code + weights | yes | yes | **no** |
| The exact object back (attributes, changes after creation) | no | with `load(..., pickle=True)` | with `load(..., pickle=True)` |
| Readable `architecture.py` and `weights.pth` for later work | yes | yes | **no** |
| Loads on another Python or torch version | yes | the rebuild yes, the pickle no | **no** |
| Storage | weights once | weights **twice** | weights once |
| Loading runs code from the file | the architecture file | the architecture file, and the pickle if you ask | the pickle |

- **Opt-in on both ends.** A pickle runs code when it is loaded, so `at.load(...)` refuses a `flavor="pickle"`
  model unless you pass `pickle=True`, and says why. The hashes in `bundle.json` are checked **before**
  anything is unpickled, so a swapped file is refused without running.
- **Your own classes go inside it.** With `flavor="pickle"`, classes and functions from your own project are
  stored in the pickle itself (cloudpickle, by value), so it loads on a machine that has none of your files.
  Library classes (torch, torchvision) are stored by name and must be installed there. With
  `pickle=True` on the folder, your classes are stored by name and found in the folder's `architecture.py`.
- **It is tested before it is stored.** Saving loads the pickle in a clean process that has none of your files,
  the way another machine would, and stops with the reason if that fails (`check=False` skips it; it loads the
  whole model once more, which takes time for very large ones).
- **Versions.** The Python and torch versions are recorded, and loading a pickle on different ones logs a warning.
- **`extras=` is the safe small version** of this: plain attributes (numbers, text, lists, dicts) are stored as
  JSON in `bundle.json` and put back on the rebuilt model, with no pickle:

```python
at.log_model(net, flavor="bundle", init_args={"hidden": 16}, extras={"class_names": ["real", "fake"], "threshold": 0.3})
net = at.load("detector", "production")                  # net.class_names, net.threshold are back
```

  They follow the model through fine-tuning: load, change `net.threshold`, save again, and the new value is kept.

### Alerts

Get told when a run fails, without watching it. Set one environment variable to a webhook URL (any URL that accepts JSON,
such as a chat tool's incoming webhook) and choose which events to send:

```bash
export AUTHTRACKER_ALERT_WEBHOOK='https://...'          # a secret: keep it out of git
export AUTHTRACKER_ALERT_ON=failed,finished             # failed, stopped, finished, custom (default: failed,stopped,custom)
authtracker test-alert                                  # sends one test message, so you can see it arrives
```

```python
at.alert("Loss is NaN", f"epoch {epoch}", level="error")     # your own messages (the "custom" event)
```

The URL is optional: with none set, nothing is sent and nothing is logged. It is masked in `authtracker config show`,
and a broken webhook never stops training. Details are in [Configuration](CONFIGURATION.md#alerts-optional).

## Move a model to another server

Changing MLflow servers, or handing a project to another team, should not mean losing the history. A model
can be exported to a folder and imported into any other MLflow store:

```python
at.export_model("churn", "churn-export")        # reads from your current store
# ... point authtracker at the new store (AUTHTRACKER_TRACKING_URI, or at.configure(...)) ...
at.import_model("churn-export")
```

or on the command line (`--tracking-uri` picks the store for that one command):

```bash
authtracker export churn --out churn-export --tracking-uri http://old-server:5000
authtracker import churn-export --tracking-uri http://new-server:5000
```

To move **everything** at once, add `--all` (or use `at.export_all` / `at.import_all`):

```bash
authtracker export --all --out everything --tracking-uri http://old-server:5000
authtracker import --all everything --tracking-uri http://new-server:5000
```

Every model that has runs gets its own sub-folder. If one model cannot be imported (for example it already exists
on the new server), it is reported as failed, the others still go in, and the command exits with 1.

What moves: all runs of the model at the three levels, parameters, tags, every metric point (with its step and
time), the files on each run, the model files, and the registered versions with their tags and aliases
(`production`, `staging`, `previous-production`). What changes: run ids (every imported run keeps its old id in the
`authtracker.imported_run_id` tag), and nothing else you can see. After the import `best_run`, `load`, `promote`
and `rollback` work as before, and the next `register` makes the next version.

Safety: the import refuses to add to a model that already has runs or versions, so nothing is duplicated. Import
under another name with `name=` / `--name` if you want a copy next to the original. The folder is plain files
(`manifest.json`, `runs/`, `models/`): copy it, zip it, or keep it as a backup. See
[`examples/09_export_import.py`](../examples/09_export_import.py).

Only MLflow supports this for now. It was built after trying the `mlflow-export-import` package, which stops on
MLflow 3 model versions (`models:/m-...`), needs an HTTP server and cannot read a SQLite or file store.

## Where the code came from

Every experiment run stores the code that produced it, by default:

| Stored | Where | What it is |
|---|---|---|
| `git_sha`, `git_branch`, `git_remote`, `git_url`, `git_dirty` | tags on every run, and `bundle.json` | The commit, the branch, where to fetch it and a link to the commit on GitHub, GitLab or Bitbucket. A token or password in a remote URL is removed. |
| `code_changes.patch` | the experiment run's files | The **uncommitted changes** since that commit, as a patch, including files that are new to git. |
| `script_train.py` (named after your script) | the experiment run's files | The **script you ran**, even outside a git repository. `git_script` says where it is in the repo. |
| `code_changes`, `code_script` | tags | What was stored, or why it was not: `saved`, `none`, `not a git repository`, `skipped: ...`. |

To reproduce a run exactly:

```bash
git clone <git_remote> && cd <repo>
git checkout <git_sha>
git apply code_changes.patch        # only if the run has one
```

- **Secrets are never copied knowingly.** A patch or script that looks like it contains a key, token, password
  or private key is **not stored**, and you are told on screen and in the `code_changes` / `code_script` tags
  (the secret itself is never repeated). The check is best effort, not a guarantee: anyone who can open the run
  can read the code, so do not rely on it to catch everything.
- **Size limits.** A patch or script larger than 1 MB is skipped, and so is a new file over 200 KB or a binary
  one. The run's tags say how many were skipped.
- **Not a script.** In a notebook there is no script file, so no script is stored (the patch still is).
  A library's entry point, such as `pytest`, is never mistaken for your script.
- **Switch it off for a run** with `at.init(..., save_code=False)`. `AUTHTRACKER_SAVE_CODE=false` sets the
  default for a whole machine or job; the argument in `init` wins.
- **Commit before runs that matter.** Then `git_dirty` is `false`, nothing is unsaved, and the commit alone is
  the whole story.

### Container image

If a run happens in a container, the image is recorded, **pinned to its digest** when that can be found out, because
a tag such as `app:latest` can point to a different image next week. A container cannot normally see its own image
name, so your build or deploy step tells authtracker:

```python
at.init("detector", "baseline", image="registry.example.com/team/app@sha256:9f2c...")   # pinned: a digest
at.init("detector", "baseline", image="registry.example.com/team/app:v3")               # a tag: recorded, with a warning
```

| Source, in order | Notes |
|---|---|
| `image=` in `init` (or `AUTHTRACKER_IMAGE`) | You pass it. With `@sha256:...` it is pinned. |
| AWS ECS / Fargate | The platform tells the container its image **and digest**, so nothing to pass. |
| GitLab CI (`CI_JOB_IMAGE`) | Gives the image name, usually only a tag. |

- A tag is **resolved to a digest** when the `docker` command is available and knows the image (typically on the
  host, not inside the container). Otherwise the run is marked `container_pinned=false` and you get one warning.
- A container with no image given and none found logs a warning that says how to record it. Outside a
  container, nothing is recorded and nothing is said.
- Recorded as tags on every run (`container_image`, `container_digest`, `container_pinned`, `container_runtime`,
  and `container_image_pinned`, the exact reference to run again) and under `container` in `bundle.json`.
- Not covered: the Dockerfile, the base image's own layers, or Kubernetes pod specs. Kubernetes does not tell a
  container its own image, so pass it in, for example from your deployment's variables.

**Which PyTorch models work?** The bundle rebuilds a model by calling your class (or a function you
give) with `init_args`, then loading the weights. So:

| Your model | Works? |
|---|---|
| Your own `nn.Module` class, in a file, in your training script, or in a notebook cell | Yes (notebooks tested in a real Jupyter kernel) |
| Several of your classes nested in each other, buffers, frozen parameters | Yes |
| Trained on `cuda` / `mps`, loaded on CPU or GPU | Yes (`device=` at load) |
| Wrapped in `nn.DataParallel`, `DistributedDataParallel` or `torch.compile` | Yes: the wrapper is removed and the model inside is saved |
| `nn.Sequential(...)`, a torchvision / timm / Hugging Face model, or a model you change after creating it (LoRA, freezing, pruning) | **Yes, with `factory=`**: a function that builds it (below). Tested with torchvision ResNet, MobileNet and EfficientNet; timm and Hugging Face are not tested |
| A model whose constructor reads files (config, vocabulary) | Yes, with `assets=` |
| Constructor arguments that are numbers, text, lists, dicts | Yes. Objects, tensors, functions or config objects are not: pass a plain dict, or use the pickle (below) |
| Extra attributes you set after creating it (`net.class_names = [...]`, `net.threshold = 0.3`) | **Yes**: `extras={...}` for plain data, or the pickle for anything |
| Two models in one save (generator and discriminator) | Not as two: put them in one `nn.Module` that holds both |
| Lazy modules (`nn.LazyLinear`), quantized models, models that need a forward pass before loading | Not tested |
| Very large models (tens of GB, sharded checkpoints) | Not designed for it: one file, no sharding |

Whenever saving cannot rebuild the model, it stops *at save time* with the reason (the model is test-built,
without allocating its weights, and its weights are checked against it), not months later at load time.

A **builder function** covers every model that has no class of yours:

```python
def build_model(n_classes=2):                           # in your own file (or your training script)
    net = torchvision.models.resnet18(weights=None, num_classes=n_classes)
    net.fc = nn.Sequential(nn.Dropout(0.2), net.fc)     # any change you make after creating it
    return net

at.log_model(net, flavor="bundle", factory=build_model, init_args={"n_classes": 2})
```

Use `weights=None` / `pretrained=False` inside it: the trained weights come from the bundle, so downloading
pretrained ones at load time would be wasted (and need the network).

**Bundle or traced model?** The next section is the other way to save PyTorch models. Use the bundle
when you want readable files, your own class back, fine-tuning and the least moving parts. Use the traced
model when you only need predictions and want a format with no code in it.

## Saving PyTorch models as a traced graph (MLflow's default)

```python
at.log_model(net, input_example=batch.cpu().numpy())      # any nn.Module, including your own classes
```

How MLflow saves it, and what that means for you:

- **A traced graph by default** (`torch.export`, the `pt2` format). It contains no pickle, so loading
  it cannot run code, and it **needs no architecture file** on the machine that loads it. That is why
  an example input is required: MLflow runs the model once on it to trace it.
- **Training on a GPU is fine.** If the model is on `cuda` or `mps`, authtracker saves its CPU
  weights and puts the model back where it was, in place, so your training loop and optimizer carry on.
- **Predicting.** `at.load("name", "production")` returns a model whose `predict()` takes and returns
  NumPy arrays, so a classifier gives you the raw outputs (apply `argmax` yourself).
- **Getting the PyTorch module back**, to fine-tune or inspect it:
  `mlflow.pytorch.load_model("models:/name@production")`. A traced model comes back as a trainable
  `GraphModule`, without your original class or its extra methods.
- **A model that cannot be traced** (an `if` that depends on tensor values, an unsupported operation)
  stops with a message that tells you what to do. Save it as a pickle instead, and ship the file that
  defines your class, otherwise loading elsewhere fails with `No module named 'net'`:

```python
at.log_model(net, input_example=batch.cpu().numpy(),
        serialization_format="pickle", code_paths=["net.py"])
```

  A pickled model can run code when loaded, so only load models you trust.
- **Checkpoints to resume training** are just files: `torch.save(state, "ckpt.pt")` then
  `at.save("ckpt.pt")`. They are stored on the run, but they are not registered models.

Tested on CPU and on an Apple `mps` GPU. Not tested: NVIDIA `cuda`, several GPUs, `torch.compile`,
very large models.

## Version a model

```python
best = at.best_run("model1", "demo", metric="accuracy")   # best run of the experiment
version = at.register(best)                               # model version 1, 2, 3, ...

at.promote("model1", version, "staging")
at.promote("model1", version, "production")               # checked by AUTHTRACKER_PROMOTE_* rules
at.rollback("model1")                                     # production goes back one version

model = at.load("model1", "production")                   # load by alias
```

A version is an integer that MLflow assigns. **Promote** moves an alias, it does not create a
version. (The "Promote model" button in the MLflow UI is different: it copies a version into
another registered model, which creates a new version.)

Promotion rules, if you set them: the metric `AUTHTRACKER_PROMOTE_METRIC` must reach
`AUTHTRACKER_PROMOTE_MIN` and, with `AUTHTRACKER_PROMOTE_MUST_BEAT_PROD=true`, beat the version that
holds the alias now. Higher is better. The rules apply to `AUTHTRACKER_GATED_ALIASES` (default
`production`). Someone using the MLflow UI or plain MLflow can still set an alias without them.

## Ship the architecture and an inference script with a model

A developer who downloads a model needs the network class, the preprocessing and a way to run one
input. Put them inside the model with `code_paths`, and wrap the model as a `pyfunc` so there is a
single `predict()` call. The folder layout below is a sketch of the pattern, not a runnable example. For plain
PyTorch, the bundle in [`examples/07_pytorch_bundle.py`](../examples/07_pytorch_bundle.py) is tested and does this for you.

```
fake_image_detector/
├── arch/
│   ├── network.py        # class FakeImageNet(nn.Module)
│   ├── preprocess.py     # image -> tensor
│   └── pyfunc_model.py   # class FakeImageModel(mlflow.pyfunc.PythonModel)
├── infer_single.py       # infer_single(net, "photo.jpg") -> {"label", "confidence", ...}
├── README_model.md       # what the model does, inputs/outputs, how to use it
└── requirements.txt
```

```python
from arch.pyfunc_model import FakeImageModel

at.save(["infer_single.py", "README_model.md", "requirements.txt"])      # visible in the UI
at.log_model(
    FakeImageModel(), flavor="pyfunc",                       # the wrapper, not the raw network
    artifacts={"weights": "weights.pt"},                     # the saved state_dict
    code_paths=["arch", "infer_single.py"],                  # copied into the model folder
    pip_requirements=["torch==2.9.1", "pillow==12.0.0"],     # pin what the model needs
)
```

What the developer does, with no architecture knowledge:

```python
import mlflow
model = mlflow.pyfunc.load_model("models:/fake-image@production")
model.predict({"image_path": ["photo.jpg"]})
# [{'label': 'fake', 'confidence': 0.93, 'probabilities': {'real': 0.07, 'fake': 0.93}, ...}]
```

Without MLflow, download the model folder (the UI **Artifacts** tab, or
`mlflow.artifacts.download_artifacts(...)`) and run:

```bash
pip install -r requirements.txt
python <model>/code/infer_single.py <model>/artifacts/weights.pt photo.jpg
```

Keep `arch/` small and self-contained, pin versions in `pip_requirements`, and test by loading the
model from a different folder (ideally a fresh virtual environment) before sharing it.

## Configure (all optional)

Everything is an environment variable, so the same code runs anywhere. Every variable, its default and
when to use it is in [Configuration](CONFIGURATION.md).

A bad value (`AUTHTRACKER_CAPTURE_ENV=maybe`) stops with a clear error, and an unknown
`AUTHTRACKER_*` name gives a warning, which catches typos.

```bash
authtracker config show     # the resolved settings (credentials masked)
authtracker doctor          # checks the backend is reachable and shows queued operations
authtracker sync            # uploads operations queued while the server was down
```

## System metrics (CPU, memory, GPU)

On by default for MLflow, so every run shows how busy the machine was. `psutil` comes with
authtracker. NVIDIA GPU charts come with it too, through `nvidia-ml-py` (installed everywhere except on a Mac; with no
NVIDIA driver it does nothing). The old `authtracker[gpu]` extra still works.

```bash
export AUTHTRACKER_SYSTEM_METRICS=false                # switch them off for this machine
export MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL=10      # seconds between samples (MLflow's default)
```

For one run, pass it to init (it wins over the variable): `at.init("fake-image", "baseline", system_metrics=False)`.

They appear as `system/...` metrics, in a **System metrics** tab on each run page. A run shorter than
the sampling interval has no points, so very short test runs look empty. The experiment run and each
run are measured separately. If `psutil` is missing, authtracker logs a warning and carries on
without them: the training script is never stopped for it. Weights & Biases records system metrics on
its own. A Mac has no NVIDIA GPU, so there you get CPU, memory, disk and network only.

## Training on several GPUs

- **One process, several GPUs (`nn.DataParallel`)**: use authtracker as you would with one GPU. When you save
  a model, the `DataParallel` wrapper is removed and the model inside is saved. Not tested on real GPUs yet.
- **One process per GPU (`torchrun`, `DistributedDataParallel`, several nodes)**: authtracker has no idea of
  ranks. If every process calls it, you get one run per process, with duplicated metrics and uploads. Call it
  from global rank 0 only:

  ```python
  rank = int(os.environ.get("RANK", 0))
  if rank == 0:
      at.init("fake-image", "baseline", resume=True)
      at.start("lr0.01")
  ...
  if rank == 0:
      at.log({"loss": loss}, step=epoch)       # average the loss across ranks first, on every rank
  ```

  Other things to plan for: save checkpoints from rank 0 only (`model.module.state_dict()`), put them where the
  other nodes can read them if you resume, and note that console output and system metrics come from rank 0's
  machine only. This pattern is general practice and has not been tried with authtracker on a real multi-GPU job.

## Outages

A tracking outage never stops training. After the retries are used up, writes (start and end of a
run, tags, params, metrics, files) are saved in the queue directory and uploaded in order when the
server answers again, and at the end of the script. Anything left over, for example after a crash,
is uploaded with `authtracker sync`. While offline, runs get temporary `offline-…` ids that are
mapped to the real ones on upload.

Saving a model, `register`, `promote`, `load` and `best_run` need an answer from the server, so they
raise `BackendError` after the retries instead of queuing. Keep your own checkpoint file too.

## Advanced: the `with` API

The simple calls above are built on explicit run objects, which you can use directly, for example
in helper functions that receive a run, or to be sure each run is closed by a block:

```python
with at.run(model="model1", experiment="churn-v2", dataset_version="2026-10") as r:
    r.log_params({"base": "resnet18"})                          # shared by all trials
    for lr in (1e-3, 1e-4):
        with r.child(f"lr{lr}", params={"lr": lr}) as c:        # one child run per trial
            c.log_metrics({"val_acc": 0.9}, step=0)
            c.log_eval(f1=0.91)                                 # final metrics
            c.log_text("epoch 1 ok\n", "train.log")
            c.log_artifact("confusion_matrix.png")
            c.log_artifacts("reports/")                         # bundle: a folder or a list
            c.log_model(model)                                  # inside this run's `with` block
```

## Adding a backend

See the [developer guide](DEVELOPMENT.md#adding-a-backend).

## Tests

See [Testing](../tests/TESTING.md) and the [developer guide](DEVELOPMENT.md).
