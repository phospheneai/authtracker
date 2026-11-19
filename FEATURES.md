# Features

What authtracker does, grouped as major, minor and patch. See the README for usage.

## Major

Tracking
- Three levels: model (an MLflow experiment), experiment (parent run), run (child run).
- `init`, `start`, `log`, `save`, `finish`. No `with` blocks or run objects needed.
- Metrics with an automatic step, or your own `step=`.
- Params, tags, text, files and folders, all through `at.save`.
- Runs close when the script ends. If it crashes, the run is marked FAILED and the traceback is saved.
- A `with` API (`authtracker.run`) for explicit scopes.

Backends
- MLflow is the default. W&B is optional (`pip install authtracker[wandb]`). A local JSON backend works with no server.
- All backends sit behind one small `Backend` class. A feature a backend lacks raises `UnsupportedError`.
- Add your own backend through the `authtracker.backends` entry point.

Reproducibility
- Git sha, branch, remote (token removed), commit URL and a dirty flag.
- Uncommitted changes, including new files, saved as `code_changes.patch`.
- Your training script saved as `script_<name>`.
- Dataset hash for a file, folder or manifest, plus name and version.
- Training YAML saved with the run and flattened into params.
- pip requirements and environment info.
- Container image, ideally pinned by digest.

Models
- MLflow flavors, with the flavor detected for you.
- PyTorch bundle: `architecture.py`, `weights.pth`, `inference.py`, `requirements.txt`, `environment.txt` and `bundle.json`. Loads as plain PyTorch, with the weights loaded using `weights_only=True`.
- Builders (`factory=`), extra files (`assets=`) and extra metadata (`extras=`).
- A loaded model keeps its code, so you can fine-tune it and save it again.

Registry
- `register`, `promote`, `rollback`, `load` and `best_run`, using aliases.
- Promotion gates: a minimum metric, beating the current production model, and which aliases are gated.

Resume
- `at.init(..., resume=True)` continues the last matching run, or starts a new one if there is none.
- `at.start(name, resume=...)` does the same for a run inside the experiment.
- Steps carry on from the last logged step.
- `at.restore(name)` downloads a file the run saved earlier (a checkpoint, say) and returns its path, or `None`.
- Params can be repeated or added on resume, but not changed.
- Files from earlier parts are kept (`.resume1`, `.resume2`).

Outages
- Failed calls are retried.
- If the server stays down, operations are queued on disk. `authtracker sync` uploads them later.
- A down server never stops training.

## Minor

- `at.watch(model, every=100)` logs the gradient norm of each layer and the total (`grad_norm/<layer>`, `grad_norm/total`). Numbers only, no histograms. PyTorch 2.1+.
- Console output (stdout, stderr, logging) saved with the run, uploaded every 60 seconds, capped at 20 MB.
- System metrics (CPU, memory, GPU) on by default. Switch them off for one run with `at.init(..., system_metrics=False)`, or for a machine with `AUTHTRACKER_SYSTEM_METRICS=false`.
- A model wrapped in `DataParallel`, `DistributedDataParallel` or `torch.compile` is saved without the wrapper.
- Pickle flavor, opt-in on save and on load: your own code is bundled, hashes are checked before unpickling, and the save checks it loads in a clean process.
- Bundling a class defined in a script or notebook.
- A hint when MLflow's traced-graph save fails because the model is on GPU.
- Settings from `AUTHTRACKER_*` environment variables. Arguments to `init()` and `save()` always win.
- Rules for required tags, allowed models and allowed metric names.
- CLI: `authtracker config`, `doctor` and `sync`.
- Files that look like secrets are left out of the saved patch and script. `save_code=False` turns code saving off.
- `authtracker.testing.check_backend` to test a custom backend.

## Patch

- Remote URLs are cleaned of tokens before they are stored.
- An image tag without a digest gives a warning.
- Image is read from the argument, ECS metadata, GitLab's `CI_JOB_IMAGE` or docker.
- Errors are specific (`ConfigError`, `ValidationError`, `PromotionError`, `BackendError`, `UnsupportedError`), all under `AuthTrackerError`.
- A backend that can't resume stops with a message instead of starting fresh.
- If the server can't be asked about an earlier run, a new run starts with a warning.
- W&B offline mode works.
- Needs Python 3.10+. Core dependencies are `pyyaml` and `cloudpickle`.

## Limits

- No rank handling for distributed training. With `torchrun` or several nodes, call authtracker from rank 0 only (see "Training on several GPUs" in the README).

## Not tested yet

- Any multi-GPU run (`DataParallel`, DDP or several nodes).

- A real EC2, S3 or login-protected server.
- CUDA, Windows and Linux.
- W&B online mode and registry.
- How the MLflow UI shows repeated steps.

The package is also not published, has no license and has no CI.
