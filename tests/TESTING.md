# Testing status

What has been tested, and what is left for later. Update this file when you test something new.

## Automated tests (no server or account needed)

```
pip install -e ".[dev]"
pip install torch          # optional: without it the PyTorch tests are skipped
pytest
```

262 tests pass and 3 are skipped (a backend/parameter combination that does not apply). The run takes
about 7 minutes, mostly MLflow start-up.

| Area | Where |
|---|---|
| The simple API, the `with` API, rules (models, tags, metric names), failures | `test_simple.py`, `test_authtracker.py` |
| Resume: steps, parameters, kept files, checkpoints | `test_resume.py` |
| Training YAML and the dataset hash | `test_config_dataset.py` |
| Git info, the saved patch and script, secret handling | `test_git_info.py`, `test_code_snapshot.py` |
| Container image detection | `test_container.py` |
| Outages: retry, offline queue, `authtracker sync` | `test_offline.py` |
| PyTorch bundles and pickles | `test_bundle.py`, `test_pickle.py` |
| Model flavor detection, `watch`, system metrics, log capture | `test_flavor_detection.py`, `test_watch.py`, `test_system_metrics.py`, `test_capture.py`, `test_periodic_upload.py` |
| W&B (the real SDK, offline mode) | `test_wandb.py` |
| The backend contract every backend must pass | `test_contract.py` (`authtracker.testing.check_backend`) |
| Packaging: extras, the core never imports MLflow | `test_packaging.py` |
| **Every example, run as a real script** | `test_examples.py` |

Backend behaviour is tested on `local`, `mlflow` (SQLite) and W&B offline with the same tests.

### The examples are tests

`test_examples.py` runs each script in `examples/` in a temporary folder, then opens the example's MLflow
database and checks what was recorded: runs and their status, metrics and steps, files, model versions and
aliases, and for the crash example a `FAILED` run with its traceback. A guard test fails if an example has no test.

## Tested on real hardware

All on one Mac (CPU and an Apple `mps` GPU), MLflow with a local SQLite store, W&B in offline mode.

| Area | Result |
|---|---|
| Tracking, files, models, registry, resume, bundles on MLflow | Works |
| Tracking, files and model artifacts on W&B (offline) | Works |
| PyTorch bundle: build, load, fine-tune, tamper check, `mps` | Works |
| Code lineage: dirty and clean runs, rebuilding a run's code | Works |
| Release workflow: build and `twine check --strict` | Works locally; the full run needs the `PYPI_API_TOKEN` secret |

## Not tested yet

- A remote or login-protected MLflow server (S3 or other remote artifact storage).
- W&B online mode, and reading runs back from the W&B server (only a stand-in for its API is tested).
- NVIDIA `cuda`, `DataParallel`, DDP or several nodes.
- Windows and Linux (the tests were run on macOS; the release workflow runs them on Linux).
- How the MLflow UI shows repeated steps.
