# Examples

Eight small, runnable scripts. Each one **checks its own claims** (it stops with an error if something
does not behave as described), and each one is run by `tests/test_examples.py`, which also opens the
resulting MLflow database to check what was recorded. So an example that stops working fails the tests.

| Example | What it shows | Needs |
|---|---|---|
| [`01_quickstart.py`](01_quickstart.py) | The shortest useful script: three runs, a text file and a model per run, `best_run`, `register`, `promote`, `load`. | scikit-learn |
| [`02_config_and_dataset.py`](02_config_and_dataset.py) (+ [`train.yaml`](train.yaml)) | Settings in a YAML (saved with the run, logged as parameters), a dataset record with a hash that changes when the data changes, arguments that override the YAML. | scikit-learn |
| [`03_registry_and_promotion.py`](03_registry_and_promotion.py) | Versions and aliases, a minimum-metric rule, "must beat production", a blocked promotion, rollback, loading by alias. | scikit-learn |
| [`04_resume_with_checkpoints.py`](04_resume_with_checkpoints.py) | Continue a stopped job: the same runs, a continuous curve, `at.save` / `at.restore` of a checkpoint, parameters that cannot change on resume. | none extra |
| [`05_rules_and_failures.py`](05_rules_and_failures.py) | Allowed models, required tags and metric names refuse bad runs. With `--crash`, a crash is saved as `FAILED` with its traceback. | none extra |
| [`06_code_lineage.py`](06_code_lineage.py) | A git project: commit, branch and remote (token removed) on every run, uncommitted changes as a patch, rebuilding a run's exact code. | git |
| [`07_pytorch_bundle.py`](07_pytorch_bundle.py) (+ [`pytorch_bundle/architecture.py`](pytorch_bundle/architecture.py)) | Plain PyTorch saved as a folder, versioned, loaded back as your own class, `extras`, `load_bundle`, fine-tuning into the next version. | torch |
| [`08_wandb_backend.py`](08_wandb_backend.py) | The same calls on Weights & Biases (offline, no account), and the clear error for the registry, which W&B does not support yet. | `authtracker[wandb]` |

## Run them

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # authtracker, MLflow, W&B, scikit-learn, pytest
pip install torch                  # only for example 07

python examples/01_quickstart.py
python examples/05_rules_and_failures.py --crash      # exits with an error on purpose
```

Run them from any folder. Each keeps its own database and artifacts in `examples/data/<example>/` and
starts from a clean state every time, so running one never affects another. To look at the results:

```bash
cd examples/data/01_quickstart
mlflow ui --backend-store-uri sqlite:///mlflow.db        # http://127.0.0.1:5000
```

Stop the UI with Ctrl+C (Ctrl+Z only suspends it and keeps the port busy).

To send an example's runs to a real MLflow server instead, set `AUTHTRACKER_TRACKING_URI` first. Their
final self-checks still apply, and the folder is not wiped, so use a server you do not mind adding runs to.

## Test them

```bash
pytest tests/test_examples.py          # about 3 minutes: every example runs as a real script
```

Examples whose libraries are missing are skipped, not failed (07 without torch, 08 without wandb, 06
without git). A guard test fails if you add a numbered example without a test for it.
