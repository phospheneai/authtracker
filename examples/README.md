# Examples

Run them from anywhere. They always write to [`data/`](data) in this folder: the database
(`mlflow_demo.db`) and the artifacts (`mlruns/`).

| Example | What it shows | Needs |
|---|---|---|
| [`sample_authtracker.py`](sample_authtracker.py) | Two models, three runs each, with a bundle of files, a log and a model per run. | scikit-learn |
| [`full_feature_demo.py`](full_feature_demo.py) (with [`train.yaml`](train.yaml)) | **One run that uses every authtracker feature on MLflow**: a training YAML, a dataset record with a hash, rules, curves, system metrics, text logs, files, bundles, a model, best run, register, promotion rules, rollback, load, a crash. Runs with no setup (local file), or against a server. | scikit-learn |
| [`pytorch_bundle/`](pytorch_bundle) | **Plain PyTorch, saved as a folder**: `architecture.py`, weights, a sample `inference.py` and requirements. `at.load` gives the real class back. | torch |
| [`code_lineage_demo.py`](code_lineage_demo.py) | **Code lineage on a local MLflow server**: a git project with a remote, a run with uncommitted changes and a clean run, what each records, and rebuilding the first run's exact code. Needs git and a running server (`AUTHTRACKER_TRACKING_URI=...`). | none |
| [`fake_image_detector/`](fake_image_detector) | A small PyTorch real/fake image classifier saved with its architecture and an `infer_single.py`, so a developer can download it and call `predict()`. MLflow only. | torch, pillow |

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e "authtracker[mlflow]"
pip install torch pillow scikit-learn          # only for the examples

python authtracker/examples/sample_authtracker.py
python authtracker/examples/fake_image_detector/sample_dl.py

cd authtracker/examples/data
mlflow ui --backend-store-uri sqlite:///mlflow_demo.db      # open http://127.0.0.1:5000
```

Start the UI with the same database the scripts write to, otherwise it shows a different (empty)
store. Stop it with Ctrl+C: Ctrl+Z only suspends it and keeps the port busy.

To use Weights & Biases instead, set `AUTHTRACKER_BACKEND=wandb` for `sample_authtracker.py`
(see the main [README](../README.md#choose-a-backend)). The image example is MLflow only.

## In the UI

- Left list: one entry per **model** (`model1`, `model2`, `fake-image`).
- Open one: each **experiment** is a parent run, and its **runs** are expandable under it.
- A run's **Artifacts** tab shows the files, the log and the model.
- To make a version from a run: open the model in the run page and use **Register model**, or run
  `at.register(at.best_run("model1", "sample", metric="accuracy"))`.
- To set `production` on a version: **Models** → the model → the version → **Aliases**. Use aliases
  for this, not the **Promote model** button, which copies a version into another registered model.

Models saved as pickles can run code when loaded, so only load models you trust.

## Moving this folder

The database stores absolute artifact paths. If you move or rename `data/`, delete it and run the
examples again, or rewrite those paths in the database.
