"""Shared set-up for the examples (not an example itself).

Every example keeps its own database and artifacts in its own folder under ``data/``, so running
one never touches another. Set ``AUTHTRACKER_EXAMPLES_DATA`` to keep that folder somewhere else
(the tests do), or ``AUTHTRACKER_TRACKING_URI`` to send the runs to an MLflow server instead.
"""
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def setup(name: str, fresh: bool = True, local_store: bool = True) -> Path:
    """Point authtracker at a local SQLite store for example `name` and return its folder.

    fresh=True wipes that folder first, so every run of an example starts from the same state.
    local_store=False leaves the tracking URI alone (for backends that are not MLflow).
    """
    folder = Path(os.environ.get("AUTHTRACKER_EXAMPLES_DATA") or HERE / "data").resolve() / name
    if fresh and folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if local_store and not os.environ.get("AUTHTRACKER_TRACKING_URI") and not os.environ.get("MLFLOW_TRACKING_URI"):
        os.chdir(folder)                      # MLflow's SQLite store writes artifacts to ./mlruns
        os.environ["AUTHTRACKER_TRACKING_URI"] = f"sqlite:///{folder / 'mlflow.db'}"
    os.environ.setdefault("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", "1")   # examples are short runs
    return folder


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def ok(message: str) -> None:
    print(f"  OK  {message}", flush=True)


def client():
    """An MLflow client on the store the example is using (for looking at what was recorded)."""
    from mlflow.tracking import MlflowClient
    import authtracker as at
    uri = at.get_settings().tracking_uri
    return MlflowClient(tracking_uri=uri, registry_uri=uri)
