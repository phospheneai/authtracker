import pytest


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    """MLflow writes artifacts to ./mlruns by default; keep that out of the project folder."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True, scope="session")
def _skip_mlflow_requirements_inference():
    """MLflow works out a model's pip requirements by loading it in a subprocess: 7 to 11 seconds on EVERY
    `save(model)`, and most tests save one. No test needs the inferred list, so use MLflow's own defaults."""
    try:
        import mlflow.models
    except ImportError:                       # MLflow not installed: nothing to speed up
        yield
        return
    patch = pytest.MonkeyPatch()
    patch.setattr(mlflow.models, "infer_pip_requirements",
                  lambda model_uri, flavor, fallback=None, **kwargs: list(fallback or []))
    yield
    patch.undo()
