import pytest


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    """MLflow writes artifacts to ./mlruns by default; keep that out of the project folder."""
    monkeypatch.chdir(tmp_path)
