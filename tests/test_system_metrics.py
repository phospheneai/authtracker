import importlib.util
import logging
import time

import pytest

pytest.importorskip("mlflow")
pytest.importorskip("psutil")

import authtracker as at  # noqa: E402
from authtracker.backends.mlflow_backend import MlflowBackend  # noqa: E402
from authtracker.settings import Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", "1")
    monkeypatch.delenv("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", raising=False)
    MlflowBackend._warned_psutil = False
    at.finish()
    yield Settings(backend="mlflow", tracking_uri=f"sqlite:///{tmp_path}/m.db", owner="t",
                   capture_env=False, queue_dir=str(tmp_path / "q"), retry_wait=0.0)
    at.finish()


def _runs(settings):
    from mlflow.tracking import MlflowClient
    c = MlflowClient(settings.tracking_uri)
    e = c.get_experiment_by_name("m1")
    return c.search_runs([e.experiment_id])


def _short_run(settings):
    at.init("m1", "e", settings=settings)
    at.start("a")
    at.log({"loss": 1.0})
    time.sleep(3.5)
    at.finish()


def test_can_be_switched_off(settings):
    import dataclasses
    _short_run(dataclasses.replace(settings, system_metrics=False))
    assert not [k for r in _runs(settings) for k in r.data.metrics if k.startswith("system/")]


def test_init_argument_switches_it_off_even_if_mlflow_s_variable_is_on(settings, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "true")
    at.init("m1", "e", system_metrics=False, settings=settings)
    at.start("a")
    time.sleep(1.2)
    at.log({"loss": 1.0})
    at.finish()
    assert not [k for r in _runs(settings) for k in r.data.metrics if k.startswith("system/")]


def test_on_by_default(settings):
    assert Settings().system_metrics is True and Settings.from_env({}).system_metrics is True
    _short_run(settings)                              # no setting given
    child = [r for r in _runs(settings) if r.data.tags["mlflow.runName"] == "a"][0]
    names = [k for k in child.data.metrics if k.startswith("system/")]
    assert any("cpu" in n for n in names) and any("memory" in n for n in names), names


def test_setting_from_environment(monkeypatch):
    monkeypatch.setenv("AUTHTRACKER_SYSTEM_METRICS", "false")
    assert Settings.from_env().system_metrics is False


def test_missing_psutil_warns_and_does_not_break_the_run(settings, monkeypatch):
    import dataclasses
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "psutil" else real(name, *a, **k))
    monkeypatch.setenv("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "true")   # MLflow's own switch
    s = dataclasses.replace(settings, system_metrics=True)
    messages = []

    class _Collect(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    collector = _Collect(level=logging.WARNING)
    logging.getLogger("authtracker").addHandler(collector)
    try:
        at.init("m1", "e", settings=s)
        at.start("a")
        at.log({"loss": 1.0})
        at.finish()
    finally:
        logging.getLogger("authtracker").removeHandler(collector)
    assert any("psutil" in m for m in messages), messages
    runs = _runs(settings)
    assert len(runs) == 2 and {r.info.status for r in runs} == {"FINISHED"}   # no orphan "running" run
