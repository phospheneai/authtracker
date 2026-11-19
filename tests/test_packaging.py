import importlib.util
from importlib import metadata

import pytest

import authtracker as at
from authtracker.backends import get_backend
from authtracker.settings import Settings


def _requirements():
    reqs = metadata.requires("authtracker") or []
    base = [r for r in reqs if "extra ==" not in r]
    extras = {}
    for r in reqs:
        if "extra ==" in r:
            name = r.split("extra ==")[1].strip(" \"'")
            extras.setdefault(name, []).append(r.split(";")[0].strip())
    return base, extras


def _named(requirements, name):
    """Requirements for exactly the package `name` (so "mlflow" does not match "mlflow-skinny"), however the
    specifier is ordered: packaging writes `mlflow<4,>=3.0`."""
    import re
    return [r for r in requirements if re.split(r"[<>=!~ ;\[(]", r, maxsplit=1)[0].lower() == name]


def test_the_tracking_platform_is_chosen_with_an_extra():
    base, extras = _requirements()
    assert not any(r.lower().startswith(("mlflow", "wandb")) for r in base)     # no platform forced on anyone
    assert any(">=3" in r for r in _named(extras["mlflow"], "mlflow"))                  # full MLflow
    assert any(">=3" in r for r in _named(extras["mlflow-skinny"], "mlflow-skinny"))    # the light client
    assert not _named(extras["mlflow-skinny"], "mlflow")                                # ...and never the full one
    assert any(r.startswith("wandb") for r in extras["wandb"])
    assert any(r.startswith("psutil") for r in extras["mlflow"] + extras["mlflow-skinny"])


def test_database_uri_needs_the_full_mlflow(monkeypatch):
    pytest.importorskip("mlflow")
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "sqlalchemy" else real(name, *a, **k))
    for uri in ("sqlite:///x.db", "postgresql://u@h/db", "mysql+pymysql://u@h/db"):
        with pytest.raises(at.BackendError, match="full MLflow"):
            get_backend(Settings(backend="mlflow", tracking_uri=uri, offline_queue=False))
    get_backend(Settings(backend="mlflow", tracking_uri="http://127.0.0.1:1", offline_queue=False))  # a server URL is fine


def test_missing_mlflow_names_both_options(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "mlflow", None)
    monkeypatch.delitem(sys.modules, "authtracker.backends.mlflow_backend", raising=False)
    with pytest.raises(at.BackendError) as e:
        get_backend(Settings(backend="mlflow", offline_queue=False))
    assert "authtracker[mlflow]" in str(e.value) and "authtracker[mlflow-skinny]" in str(e.value)
