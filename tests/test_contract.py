import subprocess
import sys
from typing import Any, Dict, List, Optional

import pytest

import authtracker as at
from authtracker.backends import Backend, available_backends, get_backend, register_backend
from authtracker.backends import base as base_module
from authtracker.settings import Settings
from authtracker.testing import check_backend


class MemoryBackend(Backend):
    """A complete third backend with ONLY the required methods (no models, no registry)."""

    def __init__(self):
        self.runs: Dict[str, dict] = {}
        self.files: Dict[str, List[str]] = {}

    def ping(self) -> None: ...

    def start_run(self, experiment, run_name, tags, parent_id):
        rid = f"r{len(self.runs)}"
        self.runs[rid] = {"run_id": rid, "name": run_name, "exp": experiment, "tags": dict(tags),
                          "params": {}, "metrics": {}, "status": "RUNNING"}
        self.files[rid] = []
        return rid

    def end_run(self, run_id, status):
        self.runs[run_id]["status"] = status

    def set_tags(self, run_id, tags):
        self.runs[run_id]["tags"].update({k: str(v) for k, v in tags.items()})

    def log_params(self, run_id, params):
        self.runs[run_id]["params"].update({k: str(v) for k, v in params.items()})

    def log_metrics(self, run_id, metrics, step):
        self.runs[run_id]["metrics"].update(metrics)

    def log_artifact(self, run_id, path):
        self.files[run_id].append(path.rsplit("/", 1)[-1])

    def get_run(self, run_id):
        return {k: v for k, v in self.runs[run_id].items() if k in
                ("run_id", "name", "metrics", "params", "tags")}

    def search_runs(self, experiment, tags, metric, mode, limit):
        hits = [self.get_run(r["run_id"]) for r in self.runs.values()
                if r["exp"] == experiment and metric in r["metrics"]
                and all(r["tags"].get(k) == v for k, v in tags.items())]
        hits.sort(key=lambda r: r["metrics"][metric], reverse=(mode == "max"))
        return hits[:limit]


@pytest.fixture
def memory(tmp_path):
    backend = MemoryBackend()
    register_backend("memory", lambda settings: backend)
    at.finish()
    s = Settings(backend="memory", owner="t", capture_env=False, queue_dir=str(tmp_path / "q"))
    yield backend, s
    at.finish()


# ---- the contract, against every backend
def test_contract_local(tmp_path):
    from sklearn.linear_model import LogisticRegression
    from authtracker.backends.local import LocalBackend
    model = LogisticRegression().fit([[0], [1]], [0, 1])
    check_backend(LocalBackend(Settings(backend="local", tracking_uri=str(tmp_path / "s"))),
                  model=model)


def test_contract_mlflow(tmp_path):
    from sklearn.linear_model import LogisticRegression
    from authtracker.backends.mlflow_backend import MlflowBackend
    model = LogisticRegression().fit([[0], [1]], [0, 1])
    check_backend(MlflowBackend(Settings(tracking_uri=f"sqlite:///{tmp_path}/m.db")), model=model)


def test_contract_minimal_third_backend():
    check_backend(MemoryBackend())


def test_contract_catches_a_broken_backend():
    class Broken(MemoryBackend):
        def search_runs(self, *a, **k):
            return super().search_runs(*a, **k)[::-1]   # wrong order
    with pytest.raises(AssertionError):
        check_backend(Broken())


# ---- a new backend works with the whole simple API, no authtracker changes
def test_simple_api_on_minimal_backend(memory):
    backend, s = memory
    at.init("m1", "exp", settings=s)
    at.start("a", config={"lr": 0.1})
    at.log({"f1": 0.9})
    at.save("hello\n", name="train.log")
    at.finish()
    best = at.best_run("m1", "exp", "f1", settings=s)
    assert best["metrics"]["f1"] == 0.9 and best["params"]["lr"] == "0.1"
    assert "train.log" in backend.files[best["run_id"]]
    assert {r["status"] for r in backend.runs.values()} == {"FINISHED"}


def test_missing_optional_features_give_clear_errors(memory):
    _, s = memory
    at.init("m1", "exp", settings=s)
    at.start("a")
    with pytest.raises(at.BackendError, match="MemoryBackend.*logging models"):
        at.save(object())
    with pytest.raises(at.BackendError, match="MemoryBackend.*model registry"):
        at.load("m1", "production", settings=s)


# ---- plugin discovery and versioning
def test_entry_point_plugin(monkeypatch):
    class EP:
        name = "plug"
        def load(self):
            return lambda settings: MemoryBackend()

    monkeypatch.setattr("authtracker.backends.metadata.entry_points", lambda **kw: [EP()])
    assert "plug" in available_backends()
    b = get_backend(Settings(backend="plug", offline_queue=False))
    assert isinstance(b, MemoryBackend)


def test_unknown_backend_lists_available():
    with pytest.raises(at.BackendError, match="mlflow"):
        get_backend(Settings(backend="nope"))


def test_api_version_mismatch(monkeypatch):
    class Old(MemoryBackend):
        api_version = base_module.BACKEND_API_VERSION + 1
    register_backend("old", lambda s: Old())
    with pytest.raises(at.BackendError, match="API version"):
        get_backend(Settings(backend="old"))


# ---- defaults and isolation
def test_default_backend_is_mlflow():
    assert Settings().backend == "mlflow"
    assert Settings.from_env({}).backend == "mlflow"


def test_core_does_not_import_mlflow():
    code = ("import sys, authtracker as at\n"
            "from authtracker.backends import get_backend\n"
            "from authtracker.settings import Settings\n"
            "get_backend(Settings(backend='local', offline_queue=False))\n"
            "assert 'mlflow' not in sys.modules, 'core imported mlflow'")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
