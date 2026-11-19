import dataclasses
from pathlib import Path

import pytest

pytest.importorskip("wandb")

import authtracker as at  # noqa: E402
from authtracker.backends import get_backend  # noqa: E402
from authtracker.backends.wandb_backend import (  # noqa: E402
    _PARENT_KEY, _TAGS_KEY, WandbBackend, _reset_process_state)
from authtracker.settings import Settings  # noqa: E402
from authtracker.testing import check_backend  # noqa: E402


@pytest.fixture
def offline(tmp_path, monkeypatch):
    """Real wandb SDK, offline mode: writes run folders to disk, needs no account."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    _reset_process_state()
    yield tmp_path
    _reset_process_state()


def _offline_runs(root: Path):
    return sorted((root / "wandb").glob("offline-run-*"))


def test_contract_offline(offline):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression().fit([[0], [1]], [0, 1])
    check_backend(WandbBackend(Settings(backend="wandb")), model=model)
    runs = _offline_runs(offline)
    assert len(runs) == 4                                        # 1 parent + 3 runs
    assert len(list(offline.glob("wandb/offline-run-*/files/note.txt"))) == 3


def test_simple_api_on_wandb(offline):
    s = Settings(backend="wandb", owner="t", capture_env=False)
    at.init("model1", "exp", dataset_version="v1", config={"base": "x"}, settings=s)
    at.start("a", config={"lr": 0.1})
    at.log({"f1": 0.7})
    at.save("hello\n", name="train.log")
    at.start("b", config={"lr": 0.01})
    at.log({"f1": 0.9})
    at.save(["pyproject.toml"] if False else [])  if False else None
    at.finish()

    best = at.best_run("model1", "exp", "f1", settings=s)
    assert best["name"] == "b" and best["metrics"]["f1"] == 0.9
    assert best["tags"]["dataset_version"] == "v1" and best["params"]["lr"] == "0.01"
    assert len(_offline_runs(offline)) == 3
    assert len(list(offline.glob("wandb/offline-run-*/files/train.log"))) == 1


def test_registry_is_clearly_unsupported(offline):
    from sklearn.linear_model import LogisticRegression
    s = Settings(backend="wandb", capture_env=False)
    at.init("model1", "exp", settings=s)
    at.start("a")
    at.log({"f1": 0.9})
    at.save(LogisticRegression().fit([[0], [1]], [0, 1]))        # a model artifact: supported
    with pytest.raises(at.BackendError, match="WandbBackend.*model registry"):
        at.register(settings=s)
    with pytest.raises(at.BackendError, match="WandbBackend.*model registry"):
        at.load("model1", "production", settings=s)


def test_log_model_options(offline, tmp_path):
    s = Settings(backend="wandb", capture_env=False)
    at.init("model1", "exp", settings=s)
    at.start("a")
    code = tmp_path / "arch"
    code.mkdir()
    (code / "net.py").write_text("x = 1")
    uri = at.save({"weights": [1, 2]}, code_paths=[str(code)], pip_requirements=["torch"])
    assert uri.startswith("wandb-artifact:model1/model-")
    with pytest.raises(at.BackendError, match="serialization_format"):
        at.save({"w": 1}, serialization_format="cloudpickle")    # an MLflow-only option


def test_wandb_backend_is_not_wrapped_in_the_queue(offline):
    b = get_backend(Settings(backend="wandb", offline_queue=True))
    assert isinstance(b, WandbBackend)
    assert type(get_backend(Settings(backend="local", offline_queue=True))).__name__ == "ResilientBackend"


# ---- the part that needs a server: reading runs from other processes (fake API, see note)
class _FakeRun:
    def __init__(self, id, name, tags, f1, lr):
        self.id, self.name = id, name
        self.config = {_TAGS_KEY: tags, _PARENT_KEY: "p/1", "lr": lr, "_wandb": {"x": 1}}
        self.summary = {"f1": f1, "_runtime": 12, "_step": 3, "note": "text", "flag": True}


class _FakeApi:
    default_entity = "team"

    def __init__(self, projects):
        self.projects = projects

    def runs(self, path):
        entity, project = path.split("/")
        assert entity == "team"
        if project not in self.projects:
            raise ValueError("Could not find project team/" + project)
        return self.projects[project]

    def run(self, path):
        _, project, rid = path.split("/")
        return next(r for r in self.projects[project] if r.id == rid)


def test_reads_runs_from_the_server_through_the_api(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    t = {"model": "m", "experiment": "e", "authtracker.level": "run"}
    api = _FakeApi({"m": [
        _FakeRun("r1", "a", t, 0.7, 0.1),
        _FakeRun("r2", "b", t, 0.9, 0.01),
        _FakeRun("r3", "parent", {**t, "authtracker.level": "experiment"}, 0.99, 0),
        _FakeRun("r4", "other-exp", {**t, "experiment": "x"}, 0.95, 0.2),
    ]})
    b = WandbBackend(Settings(backend="wandb"), api=api)
    flt = {"model": "m", "experiment": "e", "authtracker.level": "run"}
    best = b.search_runs("m", flt, "f1", "max", 5)
    assert [r["name"] for r in best] == ["b", "a"]               # filtered and ordered
    assert best[0]["run_id"] == "m/r2"
    assert best[0]["params"] == {"lr": "0.01"}                   # internal keys hidden
    assert best[0]["metrics"] == {"f1": 0.9}                     # text, bool and _runtime dropped
    assert b.search_runs("m", flt, "f1", "min", 1)[0]["name"] == "a"
    assert b.search_runs("nope", flt, "f1", "max", 1) == []      # unknown project
    assert b.get_run("m/r1")["tags"]["experiment"] == "e"
