import os

import pytest

import authtracker as at
from authtracker.settings import ConfigError, Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("AUTHTRACKER_") or k == "MLFLOW_TRACKING_URI":
            monkeypatch.delenv(k)
    at.configure(None)
    yield
    at.configure(None)


@pytest.fixture(params=["local", "mlflow"])
def settings(request, tmp_path):
    if request.param == "mlflow":
        pytest.importorskip("mlflow")
        uri = f"sqlite:///{tmp_path}/mlflow.db"
    else:
        uri = str(tmp_path / "store")
    return Settings(backend=request.param, tracking_uri=uri, owner="tester",
                    capture_env=False, queue_dir=str(tmp_path / "queue"), retry_wait=0.0)


# ---- settings
def test_env_parsing(monkeypatch):
    monkeypatch.setenv("AUTHTRACKER_BACKEND", "MLflow")
    monkeypatch.setenv("AUTHTRACKER_REQUIRED_TAGS", "owner, dataset_version")
    monkeypatch.setenv("AUTHTRACKER_PROMOTE_MIN", "0.9")
    s = Settings.from_env()
    assert s.backend == "mlflow"
    assert s.required_tags == ("owner", "dataset_version")
    assert s.promote_min == 0.9


def test_mlflow_uri_fallback(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x:5000")
    assert Settings.from_env().tracking_uri == "http://x:5000"


def test_bad_values(monkeypatch):
    monkeypatch.setenv("AUTHTRACKER_CAPTURE_ENV", "maybe")
    with pytest.raises(ConfigError):
        Settings.from_env()


def test_unknown_variable_warns(monkeypatch):
    monkeypatch.setenv("AUTHTRACKER_BACKNED", "x")
    with pytest.warns(UserWarning):
        Settings.from_env()


def test_masked():
    s = Settings(tracking_uri="postgresql://user:pw@host/db")
    assert "pw" not in str(s.masked())


# ---- validation
def test_required_tags_and_allowed_models(settings):
    import dataclasses
    s = dataclasses.replace(settings, required_tags=("dataset_version",),
                            allowed_models=("m1",))
    with pytest.raises(at.ValidationError):
        at.run("m1", "e", settings=s)  # missing dataset_version
    with pytest.raises(at.ValidationError):
        at.run("other", "e", dataset_version="v1", settings=s)
    with at.run("m1", "e", dataset_version="v1", settings=s):
        pass


def test_metric_names(settings):
    import dataclasses
    s = dataclasses.replace(settings, metric_names=("f1",))
    with at.run("m1", "e", settings=s) as r:
        r.log_eval(f1=0.5)
        with pytest.raises(at.ValidationError):
            r.log_eval(acc=0.5)


# ---- tracking
def test_hierarchy_and_best_run(settings):
    with at.run("m1", "exp", dataset_version="v1", settings=settings) as r:
        for name, f1 in [("a", 0.7), ("b", 0.9), ("c", 0.8)]:
            with r.child(name, params={"name": name}) as c:
                c.log_eval(f1=f1)
    best = at.best_run("m1", "exp", "f1", settings=settings)
    assert best["name"] == "exp-b" or best["name"] == "b"
    assert best["metrics"]["f1"] == 0.9
    assert best["tags"]["authtracker.level"] == "run"
    assert best["tags"]["experiment"] == "exp"
    assert at.best_run("m1", "nope", "f1", settings=settings) is None


def test_failed_run_status(settings):
    from authtracker.backends import get_backend
    b = get_backend(settings)
    holder = {}
    with pytest.raises(RuntimeError):
        with at.run("m1", "boom", settings=settings, backend=b) as r:
            holder["id"] = r.id
            raise RuntimeError("x")
    if settings.backend == "local":
        assert b._load(holder["id"])["status"] == "FAILED"
    else:
        assert b.client.get_run(holder["id"]).info.status == "FAILED"


# ---- versioning
def _train(settings, f1, model_name="m1"):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression().fit([[0], [1]], [0, 1])
    with at.run(model_name, "exp", settings=settings) as r:
        with r.child("t") as c:
            c.log_eval(f1=f1)
            c.log_model(model, flavor="sklearn")
            return at.register(c, settings=settings)


def test_register_promote_gates_rollback_load(settings):
    import dataclasses
    s = dataclasses.replace(settings, promote_metric="f1", promote_min=0.8,
                            promote_must_beat_prod=True)
    v1 = _train(s, 0.85)
    v_low = _train(s, 0.5)
    v2 = _train(s, 0.95)
    v_mid = _train(s, 0.9)

    with pytest.raises(at.PromotionError):          # below minimum
        at.promote("m1", v_low, "production", settings=s)
    at.promote("m1", v1, "staging", settings=s)     # staging is not gated
    at.promote("m1", v1, "production", settings=s)
    with pytest.raises(at.PromotionError):          # does not beat v2... after v2 promoted
        at.promote("m1", v2, "production", settings=s)
        at.promote("m1", v_mid, "production", settings=s)

    assert at.rollback("m1", settings=s) == v1
    model = at.load("m1", "production", settings=s)
    assert model.predict([[1]]).tolist() == [1]


def test_register_without_model_fails(settings):
    with at.run("m1", "e", settings=settings) as r:
        with pytest.raises(at.ValidationError):
            at.register(r, settings=settings)


def test_unknown_backend():
    with pytest.raises(at.BackendError):
        at.run("m", "e", settings=Settings(backend="nope"))


# ---- text and bundles
def _artifact_names(settings, run_id):
    from authtracker.backends import get_backend
    b = get_backend(settings)
    if settings.backend == "local":
        return sorted(p.name for p in b._artifacts(run_id).iterdir())
    return sorted(a.path for a in b.client.list_artifacts(run_id))


def test_log_text_and_bundle(settings, tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "a.json").write_text("{}")
    (bundle / "b.csv").write_text("x\n1\n")
    (bundle / "sub").mkdir()                       # sub-folders are not included
    with at.run("m1", "e", settings=settings) as r:
        r.log_text("epoch 1 ok\n", "train.log")
        r.log_artifacts(str(bundle))
        r.log_artifacts([str(bundle / "a.json")])  # list form, re-logging a file is fine
        run_id = r.id
        with pytest.raises(at.ValidationError):
            r.log_artifacts([str(bundle / "a.json"), str(tmp_path / "a.json")])
        with pytest.raises(at.ValidationError):
            r.log_artifacts(str(tmp_path / "missing"))
    names = _artifact_names(settings, run_id)
    assert {"train.log", "a.json", "b.csv"} <= set(names)
    assert "sub" not in names
