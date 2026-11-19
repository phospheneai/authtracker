import json
import os
import subprocess
import sys
import textwrap

import pytest

import authtracker as at
from authtracker.settings import Settings


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in list(os.environ):
        if k.startswith("AUTHTRACKER_") or k == "MLFLOW_TRACKING_URI":
            monkeypatch.delenv(k)
    at.configure(None)
    at.finish()
    yield
    at.finish()
    at.configure(None)


@pytest.fixture(params=["local", "mlflow"])
def s(request, tmp_path):
    if request.param == "mlflow":
        pytest.importorskip("mlflow")
        uri = f"sqlite:///{tmp_path}/m.db"
    else:
        uri = str(tmp_path / "store")
    return Settings(backend=request.param, tracking_uri=uri, owner="t", capture_env=False,
                    queue_dir=str(tmp_path / "q"), retry_wait=0.0)


def _history(s, run_id, key):
    from authtracker.backends import get_backend
    b = get_backend(s)
    if s.backend == "local":
        return [step for step, _ in b._load(run_id)["metrics"][key]]
    return [m.step for m in b.client.get_metric_history(run_id, key)]


def _status(s, run_id):
    from authtracker.backends import get_backend
    b = get_backend(s)
    return b._load(run_id)["status"] if s.backend == "local" else b.client.get_run(run_id).info.status


def _artifacts(s, run_id):
    from authtracker.backends import get_backend
    b = get_backend(s)
    if s.backend == "local":
        return sorted(p.name for p in b._artifacts(run_id).iterdir())
    return sorted(a.path for a in b.client.list_artifacts(run_id))


def test_three_levels_log_save_finish(s, tmp_path):
    at.init("m1", "exp", dataset_version="v1", settings=s)
    parent = at.current_run()
    r1 = at.start("a", config={"lr": 0.1})
    for loss in (3.0, 2.0, 1.0):
        at.log({"loss": loss})                       # steps count up by themselves
    at.save("epoch logs\n", name="train.log")
    f = tmp_path / "cfg.json"
    f.write_text("{}")
    at.save(str(f))
    bundle = tmp_path / "b"
    bundle.mkdir()
    (bundle / "x.csv").write_text("1")
    at.save(str(bundle))
    r2 = at.start("b")                               # closes run "a"
    at.log({"loss": 0.5}, step=7)
    at.finish()

    assert _history(s, r1.id, "loss") == [0, 1, 2]
    assert _history(s, r2.id, "loss") == [7]
    assert {"train.log", "cfg.json", "x.csv"} <= set(_artifacts(s, r1.id))
    assert {_status(s, r1.id), _status(s, r2.id), _status(s, parent.id)} == {"FINISHED"}
    best = at.best_run("m1", "exp", "loss", mode="min", settings=s)
    assert best["name"] == "b" and best["tags"]["dataset_version"] == "v1"


def test_start_default_names_and_init_closes_previous(s):
    at.init("m1", "e1", settings=s)
    first_parent = at.current_run()
    assert at.start().tags["experiment"] == "e1"
    at.init("m1", "e2", settings=s)                   # closes e1 and its run
    assert _status(s, first_parent.id) == "FINISHED"
    at.finish()
    at.finish()                                       # idempotent


def test_errors(s, tmp_path):
    with pytest.raises(at.AuthTrackerError):
        at.log({"x": 1})                              # no init
    at.init("m1", "e", settings=s)
    with pytest.raises(at.ValidationError):
        at.save(str(tmp_path / "missing.txt"))        # typo is not silently treated as text


def test_register_current_run(s):
    from sklearn.linear_model import LogisticRegression
    at.init("m1", "e", settings=s)
    at.start("a")
    at.log({"f1": 0.9})
    at.save(LogisticRegression().fit([[0], [1]], [0, 1]), flavor="sklearn")
    v = at.register(settings=s)                       # no arguments: the current run
    at.promote("m1", v, "production", settings=s)
    assert at.load("m1", "production", settings=s).predict([[1]]).tolist() == [1]


def _script(tmp_path, body):
    env = {**os.environ, "AUTHTRACKER_BACKEND": "local",
           "AUTHTRACKER_TRACKING_URI": str(tmp_path / "store"),
           "AUTHTRACKER_QUEUE_DIR": str(tmp_path / "q"), "AUTHTRACKER_CAPTURE_ENV": "false"}
    code = "import authtracker as at\n" + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)


def _runs(tmp_path):
    return [json.loads(p.read_text()) for p in (tmp_path / "store" / "runs").glob("*.json")]


def test_script_end_closes_runs_without_finish(tmp_path):
    out = _script(tmp_path, """
        at.init("m", "e")
        at.start("a")
        at.log({"x": 1})
    """)
    assert out.returncode == 0, out.stderr
    runs = _runs(tmp_path)
    assert len(runs) == 2 and {r["status"] for r in runs} == {"FINISHED"}


def test_crash_marks_runs_failed_with_traceback(tmp_path):
    out = _script(tmp_path, """
        at.init("m", "e")
        at.start("a")
        raise RuntimeError("boom")
    """)
    assert out.returncode != 0 and "boom" in out.stderr
    runs = _runs(tmp_path)
    assert len(runs) == 2 and {r["status"] for r in runs} == {"FAILED"}
    errors = list((tmp_path / "store" / "artifacts").glob("*/error.txt"))
    assert errors and "boom" in errors[0].read_text()


def test_metric_names_enforced_by_log(s):
    import dataclasses
    s = dataclasses.replace(s, metric_names=("f1", "loss"))
    at.init("m1", "e", settings=s)
    at.start("a")
    at.log({"f1": 0.9, "loss": 0.1})
    with pytest.raises(at.ValidationError):
        at.log({"acc": 0.9})


def test_tags_set_on_the_experiment_reach_later_runs(s):
    at.init("m1", "e", settings=s)
    at.set_tags({"dataset_sha256": "abc"})
    child = at.start("a")
    at.set_tags({"only_this_run": "x"})
    later = at.start("b")
    at.finish()
    from authtracker.backends import get_backend
    b = get_backend(s)
    assert b.get_run(child.id)["tags"]["dataset_sha256"] == "abc"
    assert b.get_run(child.id)["tags"]["only_this_run"] == "x"
    assert "only_this_run" not in b.get_run(later.id)["tags"]      # a run's own tags stay its own
    assert b.get_run(later.id)["tags"]["dataset_sha256"] == "abc"
