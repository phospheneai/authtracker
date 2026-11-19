import dataclasses
import json
import logging
import os
import subprocess

import pytest

import authtracker as at
from authtracker.backends import Backend, get_backend, register_backend
from authtracker.backends.local import LocalBackend
from authtracker.settings import Settings


@pytest.fixture(params=["local", "mlflow"])
def s(request, tmp_path):
    if request.param == "mlflow":
        pytest.importorskip("mlflow")
        uri = f"sqlite:///{tmp_path}/m.db"
    else:
        uri = str(tmp_path / "store")
    at.configure(None)
    at.finish()
    yield Settings(backend=request.param, tracking_uri=uri, owner="me", capture_env=False, capture_logs=False,
                   system_metrics=False, save_code=False, queue_dir=str(tmp_path / "q"), retry_wait=0.0)
    at.finish()


def _all_runs(s):
    """Every run the store has, as {id, name, level, status, tags}."""
    b = get_backend(s)
    if s.backend == "local":
        runs = [json.loads(p.read_text()) for p in (b.root / "runs").glob("*.json")]
        return [{"id": r["run_id"], "name": r["name"], "level": r["tags"]["authtracker.level"],
                 "status": r["status"], "tags": r["tags"]} for r in runs]
    exp = b.client.get_experiment_by_name("m1")
    return [{"id": r.info.run_id, "name": r.data.tags["mlflow.runName"], "level": r.data.tags["authtracker.level"],
             "status": r.info.status, "tags": dict(r.data.tags)} for r in b.client.search_runs([exp.experiment_id])]


def _count(s, level):
    return len([r for r in _all_runs(s) if r["level"] == level])


def _steps(s, run_id, key="loss"):
    b = get_backend(s)
    if s.backend == "local":
        return [step for step, _ in b._load(run_id)["metrics"][key]]
    return sorted(m.step for m in b.client.get_metric_history(run_id, key))


def _artifacts(s, run_id):
    b = get_backend(s)
    if s.backend == "local":
        folder = b._artifacts(run_id)
        return sorted(p.name for p in folder.iterdir())
    return sorted(a.path for a in b.client.list_artifacts(run_id))


def session(s, n_logs, name="abc", **init_kw):
    """One run of a training script: init, start a run, log n_logs values, finish. Returns the ids."""
    init_kw.setdefault("resume", True)
    parent = at.init("m1", "exp", dataset_version="v1", settings=s, **init_kw)
    child = at.start(name)
    for _ in range(n_logs):
        at.log({"loss": 1.0})
    ids = (parent.id, child.id)
    at.finish()
    return ids


# ---------------------------------------------------------------- continuing
def test_with_nothing_to_resume_it_starts_fresh(s):
    parent, child = session(s, 3)
    assert (_count(s, "experiment"), _count(s, "run")) == (1, 1)
    tags = {r["id"]: r["tags"] for r in _all_runs(s)}
    assert "resume_count" not in tags[parent] and "resumed" not in tags[child]


def test_running_the_same_script_again_continues_the_same_runs(s):
    p1, c1 = session(s, 3)
    p2, c2 = session(s, 2)
    assert (p2, c2) == (p1, c1)                                      # the same runs
    assert (_count(s, "experiment"), _count(s, "run")) == (1, 1)
    assert _steps(s, c1) == [0, 1, 2, 3, 4]                          # the curve continues, it does not restart
    runs = {r["id"]: r for r in _all_runs(s)}
    assert runs[c1]["status"] == "FINISHED" and runs[p1]["status"] == "FINISHED"
    assert runs[c1]["tags"]["resume_count"] == "1" and runs[c1]["tags"]["resumed"] == "true"
    assert runs[p1]["tags"]["resume_count"] == "1"
    session(s, 1)
    assert {r["id"]: r for r in _all_runs(s)}[c1]["tags"]["resume_count"] == "2"      # and it counts


def test_the_run_is_open_while_it_continues(s):
    p1, c1 = session(s, 1)
    at.init("m1", "exp", dataset_version="v1", resume=True, settings=s)
    at.start("abc")
    assert {r["id"]: r for r in _all_runs(s)}[c1]["status"] == "RUNNING"          # reopened
    at.finish()
    assert {r["id"]: r for r in _all_runs(s)}[c1]["status"] == "FINISHED"


def test_without_resume_every_session_is_new(s):
    session(s, 2, resume=False)
    session(s, 2, resume=False)
    assert (_count(s, "experiment"), _count(s, "run")) == (2, 2)


def test_a_different_name_is_a_new_run_under_the_resumed_experiment(s):
    p1, c1 = session(s, 2)
    p2, c2 = session(s, 2, name="other")
    assert p2 == p1 and c2 != c1
    assert (_count(s, "experiment"), _count(s, "run")) == (1, 2)


def test_start_can_override_the_default_of_init(s):
    p1, c1 = session(s, 2)
    parent = at.init("m1", "exp", dataset_version="v1", resume=True, settings=s)
    child = at.start("abc", resume=False)                            # a new run under the resumed experiment
    at.finish()
    assert parent.id == p1 and child.id != c1
    assert _count(s, "run") == 2


def test_resume_can_be_asked_for_on_start_only(s):
    session(s, 2, resume=False)
    parent = at.init("m1", "exp", dataset_version="v1", settings=s)               # a new experiment run ...
    child = at.start("abc", resume=True)                             # ... so there is no earlier 'abc' under it
    at.finish()
    assert _count(s, "experiment") == 2 and _count(s, "run") == 2 and "resumed" not in child.tags


def test_other_people_s_runs_are_not_resumed(s):
    session(s, 2)
    mine = _count(s, "run")
    other = dataclasses.replace(s, owner="someone-else")
    parent = at.init("m1", "exp", dataset_version="v1", resume=True, settings=other)
    at.start("abc")
    at.finish()
    assert (_count(s, "experiment"), _count(s, "run")) == (2, mine + 1)


def test_the_step_continues_for_metrics_logged_to_the_experiment_run_too(s):
    parent = at.init("m1", "exp", dataset_version="v1", resume=True, settings=s)
    at.log({"loss": 1.0}), at.log({"loss": 1.0})
    at.finish()
    at.init("m1", "exp", dataset_version="v1", resume=True, settings=s)
    at.log({"loss": 1.0}), at.log({"loss": 1.0})
    at.finish()
    assert _steps(s, parent.id) == [0, 1, 2, 3]


def test_the_with_api_resumes_too(s):
    with at.run("m1", "exp", dataset_version="v1", resume=True, settings=s) as r:
        with r.child("abc", resume=True) as c:
            c.log_metrics({"loss": 1.0}, step=0)
    with at.run("m1", "exp", dataset_version="v1", resume=True, settings=s) as r2:
        with r2.child("abc", resume=True) as c2:
            c2.log_metrics({"loss": 0.5}, step=1)
    assert (r2.id, c2.id) == (r.id, c.id) and _steps(s, c.id) == [0, 1]


# ---------------------------------------------------------------- parameters
def test_a_resumed_run_may_repeat_and_add_parameters_but_not_change_them(s):
    at.init("m1", "exp", dataset_version="v1", resume=True, config={"lr": 0.1}, settings=s)
    at.start("abc", config={"epochs": 5})
    at.finish()
    parent, child = session_with_config(s, {"lr": 0.1, "momentum": 0.9}, {"epochs": 5})       # same, plus a new one
    params = get_backend(s).get_run(parent)["params"]
    assert params == {"lr": "0.1", "momentum": "0.9"}
    assert _count(s, "experiment") == 1

    runs = _all_runs(s)
    before = {r["id"]: r["status"] for r in runs}
    with pytest.raises(at.ValidationError, match="lr was 0.1 and is now 0.01"):
        at.init("m1", "exp", dataset_version="v1", resume=True, config={"lr": 0.01}, settings=s)
    with pytest.raises(at.ValidationError, match="epochs was 5 and is now 50"):
        at.init("m1", "exp", dataset_version="v1", resume=True, settings=s)
        at.start("abc", config={"epochs": 50}, resume=True)
    at.finish()
    after = {r["id"]: r["status"] for r in _all_runs(s)}
    assert {k: v for k, v in after.items() if k in before and v != "RUNNING"} == \
        {k: v for k, v in before.items() if k in after and after[k] != "RUNNING"}     # refusing did not damage the old run
    assert "FAILED" not in after.values()


def session_with_config(s, parent_config, child_config):
    parent = at.init("m1", "exp", dataset_version="v1", resume=True, config=parent_config, settings=s)
    child = at.start("abc", config=child_config)
    ids = (parent.id, child.id)
    at.finish()
    return ids


# ---------------------------------------------------------------- earlier parts are kept
def test_files_of_earlier_parts_are_not_overwritten(s, capsys):
    s = dataclasses.replace(s, capture_logs=True, capture_env=True, log_upload_interval=0)
    at.init("m1", "exp", dataset_version="v1", resume=True, settings=s)
    child = at.start("abc")
    print("first part")
    at.finish()
    at.init("m1", "exp", dataset_version="v1", resume=True, settings=s)
    at.start("abc")
    print("second part")
    at.finish()
    files = _artifacts(s, child.id)
    assert "output.log" in files and "output.resume1.log" in files
    b = get_backend(s)
    if s.backend == "local":
        read = lambda name: (b._artifacts(child.id) / name).read_text()      # noqa: E731
    else:
        read = lambda name: open(b.client.download_artifacts(child.id, name, os.getcwd())).read()  # noqa: E731
    assert "first part" in read("output.log") and "second part" not in read("output.log")
    assert "second part" in read("output.resume1.log") and "first part" not in read("output.resume1.log")
    parent_id = [r for r in _all_runs(s) if r["level"] == "experiment"][0]["id"]
    assert {"requirements.txt", "requirements.resume1.txt"} <= set(_artifacts(s, parent_id))


GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}


def test_each_part_keeps_its_own_commit(s, tmp_path, monkeypatch):
    repo = tmp_path / "project"
    repo.mkdir()
    for args in (["init", "-q", "-b", "main"],):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=GIT_ENV)
    (repo / "a.py").write_text("x = 1\n")
    for args in (["add", "."], ["commit", "-q", "-m", "one"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=GIT_ENV)
    monkeypatch.chdir(repo)
    s = dataclasses.replace(s, capture_env=True)
    parent, _ = session(s, 1)
    first = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    (repo / "a.py").write_text("x = 2\n")
    for args in (["add", "."], ["commit", "-q", "-m", "two"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=GIT_ENV)
    second = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    parent2, _ = session(s, 1)
    assert parent2 == parent
    tags = {r["id"]: r["tags"] for r in _all_runs(s)}[parent]
    assert tags["git_sha"] == first and tags["resume1_git_sha"] == second          # the original is kept, the new one added


# ---------------------------------------------------------------- what is not supported or not reachable
class _NoResume(Backend):                                            # a backend with only the required methods
    def ping(self): ...
    def start_run(self, *a): return "r"
    def end_run(self, *a): ...
    def set_tags(self, *a): ...
    def log_params(self, *a): ...
    def log_metrics(self, *a): ...
    def log_artifact(self, *a): ...
    def get_run(self, *a): return {"tags": {}, "params": {}}
    def search_runs(self, *a): return []


def test_a_backend_that_cannot_resume_says_so():
    register_backend("noresume", lambda settings: _NoResume())
    s = Settings(backend="noresume", offline_queue=False, capture_env=False, capture_logs=False, save_code=False,
                 system_metrics=False, owner="me")
    with pytest.raises(at.UnsupportedError, match="does not support resuming runs"):
        at.init("m1", "exp", dataset_version="v1", resume=True, settings=s)
    at.init("m1", "exp", dataset_version="v1", settings=s)                       # not asking: fine
    at.finish()


def test_if_the_server_cannot_be_asked_a_new_run_starts_with_a_warning(s, monkeypatch):
    if s.backend != "local":
        pytest.skip("simulated on the local backend")

    def unreachable(self, *a):
        raise ConnectionError("server down")

    monkeypatch.setattr(LocalBackend, "find_run", unreachable)
    seen = []

    class Collect(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    handler = Collect(level=logging.WARNING)
    logging.getLogger("authtracker").addHandler(handler)
    try:
        parent = at.init("m1", "exp", dataset_version="v1", resume=True, settings=s)       # training is not stopped
        at.finish()
    finally:
        logging.getLogger("authtracker").removeHandler(handler)
    assert any("Could not look for an earlier run" in m for m in seen), seen
    assert _count(s, "experiment") == 1


# ---------------------------------------------------------------- restore
def test_restore_gives_none_the_first_time_and_the_file_after_a_restart(s, tmp_path):
    at.init("m1", "exp", settings=s, resume=True)
    at.start("abc")
    assert at.restore("ckpt.pt") is None
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("epoch 3")
    at.save(str(ckpt))
    at.finish()

    at.init("m1", "exp", settings=s, resume=True)
    at.start("abc")
    path = at.restore("ckpt.pt", folder=str(tmp_path / "back"))
    assert path and open(path).read() == "epoch 3"
    assert os.path.dirname(path) == str(tmp_path / "back")
    assert at.restore("other.pt") is None


def test_restore_without_folder_uses_a_temporary_one(s, tmp_path):
    at.init("m1", "exp", settings=s)
    (tmp_path / "a.txt").write_text("x")
    at.save(str(tmp_path / "a.txt"))
    path = at.restore("a.txt")
    assert open(path).read() == "x" and os.path.dirname(path) != str(tmp_path)


def test_a_backend_that_cannot_download_says_so():
    from authtracker.errors import UnsupportedError
    with pytest.raises(UnsupportedError):
        _NoResume().download_artifact("r", "f", ".")
