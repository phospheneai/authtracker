import pytest

import authtracker as at
from authtracker.backends.local import LocalBackend
from authtracker.backends.resilient import OFFLINE, ResilientBackend
from authtracker.settings import ConfigError, Settings


class Flaky:
    """Wraps a backend; raises ConnectionError while `down` (or for the first `fail_n` calls)."""

    def __init__(self, inner):
        self.inner, self.down, self.fail_n, self.calls = inner, False, 0, 0

    def is_transient(self, e):
        return isinstance(e, ConnectionError)

    def __getattr__(self, name):
        fn = getattr(self.inner, name)

        def wrapper(*a, **k):
            self.calls += 1
            if self.down or self.fail_n > 0:
                self.fail_n = max(0, self.fail_n - 1)
                raise ConnectionError("server down")
            return fn(*a, **k)
        return wrapper if callable(fn) else fn


@pytest.fixture
def parts(tmp_path):
    s = Settings(tracking_uri=str(tmp_path / "store"), queue_dir=str(tmp_path / "q"),
                 retries=2, retry_wait=0.0, capture_env=False, owner="t")
    flaky = Flaky(LocalBackend(s))
    rb = ResilientBackend(flaky, s, sleep=lambda _: None, cooldown=0.0)
    return s, flaky, rb


def _metrics(flaky, run_id):
    return flaky.inner.get_run(run_id)["metrics"]


def test_retry_recovers_without_queueing(parts):
    s, flaky, rb = parts
    flaky.fail_n = 2                      # fails twice, third attempt succeeds
    rid = rb.start_run("m", "r", {"model": "m"}, None)
    assert not rid.startswith(OFFLINE)
    assert rb.pending() == 0 and flaky.calls == 3


def test_queue_and_replay_in_order(parts):
    s, flaky, rb = parts
    rid = rb.start_run("m", "r", {"model": "m"}, None)
    flaky.down = True
    rb.log_metrics(rid, {"loss": 1.0}, 0)
    rb.log_metrics(rid, {"loss": 0.5}, 1)
    rb.log_params(rid, {"lr": 0.1})
    rb.end_run(rid, "FINISHED")
    assert rb.pending() == 4
    assert _metrics(flaky, rid) == {}
    flaky.down = False
    assert rb.flush() == 4 and rb.pending() == 0
    run = flaky.inner.get_run(rid)
    assert run["metrics"]["loss"] == 0.5 and run["params"]["lr"] == "0.1"
    assert flaky.inner._load(rid)["status"] == "FINISHED"


def test_start_run_offline_gets_placeholder_then_real_id(parts):
    s, flaky, rb = parts
    flaky.down = True
    parent = rb.start_run("m", "parent", {"model": "m"}, None)
    child = rb.start_run("m", "child", {"model": "m"}, parent)
    assert parent.startswith(OFFLINE) and child.startswith(OFFLINE)
    rb.log_metrics(child, {"f1": 0.9}, None)
    rb.end_run(child, "FINISHED")
    rb.end_run(parent, "FINISHED")
    flaky.down = False
    assert rb.flush() == 5
    real_child = rb._res(child)
    assert not real_child.startswith(OFFLINE)
    run = flaky.inner._load(real_child)
    assert run["metrics"]["f1"][-1][1] == 0.9
    assert run["parent"] == rb._res(parent)
    assert run["status"] == "FINISHED"


def test_artifact_is_copied_into_queue(parts, tmp_path):
    s, flaky, rb = parts
    rid = rb.start_run("m", "r", {"model": "m"}, None)
    flaky.down = True
    f = tmp_path / "note.txt"
    f.write_text("hi")
    rb.log_artifact(rid, str(f))
    f.unlink()                            # caller deletes it (like a temp dir)
    flaky.down = False
    assert rb.flush() == 1
    assert (flaky.inner._artifacts(rid) / "note.txt").read_text() == "hi"


def test_non_transient_error_is_raised_not_queued(parts):
    s, flaky, rb = parts
    rid = rb.start_run("m", "r", {"model": "m"}, None)
    rb.log_params(rid, {"lr": 0.1})
    calls = flaky.calls
    with pytest.raises(at.BackendError):
        rb.log_params(rid, {"lr": 0.2})   # conflicting param: a real error
    assert rb.pending() == 0 and flaky.calls == calls + 1


def test_reads_raise_backend_error_when_down(parts):
    s, flaky, rb = parts
    flaky.down = True
    with pytest.raises(at.BackendError, match="unreachable"):
        rb.search_runs("m", {}, "f1", "max", 1)


def test_other_process_leftovers_are_synced(parts):
    s, flaky, rb = parts
    rid = rb.start_run("m", "r", {"model": "m"}, None)
    flaky.down = True
    rb.log_metrics(rid, {"loss": 2.0}, 0)
    # "new process": different token, does not auto-flush the old one's queue
    rb2 = ResilientBackend(flaky, s, sleep=lambda _: None, cooldown=0.0)
    flaky.down = False
    assert rb2.flush() == 0 and rb2.pending() == 1
    assert rb2.flush(force=True, all_tokens=True) == 1
    assert _metrics(flaky, rid)["loss"] == 2.0


def test_run_survives_outage_end_to_end(parts):
    s, flaky, rb = parts
    with at.run("m1", "exp", settings=s, backend=rb) as r:
        with r.child("a") as c:
            flaky.down = True             # server dies mid-training
            for step in range(3):
                c.log_metrics({"loss": 1 / (step + 1)}, step)
            c.log_eval(f1=0.8)
        flaky.down = False
    rb.flush(force=True)
    assert rb.pending() == 0
    best = at.best_run("m1", "exp", "f1", settings=s, backend=rb)
    assert best["metrics"]["f1"] == 0.8


def test_settings_parsing(monkeypatch):
    monkeypatch.setenv("AUTHTRACKER_RETRIES", "5")
    monkeypatch.setenv("AUTHTRACKER_RETRY_WAIT", "0.5")
    monkeypatch.setenv("AUTHTRACKER_OFFLINE_QUEUE", "false")
    s = Settings.from_env()
    assert (s.retries, s.retry_wait, s.offline_queue) == (5, 0.5, False)
    monkeypatch.setenv("AUTHTRACKER_RETRIES", "-1")
    with pytest.raises(ConfigError):
        Settings.from_env()
