import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

import authtracker as at
from authtracker.settings import ConfigError, Settings


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in list(os.environ):
        if k.startswith("AUTHTRACKER_"):
            monkeypatch.delenv(k)
    at.configure(None)
    at.finish()
    yield
    at.finish()


def _settings(tmp_path, backend, interval):
    uri = f"sqlite:///{tmp_path}/m.db" if backend == "mlflow" else str(tmp_path / "store")
    return Settings(backend=backend, tracking_uri=uri, owner="t", capture_env=False, system_metrics=False,
                    queue_dir=str(tmp_path / "q"), retry_wait=0.0, log_upload_interval=interval)


def _stored_log(s, run_id, tmp_path):
    """The output.log that is on the server/store right now (None if there is none)."""
    from authtracker.backends import get_backend
    b = get_backend(s)
    if s.backend == "local":
        p = b._artifacts(run_id) / "output.log"
        return p.read_text() if p.exists() else None
    if "output.log" not in [a.path for a in b.client.list_artifacts(run_id)]:
        return None
    return open(b.client.download_artifacts(run_id, "output.log", str(tmp_path / f"dl{time.time_ns()}"))).read()


@pytest.mark.parametrize("backend", ["local", "mlflow"])
def test_log_is_on_the_server_while_the_run_is_still_going(tmp_path, backend):
    if backend == "mlflow":
        pytest.importorskip("mlflow")
    s = _settings(tmp_path, backend, interval=0.5)
    at.init("m1", "e", dataset_version="v1", settings=s)
    run = at.start("a")
    print("first line")
    deadline = time.time() + 8
    while time.time() < deadline and not (_stored_log(s, run.id, tmp_path) or "").count("first line"):
        time.sleep(0.3)
    assert "first line" in _stored_log(s, run.id, tmp_path)                 # visible BEFORE the run ended
    print("second line")
    at.finish()
    final = _stored_log(s, run.id, tmp_path)
    assert "first line" in final and "second line" in final                 # the final upload has it all


def test_interval_zero_uploads_only_at_the_end(tmp_path):
    s = _settings(tmp_path, "local", interval=0)
    at.init("m1", "e", dataset_version="v1", settings=s)
    run = at.start("a")
    print("only at the end")
    time.sleep(1.5)
    assert _stored_log(s, run.id, tmp_path) is None
    at.finish()
    assert "only at the end" in _stored_log(s, run.id, tmp_path)


def test_a_killed_process_keeps_its_log(tmp_path):
    env = {**os.environ, "AUTHTRACKER_BACKEND": "local", "AUTHTRACKER_TRACKING_URI": str(tmp_path / "store"),
           "AUTHTRACKER_QUEUE_DIR": str(tmp_path / "q"), "AUTHTRACKER_CAPTURE_ENV": "false",
           "AUTHTRACKER_LOG_UPLOAD_INTERVAL": "0.5"}
    code = textwrap.dedent("""
        import os, signal, time, authtracker as at
        at.init("m", "e", dataset_version="v1"); at.start("a")
        print("epoch 1 done", flush=True)
        time.sleep(2.5)                                   # long enough for a periodic upload
        os.kill(os.getpid(), signal.SIGKILL)              # no cleanup runs after this
    """)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.returncode == -signal.SIGKILL
    logs = list((tmp_path / "store" / "artifacts").glob("*/output.log"))
    assert logs and all("epoch 1 done" in p.read_text() for p in logs)       # the log survived


def test_interval_setting(monkeypatch):
    monkeypatch.setenv("AUTHTRACKER_LOG_UPLOAD_INTERVAL", "15")
    assert Settings.from_env().log_upload_interval == 15
    monkeypatch.setenv("AUTHTRACKER_LOG_UPLOAD_INTERVAL", "-1")
    with pytest.raises(ConfigError):
        Settings.from_env()
    monkeypatch.delenv("AUTHTRACKER_LOG_UPLOAD_INTERVAL")
    assert Settings.from_env().log_upload_interval == 60
