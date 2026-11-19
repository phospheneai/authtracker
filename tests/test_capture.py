import dataclasses
import json
import logging
import os
import subprocess
import sys
import textwrap

import pytest

import authtracker as at
from authtracker import capture
from authtracker.settings import Settings


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in list(os.environ):
        if k.startswith("AUTHTRACKER_"):
            monkeypatch.delenv(k)
    at.configure(None)
    at.finish()
    yield
    at.finish()


@pytest.fixture(params=["local", "mlflow"])
def s(request, tmp_path):
    if request.param == "mlflow":
        pytest.importorskip("mlflow")
        uri = f"sqlite:///{tmp_path}/m.db"
    else:
        uri = str(tmp_path / "store")
    return Settings(backend=request.param, tracking_uri=uri, owner="t", capture_env=False,
                    queue_dir=str(tmp_path / "q"), retry_wait=0.0)


def _output_log(s, run_id, tmp_path):
    from authtracker.backends import get_backend
    b = get_backend(s)
    if s.backend == "local":
        p = b._artifacts(run_id) / "output.log"
        return p.read_text() if p.exists() else None
    names = [a.path for a in b.client.list_artifacts(run_id)]
    if "output.log" not in names:
        return None
    return open(b.client.download_artifacts(run_id, "output.log", str(tmp_path / f"dl-{run_id}"))).read()


def test_print_logging_and_stderr_are_saved(s, tmp_path, capsys):
    lg = logging.getLogger("train")                    # a console handler created BEFORE the run starts
    handler = logging.StreamHandler()
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    at.init("m1", "exp", settings=s)
    parent = at.current_run()
    print("before any run")
    a = at.start("a")
    print("print in a")
    lg.info("logging in a")
    sys.stderr.write("stderr in a\n")
    b = at.start("b")
    print("print in b")
    at.finish()
    lg.removeHandler(handler)

    log_a, log_b = _output_log(s, a.id, tmp_path), _output_log(s, b.id, tmp_path)
    log_parent = _output_log(s, parent.id, tmp_path)
    assert "print in a" in log_a and "logging in a" in log_a and "stderr in a" in log_a
    assert "print in b" not in log_a and "before any run" not in log_a      # only its own part
    assert log_b.strip() == "print in b"
    for line in ("before any run", "print in a", "logging in a", "stderr in a", "print in b"):
        assert line in log_parent                                           # the parent has it all
    shown = capsys.readouterr()                                              # screen output kept
    assert "print in a" in shown.out and "stderr in a" in shown.err


def test_streams_and_handlers_are_restored(s):
    handler = logging.StreamHandler()
    lg = logging.getLogger("restore-test")
    lg.addHandler(handler)
    out, err, stream = sys.stdout, sys.stderr, handler.stream
    at.init("m1", "exp", settings=s)
    at.start("a")
    assert sys.stdout is not out and handler.stream is not stream            # active
    at.finish()
    assert sys.stdout is out and sys.stderr is err and handler.stream is stream
    lg.removeHandler(handler)


def test_can_be_switched_off(s, tmp_path):
    s = dataclasses.replace(s, capture_logs=False)
    at.init("m1", "exp", settings=s)
    r = at.start("a")
    print("not saved")
    at.finish()
    assert _output_log(s, r.id, tmp_path) is None


def test_empty_output_saves_nothing(s, tmp_path):
    at.init("m1", "exp", settings=s)
    r = at.start("a")
    at.finish()
    assert _output_log(s, r.id, tmp_path) is None


def test_log_is_capped(s, tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "MAX_BYTES", 1000)
    at.init("m1", "exp", settings=s)
    r = at.start("a")
    for _ in range(100):
        print("x" * 50)
    at.finish()
    text = _output_log(s, r.id, tmp_path)
    assert len(text) < 1300 and "truncated" in text


def test_wandb_backend_is_left_to_wandb():
    pytest.importorskip("wandb")
    from authtracker.backends.wandb_backend import WandbBackend
    assert WandbBackend.captures_console is True


def test_crash_keeps_output_and_traceback(tmp_path):
    env = {**os.environ, "AUTHTRACKER_BACKEND": "local", "AUTHTRACKER_TRACKING_URI": str(tmp_path / "store"),
           "AUTHTRACKER_QUEUE_DIR": str(tmp_path / "q"), "AUTHTRACKER_CAPTURE_ENV": "false"}
    code = textwrap.dedent("""
        import authtracker as at
        at.init("m", "e"); at.start("a")
        print("epoch 1 done")
        raise RuntimeError("boom")
    """)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.returncode != 0
    runs = {json.loads(p.read_text())["name"]: json.loads(p.read_text())
            for p in (tmp_path / "store" / "runs").glob("*.json")}
    rid = runs["a"]["run_id"]
    assert runs["a"]["status"] == "FAILED"
    art = tmp_path / "store" / "artifacts" / rid
    assert "epoch 1 done" in (art / "output.log").read_text()
    assert "boom" in (art / "error.txt").read_text()
