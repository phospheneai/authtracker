"""Webhook alerts: optional, never fatal, and the URL (a secret) never shown."""
import http.server
import json
import logging
import threading

import pytest

import authtracker as at
from authtracker import alerts
from authtracker.cli import main
from authtracker.settings import ConfigError, Settings

SECRET = "super-secret-path-123"


@pytest.fixture
def hook():
    """A real HTTP server on localhost that records what is POSTed to it."""
    received = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/{SECRET}", received
    server.shutdown()


def settings(tmp_path, **kw):
    return Settings(backend="local", tracking_uri=str(tmp_path / "store"), owner="me", capture_env=False,
                    capture_logs=False, system_metrics=False, save_code=False, queue_dir=str(tmp_path / "q"), **kw)


@pytest.fixture(autouse=True)
def _clean():
    at.finish()
    yield
    at.configure(None)
    at.finish()


# ------------------------------------------------------------------ the URL is optional
def test_no_webhook_means_no_alerts_and_no_noise(tmp_path, caplog):
    s = settings(tmp_path)                                         # alert_webhook is None
    at.configure(s)
    with caplog.at_level(logging.WARNING, logger="authtracker"):
        at.init("m", "e", dataset_version="v1", settings=s)
        assert at.alert("anything", "at all", "error") is False
        with pytest.raises(RuntimeError):
            at.finish()
            with at.run("m", "e2", dataset_version="v1", settings=s):
                raise RuntimeError("boom")
    assert [r for r in caplog.records if "alert" in r.getMessage().lower()] == []


# ------------------------------------------------------------------ sending
def test_a_generic_webhook_gets_structured_json(tmp_path, hook):
    url, received = hook
    s = settings(tmp_path, alert_webhook=url)
    at.configure(s)
    at.init("churn", "baseline", dataset_version="v1", settings=s)
    at.start("r1")
    assert at.alert("Loss is NaN", "step 1200", "error") is True
    assert len(received) == 1
    got = received[0]
    assert got["level"] == "error" and got["title"] == "Loss is NaN" and got["message"] == "step 1200"
    assert got["text"].startswith("[ERROR] Loss is NaN\nstep 1200\nmodel: churn")      # the whole message, ready to read
    assert got["model"] == "churn" and got["experiment"] == "baseline" and got["owner"] == "me" and got["run_id"]
    assert "timestamp" in got


def test_the_payload_has_a_readable_text_and_separate_fields_and_never_the_url():
    body = alerts.Alert(alerts.FAILED, "Run failed", "details", "error",
                        {"model": "churn", "run_id": None, "owner": "me"}).payload()
    assert body["text"] == "[ERROR] Run failed\ndetails\nmodel: churn\nowner: me"      # empty fields are left out
    assert (body["level"], body["title"], body["message"], body["model"]) == ("error", "Run failed", "details", "churn")
    assert "run_id" not in body and "timestamp" in body
    assert SECRET not in json.dumps(body)


def test_only_the_chosen_events_are_sent(tmp_path, hook):
    url, received = hook
    at.configure(settings(tmp_path, alert_webhook=url))                          # default: failed, stopped, custom
    assert at.alert("mine", level="info") is True
    at.configure(settings(tmp_path, alert_webhook=url, alert_on=("failed",)))     # custom is not chosen
    assert at.alert("mine again") is False
    at.configure(settings(tmp_path, alert_webhook=url, alert_on=()))              # nothing chosen: nothing sent
    assert at.alert("and again") is False
    assert [m["title"] for m in received] == ["mine"]


def test_a_typo_in_an_event_or_level_fails_at_once(tmp_path):
    with pytest.raises(ValueError, match="unknown alert event"):
        alerts.Alert("nope", "x")
    with pytest.raises(ValueError, match="unknown alert level"):
        alerts.Alert(alerts.CUSTOM, "x", level="loud")
    with pytest.raises(ValueError, match="unknown alert level"):
        at.alert("x", level="loud")                                # fails even with no webhook, so a typo is found early


# ------------------------------------------------------------------ runs alert by themselves
def test_a_crash_sends_exactly_one_alert_and_the_run_is_still_failed(tmp_path, hook):
    url, received = hook
    s = settings(tmp_path, alert_webhook=url)
    with pytest.raises(ValueError):
        with at.run("churn", "baseline", dataset_version="v1", settings=s) as parent:
            with parent.child("trial") as child:                   # a crash inside a nested run
                raise ValueError("loss is NaN\nsecond line that stays out\n")
    assert len(received) == 1, "one alert for the experiment, not one per nested run"
    got = received[0]
    assert got["title"] == "Run failed" and got["level"] == "error"
    assert got["error"] == "ValueError: loss is NaN"               # type and first line only
    assert got["model"] == "churn" and got["experiment"] == "baseline"


def test_a_finished_run_alerts_only_when_asked(tmp_path, hook):
    url, received = hook
    with at.run("m", "quiet", dataset_version="v1", settings=settings(tmp_path, alert_webhook=url)):
        pass
    assert received == []                                                       # "finished" is not chosen by default
    with at.run("m", "loud", dataset_version="v1",
                settings=settings(tmp_path, alert_webhook=url, alert_on=("finished",))):
        pass
    assert [m["title"] for m in received] == ["Run finished"] and received[0]["took"].endswith("s")


def test_a_failure_is_not_sent_when_failed_is_not_chosen(tmp_path, hook):
    url, received = hook
    with pytest.raises(ValueError):
        with at.run("m", "e", dataset_version="v1", settings=settings(tmp_path, alert_webhook=url, alert_on=("finished",))):
            raise ValueError("boom")
    assert received == []


def test_ctrl_c_is_a_warning_not_an_error(tmp_path, hook):
    url, received = hook
    with pytest.raises(KeyboardInterrupt):
        with at.run("m", "e", dataset_version="v1", settings=settings(tmp_path, alert_webhook=url)):
            raise KeyboardInterrupt()
    assert received[0]["title"] == "Run stopped" and received[0]["level"] == "warn"


# ------------------------------------------------------------------ a broken webhook never hurts the run
def test_an_unreachable_webhook_never_raises_and_never_prints_the_url(tmp_path, caplog):
    url = f"http://127.0.0.1:9/{SECRET}"                           # nothing listens on port 9
    s = settings(tmp_path, alert_webhook=url)
    at.configure(s)
    with caplog.at_level(logging.WARNING, logger="authtracker"):
        assert at.alert("hello") is False
        with pytest.raises(RuntimeError, match="real error"):      # the run's own error is what surfaces
            with at.run("m", "e", dataset_version="v1", settings=s):
                raise RuntimeError("real error")
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "could not send the alert" in text
    assert SECRET not in text and "127.0.0.1" not in text


# ------------------------------------------------------------------ settings and secrecy
def test_the_url_is_masked_wherever_settings_are_shown(tmp_path, monkeypatch, capsys):
    url = "https://hooks.example.com/services/T000/B000/" + SECRET
    s = settings(tmp_path, alert_webhook=url)
    assert s.masked()["alert_webhook"] == "https://hooks.example.com/***"
    monkeypatch.setenv("AUTHTRACKER_ALERT_WEBHOOK", url)
    assert main(["config", "show"]) == 0
    assert SECRET not in capsys.readouterr().out
    assert main(["doctor"]) in (0, 1)
    assert SECRET not in capsys.readouterr().out


def test_bad_values_are_refused_without_echoing_the_secret():
    with pytest.raises(ConfigError) as error:
        Settings.from_env({"AUTHTRACKER_ALERT_WEBHOOK": "file:///etc/passwd?" + SECRET})
    assert SECRET not in str(error.value) and "http(s)" in str(error.value)
    with pytest.raises(ConfigError, match="unknown events .*loud"):
        Settings.from_env({"AUTHTRACKER_ALERT_ON": "failed, loud"})
    assert Settings.from_env({}).alert_webhook is None                           # the URL is optional
    assert Settings.from_env({}).alert_on == ("failed", "stopped", "custom")
    s = Settings.from_env({"AUTHTRACKER_ALERT_WEBHOOK": "https://example.com/x", "AUTHTRACKER_ALERT_ON": "Failed, FINISHED"})
    assert s.alert_on == ("failed", "finished")


def test_one_line_keeps_the_type_and_the_first_line_only():
    assert alerts.one_line(ValueError("a\nb")) == "ValueError: a"
    assert len(alerts.one_line(ValueError("x" * 1000))) == 300


# ------------------------------------------------------------------ the CLI
def test_cli_test_alert(hook, monkeypatch, capsys):
    url, received = hook
    monkeypatch.delenv("AUTHTRACKER_ALERT_WEBHOOK", raising=False)
    assert main(["test-alert"]) == 1
    assert "not set" in capsys.readouterr().out
    monkeypatch.setenv("AUTHTRACKER_ALERT_WEBHOOK", url)
    monkeypatch.setenv("AUTHTRACKER_ALERT_ON", "failed")           # `test-alert` sends even when "custom" is not chosen
    assert main(["test-alert", "hello team"]) == 0
    assert capsys.readouterr().out.strip() == "sent"
    assert received[0]["title"] == "hello team" and received[0]["level"] == "info"
