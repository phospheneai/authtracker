import dataclasses
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import authtracker as at
from authtracker import utils
from authtracker.backends import get_backend
from authtracker.settings import Settings

DIGEST = "sha256:" + "a" * 64


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in ("KUBERNETES_SERVICE_HOST", "ECS_CONTAINER_METADATA_URI_V4", "CI_JOB_IMAGE", "AUTHTRACKER_IMAGE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(utils, "_exists", lambda path: False)             # not a container unless a test says so
    monkeypatch.setattr(utils, "_docker_digest", lambda ref: None)        # no docker unless a test says so
    utils._warned_image.clear()
    at.configure(None)
    at.finish()
    yield
    at.finish()


class Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def warnings_seen():
    handler, logger = Collect(), logging.getLogger("authtracker")
    logger.addHandler(handler)
    yield handler.messages
    logger.removeHandler(handler)


# ---------------------------------------------------------------- finding and pinning the image
def test_nothing_is_recorded_outside_a_container_without_an_image():
    assert utils.container_info() is None


def test_an_image_with_a_digest_is_pinned(warnings_seen):
    info = utils.container_info(f"registry.io/team/app@{DIGEST}")
    assert info["pinned"] == "true" and info["digest"] == DIGEST and info["image"].endswith(DIGEST)
    assert warnings_seen == []                                             # nothing to warn about


def test_an_image_with_only_a_tag_is_flagged_not_pinned(warnings_seen):
    info = utils.container_info("registry.io:5000/team/app:v3")
    assert info["pinned"] == "false" and "digest" not in info
    assert len(warnings_seen) == 1 and "not pinned" in warnings_seen[0] and "sha256" in warnings_seen[0]
    utils.container_info("registry.io:5000/team/app:v3")                   # the same image again: no second warning
    assert len(warnings_seen) == 1


def test_a_tag_is_pinned_when_docker_can_resolve_it(monkeypatch, warnings_seen):
    monkeypatch.setattr(utils, "_docker_digest", lambda ref: DIGEST)
    info = utils.container_info("registry.io:5000/team/app:v3")
    assert info["pinned"] == "true" and info["digest"] == DIGEST
    assert info["image"] == "registry.io:5000/team/app:v3"                 # what you said is kept
    assert info["image_pinned"] == f"registry.io:5000/team/app@{DIGEST}"   # and the exact reference to use
    assert warnings_seen == []


def test_a_container_with_an_unknown_image_says_how_to_record_it(monkeypatch, warnings_seen):
    monkeypatch.setattr(utils, "_exists", lambda path: path == "/.dockerenv")
    info = utils.container_info()
    assert info["runtime"] == "docker" and info["pinned"] == "false" and "image" not in info
    assert len(warnings_seen) == 1 and "image='registry/name@sha256" in warnings_seen[0]


@pytest.mark.parametrize("marker, runtime", [("/.dockerenv", "docker"), ("/run/.containerenv", "podman")])
def test_container_runtimes_are_detected(monkeypatch, marker, runtime):
    monkeypatch.setattr(utils, "_exists", lambda path: path == marker)
    assert utils.container_runtime() == runtime


def test_kubernetes_is_detected(monkeypatch):
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    assert utils.container_runtime() == "kubernetes"


def test_gitlab_ci_tells_the_image(monkeypatch):
    monkeypatch.setenv("CI_JOB_IMAGE", "python:3.12")
    info = utils.container_info()
    assert info["image"] == "python:3.12" and info["pinned"] == "false"


class _Ecs(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"Image": "123.dkr.ecr.eu-west-1.amazonaws.com/app:v3", "ImageID": DIGEST}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_aws_ecs_tells_the_container_its_image_and_digest(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _Ecs)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("ECS_CONTAINER_METADATA_URI_V4", f"http://127.0.0.1:{server.server_port}/v4/abc")
        info = utils.container_info()
    finally:
        server.shutdown()
    assert info["image"] == "123.dkr.ecr.eu-west-1.amazonaws.com/app:v3"
    assert info["pinned"] == "true" and info["digest"] == DIGEST


def test_an_unreachable_ecs_endpoint_is_ignored(monkeypatch):
    monkeypatch.setenv("ECS_CONTAINER_METADATA_URI_V4", "http://127.0.0.1:1/nothing")
    assert utils.container_info() is None


# ---------------------------------------------------------------- recorded with the run and the model
@pytest.fixture
def s(tmp_path):
    return Settings(backend="local", tracking_uri=str(tmp_path / "store"), owner="t", capture_logs=False,
                    system_metrics=False, save_code=False, queue_dir=str(tmp_path / "q"), retry_wait=0.0)


def test_init_records_the_image_on_every_run(s):
    parent = at.init("m", "e", dataset_version="v1", image=f"registry.io/team/app@{DIGEST}", settings=s)
    child = at.start("a")
    at.finish()
    for run in (parent, child):
        tags = get_backend(s).get_run(run.id)["tags"]
        assert tags["container_image"] == f"registry.io/team/app@{DIGEST}"
        assert tags["container_pinned"] == "true" and tags["container_digest"] == DIGEST


def test_no_container_tags_when_not_in_a_container(s):
    parent = at.init("m", "e", dataset_version="v1", settings=s)
    at.finish()
    assert not [k for k in get_backend(s).get_run(parent.id)["tags"] if k.startswith("container_")]


def test_the_image_can_come_from_the_environment_as_a_default(monkeypatch, s):
    monkeypatch.setenv("AUTHTRACKER_IMAGE", f"registry.io/team/app@{DIGEST}")
    assert Settings.from_env().image == f"registry.io/team/app@{DIGEST}"
    parent = at.init("m", "e", dataset_version="v1", settings=Settings.from_env())
    at.finish()
    assert get_backend(Settings.from_env()).get_run(parent.id)["tags"]["container_pinned"] == "true"


def test_the_saved_model_says_which_image_it_was_made_in(tmp_path, s):
    torch = pytest.importorskip("torch")
    at.init("m", "e", dataset_version="v1", image=f"registry.io/team/app@{DIGEST}", settings=s)
    at.start("a")
    run_id = at.current_run().id
    at.save(torch.nn.Linear(2, 2), flavor="pickle", check=False)
    at.finish()
    manifest = json.loads((get_backend(s).inner._artifacts(run_id) / "model_bundle" / "bundle.json").read_text())
    assert manifest["container"]["digest"] == DIGEST and manifest["container"]["pinned"] == "true"


# ---------------------------------------------------------------- per-run options in init beat the defaults
def _repo(tmp_path):
    import os
    import subprocess
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    folder = tmp_path / "project"
    folder.mkdir()
    for args in (["init", "-q", "-b", "main"],):
        subprocess.run(["git", *args], cwd=folder, check=True, capture_output=True, env=env)
    (folder / "a.py").write_text("x = 1\n")
    for args in (["add", "."], ["commit", "-q", "-m", "first"]):
        subprocess.run(["git", *args], cwd=folder, check=True, capture_output=True, env=env)
    (folder / "a.py").write_text("x = 2\n")                                 # an uncommitted change
    return folder


def test_save_code_in_init_beats_the_setting(tmp_path, monkeypatch, s):
    monkeypatch.chdir(_repo(tmp_path))
    on, off = dataclasses.replace(s, save_code=True), dataclasses.replace(s, save_code=False)

    first = at.init("m", "e", dataset_version="v1", settings=on)                    # the setting says store it
    tags = get_backend(on).get_run(first.id)["tags"]
    assert tags["code_changes"] == "saved"

    second = at.init("m", "e", dataset_version="v1", save_code=False, settings=on)  # init says do not
    assert "code_changes" not in get_backend(on).get_run(second.id)["tags"]

    third = at.init("m", "e", dataset_version="v1", save_code=True, settings=off)   # init says do
    at.finish()
    assert get_backend(off).get_run(third.id)["tags"]["code_changes"] == "saved"


def test_save_code_works_with_the_with_api_too(tmp_path, monkeypatch, s):
    monkeypatch.chdir(_repo(tmp_path))
    with at.run("m", "e", dataset_version="v1", save_code=True, settings=s) as r:
        assert r.tags["code_changes"] == "saved"
    with at.run("m", "e", dataset_version="v1", save_code=False, settings=dataclasses.replace(s, save_code=True)) as r:
        assert "code_changes" not in r.tags
