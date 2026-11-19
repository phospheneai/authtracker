import json
import os
import subprocess
import sys

import pytest

from authtracker import utils
from authtracker.settings import Settings

GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, env=GIT_ENV).stdout


@pytest.fixture
def repo(tmp_path):
    folder = tmp_path / "project"
    folder.mkdir()
    git(folder, "init", "-q", "-b", "main")
    (folder / "train.py").write_text("lr = 0.001\nprint('train')\n")
    (folder / "util.py").write_text("def f():\n    return 1\n")
    git(folder, "add", ".")
    git(folder, "commit", "-q", "-m", "first")
    return folder


# ---------------------------------------------------------------- the patch
def test_a_clean_repository_has_no_changes(repo):
    assert utils.code_changes(str(repo)) == {"patch": None, "status": "none", "untracked_skipped": 0}


def test_the_patch_rebuilds_the_exact_code_on_a_fresh_checkout(repo, tmp_path):
    (repo / "train.py").write_text("lr = 0.01\nprint('train')\n")                 # edited, not committed
    (repo / "util.py").unlink()                                                    # deleted
    (repo / "new_file.py").write_text("x = 1\n")                                   # new, never added to git
    (repo / "docs").mkdir()
    (repo / "docs" / "notes.txt").write_text("a note\n")
    result = utils.code_changes(str(repo))
    assert result["status"] == "saved"
    for expected in ("-lr = 0.001", "+lr = 0.01", "deleted file", "new_file.py", "docs/notes.txt"):
        assert expected in result["patch"], expected
    # what someone does later: clone, check out the recorded commit, apply the stored patch
    sha = git(repo, "rev-parse", "HEAD").strip()
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "-q", str(repo), str(clone))
    git(clone, "checkout", "-q", sha)
    (tmp_path / "code_changes.patch").write_text(result["patch"])
    git(clone, "apply", str(tmp_path / "code_changes.patch"))
    for relative in ("train.py", "new_file.py", "docs/notes.txt"):
        assert (clone / relative).read_text() == (repo / relative).read_text(), relative
    assert not (clone / "util.py").exists()


def test_large_or_binary_new_files_are_left_out_and_counted(repo):
    (repo / "weights.bin").write_bytes(b"\0\1\2" * 100)                            # binary
    (repo / "big.txt").write_text("x" * (utils.UNTRACKED_MAX_BYTES + 10))          # too big
    (repo / "small.txt").write_text("ok\n")
    result = utils.code_changes(str(repo))
    assert result["untracked_skipped"] == 2 and "small.txt" in result["patch"]
    assert "weights.bin" not in result["patch"] and "big.txt" not in result["patch"]


def test_a_patch_that_is_too_large_is_not_stored(repo, monkeypatch):
    monkeypatch.setattr(utils, "CODE_MAX_BYTES", 100)
    (repo / "train.py").write_text("changed = 1\n" * 50)
    assert utils.code_changes(str(repo))["status"].startswith("skipped: larger than")


@pytest.mark.parametrize("secret, label", [
    ('KEY = "AKIAABCDEFGHIJKLMNOP"', "AWS access key"),
    ('token = "ghp_' + "a" * 36 + '"', "GitHub token"),
    ('api_key = "abcdef1234567890"', "password or secret"),
    ("-----BEGIN RSA PRIVATE KEY-----", "private key"),
])
def test_a_patch_that_looks_like_it_has_a_secret_is_not_stored(repo, secret, label):
    (repo / "train.py").write_text(f"lr = 0.01\n{secret}\n")
    result = utils.code_changes(str(repo))
    assert result["patch"] is None and "contains" in result["status"] and label in result["status"]
    assert secret not in result["status"]                                          # the secret is not repeated anywhere


def test_ordinary_code_is_not_mistaken_for_a_secret():
    for text in ("lr = 0.001", "password_length = 12", "key = value", "token_count = 128", "# secret sauce",
                 "def get_api_key(): return os.environ['API_KEY']"):
        assert utils.find_secret(text) is None, text


# ---------------------------------------------------------------- the script
def test_pytests_own_entry_point_is_not_mistaken_for_your_script():
    assert utils.script_snapshot()["text"] is None            # __main__ is pytest, inside site-packages


RUN = '''
import authtracker as at
at.init("m", "e", dataset_version="v1")
at.start("a")
at.log({"x": 1})
at.finish()
'''


def _run_script(folder, store, extra_env=None):
    env = {**GIT_ENV, "AUTHTRACKER_BACKEND": "local", "AUTHTRACKER_TRACKING_URI": str(store),
           "AUTHTRACKER_QUEUE_DIR": str(store) + "_q", "AUTHTRACKER_CAPTURE_LOGS": "false",
           "AUTHTRACKER_CAPTURE_ENV": "false", **(extra_env or {})}
    return subprocess.run([sys.executable, "train.py"], cwd=folder, env=env, capture_output=True, text=True)


def _parent(store):
    runs = [json.loads(p.read_text()) for p in (store / "runs").glob("*.json")]
    return next(r for r in runs if r["tags"]["authtracker.level"] == "experiment"), store / "artifacts"


def test_a_run_stores_its_script_and_its_uncommitted_changes(repo, tmp_path):
    (repo / "train.py").write_text(RUN)
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "script")
    (repo / "util.py").write_text("def f():\n    return 2\n")                     # an uncommitted change
    store = tmp_path / "store"
    result = _run_script(repo, store)
    assert result.returncode == 0, result.stderr
    run, artifacts = _parent(store)
    files = {p.name for p in (artifacts / run["run_id"]).iterdir()}
    assert {"script_train.py", "code_changes.patch"} <= files
    assert (artifacts / run["run_id"] / "script_train.py").read_text() == RUN       # the script, exactly
    assert "+    return 2" in (artifacts / run["run_id"] / "code_changes.patch").read_text()
    tags = run["tags"]
    assert tags["code_changes"] == "saved" and tags["code_script"] == "saved" and tags["git_script"] == "train.py"


def test_a_clean_repository_stores_the_script_but_no_patch(repo, tmp_path):
    (repo / "train.py").write_text(RUN)
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "script")
    store = tmp_path / "store"
    assert _run_script(repo, store).returncode == 0
    run, artifacts = _parent(store)
    files = {p.name for p in (artifacts / run["run_id"]).iterdir()}
    assert "script_train.py" in files and "code_changes.patch" not in files
    assert run["tags"]["code_changes"] == "none"


def test_a_script_outside_a_repository_is_still_stored(tmp_path):
    folder = tmp_path / "loose"
    folder.mkdir()
    (folder / "train.py").write_text(RUN)
    store = tmp_path / "store"
    assert _run_script(folder, store).returncode == 0
    run, artifacts = _parent(store)
    assert (artifacts / run["run_id"] / "script_train.py").read_text() == RUN
    assert run["tags"]["code_changes"] == "not a git repository" and "git_script" not in run["tags"]


def test_a_script_with_a_secret_is_not_stored_and_says_why(repo, tmp_path):
    (repo / "train.py").write_text(RUN + '\nAWS_KEY = "AKIAABCDEFGHIJKLMNOP"\n')
    store = tmp_path / "store"
    result = _run_script(repo, store)
    assert result.returncode == 0, result.stderr
    run, artifacts = _parent(store)
    assert not (artifacts / run["run_id"] / "script_train.py").exists()
    assert "AWS access key" in run["tags"]["code_script"] and "AKIAABCDEFGH" not in json.dumps(run["tags"])
    assert "was not stored" in result.stderr                                        # and you are told on screen


def test_it_can_be_switched_off(repo, tmp_path):
    (repo / "train.py").write_text(RUN)
    (repo / "util.py").write_text("changed = 1\n")
    store = tmp_path / "store"
    assert _run_script(repo, store, {"AUTHTRACKER_SAVE_CODE": "false"}).returncode == 0
    run, artifacts = _parent(store)
    folder = artifacts / run["run_id"]
    names = os.listdir(folder) if folder.exists() else []               # nothing stored at all is fine
    assert not any(n.startswith(("script_", "code_changes")) for n in names)
    assert "code_changes" not in run["tags"] and "code_script" not in run["tags"]


def test_the_setting_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("AUTHTRACKER_SAVE_CODE", "false")
    assert Settings.from_env().save_code is False
    monkeypatch.delenv("AUTHTRACKER_SAVE_CODE")
    assert Settings.from_env().save_code is True                                    # on by default
