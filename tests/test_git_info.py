import os
import subprocess

import pytest

import authtracker as at
from authtracker import utils
from authtracker.settings import Settings


def _git(cwd, *args):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, env=env)


@pytest.fixture
def repo(tmp_path):
    folder = tmp_path / "project"
    folder.mkdir()
    _git(folder, "init", "-q", "-b", "main")
    (folder / "train.py").write_text("print('hi')\n")
    _git(folder, "add", ".")
    _git(folder, "commit", "-q", "-m", "first")
    return folder


@pytest.mark.parametrize("raw, clean", [
    ("https://user:ghp_secret123@github.com/org/repo.git", "https://github.com/org/repo.git"),
    ("https://ghp_secret123@github.com/org/repo.git", "https://github.com/org/repo.git"),
    ("http://x:y@git.example.com/a/b", "http://git.example.com/a/b"),
    ("git@github.com:org/repo.git", "git@github.com:org/repo.git"),            # an ssh user name is not a secret
    ("https://github.com/org/repo.git", "https://github.com/org/repo.git"),
])
def test_credentials_are_removed_from_a_remote(raw, clean):
    assert utils.clean_remote(raw) == clean
    assert "secret" not in utils.clean_remote(raw)


@pytest.mark.parametrize("remote, web, link", [
    ("git@github.com:org/repo.git", "https://github.com/org/repo", "https://github.com/org/repo/commit/abc"),
    ("https://github.com/org/repo.git", "https://github.com/org/repo", "https://github.com/org/repo/commit/abc"),
    ("https://tok@gitlab.com/g/p", "https://gitlab.com/g/p", "https://gitlab.com/g/p/commit/abc"),
    ("ssh://git@bitbucket.org/team/repo.git", "https://bitbucket.org/team/repo", "https://bitbucket.org/team/repo/commits/abc"),
    ("git@git.mycompany.com:team/repo.git", "https://git.mycompany.com/team/repo", None),    # an unknown host: no guessed link
])
def test_web_and_commit_links(remote, web, link):
    assert utils.web_url(remote) == web
    assert utils.commit_url(remote, "abc") == link


def test_git_info_for_a_repository(repo):
    _git(repo, "remote", "add", "origin", "https://user:ghp_secret123@github.com/org/repo.git")
    info = utils.git_info(str(repo))
    assert len(info["sha"]) == 40 and info["branch"] == "main" and info["dirty"] == "false"
    assert info["remote"] == "https://github.com/org/repo.git"                    # the token is gone
    assert info["url"] == f"https://github.com/org/repo/commit/{info['sha']}"
    (repo / "train.py").write_text("print('changed')\n")
    assert utils.git_info(str(repo))["dirty"] == "true"


def test_a_repository_without_a_remote_still_reports_the_commit(repo):
    info = utils.git_info(str(repo))
    assert info["sha"] and "remote" not in info and "url" not in info


def test_not_a_repository_gives_nothing(tmp_path):
    assert utils.git_info(str(tmp_path)) is None


def test_the_remote_and_commit_link_are_tagged_on_every_run(repo, monkeypatch, tmp_path):
    _git(repo, "remote", "add", "origin", "git@github.com:org/repo.git")
    monkeypatch.chdir(repo)
    s = Settings(backend="local", tracking_uri=str(tmp_path / "store"), owner="t", capture_logs=False,
                 queue_dir=str(tmp_path / "q"), retry_wait=0.0)           # capture_env stays on
    from authtracker.backends import get_backend
    at.finish()
    parent = at.init("m", "e", dataset_version="v1", settings=s)
    child = at.start("a")
    at.finish()
    tags = get_backend(s).get_run(child.id)["tags"]
    assert tags["git_remote"] == "git@github.com:org/repo.git" and tags["git_branch"] == "main"
    assert tags["git_url"] == f"https://github.com/org/repo/commit/{tags['git_sha']}"
    assert get_backend(s).get_run(parent.id)["tags"]["git_sha"] == tags["git_sha"]
