import contextlib
import importlib
import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

import authtracker as at  # noqa: E402
from authtracker import bundle  # noqa: E402
from authtracker.settings import Settings  # noqa: E402

ARCH = '''
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.body(x)
'''
X = torch.randn(5, 4, generator=torch.Generator().manual_seed(1))


@pytest.fixture
def make_net(tmp_path):
    added = []

    def _make(hidden=8, module="architecture", text=ARCH):
        folder = tmp_path / f"src_{module}"
        folder.mkdir(exist_ok=True)
        (folder / f"{module}.py").write_text(text)
        sys.path.insert(0, str(folder))
        added.append((str(folder), module))
        torch.manual_seed(0)
        return importlib.import_module(module).Net(hidden=hidden).eval()

    yield _make
    for folder, module in added:
        sys.path.remove(folder)
        sys.modules.pop(module, None)


def _fresh(code, cwd):
    """A new Python process that has none of your files on its path."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=cwd, env=env,
                          capture_output=True, text=True)


# ---------------------------------------------------------------- extras: plain attributes that are not weights
def test_extras_are_put_back_after_loading(make_net, tmp_path):
    net = make_net()
    out = tmp_path / "b"
    manifest = bundle.build(net, out, init_args={"hidden": 8}, extras={"class_names": ["real", "fake"], "threshold": 0.3})
    assert manifest["extras"] == {"class_names": ["real", "fake"], "threshold": 0.3}
    loaded = bundle.load_dir(out)
    assert loaded.class_names == ["real", "fake"] and loaded.threshold == 0.3
    plain = tmp_path / "plain"
    bundle.build(net, plain, init_args={"hidden": 8})
    assert not hasattr(bundle.load_dir(plain), "class_names")           # without extras they are not stored


def test_extras_mistakes_are_explained(make_net, tmp_path):
    net = make_net()
    for bad, match in [({"x": object()}, "JSON-friendly"), ({"body": 1}, "already used"),
                       ({"forward": 1}, "already used"), ({"not valid": 1}, "valid attribute name")]:
        with pytest.raises(at.ValidationError, match=match):
            bundle.build(net, tmp_path / "b", init_args={"hidden": 8}, extras=bad)


def test_extras_survive_fine_tuning_and_follow_changes(make_net, tmp_path):
    net = make_net()
    first = tmp_path / "v1"
    bundle.build(net, first, init_args={"hidden": 8}, extras={"threshold": 0.3, "labels": ["a", "b"]})
    loaded = bundle.load_dir(first)
    loaded.threshold = 0.9                                              # tuned later
    second = tmp_path / "v2"
    bundle.build(loaded, second)                                        # nothing passed: it remembers
    again = bundle.load_dir(second)
    assert again.threshold == 0.9 and again.labels == ["a", "b"]
    third = tmp_path / "v3"
    bundle.build(loaded, third, extras={"threshold": 0.1})              # and an explicit value wins
    assert bundle.load_dir(third).threshold == 0.1


# ---------------------------------------------------------------- the folder bundle plus model.pkl
def test_pickle_true_adds_the_exact_object_next_to_the_bundle(make_net, tmp_path):
    net = make_net()
    net.class_names, net.threshold = ["real", "fake"], 0.7              # set after creation, not in extras
    out = tmp_path / "b"
    manifest = bundle.build(net, out, init_args={"hidden": 8}, pickle=True, check=False)
    assert (out / "model.pkl").is_file() and manifest["pickle"] == "model.pkl"
    assert "model.pkl" in manifest["files"]                              # and it is hash-checked
    rebuilt = bundle.load_dir(out)                                      # the default way: architecture + weights
    assert not hasattr(rebuilt, "class_names") and torch.allclose(rebuilt(X), net(X), atol=1e-6)
    exact = bundle.load_dir(out, pickle=True)                           # the exact object
    assert exact.class_names == ["real", "fake"] and exact.threshold == 0.7
    assert type(exact).__name__ == "Net" and torch.allclose(exact(X), net(X), atol=1e-6)


def test_asking_for_a_pickle_that_was_not_saved_says_so(make_net, tmp_path):
    net = make_net()
    out = tmp_path / "b"
    bundle.build(net, out, init_args={"hidden": 8})
    with pytest.raises(at.ValidationError, match="saved without pickle=True"):
        bundle.load_dir(out, pickle=True)


# ---------------------------------------------------------------- pickle only
def test_pickle_only_stores_just_the_object_and_needs_opt_in(make_net, tmp_path):
    net = make_net()
    net.add_module("head", torch.nn.Linear(2, 3))                       # changed after creation: no class can rebuild this
    net.class_names = ["real", "fake"]
    out = tmp_path / "b"
    manifest = bundle.build_pickle(net, out, check=False)
    assert {p.name for p in out.iterdir()} == {"model.pkl", "requirements.txt", "environment.txt", "bundle.json"}
    assert manifest["kind"] == "pickle" and "torch" in (out / "requirements.txt").read_text()
    with pytest.raises(at.ValidationError, match="pickle=True"):
        bundle.load_dir(out)                                            # opt-in on this end too
    loaded = bundle.load_dir(out, pickle=True)
    assert loaded.class_names == ["real", "fake"] and torch.equal(loaded.head.weight, net.head.weight)
    assert torch.allclose(loaded(X), net(X), atol=1e-6)


def test_pickle_only_rejects_options_that_belong_to_the_folder_bundle(make_net, tmp_path):
    with pytest.raises(at.ValidationError, match="not used"):
        bundle.build_pickle(make_net(), tmp_path / "b", init_args={"hidden": 8})


def test_wrappers_are_removed_for_a_pickle_too(make_net, tmp_path):
    net = make_net()
    out = tmp_path / "b"
    bundle.build_pickle(torch.nn.DataParallel(net), out, check=False)
    assert type(bundle.load_dir(out, pickle=True)).__name__ == "Net"


def test_a_pickled_model_can_be_pickled_again(make_net, tmp_path):
    net = make_net()
    net.tag = "v1"
    first, second = tmp_path / "v1", tmp_path / "v2"
    bundle.build_pickle(net, first, check=False)
    loaded = bundle.load_dir(first, pickle=True)
    loaded.tag = "v2"
    bundle.build_pickle(loaded, second, check=False)
    assert bundle.load_dir(second, pickle=True).tag == "v2"


# ---------------------------------------------------------------- a machine that has none of your code
def _expected(tmp_path, net):
    np.save(tmp_path / "x.npy", X.numpy())
    np.save(tmp_path / "expected.npy", net(X).detach().numpy())


LOAD_ELSEWHERE = """
    import numpy as np, torch, authtracker as at
    try:
        import architecture
        print("PROBLEM: your module was importable here")
    except ImportError:
        pass
    net = at.load_bundle({path!r}, pickle=True)
    got = net(torch.tensor(np.load({x!r}))).detach().numpy()
    print(type(net).__name__, getattr(net, "class_names", None),
          "SAME" if np.allclose(got, np.load({expected!r}), atol=1e-6) else "DIFFERENT")
"""


def test_pickle_only_loads_where_none_of_your_files_exist(make_net, tmp_path):
    net = make_net()
    net.class_names = ["real", "fake"]
    out = tmp_path / "b"
    bundle.build_pickle(net, out)                                       # check=True: tests it the same way itself
    _expected(tmp_path, net)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    result = _fresh(LOAD_ELSEWHERE.format(path=str(out), x=str(tmp_path / "x.npy"),
                                          expected=str(tmp_path / "expected.npy")), elsewhere)
    assert result.returncode == 0, result.stderr
    assert "PROBLEM" not in result.stdout and "Net ['real', 'fake'] SAME" in result.stdout


def test_bundle_plus_pickle_loads_elsewhere_using_the_architecture_file_it_ships(make_net, tmp_path):
    net = make_net()
    net.class_names = ["real", "fake"]
    out = tmp_path / "b"
    bundle.build(net, out, init_args={"hidden": 8}, pickle=True)        # checked in a clean process as well
    _expected(tmp_path, net)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    result = _fresh(LOAD_ELSEWHERE.format(path=str(out), x=str(tmp_path / "x.npy"),
                                          expected=str(tmp_path / "expected.npy")), elsewhere)
    assert result.returncode == 0, result.stderr
    assert "PROBLEM" not in result.stdout and "Net ['real', 'fake'] SAME" in result.stdout


def test_a_class_defined_in_the_training_script_is_pickled_by_value(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(textwrap.dedent('''
        import sys, numpy as np, torch, torch.nn as nn
        from authtracker import bundle

        class ScriptNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2)
            def forward(self, x):
                return self.fc(x)

        net = ScriptNet().eval()
        net.class_names = ["real", "fake"]
        bundle.build_pickle(net, sys.argv[1])
        x = torch.randn(5, 4)
        np.save(sys.argv[1] + "/../x.npy", x.numpy()); np.save(sys.argv[1] + "/../expected.npy", net(x).detach().numpy())
    '''))
    out = tmp_path / "b"
    run = subprocess.run([sys.executable, str(script), str(out)], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    result = _fresh(LOAD_ELSEWHERE.format(path=str(out), x=str(tmp_path / "x.npy"),
                                          expected=str(tmp_path / "expected.npy")), elsewhere)
    assert result.returncode == 0, result.stderr
    assert "ScriptNet ['real', 'fake'] SAME" in result.stdout


def test_the_save_time_check_catches_a_pickle_that_would_not_load_elsewhere(make_net, tmp_path, monkeypatch):
    net = make_net()

    @contextlib.contextmanager
    def by_name_only():                                                 # the failure the check exists for
        import cloudpickle
        yield cloudpickle

    monkeypatch.setattr(bundle, "_pickle_user_code_by_value", by_name_only)
    with pytest.raises(at.ValidationError, match="clean process"):
        bundle.build_pickle(net, tmp_path / "b")


# ---------------------------------------------------------------- tampering and versions
MARKER_NAME = "was_executed.txt"


class _Evil:
    def __reduce__(self):                                               # what a malicious pickle would do on load
        return (os.system, (f"touch {MARKER_NAME}",))


def test_a_tampered_pickle_is_refused_before_it_is_loaded(make_net, tmp_path, monkeypatch):
    import pickle
    net = make_net()
    out = tmp_path / "b"
    bundle.build_pickle(net, out, check=False)
    (out / "model.pkl").write_bytes(pickle.dumps(_Evil()))              # swapped after saving
    monkeypatch.chdir(tmp_path)
    with pytest.raises(at.ValidationError, match="hash mismatch"):
        bundle.load_dir(out, pickle=True)
    assert not (tmp_path / MARKER_NAME).exists()                        # nothing ran


def test_a_different_python_or_torch_version_gives_a_warning(make_net, tmp_path):
    net = make_net()
    out = tmp_path / "b"
    bundle.build_pickle(net, out, check=False)
    manifest = json.loads((out / "bundle.json").read_text())
    manifest["environment"].update(python="3.8.1", torch="1.10.0")
    (out / "bundle.json").write_text(json.dumps(manifest))
    messages = []

    class Collect(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = Collect(level=logging.WARNING)
    logging.getLogger("authtracker").addHandler(handler)
    try:
        bundle.load_dir(out, pickle=True)
    finally:
        logging.getLogger("authtracker").removeHandler(handler)
    assert any("Python 3.8.1" in m and "torch 1.10.0" in m for m in messages), messages


def test_a_model_on_a_gpu_is_pickled_from_cpu_and_stays_put(make_net, tmp_path):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else None)
    if device is None:
        pytest.skip("needs a GPU (cuda or mps)")
    net = make_net().to(device)
    out = tmp_path / "b"
    bundle.build_pickle(net, out, check=False)
    assert next(net.parameters()).device.type == device
    assert next(bundle.load_dir(out, pickle=True).parameters()).device.type == "cpu"


# ---------------------------------------------------------------- where the code lives
def test_the_manifest_says_where_the_code_came_from_without_secrets(make_net, tmp_path, monkeypatch):
    repo = tmp_path / "project"
    repo.mkdir()
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    for args in (["init", "-q", "-b", "main"], ["remote", "add", "origin", "https://user:ghp_secret123@github.com/org/repo.git"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=env)
    (repo / "a.txt").write_text("x")
    for args in (["add", "."], ["commit", "-q", "-m", "first"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=env)
    monkeypatch.chdir(repo)
    net = make_net()
    out = tmp_path / "b"
    bundle.build_pickle(net, out, check=False)
    git = json.loads((out / "bundle.json").read_text())["git"]
    assert git["remote"] == "https://github.com/org/repo.git" and git["branch"] == "main"
    assert git["url"] == f"https://github.com/org/repo/commit/{git['sha']}"
    assert "ghp_secret123" not in (out / "bundle.json").read_text()      # the token never reaches the manifest
    plain = tmp_path / "plain"
    monkeypatch.chdir(tmp_path)
    # the script that is running (pytest) may itself live in a git checkout: point it somewhere that is not
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(tmp_path / "script.py"), raising=False)
    bundle.build_pickle(net, plain, check=False)
    assert json.loads((plain / "bundle.json").read_text())["git"] is None   # not a repository: no guess


# ---------------------------------------------------------------- the whole path
@pytest.fixture(params=["local", "mlflow"])
def s(request, tmp_path):
    if request.param == "mlflow":
        pytest.importorskip("mlflow")
        uri = f"sqlite:///{tmp_path}/m.db"
    else:
        uri = str(tmp_path / "store")
    at.finish()
    yield Settings(backend=request.param, tracking_uri=uri, owner="t", capture_env=False, capture_logs=False,
                   system_metrics=False, queue_dir=str(tmp_path / "q"), retry_wait=0.0)
    at.finish()


def test_save_register_load_with_pickle_through_the_registry(make_net, tmp_path, s):
    net = make_net()
    net.class_names = ["real", "fake"]
    at.init("pk", "e", dataset_version="v1", settings=s)
    at.start("only")
    at.log_model(net, flavor="pickle", check=False)
    v1 = at.register(settings=s)
    at.start("both")
    at.log_model(net, flavor="bundle", init_args={"hidden": 8}, extras={"threshold": 0.3}, pickle=True, check=False)
    v2 = at.register(settings=s)
    at.promote("pk", v1, "pickle-only", settings=s)
    at.promote("pk", v2, "both", settings=s)
    at.finish()

    with pytest.raises(at.ValidationError, match="pickle=True"):
        at.load("pk", "pickle-only", settings=s)                        # a pickle needs your say-so
    exact = at.load("pk", "pickle-only", pickle=True, settings=s)
    assert exact.class_names == ["real", "fake"] and torch.allclose(exact(X), net(X), atol=1e-6)

    rebuilt = at.load("pk", "both", settings=s)                         # the default: architecture + weights + extras
    assert rebuilt.threshold == 0.3 and not hasattr(rebuilt, "class_names")
    exact2 = at.load("pk", "both", pickle=True, settings=s)
    assert exact2.class_names == ["real", "fake"] and torch.allclose(exact2(X), net(X), atol=1e-6)
