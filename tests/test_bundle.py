import importlib
import json
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
from authtracker.backends.base import Backend  # noqa: E402
from authtracker.settings import Settings  # noqa: E402

pytestmark = pytest.mark.slow          # skipped by the release workflow: pytest -m "not slow"

ARCH = '''
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.body(x)
'''


@pytest.fixture
def make_net(tmp_path):
    """Write an architecture file, import it, and return a trained-looking instance of its class."""
    added = []

    def _make(module="architecture", hidden=8, text=ARCH, folder=None, seed=0, args=None, build=True):
        folder = Path(folder or tmp_path / f"src_{module}_{hidden}")
        folder.mkdir(exist_ok=True)
        (folder / f"{module}.py").write_text(text)
        sys.path.insert(0, str(folder))
        added.append((str(folder), module))
        mod = importlib.import_module(module)
        torch.manual_seed(seed)
        if not build:                                          # the test builds it itself
            return None, folder / f"{module}.py"
        return mod.Net(**({"hidden": hidden} if args is None else args)), folder / f"{module}.py"

    yield _make
    for folder, module in added:
        sys.path.remove(folder)
        sys.modules.pop(module, None)


X = torch.randn(5, 4, generator=torch.Generator().manual_seed(1))


def _fresh_process(code, cwd, env=None):
    """Run Python code in a new process that has none of the original source on its path."""
    full_env = {**os.environ, **(env or {})}
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=cwd, env=full_env,
                          capture_output=True, text=True)


# ---------------------------------------------------------------- the folder
def test_the_folder_has_everything_and_loads_back(make_net, tmp_path):
    net, _ = make_net(hidden=16)
    out = tmp_path / "b"
    out.mkdir()
    manifest = bundle.build(net, out, init_args={"hidden": 16})
    assert {p.name for p in out.iterdir()} >= {"architecture.py", "weights.pth", "inference.py", "requirements.txt",
                                                "environment.txt", "bundle.json"}
    assert manifest["module"] == "architecture" and manifest["class"] == "Net"
    assert manifest["environment"]["torch"] == torch.__version__
    torch.load(out / "weights.pth", weights_only=True)             # a plain state_dict: safe to load
    loaded = bundle.load_dir(out)
    assert type(loaded).__name__ == "Net" and not loaded.training  # the real class, in eval mode
    assert torch.allclose(loaded(X), net.eval()(X), atol=1e-6)


def test_it_loads_in_a_fresh_process_with_none_of_your_code(make_net, tmp_path):
    net, _ = make_net(hidden=16)
    out = tmp_path / "b"
    out.mkdir()
    bundle.build(net, out, init_args={"hidden": 16})
    np.save(tmp_path / "x.npy", X.numpy())
    np.save(tmp_path / "expected.npy", net.eval()(X).detach().numpy())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    result = _fresh_process(f"""
        import numpy as np, torch, authtracker as at
        try:
            import architecture; print("PROBLEM: the original module was importable")
        except ImportError:
            pass
        net = at.load_bundle({str(out)!r})
        out = net(torch.tensor(np.load({str(tmp_path / 'x.npy')!r}))).detach().numpy()
        print("SAME" if np.allclose(out, np.load({str(tmp_path / 'expected.npy')!r}), atol=1e-6) else "DIFFERENT")
    """, elsewhere)
    assert result.returncode == 0, result.stderr
    assert "SAME" in result.stdout and "PROBLEM" not in result.stdout


def test_the_sample_inference_script_works_without_authtracker(make_net, tmp_path):
    net, _ = make_net(hidden=16)
    out = tmp_path / "b"
    out.mkdir()
    bundle.build(net, out, init_args={"hidden": 16})
    np.save(tmp_path / "x.npy", X.numpy())
    result = _fresh_process(f"""
        import sys, numpy as np
        sys.modules["authtracker"] = None                       # authtracker must not be needed
        sys.path.insert(0, {str(out)!r})
        import inference
        print(json.dumps(inference.predict(np.load({str(tmp_path / 'x.npy')!r})).tolist()) if False else
              inference.predict(np.load({str(tmp_path / 'x.npy')!r})).tolist())
    """, tmp_path)
    assert result.returncode == 0, result.stderr
    assert np.allclose(np.array(eval(result.stdout.strip())), net.eval()(X).detach().numpy(), atol=1e-5)


def test_a_custom_inference_script_and_requirements_are_used(make_net, tmp_path):
    net, _ = make_net()
    script = tmp_path / "my_infer.py"
    script.write_text("# mine\n")
    out = tmp_path / "b"
    out.mkdir()
    bundle.build(net, out, inference=script, requirements=["torch==9.9.9", "numpy"])
    assert (out / "inference.py").read_text() == "# mine\n"
    assert (out / "requirements.txt").read_text().split() == ["torch==9.9.9", "numpy"]


def test_requirements_pin_what_the_code_imports(make_net, tmp_path):
    text = ARCH + "\nimport numpy as np\nimport json\nfrom helpers import scale\n"
    folder = tmp_path / "src"
    folder.mkdir()
    (folder / "helpers.py").write_text("def scale(x):\n    return x\n")
    net, arch = make_net(text=text, folder=folder)
    out = tmp_path / "b"
    out.mkdir()
    bundle.build(net, out, code=[arch, folder / "helpers.py"])
    lines = (out / "requirements.txt").read_text().split()
    names = {line.split("==")[0].lower() for line in lines}
    assert {"torch", "numpy"} <= names and "helpers" not in names and "json" not in names
    assert f"torch=={torch.__version__.split('+')[0]}".lower() in {line.lower() for line in lines} or \
        any(line.startswith("torch==") for line in lines)


# ---------------------------------------------------------------- mistakes are explained
def test_a_class_defined_in_the_running_script_asks_for_its_own_file(make_net, tmp_path):
    net, arch = make_net()
    type(net).__module__ = "__main__"                          # what a class defined in train.py reports
    out = tmp_path / "b"
    out.mkdir()
    with pytest.raises(at.ValidationError, match="defined in the script you are running"):
        bundle.build(net, out)
    bundle.build(net, out, code=arch)                          # ... naming the file that holds it works
    assert bundle.read_manifest(out)["module"] == "architecture"


def test_constructor_arguments_must_match_the_weights(make_net, tmp_path):
    net, _ = make_net(hidden=16)
    out = tmp_path / "b"
    out.mkdir()
    with pytest.raises(at.ValidationError, match="do not fit"):
        bundle.build(net, out)                                 # forgot init_args: hidden defaults to 8
    with pytest.raises(at.ValidationError, match="JSON-friendly"):
        bundle.build(net, out, init_args={"hidden": object()})


def test_a_changed_bundle_is_refused(make_net, tmp_path):
    net, _ = make_net()
    out = tmp_path / "b"
    out.mkdir()
    bundle.build(net, out)
    (out / "architecture.py").write_text((out / "architecture.py").read_text() + "\n# edited\n")
    with pytest.raises(at.ValidationError, match="hash mismatch"):
        bundle.load_dir(out)
    bundle.load_dir(out, verify=False)                         # unless you say you know


def test_only_modules_and_folders_that_are_bundles(tmp_path):
    with pytest.raises(at.ValidationError, match="not an authtracker bundle"):
        bundle.load_dir(tmp_path)
    with pytest.raises(at.ValidationError, match="torch.nn.Module"):
        bundle.build(object(), tmp_path)


def test_two_bundles_with_the_same_file_name_do_not_mix(make_net, tmp_path):
    small, _ = make_net(module="architecture", hidden=4, folder=tmp_path / "one")
    one, two = tmp_path / "b1", tmp_path / "b2"
    one.mkdir(), two.mkdir()
    bundle.build(small, one, init_args={"hidden": 4})
    wide_text = ARCH.replace("nn.Linear(4, hidden)", "nn.Linear(4, hidden * 2)").replace("nn.Linear(hidden, 2)", "nn.Linear(hidden * 2, 2)")
    other_dir = tmp_path / "two_src"
    other_dir.mkdir()
    (other_dir / "architecture.py").write_text(wide_text)
    sys.modules.pop("architecture", None)
    sys.path.insert(0, str(other_dir))
    try:
        wide = importlib.import_module("architecture").Net(hidden=4)
        bundle.build(wide, two, init_args={"hidden": 4})
    finally:
        sys.path.remove(str(other_dir))
        sys.modules.pop("architecture", None)
    a, b = bundle.load_dir(one), bundle.load_dir(two)
    assert a.body[0].out_features == 4 and b.body[0].out_features == 8      # each got its own class
    assert "architecture" not in sys.modules or sys.modules["architecture"].__file__ != str(one / "architecture.py")


def test_your_own_module_in_this_process_is_left_alone(make_net, tmp_path):
    net, _ = make_net(hidden=8)
    mine = sys.modules["architecture"]
    out = tmp_path / "b"
    out.mkdir()
    bundle.build(net, out, init_args={"hidden": 8})
    bundle.load_dir(out)
    assert sys.modules["architecture"] is mine


def test_the_model_stays_on_its_device_and_the_weights_are_saved_from_cpu(make_net, tmp_path):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else None)
    if device is None:
        pytest.skip("needs a GPU (cuda or mps)")
    net, _ = make_net(hidden=8)
    net.to(device)
    out = tmp_path / "b"
    out.mkdir()
    bundle.build(net, out, init_args={"hidden": 8})
    assert next(net.parameters()).device.type == device
    assert all(t.device.type == "cpu" for t in torch.load(out / "weights.pth", weights_only=True).values())
    assert bundle.read_manifest(out)["environment"]["trained_on"] == device


# ---------------------------------------------------------------- save, register, load
class _NoBundles(Backend):                                 # a backend that only implements the basics
    def ping(self): ...
    def start_run(self, *a): return "r"
    def end_run(self, *a): ...
    def set_tags(self, *a): ...
    def log_params(self, *a): ...
    def log_metrics(self, *a): ...
    def log_artifact(self, *a): ...
    def get_run(self, *a): return {}
    def search_runs(self, *a): return []


def test_a_backend_without_bundle_support_says_so(make_net, tmp_path):
    net, _ = make_net()
    with pytest.raises(at.BackendError, match="does not support saving model bundles"):
        _NoBundles().log_model_dir("r", str(tmp_path))


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


def test_save_register_promote_load_gives_back_the_real_class(make_net, tmp_path, s):
    net, _ = make_net(hidden=16)
    at.init("tiny", "bundle-check", dataset_version="v1", settings=s)
    at.start("a")
    at.log_model(net, flavor="bundle", init_args={"hidden": 16})
    v = at.register(settings=s)
    at.promote("tiny", v, "production", settings=s)
    at.finish()
    loaded = at.load("tiny", "production", settings=s)
    assert type(loaded).__name__ == "Net" and isinstance(loaded, torch.nn.Module)
    assert torch.allclose(loaded(X), net.eval()(X), atol=1e-6)
    at.promote("tiny", v, "staging", settings=s)
    assert torch.allclose(at.load("tiny", "staging", device="cpu", settings=s)(X), net(X), atol=1e-6)


def test_a_loaded_bundle_in_a_fresh_process_by_alias(make_net, tmp_path, s):
    if s.backend == "local":
        pytest.skip("same code path as above; the point here is the registry")
    net, _ = make_net(hidden=16)
    at.init("tiny", "bundle-check", dataset_version="v1", settings=s)
    at.start("a")
    at.log_model(net, flavor="bundle", init_args={"hidden": 16})
    v = at.register(settings=s)
    at.promote("tiny", v, "production", settings=s)
    at.finish()
    np.save(tmp_path / "x.npy", X.numpy())
    np.save(tmp_path / "expected.npy", net.eval()(X).detach().numpy())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    result = _fresh_process(f"""
        import numpy as np, torch, authtracker as at
        net = at.load("tiny", "production")
        out = net(torch.tensor(np.load({str(tmp_path / 'x.npy')!r}))).detach().numpy()
        print("SAME" if np.allclose(out, np.load({str(tmp_path / 'expected.npy')!r}), atol=1e-6) else "DIFFERENT")
    """, elsewhere, env={"AUTHTRACKER_BACKEND": "mlflow", "AUTHTRACKER_TRACKING_URI": s.tracking_uri,
                         "AUTHTRACKER_QUEUE_DIR": str(tmp_path / "q2"), "AUTHTRACKER_SYSTEM_METRICS": "false"})
    assert result.returncode == 0, result.stderr
    assert "SAME" in result.stdout


def test_other_models_still_load_as_before(tmp_path, s):
    if s.backend == "local":
        pytest.skip("MLflow model formats only")
    from sklearn.linear_model import LogisticRegression
    at.init("plain", "e", dataset_version="v1", settings=s)
    at.start("a")
    at.log_model(LogisticRegression().fit([[0], [1]], [0, 1]))
    v = at.register(settings=s)
    at.promote("plain", v, "production", settings=s)
    at.finish()
    assert at.load("plain", "production", settings=s).predict([[1]]).tolist() == [1]      # an MLflow model, as before


# ---------------------------------------------------------------- load, fine-tune, save again
def test_a_loaded_model_can_be_fine_tuned_and_saved_again(make_net, tmp_path):
    net, _ = make_net(hidden=16)
    first = tmp_path / "v1"
    first.mkdir()
    bundle.build(net, first, init_args={"hidden": 16})
    loaded = bundle.load_dir(first)
    import shutil
    shutil.rmtree(first)                                         # the folder it came from is gone
    loaded.train()
    opt = torch.optim.SGD(loaded.parameters(), lr=0.5)
    loaded(X).sum().backward()
    opt.step()                                                   # "fine-tuning": the weights change
    second = tmp_path / "v2"
    second.mkdir()
    bundle.build(loaded, second)                                 # no init_args, no code: it remembers them
    again = bundle.load_dir(second)
    assert torch.allclose(again(X), loaded.eval()(X), atol=1e-6)
    assert not torch.allclose(again(X), net.eval()(X), atol=1e-4)           # and it is the new weights
    assert bundle.read_manifest(second)["init_args"] == {"hidden": 16}
    assert (second / "architecture.py").read_text() == (tmp_path / "src_architecture_16" / "architecture.py").read_text()


def test_registry_round_trip_for_a_second_version(make_net, tmp_path):
    pytest.importorskip("mlflow")
    s = Settings(backend="mlflow", tracking_uri=f"sqlite:///{tmp_path}/m.db", owner="t", capture_env=False,
                 capture_logs=False, system_metrics=False, queue_dir=str(tmp_path / "q"), retry_wait=0.0)
    net, _ = make_net(hidden=16)
    at.finish()
    at.init("tiny", "ft", dataset_version="v1", settings=s)
    at.start("first")
    at.log_model(net, flavor="bundle", init_args={"hidden": 16})
    v1 = at.register(settings=s)
    at.promote("tiny", v1, "production", settings=s)
    model = at.load("tiny", "production", settings=s)                    # the registry's temp copy is gone now
    model.train()
    torch.optim.SGD(model.parameters(), lr=0.5).step() if False else None
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1)                                                   # a "fine-tuned" model
    at.start("fine-tuned")
    at.log_model(model, flavor="bundle")                                       # nothing to pass: it carries its code
    v2 = at.register(settings=s)
    at.promote("tiny", v2, "production", settings=s)
    at.finish()
    assert (v1, v2) == ("1", "2")
    latest = at.load("tiny", "production", settings=s)
    assert torch.allclose(latest(X), model.eval()(X), atol=1e-6)
    assert not torch.allclose(latest(X), net.eval()(X), atol=1e-4)


# ---------------------------------------------------------------- the class is in the training script itself
SCRIPT = '''
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import authtracker as at
from authtracker import bundle

SCALE = 0.5                                    # a plain constant the class uses
DEVICE = torch.device("cpu")                   # NOT a plain value; the class below does not use it
x = torch.randn(3, 4)                          # a global called x ... while forward() has its own x


def double(t):                                 # a helper function the class uses
    return t * 2


class Block(nn.Module):                        # a helper class the model uses
    def __init__(self, n):
        super().__init__()
        self.fc = nn.Linear(n, n)

    def forward(self, x):
        return F.relu(double(self.fc(x))) * SCALE


class Net(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.inp = nn.Linear(4, hidden)
        self.block = Block(hidden)
        self.out = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.out(self.block(self.inp(x)))


print("TRAINING HAPPENS HERE")                 # training code at the top level of the script
torch.manual_seed(0)
net = Net(hidden=16)
out_dir = sys.argv[1]
bundle.build(net, out_dir, init_args={"hidden": 16})
probe = torch.randn(5, 4, generator=torch.Generator().manual_seed(1))
np.save(out_dir + "/../probe.npy", probe.numpy())
np.save(out_dir + "/../expected.npy", net.eval()(probe).detach().numpy())
'''


def test_a_class_in_the_training_script_is_copied_out_with_what_it_needs(tmp_path):
    (tmp_path / "train.py").write_text(SCRIPT)
    out = tmp_path / "b"
    out.mkdir()
    run = subprocess.run([sys.executable, str(tmp_path / "train.py"), str(out)], cwd=tmp_path,
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    code = (out / "architecture.py").read_text()
    for needed in ("class Net", "class Block", "def double", "SCALE = 0.5", "import torch.nn as nn",
                   "import torch.nn.functional as F"):
        assert needed in code, needed
    for left_out in ("TRAINING HAPPENS HERE", "DEVICE", "OrderedDict", "import numpy", "import authtracker",
                     "manual_seed"):
        assert left_out not in code, left_out                  # the training code stays in the script
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    loaded = _fresh_process(f"""
        import numpy as np, torch, authtracker as at
        net = at.load_bundle({str(out)!r})
        got = net(torch.tensor(np.load({str(tmp_path / 'probe.npy')!r}))).detach().numpy()
        print("SAME" if np.allclose(got, np.load({str(tmp_path / 'expected.npy')!r}), atol=1e-6) else "DIFFERENT")
    """, elsewhere)
    assert loaded.returncode == 0, loaded.stderr
    assert "SAME" in loaded.stdout


def test_a_global_that_cannot_be_copied_is_named(tmp_path):
    script = tmp_path / "train2.py"
    script.write_text(textwrap.dedent('''
        import sys, torch, torch.nn as nn
        from authtracker import bundle

        DEVICE = torch.device("cpu")

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(2, 2).to(DEVICE)

            def forward(self, x):
                return self.fc(x)

        try:
            bundle.build(Net(), sys.argv[1])
        except Exception as e:
            print(type(e).__name__, "|", e)
    '''))
    out = tmp_path / "b"
    out.mkdir()
    run = subprocess.run([sys.executable, str(script), str(out)], cwd=tmp_path, capture_output=True, text=True)
    assert "ValidationError" in run.stdout and "'DEVICE'" in run.stdout, run.stdout + run.stderr
    assert "its own file" in run.stdout                        # and it says what to do instead


# ---------------------------------------------------------------- files the constructor needs (assets)
CONFIG_ARCH = '''
import json
import os

import torch.nn as nn


class Net(nn.Module):
    """Builds itself from a config file, like a Hugging Face model from a config folder."""

    def __init__(self, config_dir):
        super().__init__()
        with open(os.path.join(config_dir, "config.json")) as f:
            size = json.load(f)["hidden"]
        self.fc = nn.Linear(4, size)

    def forward(self, x):
        return self.fc(x)
'''


def _config_project(tmp_path, make_net):
    cfg = tmp_path / "backbone" / "tiny"
    cfg.mkdir(parents=True)
    (cfg / "config.json").write_text(json.dumps({"hidden": 6}))
    net, _ = make_net(text=CONFIG_ARCH, build=False)
    return cfg, net


def test_files_the_constructor_reads_travel_with_the_bundle(make_net, tmp_path):
    cfg, _ = _config_project(tmp_path, make_net)
    module = importlib.import_module("architecture")
    net = module.Net(config_dir=str(cfg))
    out = tmp_path / "b"
    out.mkdir()
    manifest = bundle.build(net, out, init_args={"config_dir": str(cfg)}, assets=[cfg])
    assert manifest["init_args"] == {"config_dir": "$BUNDLE/assets/tiny"}            # no machine path stored
    assert (out / "assets" / "tiny" / "config.json").is_file()
    assert "assets/tiny/config.json" in manifest["files"]                              # and it is hash-checked
    np.save(tmp_path / "x.npy", X.numpy())
    np.save(tmp_path / "expected.npy", net.eval()(X).detach().numpy())
    import shutil
    shutil.rmtree(tmp_path / "backbone")                                               # the original is gone
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    loaded = _fresh_process(f"""
        import numpy as np, torch, authtracker as at
        net = at.load_bundle({str(out)!r})
        got = net(torch.tensor(np.load({str(tmp_path / 'x.npy')!r}))).detach().numpy()
        print("SAME" if np.allclose(got, np.load({str(tmp_path / 'expected.npy')!r}), atol=1e-6) else "DIFFERENT")
    """, elsewhere)
    assert loaded.returncode == 0, loaded.stderr
    assert "SAME" in loaded.stdout
    script = _fresh_process(f"""
        import sys, numpy as np
        sys.path.insert(0, {str(out)!r})
        import inference
        print(inference.predict(np.load({str(tmp_path / 'x.npy')!r})).shape)      # the sample script finds the files too
    """, elsewhere)
    assert script.returncode == 0, script.stderr
    assert "torch.Size([5, 6])" in script.stdout


def test_a_constructor_that_needs_files_says_so_when_they_are_missing(make_net, tmp_path):
    cfg, _ = _config_project(tmp_path, make_net)
    net = importlib.import_module("architecture").Net(config_dir=str(cfg))
    out = tmp_path / "b"
    out.mkdir()
    with pytest.raises(at.ValidationError, match="cannot rebuild"):
        bundle.build(net, out, init_args={"config_dir": str(tmp_path / "nowhere")})    # a path that does not exist


def test_asset_mistakes_are_explained(make_net, tmp_path):
    cfg, _ = _config_project(tmp_path, make_net)
    net = importlib.import_module("architecture").Net(config_dir=str(cfg))
    out = tmp_path / "b"
    out.mkdir()
    with pytest.raises(at.ValidationError, match="does not exist"):
        bundle.build(net, out, init_args={"config_dir": str(cfg)}, assets=[tmp_path / "missing"])
    other = tmp_path / "elsewhere" / "tiny"
    other.mkdir(parents=True)
    with pytest.raises(at.ValidationError, match="two assets are called 'tiny'"):
        bundle.build(net, out, init_args={"config_dir": str(cfg)}, assets=[cfg, other])


def test_paths_inside_nested_arguments_and_inside_assets_are_rewritten(tmp_path):
    from authtracker.bundle import _resolve, _tokenize
    folder = tmp_path / "data" / "tok"
    folder.mkdir(parents=True)
    mapping = {str(folder): "$BUNDLE/assets/tok"}
    args = {"files": [str(folder / "vocab.txt"), "keep-me"], "nested": {"dir": str(folder)}, "n": 3}
    stored = _tokenize(args, mapping)
    assert stored == {"files": ["$BUNDLE/assets/tok/vocab.txt", "keep-me"], "nested": {"dir": "$BUNDLE/assets/tok"}, "n": 3}
    assert _resolve(stored, Path("/x/bundle")) == {"files": ["/x/bundle/assets/tok/vocab.txt", "keep-me"],
                                                   "nested": {"dir": "/x/bundle/assets/tok"}, "n": 3}


# ---------------------------------------------------------------- requirements through `from x import *`
def test_requirements_include_packages_that_only_arrive_through_a_star_import(make_net, tmp_path):
    helper_dir = tmp_path / "helpers"
    helper_dir.mkdir()
    (helper_dir / "starhelper.py").write_text("import torch\nimport torch.nn as nn\nfrom yaml import safe_load\n")
    sys.path.insert(0, str(helper_dir))
    try:
        text = "from starhelper import *\n\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n" \
               "        self.fc = nn.Linear(4, 2)\n\n    def forward(self, x):\n        return self.fc(x)\n"
        net, _ = make_net(text=text, args={})
        out = tmp_path / "b"
        out.mkdir()
        bundle.build(net, out)
    finally:
        sys.path.remove(str(helper_dir))
        sys.modules.pop("starhelper", None)
    names = {line.split("==")[0].lower() for line in (out / "requirements.txt").read_text().split()}
    assert "pyyaml" in names and "torch" in names                      # PyYAML is only named by the helper module
    assert "starhelper" not in names                                   # a plain local file is not a requirement


def test_only_the_matching_files_of_a_big_model_folder_are_copied(make_net, tmp_path):
    cfg, _ = _config_project(tmp_path, make_net)
    (cfg / "model.safetensors").write_bytes(b"x" * 1_000_000)             # stands in for multi-GB weights
    (cfg / "tokenizer").mkdir()
    (cfg / "tokenizer" / "vocab.json").write_text("{}")
    net = importlib.import_module("architecture").Net(config_dir=str(cfg))
    out = tmp_path / "b"
    out.mkdir()
    manifest = bundle.build(net, out, init_args={"config_dir": str(cfg)}, assets={cfg: ["*.json", "tokenizer/*.json"]})
    copied = sorted(p.relative_to(out / "assets" / "tiny").as_posix() for p in (out / "assets").rglob("*") if p.is_file())
    assert copied == ["config.json", "tokenizer/vocab.json"]                # the big file stayed behind
    assert manifest["init_args"] == {"config_dir": "$BUNDLE/assets/tiny"}
    assert bundle.load_dir(out)(X).shape == (5, 6)
    with pytest.raises(at.ValidationError, match="no file"):
        bundle.build(net, tmp_path / "b2", init_args={"config_dir": str(cfg)}, assets={cfg: ["*.nothing"]})


# ---------------------------------------------------------------- models that have no class of yours
BUILDERS = '''
import torch.nn as nn


def build_model(hidden=8):
    return nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, 2))


def build_frozen(hidden=8):
    """A model changed after it is created: a class alone cannot rebuild this."""
    net = nn.Sequential(nn.Linear(4, hidden), nn.ReLU())
    net.add_module("head", nn.Linear(hidden, 3))
    net[0].requires_grad_(False)
    return net
'''


def _builders(make_net):
    make_net(module="builders", text=BUILDERS, build=False)
    return importlib.import_module("builders")


def test_a_model_without_a_class_of_yours_asks_for_a_builder_function(tmp_path):
    seq = torch.nn.Sequential(torch.nn.Linear(4, 2))
    with pytest.raises(at.ValidationError) as e:
        bundle.build(seq, tmp_path / "b")
    msg = str(e.value)
    assert "from a library" in msg and "factory=" in msg and "code='torch'" not in msg


def test_a_builder_function_rebuilds_a_sequential_in_a_fresh_process(make_net, tmp_path):
    builders = _builders(make_net)
    net = builders.build_model(hidden=16).eval()
    out = tmp_path / "b"
    manifest = bundle.build(net, out, factory=builders.build_model, init_args={"hidden": 16})
    assert manifest["class"] == "build_model" and manifest["factory"] is True and manifest["module"] == "builders"
    np.save(tmp_path / "x.npy", X.numpy())
    np.save(tmp_path / "expected.npy", net(X).detach().numpy())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    result = _fresh_process(f"""
        import numpy as np, torch, authtracker as at
        net = at.load_bundle({str(out)!r})
        got = net(torch.tensor(np.load({str(tmp_path / 'x.npy')!r}))).detach().numpy()
        print(type(net).__name__, "SAME" if np.allclose(got, np.load({str(tmp_path / 'expected.npy')!r}), atol=1e-6) else "DIFFERENT")
    """, elsewhere)
    assert result.returncode == 0, result.stderr
    assert "Sequential SAME" in result.stdout


def test_a_model_changed_after_creation_needs_the_builder(make_net, tmp_path):
    builders = _builders(make_net)
    net = builders.build_frozen(hidden=8)
    out = tmp_path / "b"
    with pytest.raises(at.ValidationError, match="factory="):                       # the class alone cannot say how
        bundle.build(net, out, factory=lambda hidden=8: torch.nn.Sequential(torch.nn.Linear(4, hidden)),
                     init_args={"hidden": 8}) if False else bundle.build(net, out)
    bundle.build(net, out, factory=builders.build_frozen, init_args={"hidden": 8})
    loaded = bundle.load_dir(out)
    assert torch.allclose(loaded(X), net.eval()(X), atol=1e-6)
    assert not loaded[0].weight.requires_grad                                         # the freezing came back too


def test_a_builder_defined_in_the_training_script_is_copied_out(tmp_path):
    script = tmp_path / "train3.py"
    script.write_text(textwrap.dedent('''
        import sys, numpy as np, torch, torch.nn as nn
        from authtracker import bundle

        WIDTH = 12

        def build_model(depth=2):
            layers = [nn.Linear(4, WIDTH), nn.ReLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(WIDTH, WIDTH), nn.ReLU()]
            return nn.Sequential(*layers, nn.Linear(WIDTH, 2))

        torch.manual_seed(0)
        net = build_model(3).eval()
        out = sys.argv[1]
        bundle.build(net, out, factory=build_model, init_args={"depth": 3})
        probe = torch.randn(5, 4)
        np.save(out + "/../probe.npy", probe.numpy()); np.save(out + "/../expected.npy", net(probe).detach().numpy())
    '''))
    out = tmp_path / "b"
    run = subprocess.run([sys.executable, str(script), str(out)], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    code = (out / "architecture.py").read_text()
    assert "def build_model" in code and "WIDTH = 12" in code and "import torch.nn as nn" in code
    result = _fresh_process(f"""
        import numpy as np, torch, authtracker as at
        net = at.load_bundle({str(out)!r})
        got = net(torch.tensor(np.load({str(tmp_path / 'probe.npy')!r}))).detach().numpy()
        print("SAME" if np.allclose(got, np.load({str(tmp_path / 'expected.npy')!r}), atol=1e-6) else "DIFFERENT")
    """, tmp_path)
    assert "SAME" in result.stdout, result.stdout + result.stderr


# ---------------------------------------------------------------- wrappers around your model
def test_dataparallel_and_compile_wrappers_are_removed(make_net, tmp_path):
    net, _ = make_net(hidden=8)
    wrapped = torch.nn.DataParallel(net)
    out = tmp_path / "b"
    bundle.build(wrapped, out, init_args={"hidden": 8})
    saved = torch.load(out / "weights.pth", weights_only=True)
    assert not any(k.startswith("module.") for k in saved)                       # no wrapper prefix
    assert torch.allclose(bundle.load_dir(out)(X), net.eval()(X), atol=1e-6)

    class FakeCompiled(torch.nn.Module):                                          # torch.compile keeps the model in _orig_mod
        def __init__(self, inner):
            super().__init__()
            self._orig_mod = inner

    out2 = tmp_path / "b2"
    bundle.build(FakeCompiled(net), out2, init_args={"hidden": 8})
    assert not any(k.startswith("_orig_mod") for k in torch.load(out2 / "weights.pth", weights_only=True))


# ---------------------------------------------------------------- odd state
class _Obj:
    pass


def test_state_that_is_not_plain_tensors_is_caught_when_saving(make_net, tmp_path):
    text = ARCH + '''
    def get_extra_state(self):
        return {"tag": "ok", "n": 3}

    def set_extra_state(self, state):
        self.extra = state
'''
    net, _ = make_net(text=text)                                    # extra state made of plain types: fine
    out = tmp_path / "plain"
    bundle.build(net, out)
    assert bundle.load_dir(out).extra == {"tag": "ok", "n": 3}

    bad_text = ARCH + '''
    def get_extra_state(self):
        return object()

    def set_extra_state(self, state):
        pass
'''
    bad, _ = make_net(module="architecture_bad", text=bad_text)
    with pytest.raises(at.ValidationError, match="not plain tensors"):                  # an arbitrary object is not
        bundle.build(bad, tmp_path / "bad")


# ---------------------------------------------------------------- the whole path with a builder
def test_builder_models_are_registered_loaded_and_fine_tuned_again(make_net, tmp_path):
    pytest.importorskip("mlflow")
    builders = _builders(make_net)
    s = Settings(backend="mlflow", tracking_uri=f"sqlite:///{tmp_path}/m.db", owner="t", capture_env=False,
                 capture_logs=False, system_metrics=False, queue_dir=str(tmp_path / "q"), retry_wait=0.0)
    net = builders.build_model(hidden=16)
    at.finish()
    at.init("seq", "e", dataset_version="v1", settings=s)
    at.start("a")
    at.log_model(net, flavor="bundle", factory=builders.build_model, init_args={"hidden": 16})
    v1 = at.register(settings=s)
    at.promote("seq", v1, "production", settings=s)
    model = at.load("seq", "production", settings=s)
    assert isinstance(model, torch.nn.Sequential) and torch.allclose(model(X), net.eval()(X), atol=1e-6)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1)                                              # "fine-tuned"
    at.start("b")
    at.log_model(model, flavor="bundle")                                  # nothing to pass: builder and arguments are remembered
    v2 = at.register(settings=s)
    at.promote("seq", v2, "production", settings=s)
    at.finish()
    again = at.load("seq", "production", settings=s)
    assert (v1, v2) == ("1", "2") and torch.allclose(again(X), model.eval()(X), atol=1e-6)


# ---------------------------------------------------------------- classes defined in a notebook cell
NOTEBOOK_CELL = '''
import torch.nn as nn
import torch.nn.functional as F

SCALE = 0.5


def double(t):
    return t * 2


class Block(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.fc = nn.Linear(n, n)

    def forward(self, x):
        return F.relu(double(self.fc(x))) * SCALE


class CellNet(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.inp = nn.Linear(4, hidden)
        self.block = Block(hidden)
        self.out = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.out(self.block(self.inp(x)))


def build_cell_seq(hidden=8):
    return nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, 2))
'''


def test_classes_and_builders_defined_in_a_notebook_cell_are_copied_out(tmp_path):
    """A notebook has no script file behind __main__: only the cell text Python keeps in linecache."""
    driver = f'''
import linecache, sys, numpy as np, torch
from authtracker import bundle

source = {NOTEBOOK_CELL!r}
filename = "<ipython-input-1-abc>"                                   # what a notebook cell is called
linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
exec(compile(source, filename, "exec"), vars(sys.modules["__main__"]))     # "run the cell"
import inspect
try:
    inspect.getsource(CellNet)
    print("NOTE: plain inspect could find the class, so this did not reproduce a notebook")
except OSError:
    pass                                                             # exactly the notebook situation

torch.manual_seed(0)
probe = torch.randn(5, 4)
net = CellNet(16).eval()
bundle.build(net, {str(tmp_path / "b_class")!r}, init_args={{"hidden": 16}})
seq = build_cell_seq(8).eval()
bundle.build(seq, {str(tmp_path / "b_seq")!r}, factory=build_cell_seq, init_args={{"hidden": 8}})
np.save({str(tmp_path / "probe.npy")!r}, probe.numpy())
np.save({str(tmp_path / "e_class.npy")!r}, net(probe).detach().numpy())
np.save({str(tmp_path / "e_seq.npy")!r}, seq(probe).detach().numpy())
'''
    run = subprocess.run([sys.executable, "-c", driver], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "NOTE" not in run.stdout
    code = (tmp_path / "b_class" / "architecture.py").read_text()
    for needed in ("class CellNet", "class Block", "def double", "SCALE = 0.5", "import torch.nn.functional as F"):
        assert needed in code, needed
    assert "build_cell_seq" not in code and "probe" not in code               # only what the class needs
    assert "def build_cell_seq" in (tmp_path / "b_seq" / "architecture.py").read_text()
    result = _fresh_process(f"""
        import numpy as np, torch, authtracker as at
        probe = torch.tensor(np.load({str(tmp_path / 'probe.npy')!r}))
        for name, expected in (("b_class", "e_class"), ("b_seq", "e_seq")):
            net = at.load_bundle({str(tmp_path)!r} + "/" + name)
            same = np.allclose(net(probe).detach().numpy(), np.load({str(tmp_path)!r} + "/" + expected + ".npy"), atol=1e-6)
            print(name, type(net).__name__, "SAME" if same else "DIFFERENT")
    """, tmp_path / "b_class")
    assert result.returncode == 0, result.stderr
    assert "b_class CellNet SAME" in result.stdout and "b_seq Sequential SAME" in result.stdout


def test_a_bundle_is_linked_to_its_run_like_any_other_model(make_net, s):
    if s.backend == "local":
        pytest.skip("run-to-model links are MLflow's model section")
    from authtracker.backends import get_backend
    net, _ = make_net(hidden=16)
    at.init("tiny", "link-check", dataset_version="v1", settings=s)
    trial = at.start("a")
    uri = at.log_model(net, flavor="bundle", init_args={"hidden": 16})
    run = get_backend(s).client.get_run(trial.id)
    assert [m.model_id for m in run.outputs.model_outputs] == [uri.split("/", 1)[1]], "the run lists its model"


def test_a_bundle_can_go_to_the_experiment_while_a_trial_is_open(make_net, s):
    net, _ = make_net(hidden=16)
    experiment = at.init("tiny", "on-experiment", dataset_version="v1", settings=s)
    at.start("a")
    at.log_model(net, flavor="bundle", init_args={"hidden": 16}, experiment=True)
    from authtracker.backends import get_backend
    assert get_backend(s).get_run(experiment.id)["tags"].get("authtracker.model_uri")
    version = at.register(experiment, settings=s)
    at.promote("tiny", version, "production", settings=s)
    assert torch.allclose(at.load("tiny", "production", settings=s)(X), net.eval()(X), atol=1e-6)
