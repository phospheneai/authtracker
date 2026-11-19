import pytest

pytest.importorskip("mlflow")

import authtracker as at  # noqa: E402
from authtracker.backends.mlflow_backend import _detect_flavor  # noqa: E402


def _base(module, name="Base"):
    """A stand-in for a library class (reports the library's module name), subclassed below."""
    cls = type(name, (), {})
    cls.__module__ = module
    return cls


def test_your_own_class_is_recognised_through_its_parent():
    class MyNet(_base("torch.nn.modules.module", "Module")):
        pass

    class MyEstimator(_base("sklearn.base", "BaseEstimator")):
        pass

    assert _detect_flavor(MyNet()) == "pytorch"
    assert _detect_flavor(MyEstimator()) == "sklearn"


def test_the_class_closest_to_you_wins():
    sk = _base("sklearn.base", "BaseEstimator")
    xgb = type("XGBClassifier", (sk,), {})
    xgb.__module__ = "xgboost.sklearn"                     # an xgboost model is also a sklearn estimator

    class Mine(xgb):
        pass

    assert _detect_flavor(Mine()) == "xgboost"
    assert _detect_flavor(xgb()) == "xgboost"


def test_unknown_objects_still_say_what_to_do():
    with pytest.raises(at.BackendError, match="pass flavor="):
        _detect_flavor(object())


def test_a_custom_pytorch_model_saves_registers_and_loads(tmp_path):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    nn = torch.nn

    class TinyNet(nn.Module):                              # defined in the test, like your own code
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x)

    torch.manual_seed(0)
    X = torch.randn(32, 4)
    net = TinyNet()
    s = at.Settings(backend="mlflow", tracking_uri=f"sqlite:///{tmp_path}/m.db", owner="t",
                    capture_env=False, capture_logs=False, system_metrics=False,
                    queue_dir=str(tmp_path / "q"), retry_wait=0.0)
    at.finish()
    at.init("tiny", "check", dataset_version="v1", settings=s)
    at.start("a")
    at.save(net, input_example=X[:2].numpy())              # no flavor= needed
    v = at.register(settings=s)
    at.promote("tiny", v, "production", settings=s)
    at.finish()
    out = np.asarray(at.load("tiny", "production", settings=s).predict(X[:8].numpy()))
    assert np.allclose(out, net(X[:8]).detach().numpy(), atol=1e-5)    # the loaded model gives the same numbers


def test_a_model_on_a_gpu_is_saved_from_cpu_weights_and_put_back(tmp_path):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else None)
    if device is None:
        pytest.skip("needs a GPU (cuda or mps)")
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x)

    net = Net().to(device)
    X = torch.randn(8, 4)
    s = at.Settings(backend="mlflow", tracking_uri=f"sqlite:///{tmp_path}/m.db", owner="t",
                    capture_env=False, capture_logs=False, system_metrics=False,
                    queue_dir=str(tmp_path / "q"), retry_wait=0.0)
    at.finish()
    at.init("gpu", "check", dataset_version="v1", settings=s)
    at.start("a")
    at.save(net, input_example=X.numpy())                       # a CPU example, a model on the GPU
    v = at.register(settings=s)
    at.promote("gpu", v, "production", settings=s)
    at.finish()
    assert next(net.parameters()).device.type == device         # the model is back where training left it
    out = np.asarray(at.load("gpu", "production", settings=s).predict(X.numpy()))
    assert np.allclose(out, net(X.to(device)).detach().cpu().numpy(), atol=1e-4)


# ---------------------------------------------------------------- plain PyTorch: clear errors and the fallback
def _torch_setup(tmp_path):
    torch = pytest.importorskip("torch")
    nn = torch.nn
    s = at.Settings(backend="mlflow", tracking_uri=f"sqlite:///{tmp_path}/m.db", owner="t",
                    capture_env=False, capture_logs=False, system_metrics=False,
                    queue_dir=str(tmp_path / "q"), retry_wait=0.0)
    at.finish()
    at.init("torch-errors", "check", dataset_version="v1", settings=s)
    at.start("a")
    return torch, nn, s


def test_saving_a_pytorch_model_without_an_example_says_what_to_do(tmp_path):
    torch, nn, _ = _torch_setup(tmp_path)
    with pytest.raises(at.BackendError) as e:
        at.save(nn.Linear(4, 2))
    msg = str(e.value)
    assert "input_example=" in msg and "serialization_format='pickle'" in msg and "code_paths" in msg


def test_a_model_that_cannot_be_traced_says_what_to_do_and_the_fallback_works(tmp_path):
    torch, nn, s = _torch_setup(tmp_path)
    np = pytest.importorskip("numpy")

    class Branchy(nn.Module):                              # which branch runs depends on the data
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            if x.sum() > 0:
                return self.fc(x)
            return -self.fc(x)

    net, X = Branchy().eval(), torch.randn(4, 4)
    with pytest.raises(at.BackendError) as e:
        at.save(net, input_example=X.numpy())
    assert "could not be traced" in str(e.value) and "serialization_format='pickle'" in str(e.value)
    assert e.value.__cause__ is not None                   # the original error is kept for debugging
    at.save(net, input_example=X.numpy(), serialization_format="pickle")      # the advice works
    v = at.register(settings=s)
    at.promote("torch-errors", v, "production", settings=s)
    at.finish()
    out = np.asarray(at.load("torch-errors", "production", settings=s).predict(X.numpy()))
    assert np.allclose(out, net(X).detach().numpy(), atol=1e-5)


def test_the_real_module_can_be_loaded_back_for_more_training(tmp_path):
    torch, nn, s = _torch_setup(tmp_path)
    import mlflow
    at.save(nn.Linear(4, 2), input_example=torch.randn(2, 4).numpy())
    v = at.register(settings=s)
    at.promote("torch-errors", v, "production", settings=s)
    at.finish()
    mlflow.set_tracking_uri(s.tracking_uri)
    mlflow.set_registry_uri(s.tracking_uri)
    module = mlflow.pytorch.load_model("models:/torch-errors@production")
    params = list(module.parameters())
    assert params and all(p.requires_grad for p in params)                 # a trainable PyTorch module
