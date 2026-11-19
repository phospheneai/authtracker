import json

import pytest

torch = pytest.importorskip("torch")

import authtracker as at  # noqa: E402
from authtracker.backends import get_backend  # noqa: E402
from authtracker.settings import Settings  # noqa: E402


@pytest.fixture
def s(tmp_path):
    at.configure(None)
    at.finish()
    yield Settings(backend="local", tracking_uri=str(tmp_path / "store"), owner="me", capture_env=False,
                   capture_logs=False, system_metrics=False, save_code=False, queue_dir=str(tmp_path / "q"))
    at.finish()


def _metrics(s, run):
    return get_backend(s)._load(run.id)["metrics"]


def _train(model, steps):
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(steps):
        opt.zero_grad()
        model(torch.randn(4, 3)).sum().backward()
        opt.step()


def test_logs_gradient_norms_every_n_steps(s):
    run = at.init("m1", "e", settings=s)
    model = torch.nn.Linear(3, 2)
    at.watch(model, every=2)
    _train(model, 5)
    at.finish()
    metrics = _metrics(s, run)
    assert {"grad_norm/weight", "grad_norm/bias", "grad_norm/total"} <= set(metrics)
    assert [step for step, _ in metrics["grad_norm/total"]] == [0, 2]    # step 4 is written by the next step
    total = dict(metrics["grad_norm/total"])[0]
    parts = [dict(metrics[f"grad_norm/{n}"])[0] for n in ("weight", "bias")]
    assert total == pytest.approx(sum(p * p for p in parts) ** 0.5)


def test_stop_ends_it_and_writes_what_is_waiting(s):
    run = at.init("m1", "e", settings=s)
    model = torch.nn.Linear(3, 2)
    stop = at.watch(model, every=1)
    _train(model, 2)
    stop()
    _train(model, 2)
    at.finish()
    assert [step for step, _ in _metrics(s, run)["grad_norm/total"]] == [0, 1]


def test_a_wrapped_model_is_watched_by_its_layers(s):
    run = at.init("m1", "e", settings=s)
    model = torch.nn.DataParallel(torch.nn.Linear(3, 2))
    at.watch(model, every=1)
    _train(model, 2)
    at.finish()
    assert "grad_norm/weight" in _metrics(s, run)          # not "module.weight"


def test_training_goes_on_after_the_run_is_closed(s):
    at.init("m1", "e", settings=s)
    model = torch.nn.Linear(3, 2)
    at.watch(model, every=1)
    _train(model, 1)
    at.finish()
    _train(model, 2)                                        # must not raise


def test_needs_an_open_run_and_a_sensible_every(s):
    with pytest.raises(at.AuthTrackerError):
        at.watch(torch.nn.Linear(3, 2))
    at.init("m1", "e", settings=s)
    with pytest.raises(at.ValidationError):
        at.watch(torch.nn.Linear(3, 2), every=0)
