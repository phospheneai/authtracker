"""Moving a model between stores: export to a folder, import somewhere else."""
import json

import pytest

mlflow = pytest.importorskip("mlflow")
from mlflow.tracking import MlflowClient  # noqa: E402

import authtracker as at  # noqa: E402
from authtracker.backends.mlflow_transfer import IMPORTED_FROM  # noqa: E402
from authtracker.errors import AuthTrackerError, UnsupportedError, ValidationError  # noqa: E402
from authtracker.settings import Settings  # noqa: E402


def store(tmp_path, name):
    return Settings(tracking_uri=f"sqlite:///{tmp_path}/{name}.db", owner="me", capture_env=False,
                    capture_logs=False, system_metrics=False, save_code=False, queue_dir=str(tmp_path / "q"),
                    retry_wait=0.0)


def client(s):
    return MlflowClient(tracking_uri=s.tracking_uri, registry_uri=s.tracking_uri)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """Built ONCE for the module (it is only ever read): model 'churn' with two runs, steps, files, sklearn
    models, two versions and aliases, plus a second model whose name is not a safe folder name."""
    from sklearn.linear_model import LogisticRegression
    s = store(tmp_path_factory.mktemp("source"), "source")
    at.configure(s)
    at.finish()
    at.init("churn", "baseline", dataset_version="v1", config={"base": "x"}, settings=s)
    for name, C in (("weak", 0.001), ("strong", 10.0)):
        at.start(name, config={"C": C})
        clf = LogisticRegression(C=C).fit([[0], [1], [2], [3]], [0, 0, 1, 1])
        for epoch in range(3):
            at.log({"loss": 1.0 / (epoch + 1)})
        at.log({"f1": 0.5 if name == "weak" else 0.9})
        at.save(f"notes for {name}\n", name="notes.txt")
        at.log_model(clf)
        v = at.register()
        at.promote("churn", v, "staging")
    at.finish()
    at.promote("churn", "2", "production")
    at.promote("churn", "1", "production")                    # production -> 1, previous-production -> 2
    at.init("team/fraud v2", "e", dataset_version="v1", settings=s)
    at.start("a")
    at.log({"acc": 0.8})
    at.finish()
    yield s
    at.configure(None)
    at.finish()


@pytest.fixture
def two_models(source):
    return source


def test_round_trip_keeps_runs_metrics_files_versions_and_aliases(source, tmp_path):
    out = tmp_path / "export"
    summary = at.export_model("churn", out, settings=source)
    assert summary["runs"] == 3 and summary["versions"] == 2 and summary["models"] == 2
    assert (out / "manifest.json").is_file()

    dest = store(tmp_path, "dest")
    result = at.import_model(out, settings=dest)
    assert result["model"] == "churn" and result["versions"] == {"1": "1", "2": "2"}

    cs, cd = client(source), client(dest)
    def runs(c):
        exp = c.get_experiment_by_name("churn")
        return {r.data.tags["mlflow.runName"]: r for r in c.search_runs([exp.experiment_id])}
    a, b = runs(cs), runs(cd)
    assert set(a) == set(b) == {"baseline", "weak", "strong"}
    for name in a:
        assert a[name].data.params == b[name].data.params
        assert a[name].info.status == b[name].info.status == "FINISHED"
        assert a[name].info.start_time == b[name].info.start_time
        assert b[name].data.tags[IMPORTED_FROM] == a[name].info.run_id
        assert b[name].info.run_id != a[name].info.run_id
        for key in a[name].data.metrics:
            old = [(m.step, m.value) for m in cs.get_metric_history(a[name].info.run_id, key)]
            new = [(m.step, m.value) for m in cd.get_metric_history(b[name].info.run_id, key)]
            assert old == new, key
    assert [m.step for m in cd.get_metric_history(b["weak"].info.run_id, "loss")] == [0, 1, 2]

    # the three levels survive: the children point at the NEW parent
    parent = b["baseline"].info.run_id
    assert b["weak"].data.tags["mlflow.parentRunId"] == parent == b["strong"].data.tags["mlflow.parentRunId"]
    assert b["baseline"].data.tags["authtracker.level"] == "experiment"

    # files
    path = cd.download_artifacts(b["strong"].info.run_id, "notes.txt", str(tmp_path / "dl"))
    assert open(path).read() == "notes for strong\n"

    # versions, their runs, and aliases (including previous-production)
    versions = {str(v.version): v for v in cd.search_model_versions("name='churn'")}
    assert set(versions) == {"1", "2"}
    assert versions["1"].tags["authtracker.run_id"] == b["weak"].info.run_id
    assert versions["2"].tags["authtracker.run_id"] == b["strong"].info.run_id
    assert cd.get_registered_model("churn").aliases == cs.get_registered_model("churn").aliases


def test_the_imported_model_works_through_the_normal_api(source, tmp_path):
    out = tmp_path / "export"
    at.export_model("churn", out, settings=source)
    dest = store(tmp_path, "dest")
    at.import_model(out, settings=dest)
    at.configure(dest)
    best = at.best_run("churn", "baseline", metric="f1")
    assert best["name"] == "strong" and best["tags"][IMPORTED_FROM]
    assert at.load("churn", "production").predict([[3]]).tolist() == [1]      # production -> version 1 (weak, C=0.001)
    # a new version can be registered on top, so work carries on where it was left
    at.init("churn", "baseline-2", dataset_version="v1", settings=dest)
    at.start("next")
    from sklearn.linear_model import LogisticRegression
    at.log_model(LogisticRegression().fit([[0], [1]], [0, 1]))
    assert at.register() == "3"
    at.finish()


def test_import_under_a_new_name_retags_the_runs(source, tmp_path):
    out = tmp_path / "export"
    at.export_model("churn", out, settings=source)
    dest = store(tmp_path, "dest")
    result = at.import_model(out, "churn-copy", settings=dest)
    assert result["model"] == "churn-copy"
    at.configure(dest)
    best = at.best_run("churn-copy", "baseline", metric="f1")          # searches by the model tag
    assert best["tags"]["model"] == "churn-copy"
    assert at.load("churn-copy", "staging") is not None


def test_importing_twice_is_refused_and_changes_nothing(source, tmp_path):
    out = tmp_path / "export"
    at.export_model("churn", out, settings=source)
    dest = store(tmp_path, "dest")
    at.import_model(out, settings=dest)
    with pytest.raises(ValidationError, match="already has runs"):
        at.import_model(out, settings=dest)
    exp = client(dest).get_experiment_by_name("churn")
    assert len(client(dest).search_runs([exp.experiment_id])) == 3


def test_export_refuses_a_missing_model_and_a_folder_with_files(source, tmp_path):
    with pytest.raises(ValidationError, match="no model 'nope'"):
        at.export_model("nope", tmp_path / "x", settings=source)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "file").write_text("x")
    with pytest.raises(ValidationError, match="not empty"):
        at.export_model("churn", busy, settings=source)


def test_import_refuses_what_is_not_an_export(tmp_path):
    dest = store(tmp_path, "dest")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValidationError, match="manifest.json is missing"):
        at.import_model(tmp_path / "empty", settings=dest)
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "manifest.json").write_text(json.dumps({"format": "something-else"}))
    with pytest.raises(ValidationError, match="not an authtracker export"):
        at.import_model(bad, settings=dest)
    newer = tmp_path / "newer"
    newer.mkdir()
    (newer / "manifest.json").write_text(json.dumps({"format": "authtracker-export", "format_version": 99}))
    with pytest.raises(ValidationError, match="Upgrade authtracker"):
        at.import_model(newer, settings=dest)


def test_a_model_with_runs_but_no_versions_moves_too(tmp_path):
    s = store(tmp_path, "source")
    at.configure(s)
    at.init("plain", "e", dataset_version="v1", settings=s)
    at.start("a")
    at.log({"acc": 0.7})
    at.finish()
    out = tmp_path / "export"
    summary = at.export_model("plain", out, settings=s)
    assert summary["versions"] == 0 and summary["models"] == 0
    dest = store(tmp_path, "dest")
    result = at.import_model(out, settings=dest)
    assert result["runs"] == 2 and result["versions"] == {}
    at.configure(dest)
    assert at.best_run("plain", "e", metric="acc")["metrics"]["acc"] == 0.7
    at.configure(None)


def test_other_backends_say_it_is_not_supported(tmp_path):
    s = Settings(backend="local", tracking_uri=str(tmp_path / "store"), owner="me", queue_dir=str(tmp_path / "q"))
    with pytest.raises(UnsupportedError, match="exporting models"):
        at.export_model("m", tmp_path / "out", settings=s)
    with pytest.raises(AuthTrackerError):
        at.import_model(tmp_path / "out", settings=s)


def test_a_pytorch_bundle_survives_the_trip(tmp_path):
    torch = pytest.importorskip("torch")
    import sys
    import textwrap
    (tmp_path / "arch.py").write_text(textwrap.dedent("""
        import torch.nn as nn
        class Tiny(nn.Module):
            def __init__(self, n=3):
                super().__init__()
                self.fc = nn.Linear(n, 2)
            def forward(self, x):
                return self.fc(x)
    """))
    sys.path.insert(0, str(tmp_path))
    try:
        import arch
        s = store(tmp_path, "source")
        at.configure(s)
        torch.manual_seed(0)
        net = arch.Tiny()
        at.init("tiny", "e", dataset_version="v1", settings=s)
        at.start("r")
        at.log_model(net, flavor="bundle", init_args={"n": 3}, extras={"class_names": ["a", "b"]})
        v = at.register()
        at.promote("tiny", v, "production")
        at.finish()
        out = tmp_path / "export"
        at.export_model("tiny", out, settings=s)
        dest = store(tmp_path, "dest")
        at.import_model(out, settings=dest)
        at.configure(dest)
        loaded = at.load("tiny", "production")
        x = torch.randn(4, 3)
        assert torch.allclose(loaded(x), net.eval()(x), atol=1e-6)
        assert loaded.class_names == ["a", "b"]
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("arch", None)
        at.configure(None)
        at.finish()


def test_cli_export_and_import(source, tmp_path, capsys):
    from authtracker.cli import main
    out = tmp_path / "cli-export"
    assert main(["export", "churn", "--out", str(out), "--tracking-uri", source.tracking_uri]) == 0
    assert "3 runs" in capsys.readouterr().out
    dest = store(tmp_path, "dest")
    assert main(["import", str(out), "--name", "churn-cli", "--tracking-uri", dest.tracking_uri]) == 0
    text = capsys.readouterr().out
    assert "versions (old -> new): 1->1, 2->2" in text and "production=v1" in text
    assert main(["import", str(out), "--name", "churn-cli", "--tracking-uri", dest.tracking_uri]) == 1   # already there
    assert "already has runs" in capsys.readouterr().out
    assert main(["export", "nope", "--out", str(tmp_path / "o2"), "--tracking-uri", source.tracking_uri]) == 1


# ------------------------------------------------------------------ all models at once
def test_export_all_and_import_all_move_every_model(two_models, tmp_path):
    out = tmp_path / "all"
    summary = at.export_all(out, settings=two_models)
    assert summary["count"] == 2 and set(summary["models"]) == {"churn", "team/fraud v2"}
    assert (out / "manifest.json").is_file() and (out / "team%2Ffraud%20v2" / "manifest.json").is_file()

    dest = store(tmp_path, "dest")
    result = at.import_all(out, settings=dest)
    assert result["count"] == 2 and result["failed"] == {}
    assert result["imported"]["churn"]["versions"] == {"1": "1", "2": "2"}
    c = client(dest)
    assert {e.name for e in c.search_experiments()} >= {"churn", "team/fraud v2"}
    at.configure(dest)
    assert at.best_run("team/fraud v2", "e", metric="acc")["metrics"]["acc"] == 0.8
    assert at.load("churn", "staging") is not None
    at.configure(None)


def test_export_all_skips_empty_experiments(tmp_path):
    s = store(tmp_path, "source")
    at.configure(s)
    client(s).create_experiment("nothing-here")                     # an experiment with no runs
    out = tmp_path / "all"
    assert at.export_all(out, settings=s)["count"] == 0
    at.configure(None)


def test_import_all_reports_a_model_that_already_exists_and_imports_the_rest(two_models, tmp_path):
    out = tmp_path / "all"
    at.export_all(out, settings=two_models)
    dest = store(tmp_path, "dest")
    at.import_model(tmp_path / "all" / "churn", settings=dest)       # 'churn' is already there
    result = at.import_all(out, settings=dest)
    assert list(result["imported"]) == ["team/fraud v2"]
    assert "already has runs" in result["failed"]["churn"]
    assert len(client(dest).search_runs([client(dest).get_experiment_by_name("churn").experiment_id])) == 3


def test_one_folder_and_all_folder_are_not_mixed_up(two_models, tmp_path):
    one, many = tmp_path / "one", tmp_path / "many"
    at.export_model("churn", one, settings=two_models)
    at.export_all(many, settings=two_models)
    dest = store(tmp_path, "dest")
    with pytest.raises(ValidationError, match="import_all"):
        at.import_model(many, settings=dest)
    with pytest.raises(ValidationError, match="holds one model"):
        at.import_all(one, settings=dest)


def test_cli_all_flags(two_models, tmp_path, capsys):
    from authtracker.cli import main
    out = tmp_path / "all"
    assert main(["export", "--all", "--out", str(out), "--tracking-uri", two_models.tracking_uri]) == 0
    assert "2 models" in capsys.readouterr().out
    dest = store(tmp_path, "dest")
    assert main(["import", "--all", str(out), "--tracking-uri", dest.tracking_uri]) == 0
    assert "2 imported, 0 failed" in capsys.readouterr().out
    assert main(["import", "--all", str(out), "--tracking-uri", dest.tracking_uri]) == 1      # all exist now
    assert "0 imported, 2 failed" in capsys.readouterr().out
    # argument mistakes are refused with a message, not a traceback
    assert main(["export", "--out", str(tmp_path / "x"), "--tracking-uri", dest.tracking_uri]) == 1
    assert main(["export", "churn", "--all", "--out", str(tmp_path / "x"), "--tracking-uri", dest.tracking_uri]) == 1
    assert main(["import", "--all", "--name", "x", str(out), "--tracking-uri", dest.tracking_uri]) == 1
    assert "cannot be used with --all" in capsys.readouterr().out


def test_a_files_model_survives_the_trip(tmp_path):
    (tmp_path / "weights.pth").write_text("fake weights")
    s = store(tmp_path, "source")
    at.configure(s)
    at.init("filesmodel", "e", dataset_version="v1", settings=s)
    trial = at.start("r")
    at.log_model(str(tmp_path / "weights.pth"))
    version = at.register(settings=s)
    at.promote("filesmodel", version, "production", settings=s)
    at.finish()
    out = tmp_path / "export"
    at.export_model("filesmodel", out, settings=s)
    dest = store(tmp_path, "dest")
    at.import_model(out, settings=dest)
    folder = at.load("filesmodel", "production", settings=dest)       # still a files model: a folder comes back
    assert open(f"{folder}/weights.pth").read() == "fake weights"
    cd = client(dest)
    imported = [r for r in cd.search_runs([cd.get_experiment_by_name("filesmodel").experiment_id])
                if r.data.tags.get("mlflow.runName") == "r"][0]
    linked = [m.model_id for m in imported.outputs.model_outputs]
    assert linked == [imported.data.tags["authtracker.model_uri"].split("/", 1)[1]], "the imported run lists its model"
    at.configure(None)
