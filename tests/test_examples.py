"""Run every example as a real script and check what it left in the store.

The examples assert their own claims, but a script that merely exits 0 proves little, so each test
also opens the example's MLflow database and checks the result independently.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def data_dir(tmp_path):
    """A folder per test: MLflow caches its database connection per path, so reusing one would show stale data."""
    return tmp_path


def _env(data_dir):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("AUTHTRACKER_", "MLFLOW_", "WANDB_"))}
    env["AUTHTRACKER_EXAMPLES_DATA"] = str(data_dir)
    env["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    return env


def run_example(data_dir, name, *args, expect=0):
    result = subprocess.run([sys.executable, str(EXAMPLES / name), *args], cwd=data_dir, env=_env(data_dir),
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == expect, (
        f"{name} exited {result.returncode}, expected {expect}\n--- stdout\n{result.stdout[-3000:]}\n--- stderr\n{result.stderr[-3000:]}")
    return result


def client(data_dir, example):
    mlflow = pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient
    uri = f"sqlite:///{Path(data_dir) / example / 'mlflow.db'}"
    return MlflowClient(tracking_uri=uri, registry_uri=uri)    # both: other tests may have set a global registry


def runs(c, model, level=None):
    experiment = c.get_experiment_by_name(model)
    assert experiment is not None, f"no experiment {model!r}"
    found = c.search_runs([experiment.experiment_id], max_results=1000)
    return [r for r in found if level is None or r.data.tags["authtracker.level"] == level]


COVERED = {"01_quickstart.py", "02_config_and_dataset.py", "03_registry_and_promotion.py",
           "04_resume_with_checkpoints.py", "05_rules_and_failures.py", "06_code_lineage.py",
           "07_pytorch_bundle.py", "08_wandb_backend.py", "09_export_import.py"}


def test_every_example_has_a_test():
    scripts = {p.name for p in EXAMPLES.glob("[0-9][0-9]_*.py")}
    assert scripts, "no examples found"
    assert scripts == COVERED, f"examples without a test (or tests without an example): {scripts ^ COVERED}"


def test_01_quickstart(data_dir):
    out = run_example(data_dir, "01_quickstart.py").stdout
    assert "version 1 is 'production'" in out
    c = client(data_dir, "01_quickstart")
    assert len(runs(c, "iris", "experiment")) == 1
    children = runs(c, "iris", "run")
    assert sorted(r.data.tags["mlflow.runName"] for r in children) == ["C=0.001", "C=0.1", "C=10"]
    assert all(r.info.status == "FINISHED" and "accuracy" in r.data.metrics for r in children)
    assert all("summary.txt" in {a.path for a in c.list_artifacts(r.info.run_id)} for r in children)
    best = max(children, key=lambda r: r.data.metrics["accuracy"])
    version = c.get_model_version_by_alias("iris", "production")
    assert version.run_id == best.info.run_id, "production must be the best run"


def test_an_example_starts_from_a_clean_state_every_time(data_dir):
    run_example(data_dir, "05_rules_and_failures.py")
    run_example(data_dir, "05_rules_and_failures.py")             # fresh=True: the second run must not trip over the first
    c = client(data_dir, "05_rules_and_failures")
    assert len(runs(c, "fraud-detector", "run")) == 1             # not 2: the first run's data was wiped


def test_02_config_and_dataset(data_dir):
    run_example(data_dir, "02_config_and_dataset.py")
    c = client(data_dir, "02_config_and_dataset")
    parents = runs(c, "churn", "experiment")
    assert len(parents) == 4
    hashes = {r.data.tags["dataset_hash"] for r in parents if r.data.tags["experiment"] == "gradient-boosting"}
    assert len(hashes) == 2, "same data twice, then changed data: two distinct hashes"
    assert {r.data.tags["experiment"] for r in parents} == {"gradient-boosting", "bigger-trees"}
    for r in parents:
        assert r.data.params["optimizer.n_estimators"] == "50"
        assert r.data.tags["team"] == "ml-platform"
        assert {"train.yaml", "dataset.json"} <= {a.path for a in c.list_artifacts(r.info.run_id)}


def test_03_registry_and_promotion(data_dir):
    run_example(data_dir, "03_registry_and_promotion.py")
    c = client(data_dir, "03_registry_and_promotion")
    versions = {str(v.version): v for v in c.search_model_versions("name='spam-filter'")}
    assert set(versions) == {"1", "2", "3"}
    production = c.get_model_version_by_alias("spam-filter", "production")
    assert str(production.version) == "1", "production ends on the best version after the rollback"
    assert str(c.get_model_version_by_alias("spam-filter", "staging").version) == "1"
    f1 = {v: c.get_run(versions[v].run_id).data.metrics["f1"] for v in versions}
    assert f1["1"] == f1["3"] > f1["2"]


def test_04_resume_with_checkpoints(data_dir):
    run_example(data_dir, "04_resume_with_checkpoints.py")
    c = client(data_dir, "04_resume_with_checkpoints")
    assert len(runs(c, "regressor", "experiment")) == 1
    children = runs(c, "regressor", "run")
    assert len(children) == 2
    resumed = [r for r in children if r.data.tags.get("resumed") == "true"]
    assert len(resumed) == 1
    steps = sorted(m.step for m in c.get_metric_history(resumed[0].info.run_id, "loss"))
    assert steps == list(range(10))
    losses = [m.value for m in sorted(c.get_metric_history(resumed[0].info.run_id, "loss"), key=lambda m: m.step)]
    assert losses == sorted(losses, reverse=True), "the loss keeps falling across the restart"


def test_05_rules(data_dir):
    run_example(data_dir, "05_rules_and_failures.py")
    c = client(data_dir, "05_rules_and_failures")
    found = runs(c, "fraud-detector")
    assert found and all(r.info.status == "FINISHED" for r in found)
    assert c.get_experiment_by_name("some-other-model") is None, "a refused model must not create an experiment"
    good = [r for r in runs(c, "fraud-detector", "run") if r.data.tags["mlflow.runName"] == "good"][0]
    assert set(good.data.metrics) >= {"loss", "accuracy"} and "made_up_metric" not in good.data.metrics


def test_05_crash_is_recorded_as_failed(data_dir):
    result = run_example(data_dir, "05_rules_and_failures.py", "--crash", expect=1)
    assert "loss is NaN" in result.stderr
    c = client(data_dir, "05_rules_and_failures")
    failed = [r for r in runs(c, "fraud-detector") if r.info.status == "FAILED"]
    assert len(failed) == 2, "both the run and its experiment run are FAILED"
    names = {r.data.tags["mlflow.runName"] for r in failed}
    assert "will-fail" in names
    error_file = next(r for r in failed if "error.txt" in {a.path for a in c.list_artifacts(r.info.run_id)})
    text = Path(c.download_artifacts(error_file.info.run_id, "error.txt", str(data_dir))).read_text()
    assert "RuntimeError" in text and "loss is NaN" in text


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_06_code_lineage(data_dir):
    run_example(data_dir, "06_code_lineage.py")
    c = client(data_dir, "06_code_lineage")
    parents = sorted(runs(c, "code-lineage", "experiment"), key=lambda r: r.info.start_time)
    assert [r.data.tags["git_dirty"] for r in parents] == ["true", "false"]
    assert all("TOKEN" not in v for r in parents for v in r.data.tags.values()), "no token may leak into any tag"
    assert "code_changes.patch" in {a.path for a in c.list_artifacts(parents[0].info.run_id)}


def test_07_pytorch_bundle(data_dir):
    pytest.importorskip("torch")
    run_example(data_dir, "07_pytorch_bundle.py")
    c = client(data_dir, "07_pytorch_bundle")
    versions = c.search_model_versions("name='bundle-demo'")
    assert sorted(str(v.version) for v in versions) == ["1", "2"]
    assert str(c.get_model_version_by_alias("bundle-demo", "production").version) == "2"
    assert str(c.get_model_version_by_alias("bundle-demo", "previous-production").version) == "1"


def test_08_wandb_backend(data_dir):
    pytest.importorskip("wandb")
    out = run_example(data_dir, "08_wandb_backend.py").stdout
    assert "best run is C=10" in out
    assert len(list((Path(data_dir) / "08_wandb_backend" / "wandb").glob("offline-run-*"))) == 4


def test_09_export_import(data_dir):
    run_example(data_dir, "09_export_import.py")
    from mlflow.tracking import MlflowClient
    folder = Path(data_dir) / "09_export_import"
    for store, runs_expected, versions_expected in (("old", 3, {"1", "2"}), ("new", 5, {"1", "2", "3"})):
        uri = f"sqlite:///{folder / (store + '.db')}"
        c = MlflowClient(tracking_uri=uri, registry_uri=uri)
        assert len(runs(c, "churn")) == runs_expected, store
        assert {str(v.version) for v in c.search_model_versions("name='churn'")} == versions_expected, store
    new = MlflowClient(tracking_uri=f"sqlite:///{folder / 'new.db'}", registry_uri=f"sqlite:///{folder / 'new.db'}")
    assert str(new.get_model_version_by_alias("churn", "production").version) == "2"
    imported = [r for r in runs(new, "churn") if "authtracker.imported_run_id" in r.data.tags]
    assert len(imported) == 3, "the three moved runs say where they came from; the two new ones do not"
