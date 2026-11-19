"""Keep the settings in a YAML file and record which data the run used.

    python examples/02_config_and_dataset.py

Shows: a training YAML saved with the run and logged as parameters, a dataset record with a
fingerprint (hash), and how the hash changes when the data changes.
"""
import tempfile
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from _common import client, ok, section, setup

setup("02_config_and_dataset")

import authtracker as at  # noqa: E402

CONFIG = Path(__file__).resolve().parent / "train.yaml"

# A small "dataset" on disk, so there is something to fingerprint.
data_dir = Path(tempfile.mkdtemp(prefix="authtracker-dataset-"))
X, y = make_classification(n_samples=600, n_features=10, random_state=0)
np.save(data_dir / "features.npy", X)
np.save(data_dir / "labels.npy", y)


def train(note: str, **overrides):
    """One experiment run with one run inside it. Returns the experiment run's tags."""
    parent = at.init(config=CONFIG, dataset={"name": "churn-data", "version": "v1", "path": str(data_dir)}, **overrides)
    tags = dict(parent.tags)
    at.start(note)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=0).fit(X_train, y_train)
    at.log({"accuracy": clf.score(X_test, y_test)})
    at.finish()
    return tags


section("1. The YAML names the model and experiment, and its settings become parameters")
first = train("first")
assert (first["model"], first["experiment"]) == ("churn", "gradient-boosting")
assert first["team"] == "ml-platform"
ok(f"model={first['model']} experiment={first['experiment']} team={first['team']} (all from train.yaml)")

mlflow = client()
experiment = mlflow.get_experiment_by_name("churn")
parent = mlflow.search_runs([experiment.experiment_id], "tags.`authtracker.level` = 'experiment'")[0]
params = parent.data.params
for key in ("optimizer.learning_rate", "optimizer.n_estimators", "split.train"):
    print(f"  param {key:<26} = {params[key]}")
assert params["optimizer.learning_rate"] == "0.1" and params["optimizer.n_estimators"] == "50"
files = sorted(a.path for a in mlflow.list_artifacts(parent.info.run_id))
assert "train.yaml" in files and "dataset.json" in files
ok(f"the YAML and dataset.json are saved with the run: {files}")

section("2. The dataset is recorded with a hash: same data, same hash")
second = train("second")
assert first["dataset_hash"] == second["dataset_hash"]
ok(f"two runs on the same data share dataset_hash={first['dataset_hash']}, {first['dataset_files']} files, {first['dataset_bytes']} bytes")

section("3. Change the data and the hash changes")
np.save(data_dir / "extra_rows.npy", X[:5])               # a new file appears in the folder
third = train("third")
assert third["dataset_hash"] != first["dataset_hash"]
ok(f"after adding a file: dataset_hash={third['dataset_hash']}, {third['dataset_files']} files")

section("4. Arguments to init() win over the YAML")
overridden = train("override", experiment="bigger-trees")
assert overridden["experiment"] == "bigger-trees" and overridden["model"] == "churn"
ok("experiment overridden to 'bigger-trees', model still from the YAML")
