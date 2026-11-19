"""Sample runs with a model + an artifact bundle per run (nothing is registered).

Run (from anywhere):   python authtracker/examples/sample_authtracker.py
View:  cd authtracker/examples/data && mlflow ui --backend-store-uri sqlite:///mlflow_demo.db
"""
import json
import os
import tempfile
from pathlib import Path

import authtracker as at  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

HERE = Path(__file__).resolve().parent
# All data (database + artifacts) goes to examples/data, wherever the script is run from.
# MLflow's sqlite store writes artifacts to ./mlruns, so run from inside that folder.
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
os.chdir(DATA)
os.environ.setdefault("AUTHTRACKER_TRACKING_URI", f"sqlite:///{DATA / 'mlflow_demo.db'}")

X = [[0, 1], [1, 0], [2, 1], [3, 2], [4, 3], [5, 3], [6, 5], [7, 5]]
y = [0, 0, 0, 0, 1, 1, 1, 1]

TRIALS = {
    "model1": [(f"C{C}", LogisticRegression(C=C), {"C": C}) for C in (0.1, 1.0, 10.0)],
    "model2": [(f"trees{n}", RandomForestClassifier(n_estimators=n, random_state=0),
                {"n_estimators": n}) for n in (5, 20, 50)],
}

for model_name, trials in TRIALS.items():
    at.init(model=model_name, experiment="sample", dataset_version="v1")
    for trial_name, clf, params in trials:
        at.start(trial_name, config=params)
        clf.fit(X, y)
        acc = clf.score(X, y)
        at.log({"accuracy": acc})
        at.save(f"{trial_name}: accuracy={acc:.3f}\n", name="train.log")

        with tempfile.TemporaryDirectory() as d:             # the bundle
            bundle = Path(d)
            (bundle / "config.json").write_text(json.dumps({**params, "model": model_name}))
            (bundle / "labels.csv").write_text("id,label\n0,real\n1,fake\n")
            (bundle / "report.json").write_text(json.dumps({"accuracy": acc}))
            at.save(str(bundle))

        # cloudpickle: MLflow's default "skops" format rejects some sklearn types
        # (RandomForest). Only load models you trained yourself with pickle formats.
        at.save(clf, serialization_format="cloudpickle")
        print(f"{model_name} / {trial_name}: accuracy={acc:.3f}")

at.finish()
print("Done. Open the UI and register a model from a run.")
