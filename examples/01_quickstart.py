"""The shortest useful script: track three trials, keep the best, load it back.

    python examples/01_quickstart.py

Shows: init -> start -> log -> save (text and a model) -> best_run -> register -> promote -> load.
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from _common import ok, section, setup

setup("01_quickstart")

import authtracker as at  # noqa: E402

MODEL, EXPERIMENT = "iris", "logistic-regression"

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

section("1. One experiment, three runs (one per value of C)")
at.init(model=MODEL, experiment=EXPERIMENT, dataset_version="sklearn-iris")
for C in (0.001, 0.1, 10.0):
    at.start(f"C={C:g}", config={"C": C})                     # starts a run and closes the previous one
    clf = LogisticRegression(C=C, max_iter=200).fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    at.log({"accuracy": accuracy})                           # step counts up by itself
    at.save(f"C={C:g} accuracy={accuracy:.4f}\n", name="summary.txt")   # text -> a file in the run
    at.log_model(clf)                                             # the model (sklearn flavor is detected)
    print(f"  C={C:<6g} accuracy={accuracy:.4f}")
at.finish()
ok("three runs logged")

section("2. Find the best run, version it, load it by name")
best = at.best_run(MODEL, EXPERIMENT, metric="accuracy")
print(f"  best run: {best['name']}  accuracy={best['metrics']['accuracy']:.4f}")
version = at.register(best)
at.promote(MODEL, version, "production")
model = at.load(MODEL, "production")                         # by alias, never by run id
predictions = np.asarray(model.predict(X_test))
accuracy = float((predictions == y_test).mean())
assert abs(accuracy - best["metrics"]["accuracy"]) < 1e-9, "the loaded model must behave like the logged one"
ok(f"version {version} is 'production'; the loaded model scores {accuracy:.4f}, the same as the run")
