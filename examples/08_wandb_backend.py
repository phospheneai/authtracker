"""The same tracking code on Weights & Biases (offline, so no account is needed).

    python examples/08_wandb_backend.py          (needs authtracker[wandb])

Shows: switching backend with one environment variable, the same calls producing the same three levels
(model = W&B project, experiment = group + parent run, run = run in that group), best_run, and the clear
error for what W&B does not support yet (the model registry). To use a real W&B project, drop the two
WANDB_* lines below and export WANDB_API_KEY instead.
"""
import os

from _common import ok, section, setup

folder = setup("08_wandb_backend", local_store=False)
os.environ["AUTHTRACKER_BACKEND"] = "wandb"
os.environ["WANDB_MODE"] = "offline"                     # write runs to ./wandb instead of the server
os.environ["WANDB_DIR"] = str(folder)
os.environ["WANDB_SILENT"] = "true"

from sklearn.datasets import load_iris  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import authtracker as at  # noqa: E402

MODEL, EXPERIMENT = "iris-wandb", "logistic-regression"

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

section("1. The same calls as the quickstart")
at.init(MODEL, EXPERIMENT, dataset_version="sklearn-iris")
for C in (0.001, 0.1, 10.0):
    at.start(f"C={C:g}", config={"C": C})
    clf = LogisticRegression(C=C, max_iter=200).fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    at.log({"accuracy": accuracy})
    at.save(f"C={C:g} accuracy={accuracy:.4f}\n", name="summary.txt")
    at.save(clf)                                          # on W&B this is a model artifact
    print(f"  C={C:<6g} accuracy={accuracy:.4f}")
at.finish()

section("2. best_run works the same")
best = at.best_run(MODEL, EXPERIMENT, metric="accuracy")
assert best["name"] == "C=10" and best["tags"]["dataset_version"] == "sklearn-iris"
ok(f"best run is {best['name']} with accuracy {best['metrics']['accuracy']:.4f}")

section("3. What W&B does not support yet fails clearly")
try:
    at.register(best)
    raise SystemExit("register should not be supported on W&B yet")
except at.BackendError as error:
    ok(f"{type(error).__name__}: {error}")

runs = sorted(p.name for p in (folder / "wandb").glob("offline-run-*"))
assert len(runs) == 4, runs                              # 1 experiment run + 3 runs
ok(f"{len(runs)} W&B runs written to {folder / 'wandb'} (sync them later with `wandb sync`)")
