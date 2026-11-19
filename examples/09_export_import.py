"""Move a model, with everything it has, from one tracking store to another.

    python examples/09_export_import.py

Shows: at.export_model writes a folder (runs, metrics, files, model files, versions, aliases);
at.import_model reads it into another store, for example another MLflow server. Run ids change, version
numbers and aliases are kept, every imported run remembers the id it had, and the model loads and
registers new versions as if it had always been there.

On the command line the same two steps are:

    authtracker export churn --out churn-export --tracking-uri http://old-server:5000
    authtracker import churn-export --tracking-uri http://new-server:5000
"""
import dataclasses
import tempfile
from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from _common import ok, section, setup

folder = setup("09_export_import", local_store=False)

import authtracker as at  # noqa: E402
from authtracker.settings import Settings  # noqa: E402

OLD = Settings(tracking_uri=f"sqlite:///{folder / 'old.db'}")        # "the old server"
NEW = Settings(tracking_uri=f"sqlite:///{folder / 'new.db'}")        # "the new server"
(folder / "old").mkdir(exist_ok=True)
(folder / "new").mkdir(exist_ok=True)

X, y = make_classification(n_samples=800, n_features=10, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

section("1. Some work on the old store: two runs, two versions, aliases")
import os  # noqa: E402
os.chdir(folder / "old")                                              # artifacts of the old store live here
at.configure(OLD)
scores = {}                                                          # version -> f1, to check against later
at.init("churn", "baseline", dataset_version="v1")
for C in (0.001, 1.0):
    at.start(f"C={C:g}", config={"C": C})
    clf = LogisticRegression(C=C, max_iter=500).fit(X_train, y_train)
    for epoch in range(3):
        at.log({"loss": 1.0 / (epoch + 1)})
    f1 = f1_score(y_test, clf.predict(X_test))
    at.log({"f1": f1})
    at.save(f"trained with C={C:g}\n", name="notes.txt")
    at.log_model(clf)
    version = at.register()
    scores[version] = f1
    at.promote("churn", version, "staging")
at.finish()
at.promote("churn", "2", "production")
ok("model 'churn': 3 runs, versions 1 and 2, aliases staging and production")

section("2. Export it to a folder")
export_dir = Path(tempfile.mkdtemp(prefix="authtracker-export-")) / "churn"
summary = at.export_model("churn", export_dir)
print(f"  {summary['runs']} runs, {summary['metrics']} metric points, {summary['models']} model folders, "
      f"{summary['versions']} versions, aliases {summary['aliases']}")
assert summary["runs"] == 3 and summary["versions"] == 2
ok(f"written to {export_dir}")

section("3. Import it into the new store")
os.chdir(folder / "new")
at.configure(NEW)
result = at.import_model(export_dir)
assert result["versions"] == {"1": "1", "2": "2"} and result["aliases"]["production"] == "2"
ok(f"imported: versions old->new {result['versions']}, aliases {result['aliases']}")

section("4. It works on the new store like it was always there")
best = at.best_run("churn", "baseline", metric="f1")
assert best["tags"]["authtracker.imported_run_id"], "an imported run remembers its old id"
ok(f"best run is {best['name']} (f1={best['metrics']['f1']:.4f}), imported from run {best['tags']['authtracker.imported_run_id'][:8]}")
production = at.load("churn", "production")
assert abs(f1_score(y_test, production.predict(X_test)) - scores["2"]) < 1e-9
ok("the 'production' model loaded from the new store scores exactly what version 2 scored on the old store")

at.init("churn", "after-the-move", dataset_version="v1")
at.start("C=10", config={"C": 10})
at.log_model(LogisticRegression(C=10, max_iter=500).fit(X_train, y_train))
assert at.register() == "3"
at.finish()
ok("a new run registers version 3 on top of the imported versions")

section("5. Importing twice is refused, so nothing is duplicated")
try:
    at.import_model(export_dir)
    raise SystemExit("importing twice should be refused")
except at.ValidationError as error:
    ok(f"refused: {error}")
