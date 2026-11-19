"""Version models and ship them safely: aliases, promotion rules, rollback.

    python examples/03_registry_and_promotion.py

Shows: register a version from a run, promote with a minimum-metric rule and a "must beat production"
rule, a blocked promotion, rollback, and loading by alias.
"""
import dataclasses

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from _common import ok, section, setup

setup("03_registry_and_promotion")

import authtracker as at  # noqa: E402

MODEL, EXPERIMENT = "spam-filter", "sgd"

X, y = make_classification(n_samples=1200, n_features=12, n_informative=6, class_sep=0.8, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

section("1. Train two models that differ in regularisation (alpha)")
at.init(MODEL, EXPERIMENT, dataset_version="synthetic-v1")
for alpha in (1e-5, 5.0):
    at.start(f"alpha={alpha:g}", config={"alpha": alpha})
    clf = SGDClassifier(loss="log_loss", alpha=alpha, random_state=0).fit(X_train, y_train)
    f1 = f1_score(y_test, clf.predict(X_test))
    at.log({"f1": f1})
    at.log_model(clf, input_example=X_test[:3])
    print(f"  alpha={alpha:<8g} f1={f1:.4f}")
at.finish()

best = at.best_run(MODEL, EXPERIMENT, metric="f1")
worst = at.best_run(MODEL, EXPERIMENT, metric="f1", mode="min")
assert best["metrics"]["f1"] > worst["metrics"]["f1"]
v_best, v_worst = at.register(best), at.register(worst)
ok(f"registered version {v_best} (best, f1={best['metrics']['f1']:.4f}) and {v_worst} (worst, f1={worst['metrics']['f1']:.4f})")

section("2. Rules for 'production': f1 must reach a minimum, and must beat the current production model")
settings = at.get_settings()
threshold = round((best["metrics"]["f1"] + worst["metrics"]["f1"]) / 2, 4)
at.configure(dataclasses.replace(settings, promote_metric="f1", promote_min=threshold, promote_must_beat_prod=True))
print(f"  rule: f1 >= {threshold}")

try:
    at.promote(MODEL, v_worst, "production")
    raise SystemExit("the weak model should have been blocked")
except at.PromotionError as error:
    ok(f"weak model blocked: {error}")

at.promote(MODEL, v_best, "staging")                           # only 'production' is gated by default
at.promote(MODEL, v_best, "production")
ok(f"version {v_best} is now 'staging' and 'production'")

v_same = at.register(best)                                     # a new version with exactly the same f1
try:
    at.promote(MODEL, v_same, "production")
    raise SystemExit("a version that does not beat production should be blocked")
except at.PromotionError as error:
    ok(f"a version that only ties production is blocked: {error}")

section("3. Rollback")
at.configure(dataclasses.replace(settings, promote_metric=None))      # rules off: force the weak one in
at.promote(MODEL, v_worst, "production")
ok(f"production moved to version {v_worst}")
back = at.rollback(MODEL)
assert back == v_best
ok(f"rollback: production is version {back} again")

section("4. Load by alias, never by run id")
at.configure(None)
production = at.load(MODEL, "production")
staging = at.load(MODEL, "staging")
same = np.array_equal(np.asarray(production.predict(X_test)), np.asarray(staging.predict(X_test)))
assert same, "staging and production both point at the best version"
score = f1_score(y_test, np.asarray(production.predict(X_test)))
assert abs(score - best["metrics"]["f1"]) < 1e-9
ok(f"the production model scores f1={score:.4f}, the same as the best run")
