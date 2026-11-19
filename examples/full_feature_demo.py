"""One script that uses every authtracker feature on MLflow.

Runs as is: with no setup it keeps its data in examples/data (a local SQLite file). To use an MLflow
server instead, set AUTHTRACKER_TRACKING_URI first.

    python authtracker/examples/full_feature_demo.py            # the full tour
    python authtracker/examples/full_feature_demo.py --crash    # also adds one deliberately failing run

Everything is configured with environment variables. The rules below (allowed model names, required
tags, metric names, promotion thresholds) are examples that show the features; all of them are
optional and off unless you set them. In real use they live in your shell, container or CI settings.
"""
import dataclasses
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------------------------
# 1. CONFIGURATION: only environment variables, no config file.
# ---------------------------------------------------------------------------------------------
DATA = Path(__file__).resolve().parent / "data"                                  # where the demo keeps its files
if "AUTHTRACKER_TRACKING_URI" not in os.environ:                                # no server given: use a local file
    DATA.mkdir(exist_ok=True)
    os.chdir(DATA)                       # MLflow's sqlite store writes artifacts to ./mlruns
    os.environ["AUTHTRACKER_TRACKING_URI"] = f"sqlite:///{DATA / 'full_demo.db'}"
os.environ.setdefault("AUTHTRACKER_ALLOWED_MODELS", "classifier-demo")           # naming rule
os.environ.setdefault("AUTHTRACKER_REQUIRED_TAGS", "owner,dataset_version")      # every run needs these
os.environ.setdefault("AUTHTRACKER_METRIC_NAMES",                                # metric naming rule
                      "train_loss,val_acc,accuracy,f1,precision,recall")
os.environ.setdefault("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", "1")            # CPU/memory charts are on by default; sample every second here (demo runs are short)
os.environ.setdefault("AUTHTRACKER_PROMOTE_METRIC", "f1")                        # promotion rule ...
os.environ.setdefault("AUTHTRACKER_PROMOTE_MUST_BEAT_PROD", "true")              # ... must beat production

import numpy as np  # noqa: E402
from sklearn.datasets import make_classification  # noqa: E402
from sklearn.linear_model import SGDClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import authtracker as at  # noqa: E402

MODEL = "classifier-demo"            # these two also come from train.yaml (arguments to init would override)
EXPERIMENT = "full-demo"
EPOCHS = 8
CONFIG = Path(__file__).resolve().parent / "train.yaml"
ALPHAS = (1e-5, 1e-3, 0.5)         # one run (trial) per value


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def ok(message):
    print(f"  OK  {message}", flush=True)


# A logger created BEFORE the run starts: authtracker still captures what it prints.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("train")

# ---------------------------------------------------------------------------------------------
# Data (a synthetic dataset, so the demo needs nothing downloaded)
# ---------------------------------------------------------------------------------------------
X, y = make_classification(n_samples=1500, n_features=12, n_informative=6, class_sep=0.8, random_state=0)
X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)
DATA_DIR = Path(tempfile.mkdtemp(prefix="authtracker-demo-data-"))     # the demo "dataset" lives here
np.save(DATA_DIR / "features.npy", X)
np.save(DATA_DIR / "labels.npy", y)

section("0. Resolved settings (authtracker config show)")
settings = at.get_settings()
for key in ("backend", "tracking_uri", "owner", "required_tags", "allowed_models", "metric_names",
            "promote_metric", "promote_must_beat_prod", "capture_logs", "capture_env", "offline_queue"):
    print(f"  {key:<24}= {getattr(settings, key)}")

# ---------------------------------------------------------------------------------------------
section("1. Rules refuse bad runs BEFORE anything is logged")
try:
    at.init("some-other-model", EXPERIMENT, dataset_version="v1")
except at.ValidationError as e:
    ok(f"unknown model name refused:   {e}")
try:
    at.init(MODEL, EXPERIMENT)                                  # no dataset_version
except at.ValidationError as e:
    ok(f"missing required tag refused: {e}")

# ---------------------------------------------------------------------------------------------
section("2. Track an experiment: init (the experiment) -> start (one run per trial) -> log -> save")
at.init(
    config=CONFIG,                                 # train.yaml: names, tags and parameters; saved with the run
    dataset={"name": "synthetic", "version": "synthetic-v1", "path": str(DATA_DIR)},   # stored with a hash
    # model=..., experiment=... would override what train.yaml says
)
at.set_tags({"algorithm": "sgd-logistic-regression"})            # a tag added later reaches the runs started after it
ok(f"experiment run open: {at.current_run().id}")

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    for alpha in ALPHAS:
        run = at.start(f"alpha={alpha:g}", config={"alpha": alpha, "loss": "log_loss"})
        print(f"\n--- run {run.tags['experiment']} / alpha={alpha:g}")      # print(): saved to output.log

        clf = SGDClassifier(loss="log_loss", alpha=alpha, random_state=0)
        classes = np.unique(y_train)
        for epoch in range(EPOCHS):
            clf.partial_fit(X_train, y_train, classes=classes)
            loss = log_loss(y_train, clf.predict_proba(X_train))
            val_acc = clf.score(X_val, y_val)
            print(f"epoch {epoch + 1}/{EPOCHS}  train_loss={loss:.4f}  val_acc={val_acc:.4f}")
            log.info("finished epoch %d", epoch + 1)                        # logging: saved to output.log
            at.log({"train_loss": loss, "val_acc": val_acc})                # a curve; the step counts itself

        if alpha == ALPHAS[0]:
            try:
                at.log({"made_up_metric": 1.0})                             # not in AUTHTRACKER_METRIC_NAMES
            except at.ValidationError as e:
                ok(f"unknown metric name refused: {e}")

        pred = clf.predict(X_test)
        final = {"accuracy": accuracy_score(y_test, pred), "f1": f1_score(y_test, pred),
                 "precision": precision_score(y_test, pred), "recall": recall_score(y_test, pred)}
        at.log(final)
        print("test:", {k: round(v, 4) for k, v in final.items()})

        # --- save: text, one file, a folder (bundle), a list of files (bundle), and the model
        at.save(f"alpha={alpha:g}: test f1={final['f1']:.4f}\n", name="summary.txt")        # text -> file

        (tmp / "train.yaml").write_text(
            f"model: {MODEL}\nexperiment: {EXPERIMENT}\ndataset_version: synthetic-v1\n"
            f"optimizer:\n  name: sgd\n  alpha: {alpha}\nepochs: {EPOCHS}\n")
        at.save(str(tmp / "train.yaml"))                                                   # one file

        report = tmp / f"report-{alpha:g}"
        report.mkdir()
        (report / "metrics.json").write_text(json.dumps(final, indent=2))
        (report / "confusion_matrix.csv").write_text(
            "\n".join(",".join(map(str, row)) for row in confusion_matrix(y_test, pred)) + "\n")
        at.save(str(report))                                                               # a folder = a bundle

        (tmp / "class_balance.json").write_text(json.dumps({"train": {"0": int((y_train == 0).sum()), "1": int((y_train == 1).sum())}}))
        (tmp / "labels.json").write_text(json.dumps({"0": "negative", "1": "positive"}))
        at.save([str(tmp / "class_balance.json"), str(tmp / "labels.json")])               # a list = a bundle

        at.save(clf, input_example=X_test[:3])      # the model (MLflow also records its input format)
        ok(f"alpha={alpha:g}: logged curves, metrics, files, bundles and the model")
at.finish()                                         # optional: runs also close when the script ends
ok("all runs closed")

# ---------------------------------------------------------------------------------------------
section("3. The 'with' API (advanced): the same thing with explicit run objects")
with at.run(MODEL, "with-api-demo", dataset_version="synthetic-v1") as r:
    r.log_params({"note": "explicit parent run"})
    with r.child("manual-trial", params={"alpha": 0.01}) as c:
        print("printed inside a with-block run")
        c.log_metrics({"train_loss": 0.5, "val_acc": 0.8}, step=0)
        c.log_metric("val_acc", 0.85, step=1)
        c.log_eval(accuracy=0.85, f1=0.84)           # final metrics, name-checked
        c.log_text("hello from the with-API\n", "note.txt")
        c.set_tags({"api": "with"})
ok("with-API parent and child run closed")

# ---------------------------------------------------------------------------------------------
section("4. Find the best run, register versions, promotion rules, rollback, load")
best = at.best_run(MODEL, EXPERIMENT, metric="f1")                  # higher is better
worst = at.best_run(MODEL, EXPERIMENT, metric="f1", mode="min")
print(f"  best run : {best['name']:<12} f1={best['metrics']['f1']:.4f}")
print(f"  worst run: {worst['name']:<12} f1={worst['metrics']['f1']:.4f}")

v_best, v_worst = at.register(best), at.register(worst)            # model versions (database records)
ok(f"registered version {v_best} (best) and {v_worst} (worst)")

# Set the promotion threshold half way between the two runs, so the rule has something to decide.
threshold = round((best["metrics"]["f1"] + worst["metrics"]["f1"]) / 2, 4)
at.configure(dataclasses.replace(settings, promote_min=threshold))
print(f"  promotion rule: f1 >= {threshold} and must beat the current production version")

try:
    at.promote(MODEL, v_worst, "production")
    print("  UNEXPECTED: the weak model was promoted")
except at.PromotionError as e:
    ok(f"weak model blocked: {e}")
at.promote(MODEL, v_best, "staging")                               # staging is not gated
at.promote(MODEL, v_best, "production")
ok(f"version {v_best} is now 'staging' and 'production'")

# Turn the rules off to show rollback (production moves to another version, then back).
at.configure(dataclasses.replace(settings, promote_metric=None))
at.promote(MODEL, v_worst, "production")
ok(f"production moved to version {v_worst}")
back = at.rollback(MODEL)
ok(f"rollback: production is version {back} again")
at.configure(None)

model = at.load(MODEL, "production")                               # load by alias, never by run id
print("  predictions from the production model :", model.predict(X_test[:8]).tolist())
print("  true labels                           :", y_test[:8].tolist())

# ---------------------------------------------------------------------------------------------
if "--crash" in sys.argv:
    section("5. A deliberate crash (the run is saved as FAILED, with the traceback)")
    at.init(MODEL, "crash-demo", dataset_version="synthetic-v1")
    at.start("will-fail")
    print("epoch 1 ok")
    raise RuntimeError("loss is NaN")

section("Done")
print(f"  Open {os.environ['AUTHTRACKER_TRACKING_URI']} and look at the model '{MODEL}'.")
