"""Team rules that refuse bad runs, and what a crashed run looks like.

    python examples/05_rules_and_failures.py            # the rules
    python examples/05_rules_and_failures.py --crash    # ... then a run that crashes (exit code 1)

Shows: allowed model names, required tags, allowed metric names (all set with environment variables),
errors that name the problem, and a crash that is saved as FAILED with its traceback.
"""
import os
import sys

from _common import client, ok, section, setup

setup("05_rules_and_failures")

# Rules live in the environment (your shell, container or CI), never in code. Set here to show them.
os.environ["AUTHTRACKER_ALLOWED_MODELS"] = "fraud-detector"
os.environ["AUTHTRACKER_REQUIRED_TAGS"] = "owner,dataset_version"
os.environ["AUTHTRACKER_METRIC_NAMES"] = "loss,accuracy"

import authtracker as at  # noqa: E402

MODEL = "fraud-detector"

section("1. A model name that is not allowed is refused before anything is logged")
try:
    at.init("some-other-model", "e1", dataset_version="v1")
    raise SystemExit("should have been refused")
except at.ValidationError as error:
    ok(f"refused: {error}")

section("2. A missing required tag is refused")
try:
    at.init(MODEL, "e1")                               # no dataset_version
    raise SystemExit("should have been refused")
except at.ValidationError as error:
    ok(f"refused: {error}")

section("3. Only the allowed metric names can be logged")
at.init(MODEL, "e1", dataset_version="v1")
at.start("good")
at.log({"loss": 0.5, "accuracy": 0.8})
ok("loss and accuracy accepted")
try:
    at.log({"made_up_metric": 1.0})
    raise SystemExit("should have been refused")
except at.ValidationError as error:
    ok(f"refused: {error}")
at.finish()

if "--crash" in sys.argv:
    section("4. A crash: the run is saved as FAILED, with the traceback")
    at.init(MODEL, "crashing", dataset_version="v1")
    at.start("will-fail")
    at.log({"loss": 0.9})
    print("epoch 1 ok")
    raise RuntimeError("loss is NaN")                  # unhandled: authtracker closes the runs as FAILED
else:
    section("4. Run again with --crash to see a failed run")
    mlflow = client()
    experiment = mlflow.get_experiment_by_name(MODEL)
    runs = mlflow.search_runs([experiment.experiment_id])
    assert runs and all(r.info.status == "FINISHED" for r in runs)
    ok(f"{len(runs)} runs recorded, all FINISHED (the refused ones never created a run)")
