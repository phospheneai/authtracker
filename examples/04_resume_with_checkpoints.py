"""Continue a training job after it stopped: same runs, same curve, same checkpoint.

    python examples/04_resume_with_checkpoints.py

Shows: init(resume=True), steps that carry on, at.save of a checkpoint and at.restore of it, and parameters
that may be repeated but not changed. The "interruption" is simulated by closing the run half way, which is
what a killed job leaves behind.
"""
import json
from pathlib import Path

from _common import client, ok, section, setup

folder = setup("04_resume_with_checkpoints")

import authtracker as at  # noqa: E402

MODEL, EXPERIMENT, EPOCHS = "regressor", "resume-demo", 10


def train(until: int) -> int:
    """Train until `until` epochs are done. Safe to call again after any interruption."""
    at.init(MODEL, EXPERIMENT, dataset_version="synthetic-v1", resume=True, config={"lr": 0.1})
    at.start("main", config={"epochs": EPOCHS})
    checkpoint = at.restore("ckpt.json")                     # None the first time
    state = json.loads(Path(checkpoint).read_text()) if checkpoint else {"epoch": -1, "weight": 0.0}
    start = state["epoch"] + 1
    print(f"  starting at epoch {start}" + (" (restored from the checkpoint)" if checkpoint else " (fresh)"))
    for epoch in range(start, until):
        state["weight"] += 0.5 * (3.0 - state["weight"])      # converges to 3.0
        state["epoch"] = epoch
        at.log({"loss": (3.0 - state["weight"]) ** 2})        # no step= needed: it carries on by itself
        Path("ckpt.json").write_text(json.dumps(state))
        at.save("ckpt.json")
    at.finish()
    return start


def runs():
    mlflow = client()
    experiment = mlflow.get_experiment_by_name(MODEL)
    found = mlflow.search_runs([experiment.experiment_id])
    parents = [r for r in found if r.data.tags["authtracker.level"] == "experiment"]
    children = [r for r in found if r.data.tags["authtracker.level"] == "run"]
    return mlflow, parents, children


section("1. First session: trains 4 epochs, then the job is 'killed'")
assert train(until=4) == 0

section("2. Second session: the same script continues from epoch 4")
assert train(until=EPOCHS) == 4

mlflow, parents, children = runs()
assert len(parents) == 1 and len(children) == 1, "still one experiment run and one run"
ok("still ONE experiment run and ONE run after two sessions")

steps = sorted(m.step for m in mlflow.get_metric_history(children[0].info.run_id, "loss"))
assert steps == list(range(EPOCHS)), steps
ok(f"the loss curve is continuous: steps {steps[0]}..{steps[-1]} with no gaps or repeats")

tags = children[0].data.tags
assert tags["resumed"] == "true" and tags["resume_count"] == "1"
ok("the run is tagged resumed=true, resume_count=1")
files = sorted(a.path for a in mlflow.list_artifacts(children[0].info.run_id))
assert "output.log" in files and "output.resume1.log" in files
ok(f"the earlier part's log is kept next to the new one: {[f for f in files if f.startswith('output')]}")

section("3. A parameter may be repeated, but not changed")
try:
    at.init(MODEL, EXPERIMENT, dataset_version="synthetic-v1", resume=True, config={"lr": 0.01})
    raise SystemExit("changing lr on resume should be refused")
except at.ValidationError as error:
    ok(f"refused: {error}")
at.finish()
