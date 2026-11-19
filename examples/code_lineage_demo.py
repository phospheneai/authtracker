"""Code lineage, end to end: from a git repository to a run in MLflow, and back to the exact code.

What it does, in a throw-away folder:

  1. makes a small git project (a training script) with a remote and commits it;
  2. edits the script and adds a new file WITHOUT committing  -> run A is "dirty";
  3. commits everything and runs again                       -> run B is clean;
  4. shows what each run recorded, and rebuilds run A's exact code from it.

Start an MLflow server first (for example `mlflow server --port 5000`), then:

    python authtracker/examples/code_lineage_demo.py                      # server on http://127.0.0.1:5000
    AUTHTRACKER_TRACKING_URI=http://127.0.0.1:5001 python authtracker/examples/code_lineage_demo.py

and open the server in a browser: experiment "code-lineage-demo".
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SERVER = os.environ.get("AUTHTRACKER_TRACKING_URI", "http://127.0.0.1:5000")
GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "demo", "GIT_AUTHOR_EMAIL": "demo@example.com",
           "GIT_COMMITTER_NAME": "demo", "GIT_COMMITTER_EMAIL": "demo@example.com"}

TRAIN = '''\
import authtracker as at

LEARNING_RATE = {lr}

at.init("code-lineage-demo", "baseline", dataset_version="demo-v1", config={{"lr": LEARNING_RATE}})
at.start("{name}")
for epoch in range(3):
    at.log({{"loss": 1.0 / (epoch + 1) / LEARNING_RATE / 100}})
at.finish()
'''


def git(folder, *args):
    return subprocess.run(["git", *args], cwd=folder, check=True, capture_output=True, text=True, env=GIT_ENV).stdout


def train(folder, name):
    """Run the project's train.py, the way you would from a terminal."""
    env = {**GIT_ENV, "AUTHTRACKER_TRACKING_URI": SERVER}
    result = subprocess.run([sys.executable, "train.py"], cwd=folder, env=env, capture_output=True, text=True)
    if result.returncode:
        sys.exit(f"run {name} failed:\n{result.stderr[-1500:]}")


def show(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main():
    import mlflow
    from mlflow.tracking import MlflowClient

    work = Path(tempfile.mkdtemp(prefix="authtracker-lineage-"))
    project = work / "detector"
    project.mkdir()
    git(project, "init", "-q", "-b", "main")
    # A remote with a token in it, the way people paste them. authtracker stores it WITHOUT the token.
    git(project, "remote", "add", "origin", "https://demo-user:EXAMPLE_TOKEN@github.com/example-org/detector.git")

    show("1. A project with one commit")
    (project / "train.py").write_text(TRAIN.format(lr="0.001", name="A (uncommitted edits)"))
    git(project, "add", ".")
    git(project, "commit", "-q", "-m", "baseline training script")
    print("  commit:", git(project, "rev-parse", "--short", "HEAD").strip(), "on branch main")

    show("2. Edit the script and add a new file, but do NOT commit; run A")
    (project / "train.py").write_text(TRAIN.format(lr="0.01", name="A (uncommitted edits)"))
    (project / "augment.py").write_text("def flip(x):\n    return x[::-1]\n")
    print("  git status:", git(project, "status", "--short").replace("\n", " | ").strip())
    train(project, "A")

    show("3. Commit everything; run B")
    (project / "train.py").write_text(TRAIN.format(lr="0.01", name="B (clean)"))
    git(project, "add", ".")
    git(project, "commit", "-q", "-m", "learning rate 0.01, add augment")
    train(project, "B")

    # ---------------------------------------------------------------- what MLflow recorded
    mlflow.set_tracking_uri(SERVER)
    client = MlflowClient()
    experiment = client.get_experiment_by_name("code-lineage-demo")
    parent_runs = sorted(client.search_runs([experiment.experiment_id], "tags.authtracker.level = 'experiment'"),
                         key=lambda r: r.info.start_time)[-2:]            # the two runs this demo just made: A, then B

    for label, parent in zip(("A", "B"), parent_runs):
        show(f"4. What MLflow recorded for run {label}   ({SERVER}/#/experiments/{experiment.experiment_id}/runs/{parent.info.run_id})")
        tags = parent.data.tags
        for key in ("git_sha", "git_branch", "git_dirty", "git_remote", "git_url", "git_script",
                    "code_changes", "code_script"):
            print(f"  {key:<14} {tags.get(key, '-')}")
        print("  artifacts     ", sorted(a.path for a in client.list_artifacts(parent.info.run_id)))

    # ---------------------------------------------------------------- go back to run A's exact code
    run_a = parent_runs[0]
    show("5. Rebuild run A's exact code from what it recorded")
    sha = run_a.data.tags["git_sha"]
    patch_file = client.download_artifacts(run_a.info.run_id, "code_changes.patch", str(work))
    script_file = client.download_artifacts(run_a.info.run_id, "script_train.py", str(work))
    rebuilt = work / "rebuilt"
    git(work, "clone", "-q", str(project), str(rebuilt))       # (a real run would clone git_remote)
    git(rebuilt, "checkout", "-q", sha)
    git(rebuilt, "apply", patch_file)
    same = (rebuilt / "train.py").read_text() == Path(script_file).read_text()
    print(f"  git checkout {sha[:8]} && git apply code_changes.patch")
    print("  train.py LEARNING_RATE after applying:", [l for l in (rebuilt / "train.py").read_text().splitlines() if l.startswith("LEARNING_RATE")][0])
    print("  the rebuilt train.py is identical to the script run A saved:", same)
    print("  new file from the patch is there:", (rebuilt / "augment.py").exists())

    print(f"\nOpen {SERVER} -> experiment 'code-lineage-demo' -> the two 'baseline' runs.")
    print("  Overview tab: the tags above.   Artifacts tab: script_train.py and (run A only) code_changes.patch.")


if __name__ == "__main__":
    try:
        main()
    finally:                                              # the throw-away project and the rebuilt copy
        for leftover in Path(tempfile.gettempdir()).glob("authtracker-lineage-*"):
            shutil.rmtree(leftover, ignore_errors=True)
