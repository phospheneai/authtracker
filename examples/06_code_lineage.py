"""Know exactly which code produced a run, and rebuild it.

    python examples/06_code_lineage.py          (needs git)

Shows, in a throw-away git project: the commit, branch and remote (token removed) recorded on every run,
the uncommitted changes saved as a patch, the script saved even without git, and rebuilding the first
run's exact code from what it recorded.
"""
import os
import subprocess
import sys
from pathlib import Path

from _common import client, ok, section, setup

folder = setup("06_code_lineage")

GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "demo", "GIT_AUTHOR_EMAIL": "demo@example.com",
           "GIT_COMMITTER_NAME": "demo", "GIT_COMMITTER_EMAIL": "demo@example.com"}
TRAIN = '''\
import authtracker as at

LEARNING_RATE = {lr}

at.init("code-lineage", "baseline", dataset_version="demo-v1", config={{"lr": LEARNING_RATE}})
at.start("{name}")
for epoch in range(3):
    at.log({{"loss": 1.0 / (epoch + 1) / LEARNING_RATE / 100}})
at.finish()
'''


def git(folder, *args):
    return subprocess.run(["git", *args], cwd=folder, check=True, capture_output=True, text=True, env=GIT_ENV).stdout


def train(folder):
    """Run the project's train.py the way you would from a terminal."""
    result = subprocess.run([sys.executable, "train.py"], cwd=folder, env=GIT_ENV, capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(f"train.py failed:\n{result.stderr[-1500:]}")


work = folder                    # the project and its tracking files stay here, so you can browse them
project = work / "detector"
project.mkdir()
(project / ".gitignore").write_text("mlruns/\n*.db\n")     # tracking files are not part of the project
git(project, "init", "-q", "-b", "main")
# A remote with a token in it, the way people paste them. authtracker keeps it WITHOUT the token.
git(project, "remote", "add", "origin", "https://demo-user:EXAMPLE_TOKEN@github.com/example-org/detector.git")
(project / "train.py").write_text(TRAIN.format(lr="0.001", name="A"))
git(project, "add", ".")
git(project, "commit", "-q", "-m", "baseline")

section("1. Edit the script and add a file WITHOUT committing, then run (run A)")
(project / "train.py").write_text(TRAIN.format(lr="0.01", name="A"))
(project / "augment.py").write_text("def flip(x):\n    return x[::-1]\n")
print("  git status:", git(project, "status", "--short").replace("\n", " | ").strip())
train(project)

section("2. Commit everything and run again (run B)")
(project / "train.py").write_text(TRAIN.format(lr="0.01", name="B"))
git(project, "add", ".")
git(project, "commit", "-q", "-m", "lr 0.01, add augment")
train(project)

section("3. What each run recorded")
mlflow = client()
experiment = mlflow.get_experiment_by_name("code-lineage")
parents = sorted(mlflow.search_runs([experiment.experiment_id], "tags.`authtracker.level` = 'experiment'"),
                 key=lambda r: r.info.start_time)
assert len(parents) == 2
run_a, run_b = parents
for label, parent in (("A", run_a), ("B", run_b)):
    tags = parent.data.tags
    print(f"  run {label}: " + "  ".join(f"{k}={tags.get(k, '-')}" for k in ("git_sha", "git_branch", "git_dirty", "code_changes")))
assert run_a.data.tags["git_dirty"] == "true" and run_b.data.tags["git_dirty"] == "false"
assert run_a.data.tags["git_branch"] == "main"
assert "EXAMPLE_TOKEN" not in run_a.data.tags["git_remote"], "the token must be removed"
assert run_a.data.tags["git_remote"].endswith("github.com/example-org/detector.git")
ok(f"A is dirty, B is clean; the remote is stored without its token: {run_a.data.tags['git_remote']}")
files_a = {a.path for a in mlflow.list_artifacts(run_a.info.run_id)}
files_b = {a.path for a in mlflow.list_artifacts(run_b.info.run_id)}
assert {"code_changes.patch", "script_train.py"} <= files_a
assert "script_train.py" in files_b and "code_changes.patch" not in files_b
ok("A saved script_train.py and code_changes.patch; B (clean) saved only the script")

section("4. Rebuild run A's exact code from what it recorded")
sha = run_a.data.tags["git_sha"]
patch = mlflow.download_artifacts(run_a.info.run_id, "code_changes.patch", str(work))
script = mlflow.download_artifacts(run_a.info.run_id, "script_train.py", str(work))
rebuilt = work / "rebuilt"
git(work, "clone", "-q", str(project), str(rebuilt))       # a real run would clone git_remote
git(rebuilt, "checkout", "-q", sha)
git(rebuilt, "apply", patch)
assert (rebuilt / "train.py").read_text() == Path(script).read_text()
assert (rebuilt / "augment.py").exists()
ok(f"git checkout {sha[:8]} && git apply code_changes.patch gives the same train.py and the new augment.py")
