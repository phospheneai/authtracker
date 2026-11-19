"""Plain PyTorch, saved as a folder, versioned, loaded back as the real class, fine-tuned.

    python examples/07_pytorch_bundle.py          (needs torch)

Shows: at.log_model(net, flavor="bundle"), the files a bundle holds, register/promote, at.load returning your
own class with its weights, extras (plain attributes), at.load_bundle on a downloaded folder, and fine-tuning
a loaded model into the next version.
"""
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from _common import client, ok, section, setup

setup("07_pytorch_bundle")
sys.path.insert(0, str(Path(__file__).resolve().parent / "pytorch_bundle"))

import authtracker as at  # noqa: E402
from architecture import Classifier  # noqa: E402

MODEL = "bundle-demo"
ARGS = {"n_features": 8, "hidden": 32, "n_classes": 2}          # how to build the class again later

torch.manual_seed(0)
X = torch.randn(600, 8)
y = (X[:, :3].sum(1) > 0).long()                                # a learnable rule
X_train, y_train, X_val, y_val = X[:400], y[:400], X[400:], y[400:]


def fit(net, epochs, lr):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        net.train()
        opt.zero_grad()
        loss = F.cross_entropy(net(X_train), y_train)
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            accuracy = (net(X_val).argmax(1) == y_val).float().mean().item()
        at.log({"train_loss": loss.item(), "val_acc": accuracy})
    return accuracy


section("1. Train and save as a bundle")
at.init(MODEL, "baseline", dataset_version="synthetic-v1", config=ARGS)
at.start("first", config={"lr": 0.05, "epochs": 5})
net = Classifier(**ARGS)
accuracy = fit(net, epochs=5, lr=0.05)
net.eval()
at.log({"accuracy": accuracy})
at.log_model(net, flavor="bundle", init_args=ARGS, extras={"class_names": ["negative", "positive"]})
v1 = at.register()
at.promote(MODEL, v1, "production")
at.finish()
ok(f"saved, registered as version {v1} and promoted (accuracy {accuracy:.3f})")

section("2. What the bundle holds")
import mlflow.artifacts  # noqa: E402

folder = Path(tempfile.mkdtemp(prefix="authtracker-bundle-"))
mlflow_model = mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{MODEL}@production", dst_path=str(folder))
found = sorted(p.name for p in Path(mlflow_model).rglob("*") if p.is_file())
print("  files:", [n for n in found])
for needed in ("architecture.py", "weights.pth", "inference.py", "requirements.txt", "bundle.json"):
    assert needed in found, f"{needed} missing from the bundle"
ok("architecture.py, weights.pth, inference.py, requirements.txt and bundle.json are all there")

section("3. Load it back: your own class, with its weights")
loaded = at.load(MODEL, "production")
assert type(loaded).__name__ == "Classifier"
with torch.no_grad():
    assert torch.allclose(loaded(X_val), net(X_val), atol=1e-6)
assert loaded.class_names == ["negative", "positive"], "extras are put back on the model"
ok("same class, same outputs, and class_names restored from extras")

section("4. Load the folder directly, with no server (what a downloaded bundle gives you)")
direct = at.load_bundle(next(p.parent for p in Path(mlflow_model).rglob("bundle.json")))
with torch.no_grad():
    assert torch.allclose(direct(X_val), net(X_val), atol=1e-6)
ok("load_bundle(folder) gives the same model")

section("5. Fine-tune the loaded model and save the next version (nothing to pass again)")
at.init(MODEL, "fine-tune", dataset_version="synthetic-v1")
at.start("second", config={"lr": 0.01, "epochs": 20})
accuracy2 = fit(loaded, epochs=20, lr=0.01)
loaded.eval()
at.log({"accuracy": accuracy2})
at.log_model(loaded, flavor="bundle")                               # it remembers its code and arguments
v2 = at.register()
at.finish()
assert int(v2) == int(v1) + 1
ok(f"version {v2} saved from the fine-tuned model (accuracy {accuracy2:.3f})")
at.promote(MODEL, v2, "production")
newer = at.load(MODEL, "production")
with torch.no_grad():
    assert torch.allclose(newer(X_val), loaded(X_val), atol=1e-6)
    assert not torch.allclose(newer(X_val), net(X_val), atol=1e-6), "version 2 must differ from version 1"
ok("production now serves the fine-tuned weights")
