"""Train a small classifier in plain PyTorch, save it as a bundle, version it, load it back.

    python authtracker/examples/pytorch_bundle/train.py

With no setup it keeps its data in examples/data (a local SQLite file). Set AUTHTRACKER_TRACKING_URI
to use an MLflow server instead.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
if "AUTHTRACKER_TRACKING_URI" not in os.environ:
    DATA.mkdir(exist_ok=True)
    os.chdir(DATA)                         # MLflow's sqlite store writes artifacts to ./mlruns
    os.environ["AUTHTRACKER_TRACKING_URI"] = f"sqlite:///{DATA / 'bundle_demo.db'}"

import authtracker as at  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from architecture import Classifier  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
ARGS = {"n_features": 8, "hidden": 32, "n_classes": 2}          # how to build the class again later

torch.manual_seed(0)
X = torch.randn(600, 8)
y = (X[:, :3].sum(1) > 0).long()                                # a learnable rule
X_train, y_train, X_val, y_val = X[:400].to(DEVICE), y[:400].to(DEVICE), X[400:].to(DEVICE), y[400:].to(DEVICE)

at.init("bundle-demo", "baseline", dataset_version="synthetic-v1", config={"device": DEVICE, **ARGS})
at.start("lr0.05", config={"lr": 0.05, "epochs": 40})

net = Classifier(**ARGS).to(DEVICE)
opt = torch.optim.Adam(net.parameters(), lr=0.05)
for epoch in range(40):
    net.train()
    opt.zero_grad()
    loss = F.cross_entropy(net(X_train), y_train)
    loss.backward()
    opt.step()
    net.eval()
    with torch.no_grad():
        acc = (net(X_val).argmax(1) == y_val).float().mean().item()
    at.log({"train_loss": loss.item(), "val_acc": acc})
at.log({"accuracy": acc})
print(f"trained on {DEVICE}: val accuracy {acc:.3f}")

# The model, as a folder: architecture.py + weights.pth + inference.py + requirements.txt + ...
at.save(net, flavor="bundle", init_args=ARGS)
version = at.register()
at.promote("bundle-demo", version, "production")
print(f"saved, registered as version {version}, promoted to production")
at.finish()

# Anyone, anywhere: the real class with its weights back, by name and label.
model = at.load("bundle-demo", "production")
with torch.no_grad():
    same = torch.allclose(model(X_val.cpu()), net.cpu().eval()(X_val.cpu()), atol=1e-5)
print(f"loaded {type(model).__name__} back; gives the same outputs as the trained model: {same}")
