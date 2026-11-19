"""Train a tiny real/fake image classifier and log it so a developer can download and use it.

Run (from anywhere):
    python authtracker/examples/fake_image_detector/sample_dl.py
View:
    cd authtracker/examples/data && mlflow ui --backend-store-uri sqlite:///mlflow_demo.db
"""
import os
import tempfile
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent

import authtracker as at  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from arch.network import IMAGE_SIZE, FakeImageNet  # noqa: E402
from arch.pyfunc_model import FakeImageModel  # noqa: E402


# All data (database + artifacts) goes to examples/data, wherever the script is run from.
# MLflow's sqlite store writes artifacts to ./mlruns, so run from inside that folder.
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)
os.chdir(DATA)
os.environ.setdefault("AUTHTRACKER_TRACKING_URI", f"sqlite:///{DATA / 'mlflow_demo.db'}")


def make_data(n=128, seed=0):
    """Toy 'images': bright = fake, dark = real."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 2, (n,), generator=g)
    base = torch.where(labels == 1, 0.7, 0.3).view(-1, 1, 1, 1)
    x = (base + 0.15 * torch.randn(n, 3, IMAGE_SIZE, IMAGE_SIZE, generator=g)).clamp(0, 1)
    return x, labels


def accuracy(net, x, y):
    with torch.no_grad():
        return (net(x).argmax(1) == y).float().mean().item()


X, Y = make_data(128, seed=0)
XV, YV = make_data(64, seed=1)
PINNED = [f"{p}=={metadata.version(p)}" for p in ("torch", "pillow", "numpy")]

at.init(model="fake-image", experiment="baseline", dataset_version="toy-v1",
        config={"architecture": "FakeImageNet", "epochs": 15})

for lr in (1e-2, 1e-3):
    at.start(f"lr{lr}", config={"lr": lr})
    torch.manual_seed(0)
    net = FakeImageNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(15):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(net(X), Y)
        loss.backward()
        opt.step()
        at.log({"train_loss": loss.item(), "val_acc": accuracy(net, XV, YV)})
    val_acc = accuracy(net, XV, YV)
    at.log({"accuracy": val_acc})
    at.save(f"lr={lr} final val_acc={val_acc:.3f}\n", name="train.log")

    with tempfile.TemporaryDirectory() as d:
        weights = os.path.join(d, "weights.pt")
        torch.save(net.state_dict(), weights)

        # Plain files for developers who open the run in the UI.
        at.save([str(HERE / "infer_single.py"), str(HERE / "README_model.md"),
                 str(HERE / "requirements.txt")])

        # The model: weights + architecture + inference code, one predict() call.
        at.save(
            FakeImageModel(), flavor="pyfunc",
            artifacts={"weights": weights},
            code_paths=[str(HERE / "arch"), str(HERE / "infer_single.py")],
            pip_requirements=PINNED + [f"mlflow=={metadata.version('mlflow')}"],
        )
    print(f"lr={lr}: val_acc={val_acc:.3f}")

at.finish()
print("Done. Register a version from the UI, or with at.best_run + at.register.")
