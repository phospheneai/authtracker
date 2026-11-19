"""The model class. It lives in its own file, so a bundle can ship it."""
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, n_features=8, hidden=32, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)
