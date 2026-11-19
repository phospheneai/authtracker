"""The model architecture. Shipped inside every logged model via code_paths."""
import torch.nn as nn

LABELS = ["real", "fake"]
IMAGE_SIZE = 32


class FakeImageNet(nn.Module):
    """Tiny CNN: image (3 x 32 x 32) -> logits for [real, fake]."""

    def __init__(self, num_classes: int = len(LABELS)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(16, num_classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))
