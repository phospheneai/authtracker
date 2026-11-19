"""Image -> tensor. Must match what the model was trained with."""
import numpy as np
import torch
from PIL import Image

from .network import IMAGE_SIZE


def load_image(path: str) -> torch.Tensor:
    """Return a (1, 3, IMAGE_SIZE, IMAGE_SIZE) float tensor scaled to [0, 1]."""
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(img, dtype="float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
