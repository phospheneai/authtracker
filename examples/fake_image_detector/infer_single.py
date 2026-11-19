"""Classify one image as real or fake.

As a library:   from infer_single import infer_single; infer_single(net, "photo.jpg")
Standalone:     python infer_single.py weights.pt photo.jpg
"""
import sys

import torch

from arch.network import LABELS, FakeImageNet
from arch.preprocess import load_image


def infer_single(net, image_path: str) -> dict:
    net.eval()
    with torch.no_grad():
        probs = torch.softmax(net(load_image(image_path)), dim=1)[0]
    best = int(probs.argmax())
    return {
        "image_path": image_path,
        "label": LABELS[best],
        "confidence": round(float(probs[best]), 4),
        "probabilities": {name: round(float(p), 4) for name, p in zip(LABELS, probs)},
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python infer_single.py <weights.pt> <image>")
    net = FakeImageNet()
    net.load_state_dict(torch.load(sys.argv[1], map_location="cpu"))
    print(infer_single(net, sys.argv[2]))
