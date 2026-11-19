"""Read a training YAML file.

    authtracker:                       # optional block: where this run belongs
      model: fake-image
      experiment: baseline
      dataset: {name: faces, version: v3, path: /data/faces}
      tags: {team: vision}
    optimizer: {name: adam, lr: 0.001} # everything else becomes parameters of the run
    epochs: 15

Everything outside the `authtracker:` block is logged as parameters (nested keys are joined with
dots: `optimizer.lr`). Arguments passed to `authtracker.init(...)` override the block.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, Mapping, Tuple

from .errors import ValidationError

BLOCK = "authtracker"
BLOCK_KEYS = {"model", "experiment", "dataset", "dataset_version", "tags", "owner"}


def _max_param_length() -> int:
    try:
        from mlflow.utils.validation import MAX_PARAM_VAL_LENGTH
        return int(MAX_PARAM_VAL_LENGTH)
    except Exception:
        return 500


def _scalar(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)):
        text = value
    elif value is None:
        text = "null"
    else:                                   # lists and anything else: readable JSON
        text = json.dumps(value, default=str)
    limit = _max_param_length()
    if isinstance(text, str) and len(text) > limit:
        text = text[: limit - 20] + "...(truncated)"
    return text


def flatten(data: Mapping[str, Any], prefix: str = "") -> Iterator[Tuple[str, Any]]:
    """{'a': {'b': 1}} -> ('a.b', 1). Lists become JSON text."""
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            yield from flatten(value, name + ".")
        else:
            yield name, _scalar(value)


def flat_params(data: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(flatten(data))


def load(path) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Return (params, tracking block, absolute path) for the YAML file at `path`."""
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise ValidationError(f"Training config {path!r} does not exist")
    try:
        import yaml
    except ImportError as e:                                    # pragma: no cover
        raise ValidationError("Reading a YAML config needs PyYAML: pip install pyyaml") from e
    with open(path, encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValidationError(f"{path!r} is not valid YAML: {e}") from e
    if not isinstance(data, dict):
        raise ValidationError(f"{path!r} must contain a mapping at the top level")
    block = data.pop(BLOCK, None) or {}
    if not isinstance(block, dict):
        raise ValidationError(f"'{BLOCK}:' in {path!r} must be a mapping")
    unknown = sorted(set(block) - BLOCK_KEYS)
    if unknown:
        raise ValidationError(
            f"Unknown keys {unknown} in the '{BLOCK}:' block of {path!r}; allowed: {sorted(BLOCK_KEYS)}")
    return flat_params(data), block, os.path.abspath(path)
