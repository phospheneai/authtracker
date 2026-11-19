"""Describe the data a run used, with a fingerprint (hash) of it.

    dataset={"name": "faces", "version": "v3", "path": "/data/faces"}
    dataset={"name": "faces", "version": "v3", "path": "/data/faces", "hash": "contents"}
    dataset="/data/faces"                       # just a path

Hash modes:
  manifest (default)  fingerprint of the list of files: relative path and size of each. Fast even
                      for millions of files, and it catches files that were added, removed,
                      renamed or changed size. It does not read the files.
  contents            also reads every byte (slow on big data); catches a changed file of the same size.
  none                no hash.

Only local paths can be hashed. For a remote path (s3://, gs://, ...) the path is recorded and no
hash is computed.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Mapping, Optional, Union

from .errors import ValidationError

MODES = ("manifest", "contents", "none")
_CHUNK = 1024 * 1024


def _files(path: str):
    """Yield (relative path with '/', absolute path, size) in a stable order."""
    if os.path.isfile(path):
        yield os.path.basename(path), path, os.path.getsize(path)
        return
    entries = []
    for root, dirs, names in os.walk(path):
        dirs.sort()
        for name in sorted(names):
            full = os.path.join(root, name)
            if os.path.isfile(full):
                entries.append((os.path.relpath(full, path).replace(os.sep, "/"), full, os.path.getsize(full)))
    yield from sorted(entries)


def fingerprint(path: str, mode: str = "manifest") -> Dict[str, Any]:
    """Hash a local file or folder. Returns {'hash', 'files', 'bytes'}."""
    h = hashlib.sha256()
    count = total = 0
    for rel, full, size in _files(path):
        h.update(f"{rel}\t{size}\n".encode())
        if mode == "contents":
            with open(full, "rb") as f:
                for chunk in iter(lambda: f.read(_CHUNK), b""):
                    h.update(chunk)
        count += 1
        total += size
    return {"hash": h.hexdigest(), "files": count, "bytes": total}


def describe(spec: Union[str, os.PathLike, Mapping[str, Any]]) -> Dict[str, Any]:
    """Turn the `dataset=` argument into a record: name, version, path, hash, hash_mode, files, bytes."""
    if isinstance(spec, (str, os.PathLike)):
        spec = {"path": os.fspath(spec)}
    if not isinstance(spec, Mapping):
        raise ValidationError("dataset must be a path or a dict with name / version / path")
    allowed = {"name", "version", "path", "hash"}
    unknown = sorted(set(spec) - allowed)
    if unknown:
        raise ValidationError(f"Unknown dataset keys {unknown}; allowed: {sorted(allowed)}")

    path: Optional[str] = os.fspath(spec["path"]) if spec.get("path") else None
    mode = spec.get("hash", "manifest")
    if mode not in MODES:
        raise ValidationError(f"dataset hash must be one of {list(MODES)}, got {mode!r}")
    name = spec.get("name") or (os.path.basename(path.rstrip("/")) if path else None)
    if not name and not path:
        raise ValidationError("dataset needs at least a name or a path")

    record: Dict[str, Any] = {"name": name, "version": spec.get("version"), "path": path,
                              "hash": None, "hash_mode": "none", "files": None, "bytes": None}
    if path and mode != "none":
        if "://" in path:
            record["hash_mode"] = "remote path: not hashed"
        elif not os.path.exists(path):
            raise ValidationError(f"dataset path {path!r} does not exist")
        else:
            record.update(fingerprint(path, mode))
            record["hash_mode"] = mode
    return record


def as_tags(record: Mapping[str, Any]) -> Dict[str, str]:
    """The record as run tags, so you can search runs by dataset in the UI."""
    keys = {"name": "dataset_name", "path": "dataset_path", "hash": "dataset_hash",
            "files": "dataset_files", "bytes": "dataset_bytes", "hash_mode": "dataset_hash_mode"}
    tags = {tag: str(record[k]) for k, tag in keys.items() if record.get(k) is not None}
    if record.get("hash"):
        tags["dataset_hash"] = record["hash"][:16]          # short form: readable in the UI
    return tags
