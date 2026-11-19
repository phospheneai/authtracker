"""Contract checks for backends. Run them against every backend you write:

    from authtracker.testing import check_backend

    def test_my_backend():
        check_backend(MyBackend(...))                       # required behaviour
        check_backend(MyBackend(...), model=my_model)       # also model + registry, if supported
"""
from __future__ import annotations

import os
import tempfile
import uuid
from typing import Any, Optional

from .backends.base import TAG_LEVEL, Backend


def _supports(backend: Backend, method: str) -> bool:
    inner = getattr(backend, "inner", backend)  # look through the resilient wrapper
    return getattr(type(inner), method) is not getattr(Backend, method)


def check_backend(backend: Backend, *, model: Optional[Any] = None) -> None:
    exp = f"contract-{uuid.uuid4().hex[:8]}"
    base_tags = {"model": exp, "experiment": "e"}

    backend.ping()
    parent = backend.start_run(exp, "parent", {**base_tags, TAG_LEVEL: "experiment"}, None)
    assert isinstance(parent, str) and parent, "start_run must return a non-empty string id"

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "note.txt")
        with open(path, "w") as f:
            f.write("hi")

        ids = []
        for i, (name, f1) in enumerate([("a", 0.7), ("b", 0.9), ("c", 0.8)]):
            rid = backend.start_run(exp, name, {**base_tags, TAG_LEVEL: "run"}, parent)
            ids.append(rid)
            backend.log_params(rid, {"lr": 0.1 * (i + 1)})
            backend.log_metrics(rid, {"loss": 1.0}, 0)
            backend.log_metrics(rid, {"loss": 0.5}, 1)
            backend.log_metrics(rid, {"f1": f1}, None)
            backend.set_tags(rid, {"extra": "x"})
            backend.log_artifact(rid, path)
            if i == 0 and model is not None and _supports(backend, "log_model"):
                _check_models(backend, rid, model, exp)
            backend.end_run(rid, "FINISHED")

    info = backend.get_run(ids[0])
    assert {"run_id", "name", "metrics", "params", "tags"} <= set(info), "get_run keys"
    assert info["run_id"] == ids[0]
    assert info["name"] == "a", "get_run must report the run name"
    assert info["metrics"]["loss"] == 0.5, "metrics must report the latest value"
    assert abs(float(info["params"]["lr"]) - 0.1) < 1e-9
    assert info["tags"]["extra"] == "x" and info["tags"]["model"] == exp

    backend.end_run(parent, "FINISHED")

    flt = {"model": exp, TAG_LEVEL: "run"}
    best = backend.search_runs(exp, flt, "f1", "max", 2)
    assert [r["metrics"]["f1"] for r in best] == [0.9, 0.8], "search_runs max order/limit"
    assert backend.search_runs(exp, flt, "f1", "min", 1)[0]["metrics"]["f1"] == 0.7
    assert len(backend.search_runs(exp, flt, "f1", "max", 10)) == 3, "parent run must be excluded"
    assert backend.search_runs(exp, flt, "no_such_metric", "max", 5) == []
    assert backend.search_runs(exp + "-missing", {}, "f1", "max", 1) == [], "unknown experiment"


def _check_models(backend: Backend, run_id: str, model: Any, name: str) -> None:
    uri = backend.log_model(run_id, model, None)
    assert uri, "log_model must return a URI"
    assert backend.get_run(run_id)["tags"].get("authtracker.model_uri") == uri
    if not _supports(backend, "register"):
        return
    version = backend.register(run_id, name, uri)
    assert isinstance(version, str)
    assert backend.get_alias(name, "production") is None
    backend.set_alias(name, version, "production")
    assert backend.get_alias(name, "production") == version
    assert backend.version_run_id(name, version) == run_id
    assert backend.load(name, "production") is not None
