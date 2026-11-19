import json
import os
import textwrap

import pytest

import authtracker as at
from authtracker import dataset as ds
from authtracker.settings import Settings


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in list(os.environ):
        if k.startswith("AUTHTRACKER_"):
            monkeypatch.delenv(k)
    at.configure(None)
    at.finish()
    yield
    at.finish()


@pytest.fixture(params=["local", "mlflow"])
def s(request, tmp_path):
    if request.param == "mlflow":
        pytest.importorskip("mlflow")
        uri = f"sqlite:///{tmp_path}/m.db"
    else:
        uri = str(tmp_path / "store")
    return Settings(backend=request.param, tracking_uri=uri, owner="t", capture_env=False,
                    capture_logs=False, system_metrics=False, queue_dir=str(tmp_path / "q"),
                    retry_wait=0.0)


def _get(s, run_id):
    from authtracker.backends import get_backend
    return get_backend(s).get_run(run_id)


def _artifacts(s, run_id):
    from authtracker.backends import get_backend
    b = get_backend(s)
    if s.backend == "local":
        return sorted(p.name for p in b._artifacts(run_id).iterdir())
    return sorted(a.path for a in b.client.list_artifacts(run_id))


def _yaml(tmp_path, body, name="train.yaml"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return str(p)


FULL = """\
    authtracker:
      model: fake-image
      experiment: baseline
      tags: {team: vision}
    optimizer: {name: adam, lr: 0.001}
    epochs: 15
    layers: [64, 32]
    note: null
"""


# ---------------------------------------------------------------- training YAML
def test_yaml_sets_names_params_tags_and_is_saved(s, tmp_path):
    path = _yaml(tmp_path, FULL)
    parent = at.init(config=path, dataset_version="v1", settings=s)
    child = at.start("a")
    at.finish()
    p = _get(s, parent.id)
    assert p["tags"]["model"] == "fake-image" and p["tags"]["experiment"] == "baseline"
    assert p["tags"]["team"] == "vision"
    assert p["params"]["optimizer.name"] == "adam" and p["params"]["optimizer.lr"] == "0.001"
    assert p["params"]["epochs"] == "15" and p["params"]["layers"] == "[64, 32]"
    assert p["params"]["note"] == "null"
    assert "authtracker" not in " ".join(p["params"])                # the block is not a parameter
    assert "train.yaml" in _artifacts(s, parent.id)
    assert _get(s, child.id)["tags"]["team"] == "vision"             # runs inherit the tags


def test_arguments_override_the_yaml(s, tmp_path):
    path = _yaml(tmp_path, FULL)
    parent = at.init("other-model", "other-exp", config=path, tags={"team": "nlp"},
                     dataset_version="v1", settings=s)
    at.finish()
    t = _get(s, parent.id)["tags"]
    assert t["model"] == "other-model" and t["experiment"] == "other-exp" and t["team"] == "nlp"


def test_yaml_errors_are_clear(s, tmp_path):
    with pytest.raises(at.ValidationError, match="model and experiment are required"):
        at.init(config=_yaml(tmp_path, "epochs: 3\n"), settings=s)
    with pytest.raises(at.ValidationError, match="Unknown keys.*experiement"):
        at.init(config=_yaml(tmp_path, "authtracker: {model: m, experiement: e}\n", "typo.yaml"), settings=s)
    with pytest.raises(at.ValidationError, match="does not exist"):
        at.init("m", "e", config=str(tmp_path / "missing.yaml"), settings=s)
    with pytest.raises(at.ValidationError, match="not valid YAML"):
        at.init("m", "e", config=_yaml(tmp_path, "a: [1, 2\n", "bad.yaml"), settings=s)
    with pytest.raises(at.ValidationError, match="mapping"):
        at.init("m", "e", config=_yaml(tmp_path, "- 1\n- 2\n", "list.yaml"), settings=s)


def test_nested_dict_config_is_flattened(s):
    parent = at.init("m", "e", config={"opt": {"lr": 0.1}, "epochs": 3}, dataset_version="v1", settings=s)
    at.finish()
    assert _get(s, parent.id)["params"] == {"opt.lr": "0.1", "epochs": "3"}


def test_a_very_long_value_is_cut_instead_of_failing(s, tmp_path):
    path = _yaml(tmp_path, "authtracker: {model: m, experiment: e}\nbig: '" + "x" * 9000 + "'\n")
    parent = at.init(config=path, dataset_version="v1", settings=s)
    at.finish()
    assert _get(s, parent.id)["params"]["big"].endswith("(truncated)")


# ---------------------------------------------------------------- dataset record
def _make_data(root, files=None):
    root.mkdir(exist_ok=True)
    for name, text in (files or {"a.txt": "hello", "sub/b.txt": "world!"}).items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return str(root)


def test_hash_is_stable_and_follows_the_data(tmp_path):
    one = ds.describe({"path": _make_data(tmp_path / "one")})
    copy = ds.describe({"path": _make_data(tmp_path / "copy")})
    assert one["hash"] == copy["hash"] and one["files"] == 2 and one["bytes"] == 11   # a copy has the same fingerprint
    (tmp_path / "copy" / "c.txt").write_text("new file")
    assert ds.describe({"path": str(tmp_path / "copy")})["hash"] != one["hash"]       # a file was added
    (tmp_path / "copy" / "c.txt").unlink()
    (tmp_path / "copy" / "a.txt").write_text("hello world")
    assert ds.describe({"path": str(tmp_path / "copy")})["hash"] != one["hash"]       # a file changed size


def test_manifest_vs_contents_hash(tmp_path):
    base = _make_data(tmp_path / "d", {"a.txt": "hello"})
    m1, c1 = ds.describe({"path": base})["hash"], ds.describe({"path": base, "hash": "contents"})["hash"]
    (tmp_path / "d" / "a.txt").write_text("HELLO")                                     # same size, new bytes
    assert ds.describe({"path": base})["hash"] == m1                                   # manifest cannot see it
    assert ds.describe({"path": base, "hash": "contents"})["hash"] != c1               # contents can


def test_hash_modes_and_special_paths(tmp_path):
    base = _make_data(tmp_path / "d")
    assert ds.describe({"path": base, "hash": "none"})["hash"] is None
    remote = ds.describe({"name": "faces", "version": "v3", "path": "s3://bucket/faces"})
    assert remote["hash"] is None and "not hashed" in remote["hash_mode"] and remote["version"] == "v3"
    single = ds.describe(str(tmp_path / "d" / "a.txt"))                                # a plain path, a single file
    assert single["name"] == "a.txt" and single["files"] == 1
    with pytest.raises(at.ValidationError, match="does not exist"):
        ds.describe({"path": str(tmp_path / "nope")})
    with pytest.raises(at.ValidationError, match="Unknown dataset keys"):
        ds.describe({"path": base, "verison": "v1"})
    with pytest.raises(at.ValidationError, match="hash must be"):
        ds.describe({"path": base, "hash": "md5"})


def test_dataset_is_recorded_on_every_run(s, tmp_path):
    path = _make_data(tmp_path / "faces")
    parent = at.init("m", "e", dataset={"name": "faces", "version": "v3", "path": path}, settings=s)
    child = at.start("a")
    at.finish()
    full = ds.describe({"path": path})["hash"]
    for run in (parent, child):
        t = _get(s, run.id)["tags"]
        assert t["dataset_version"] == "v3" and t["dataset_name"] == "faces"
        assert t["dataset_hash"] == full[:16] and t["dataset_files"] == "2" and t["dataset_bytes"] == "11"
    record = json.loads(_read_artifact(s, parent.id, "dataset.json", tmp_path))
    assert record["hash"] == full and record["path"] == path


def _read_artifact(s, run_id, name, tmp_path):
    from authtracker.backends import get_backend
    b = get_backend(s)
    if s.backend == "local":
        return (b._artifacts(run_id) / name).read_text()
    return open(b.client.download_artifacts(run_id, name, str(tmp_path / f"dl-{run_id}"))).read()


def test_dataset_version_precedence_and_required_tags(s, tmp_path):
    import dataclasses
    s = dataclasses.replace(s, required_tags=("dataset_version",))
    path = _make_data(tmp_path / "d")
    at.init("m", "e", dataset={"path": path, "version": "from-dataset"}, settings=s)   # satisfies the rule
    assert _get(s, at.current_run().id)["tags"]["dataset_version"] == "from-dataset"
    at.init("m", "e", dataset={"path": path, "version": "from-dataset"}, dataset_version="explicit", settings=s)
    assert _get(s, at.current_run().id)["tags"]["dataset_version"] == "explicit"
    at.finish()
    with pytest.raises(at.ValidationError, match="dataset_version"):
        at.init("m", "e", settings=s)


def test_dataset_can_come_from_the_yaml(s, tmp_path):
    data = _make_data(tmp_path / "d")
    path = _yaml(tmp_path, f"authtracker:\n  model: m\n  experiment: e\n  dataset: {{name: faces, version: v9, path: {data}}}\n")
    parent = at.init(config=path, settings=s)
    at.finish()
    t = _get(s, parent.id)["tags"]
    assert t["dataset_version"] == "v9" and t["dataset_name"] == "faces" and "dataset_hash" in t
