"""A PyTorch model as a plain folder you can read, run and reproduce.

    at.log_model(net, flavor="bundle", init_args={"hidden": 64})

stores this folder with the run (and `at.register()` can version it):

    architecture.py     the file that defines your model class (and any other code you list)
    weights.pth         the weights, as a plain state_dict (loaded with weights_only=True)
    inference.py        a sample inference script. It does not need authtracker
    requirements.txt    pinned versions of torch and the packages your code imports
    environment.txt     every package installed when it was saved
    bundle.json         what to import, how to build it, file hashes, python / torch / cuda versions

`at.load(name, alias)` (or `at.load_bundle(folder)`) imports the class from the folder, builds it with
`init_args`, loads the weights and returns the real `torch.nn.Module`. There is no MLflow model
format and no pickle of your model: the weights are a state_dict, the architecture is source code.

Loading runs the architecture file, so only load bundles you trust.

Two optional extras, for when you want the exact object back (like fastai's `export.pkl`):

    at.log_model(net, flavor="bundle", pickle=True)    # the folder above, plus model.pkl
    at.log_model(net, flavor="pickle")                 # only model.pkl, requirements and the environment
    at.load("name", "production", pickle=True)    # the exact object, attributes and all

A pickle runs code when it is loaded, and only loads on the same Python and torch versions, so it is
opt-in on both ends. The hashes in bundle.json are checked before anything is unpickled.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import linecache
import logging
import os
import pickle as _pickle
import platform
import re
import shutil
import subprocess
import symtable
import sys
import sysconfig
import textwrap
import tempfile
import time
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from .errors import ValidationError
from .utils import container_info, git_info, requirements_text

_log = logging.getLogger("authtracker")

FLAVOR = "bundle"                  # the value to pass as at.log_model(model, flavor=...). Rename it here.
PICKLE_FLAVOR = "pickle"           # at.log_model(model, flavor="pickle"): only the pickled object
MODEL_TYPE = "authtracker-bundle"  # recorded on the stored model so it can be recognised when loading
FILES_TYPE = "authtracker-files"   # a model that is just files (at.log_model("weights.pth")): no loader, loads as a folder
MANIFEST = "bundle.json"
WEIGHTS = "weights.pth"
INFERENCE = "inference.py"
REQUIREMENTS = "requirements.txt"
ENVIRONMENT = "environment.txt"
PICKLE = "model.pkl"
FORMAT_VERSION = 1
_RESERVED = {MANIFEST, WEIGHTS, INFERENCE, REQUIREMENTS, ENVIRONMENT, PICKLE}
TOKEN = "$BUNDLE"                  # in init_args: "the folder this bundle is in", filled in at load
ASSETS = "assets"
_CARRY = "_authtracker_bundle"       # set on a loaded model: its code and arguments, so it can be saved again
_CARRY_LIMIT = 5 * 1024 * 1024       # carry the code only if it is small

PathLike = Union[str, Path]


def _torch():
    try:
        import torch
    except ImportError as e:                               # pragma: no cover
        raise ValidationError("A PyTorch bundle needs torch: pip install torch") from e
    return torch


# ---------------------------------------------------------------------------------------- build
def _as_paths(code: Union[None, PathLike, Sequence[PathLike]]) -> List[Path]:
    if code is None:
        return []
    if isinstance(code, (str, Path)):
        return [Path(code)]
    return [Path(c) for c in code]


def _is_library(source: Optional[str]) -> bool:
    """True for code that belongs to an installed library (torch, torchvision, timm...), not to you."""
    if not source:
        return False
    stdlib = sysconfig.get_paths().get("stdlib", "\0")
    return "site-packages" in source or "dist-packages" in source or source.startswith(stdlib)


def _code_entries(cls: Any, code) -> tuple:
    """Which files hold the architecture, and the module name to import the class (or builder) from."""
    entries = _as_paths(code)
    module = cls.__module__
    if not entries:
        if module == "__main__":
            raise ValidationError(
                f"{cls.__name__} is defined in the script you are running. Put the class in its own "
                "file (for example architecture.py), import it from there, or name the file: "
                "code='architecture.py'")        # (build() copies it out itself before it gets here)
        try:
            source = inspect.getsourcefile(cls)
        except TypeError:
            source = None
        if _is_library(source):
            raise ValidationError(
                f"This model is a {module}.{cls.__name__} from a library, so there is no class of yours to "
                "store. Give a function that builds it, in your own file or script, and pass it as "
                "factory=:\n    def build_model(**kw):\n        return nn.Sequential(...)   # or torchvision.models."
                "resnet18(num_classes=2), timm.create_model(...)\n"
                "    at.log_model(net, flavor='bundle', factory=build_model, init_args={...})")
        if "." in module:
            raise ValidationError(
                f"{cls.__name__} lives in the package module {module!r}. Pass code='{module.split('.')[0]}' "
                "(the folder) so its structure is kept.")
        if not source:
            raise ValidationError(f"Cannot find the source file of {cls.__name__}; pass code='file.py'")
        entries = [Path(source)]
    for p in entries:
        if not p.exists():
            raise ValidationError(f"code path {str(p)!r} does not exist")
        if p.name in _RESERVED:
            raise ValidationError(f"{p.name!r} is a name the bundle uses itself; rename that file")
    if module == "__main__":                               # the class was defined in a script and `code=` names the file
        pattern = re.compile(rf"^(class|def)\s+{re.escape(cls.__name__)}\b", re.M)
        found = [p for p in entries if p.is_file() and pattern.search(p.read_text(encoding="utf-8"))]
        if not found:
            raise ValidationError(f"None of the code files defines {cls.__name__}")
        module = found[0].stem
    return entries, module


def _globals_used(source: str) -> List[str]:
    """Names a piece of source reads from the module around it (imports, helpers, constants)."""
    table = symtable.symtable(textwrap.dedent(source), "<source>", "exec")
    used: List[str] = []

    def walk(t, top: bool) -> None:
        for sym in t.get_symbols():
            if not sym.is_referenced() or sym.get_name() in used:
                continue
            if (top and not sym.is_assigned()) or (not top and sym.is_global()):
                used.append(sym.get_name())
        for child in t.get_children():
            walk(child, False)

    walk(table, True)
    return used


def _source_of(obj: Any) -> str:
    """The source of a class or function, including ones defined in a notebook cell.

    Python cannot find the source of a *class* defined in `__main__` when there is no script file
    (a notebook): the module has no file. But the class's methods remember which cell they came from,
    and the notebook keeps every cell's text, so the class is found through one of its methods."""
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        if not inspect.isclass(obj):
            raise
    for member in vars(obj).values():
        code = getattr(getattr(member, "__func__", member), "__code__", None)
        lines = linecache.getlines(code.co_filename) if code is not None else []
        if not lines:
            continue
        for node in ast.walk(ast.parse("".join(lines))):
            if isinstance(node, ast.ClassDef) and node.name == obj.__name__ \
                    and node.lineno <= code.co_firstlineno <= node.end_lineno:
                first = min([node.lineno] + [d.lineno for d in node.decorator_list])
                return "".join(lines[first - 1:node.end_lineno])
    raise OSError("source code not available")


def _extract_main_class(cls: type) -> str:
    """Write architecture.py for a class defined in the running script or notebook.

    It holds the class and what it needs from the script around it: the imports it uses, helper
    functions and classes, and simple constants. The training code is left out.
    """
    namespace = vars(sys.modules["__main__"])
    imports: Dict[str, str] = {}
    consts: Dict[str, str] = {}
    defs: List[str] = []
    seen = {cls.__name__}

    def visit(source: str) -> None:
        for name in _globals_used(source):
            if name in seen or name not in namespace:
                continue                                    # already handled, or a builtin
            value = namespace[name]
            if isinstance(value, types.ModuleType):
                imports[name] = f"import {value.__name__}" + (f" as {name}" if name != value.__name__ else "")
            elif isinstance(value, (type, types.FunctionType)):
                if value.__module__ == "__main__":
                    seen.add(name)
                    text = textwrap.dedent(_source_of(value))
                    visit(text)                             # what this helper needs comes first
                    defs.append(text)
                elif "<locals>" in value.__qualname__:
                    raise ValidationError(f"{name!r} is defined inside a function, which cannot be copied")
                else:
                    imports[name] = (f"from {value.__module__} import {value.__qualname__}"
                                     + (f" as {name}" if name != value.__qualname__ else ""))
            else:
                try:
                    literal = repr(value)
                    ast.literal_eval(literal)
                except (ValueError, SyntaxError):
                    raise ValidationError(
                        f"the class uses the global {name!r} ({type(value).__name__}), which is not a plain "
                        "number, text, list or dict, so it cannot be copied") from None
                consts[name] = f"{name} = {literal}"

    source = textwrap.dedent(_source_of(cls))
    visit(source)
    header = '"""Copied by authtracker from the script that trained the model: the class and what it needs."""'
    parts = [header, "\n".join(sorted(imports.values())), "\n".join(consts.values()), *defs, source]
    return "\n\n\n".join(p.strip("\n") for p in parts if p.strip()) + "\n"


def _tokenize(value: Any, mapping: Mapping[str, str]) -> Any:
    """Replace paths of copied assets (in init_args) by TOKEN/assets/<name>."""
    if isinstance(value, str):
        absolute = os.path.abspath(value)
        for original, token in mapping.items():
            if absolute == original or absolute.startswith(original + os.sep):
                return token + absolute[len(original):].replace(os.sep, "/")
        return value
    if isinstance(value, Mapping):
        return {k: _tokenize(v, mapping) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tokenize(v, mapping) for v in value]
    return value


def _resolve(value: Any, root: Path) -> Any:
    """The reverse: TOKEN/... becomes the real path inside the bundle folder."""
    if isinstance(value, str) and value.startswith(TOKEN):
        return str(root) + value[len(TOKEN):]
    if isinstance(value, Mapping):
        return {k: _resolve(v, root) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_resolve(v, root) for v in value]
    return value


def _copy_assets(assets: Union[Sequence[PathLike], Mapping[PathLike, Sequence[str]]],
                 dest: Path) -> Dict[str, str]:
    """Copy files the constructor reads into assets/. A list copies each path whole; a dict
    {folder: ["*.json", ...]} copies only the files that match, so a model folder with multi-GB
    weights next to its config.json can contribute just the config."""
    items = assets.items() if isinstance(assets, Mapping) else [(a, None) for a in assets]
    mapping: Dict[str, str] = {}
    for item, patterns in items:
        source = Path(item)
        if not source.exists():
            raise ValidationError(f"asset {str(item)!r} does not exist")
        target = dest / ASSETS / source.name
        if target.exists():
            raise ValidationError(f"two assets are called {source.name!r}; rename one")
        target.parent.mkdir(exist_ok=True)
        if patterns is not None:
            if not source.is_dir():
                raise ValidationError(f"patterns only make sense for a folder, and {str(item)!r} is a file")
            files = sorted({f for pattern in patterns for f in source.glob(pattern) if f.is_file()})
            if not files:
                raise ValidationError(f"no file in {str(item)!r} matches {list(patterns)}")
            for f in files:
                (target / f.relative_to(source)).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target / f.relative_to(source))
        elif source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(source, target)
        mapping[os.path.abspath(source)] = f"{TOKEN}/{ASSETS}/{source.name}"
    return mapping


def _unwrap(model: Any) -> Any:
    """The model inside DataParallel, DistributedDataParallel and torch.compile wrappers.

    Those wrappers add a `module.` prefix to the weights and are not your architecture: what you want
    to keep is the model inside them."""
    torch = _torch()
    for _ in range(5):
        if hasattr(model, "_orig_mod"):                                     # torch.compile
            model = model._orig_mod
        elif isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model = model.module
        else:
            break
    return model


def _validate_extras(model: Any, extras: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Plain attributes to put back on the model after loading (class names, a threshold...)."""
    extras = dict(extras or {})
    try:
        json.dumps(extras)
    except TypeError as e:
        raise ValidationError(f"extras must be JSON-friendly (numbers, text, lists, dicts): {e}") from e
    taken = ({n for n, _ in model.named_children()} | {n for n, _ in model.named_parameters(recurse=False)}
             | {n for n, _ in model.named_buffers(recurse=False)})
    for name in extras:
        if not name.isidentifier():
            raise ValidationError(f"extras name {name!r} is not a valid attribute name")
        if name in taken or callable(getattr(type(model), name, None)):
            raise ValidationError(f"extras name {name!r} is already used by the model (a layer, a weight or a method)")
    return extras


@contextlib.contextmanager
def _on_cpu(model: Any):
    """The model's tensors on the CPU while it is saved, and back where they were afterwards."""
    devices = {t.device for t in list(model.parameters()) + list(model.buffers())}
    if len(devices) != 1 or next(iter(devices)).type == "cpu":
        yield
        return
    original = next(iter(devices))
    model.to("cpu")
    try:
        yield
    finally:
        model.to(original)


def _user_modules() -> list:
    """Modules loaded from your own project (not from an installed library, not authtracker)."""
    found = []
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if module is None or name == "__main__" or name.split(".")[0] == "authtracker":
            continue
        if file and file.endswith(".py") and not _is_library(file):
            found.append(module)
    return found


@contextlib.contextmanager
def _pickle_user_code_by_value():
    """Make cloudpickle store *your* classes and functions inside the pickle, not just their names,
    so the pickle loads on a machine that does not have your files."""
    try:
        import cloudpickle
    except ImportError as e:                                           # pragma: no cover
        raise ValidationError("Pickling needs cloudpickle: pip install cloudpickle") from e
    registered = []
    for module in _user_modules():
        try:
            cloudpickle.register_pickle_by_value(module)
            registered.append(module)
        except Exception:
            continue
    try:
        yield cloudpickle
    finally:
        for module in registered:
            try:
                cloudpickle.unregister_pickle_by_value(module)
            except Exception:
                pass


def _object_packages(model: Any) -> set:
    """Top-level packages of every class in the model, to pin them when only a pickle is stored."""
    tops = set()
    for module in model.modules():
        for obj in (module, *vars(module).values()):
            name = getattr(type(obj), "__module__", None)
            if isinstance(name, str) and name:
                tops.add(name.split(".")[0])
    tops.discard("builtins")
    return tops


def _copy_code(entries: Sequence[Path], dest: Path) -> None:
    for p in entries:
        if p.is_dir():
            shutil.copytree(p, dest / p.name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(p, dest / p.name)


def _imports(files: Iterable[Path]) -> set:
    names = set()
    for f in files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def _namespace_packages(module: Any) -> set:
    """Top-level packages behind every name in a module, including names that came in through
    `from something import *`, which reading the source cannot see."""
    tops = set()
    for value in vars(module).values():
        name = value.__name__ if isinstance(value, types.ModuleType) else getattr(value, "__module__", None)
        if isinstance(name, str) and name:
            tops.add(name.split(".")[0])
    return tops


def _auto_requirements(dest: Path, extra: Iterable[str] = ()) -> List[str]:
    """Pinned versions of torch and of every installed package the bundle's code uses."""
    local = {p.stem for p in dest.glob("*.py")} | {p.name for p in dest.iterdir() if p.is_dir()}
    code = [p for p in dest.rglob("*.py") if ASSETS not in p.relative_to(dest).parts[:1]]
    owners = importlib.metadata.packages_distributions()
    pins = set()
    for name in _imports(code) | set(extra) | {"torch"}:
        if name in local or name in sys.stdlib_module_names or name not in owners:
            continue
        dist = owners[name][0]
        pins.add(f"{dist}=={importlib.metadata.version(dist)}")
    return sorted(pins, key=str.lower)


_TEMPLATE = '''"""Sample inference for this bundle. It needs torch and requirements.txt, not authtracker.

    python inference.py input.npy        # a NumPy file of shape (batch, ...); prints the model output

From your own code:

    from inference import load_model, predict
    model = load_model()                 # the real torch.nn.Module with its weights, in eval mode
    out = predict(x)                     # x: a NumPy array or tensor of float32 inputs
"""
import importlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _resolve(value):
    if isinstance(value, str) and value.startswith("$BUNDLE"):
        return str(HERE) + value[len("$BUNDLE"):]
    if isinstance(value, dict):
        return {k: _resolve(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v) for v in value]
    return value


def load_model(device="cpu"):
    manifest = json.loads((HERE / "bundle.json").read_text())
    sys.path.insert(0, str(HERE))
    try:
        module = importlib.import_module(manifest["module"])
    finally:
        sys.path.remove(str(HERE))
    net = getattr(module, manifest["class"])(**_resolve(manifest["init_args"]))
    net.load_state_dict(torch.load(HERE / manifest["weights"], map_location=device, weights_only=True))
    return net.to(device).eval()


def predict(x, device="cpu"):
    model = load_model(device)
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32, device=device))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python inference.py input.npy")
    import numpy as np
    print(predict(np.load(sys.argv[1])))
'''


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hashes(root: Path) -> Dict[str, str]:
    return {p.relative_to(root).as_posix(): _sha256(p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != MANIFEST and "__pycache__" not in p.parts}


def _environment(model) -> Dict[str, Any]:
    torch = _torch()
    devices = {t.device.type for t in list(model.parameters()) + list(model.buffers())}
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "trained_on": sorted(devices)[0] if len(devices) == 1 else sorted(devices),
        "created_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }


def build(model: Any, dest: PathLike, *, init_args: Optional[Mapping[str, Any]] = None,
          code: Union[None, PathLike, Sequence[PathLike]] = None, inference: Optional[PathLike] = None,
          requirements: Optional[Sequence[str]] = None,
          assets: Union[None, Sequence[PathLike], Mapping[PathLike, Sequence[str]]] = None,
          factory: Optional[Any] = None, extras: Optional[Mapping[str, Any]] = None,
          pickle: bool = False, check: bool = True,
          context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Write the bundle for `model` into the (empty) folder `dest`. Returns the manifest.

    extras: plain attributes to put back on the model after loading, for things that are not weights,
    such as class_names=[...] or threshold=0.3 (numbers, text, lists, dicts).
    pickle: also store model.pkl, the whole object, loadable later with pickle=True.

    factory: a function `f(**init_args)` that returns a new, untrained model like `model`. Use it when
    the model has no class of yours: nn.Sequential(...), torchvision / timm models, or a model you
    change after creating it (adding LoRA, freezing). Without it, the model's class is shipped.
    """
    torch = _torch()
    model = _unwrap(model)
    if not isinstance(model, torch.nn.Module):
        raise ValidationError(f"flavor='{FLAVOR}' is for a torch.nn.Module, not {type(model).__name__}")
    if factory is not None and not callable(factory):
        raise ValidationError("factory must be a function that builds the model: factory=build_model")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    carried = getattr(model, _CARRY, None)               # a model that came from a bundle carries its code
    if carried is not None and (code is not None or factory is not None or (
            not carried.get("factory") and carried["class"] != type(model).__name__)):
        carried = None
    if init_args is None and carried is not None:
        init_args = carried["init_args"]
    init_args = dict(init_args or {})
    kept = {k: getattr(model, k, v) for k, v in (carried or {}).get("extras", {}).items()}   # as they are now
    extras = _validate_extras(model, {**kept, **dict(extras or {})}) if (kept or extras) else {}
    try:
        json.dumps(init_args)
    except TypeError as e:
        raise ValidationError(f"init_args must be JSON-friendly (numbers, text, lists, dicts): {e}") from e

    cls = factory if factory is not None else type(model)              # what the loader will call
    extracted = False                                                  # was the code copied out of a script or notebook?
    if carried is not None:
        module = carried["module"]
        for name, data in carried["files"].items():
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    elif code is None and cls.__module__ == "__main__":
        extracted = True
        try:
            text = _extract_main_class(cls)
        except (OSError, TypeError, ValidationError) as e:
            raise ValidationError(
                f"{cls.__name__} is defined in the script you are running and authtracker could not copy "
                f"it out ({e}). Put the class in its own file (for example architecture.py), import it, "
                "or name the file: code='architecture.py'") from e
        (dest / "architecture.py").write_text(text, encoding="utf-8")
        module = "architecture"
    else:
        entries, module = _code_entries(cls, code)
        _copy_code(entries, dest)

    if assets:
        init_args = _tokenize(init_args, _copy_assets(assets, dest))        # files the constructor needs
    state = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)      # the model itself stays put
             for k, v in model.state_dict().items()}
    torch.save(state, dest / WEIGHTS)

    if inference is not None:
        if not Path(inference).is_file():
            raise ValidationError(f"inference script {str(inference)!r} does not exist")
        shutil.copy2(inference, dest / INFERENCE)
    elif not (dest / INFERENCE).exists():
        (dest / INFERENCE).write_text(_TEMPLATE, encoding="utf-8")

    (dest / ENVIRONMENT).write_text(requirements_text(), encoding="utf-8")
    entry = carried["class"] if carried is not None else cls.__name__                # what the loader calls
    is_factory = bool(carried["factory"]) if carried is not None else factory is not None
    manifest = {"format": MODEL_TYPE, "version": FORMAT_VERSION, "kind": "bundle", "module": module,
                "class": entry, "factory": is_factory, "init_args": init_args, "weights": WEIGHTS,
                "inference": INFERENCE, "extras": extras, "pickle": PICKLE if pickle else None,
                "environment": _environment(model), "git": git_info(), "container": _container(context)}
    try:
        used = _check(dest, manifest, state) if check else set()
    except ValidationError as e:
        if not extracted:
            raise
        raise ValidationError(
            f"{cls.__name__} is defined in the script you are running, and the copy authtracker made of it "
            f"does not work: {e}\nThe copy only gets what the class uses directly. Put the class in its own "
            "file (for example architecture.py) and import it, or name the file: code='architecture.py'") from e
    pins = list(requirements) if requirements is not None else _auto_requirements(dest, used)
    (dest / REQUIREMENTS).write_text("\n".join(pins) + "\n", encoding="utf-8")
    if pickle:
        _write_pickle(model, dest, by_value=False)         # by name: architecture.py in this folder is what it finds
        (dest / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if check:
            _check_pickle(dest)
    manifest["files"] = _hashes(dest)
    (dest / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _write_pickle(model: Any, dest: Path, by_value: bool) -> None:
    """cloudpickle the whole model, from CPU tensors. `by_value` stores your own classes inside the
    pickle (so it loads without your files); otherwise they are stored by name."""
    with contextlib.ExitStack() as stack:
        if by_value:
            cloudpickle = stack.enter_context(_pickle_user_code_by_value())
        else:
            import cloudpickle
        stack.enter_context(_on_cpu(model))
        out = stack.enter_context(open(dest / PICKLE, "wb"))
        cloudpickle.dump(model, out, protocol=_pickle.HIGHEST_PROTOCOL)


def _check_pickle(root: Path) -> None:
    """Load the pickle in a clean process that has none of your files, the way another machine would."""
    import authtracker
    library = str(Path(authtracker.__file__).resolve().parent.parent)
    code = ("import sys\nsys.path.insert(0, %r)\nfrom authtracker import bundle\n"
            "net = bundle.load_dir(%r, pickle=True, verify=False)\nprint(type(net).__name__)") % (library, str(root))
    with tempfile.TemporaryDirectory() as empty:
        env = {**os.environ, "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run([sys.executable, "-c", code], cwd=empty, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        reason = (result.stderr.strip().splitlines() or ["no error message"])[-1][:250]
        raise ValidationError(
            f"The pickle cannot be loaded in a clean process, so it would not load on another machine: "
            f"{reason}. A class or function from a module that is not on the path is the usual reason. "
            "Use the folder bundle (without pickle=True), or check=False to skip this test.")


def _container(context: Optional[Mapping[str, Any]]) -> Optional[Dict[str, str]]:
    """The container image of the run this model belongs to (passed in by the run), else what can be found."""
    if context and "container" in context:
        return dict(context["container"]) if context["container"] else None
    return container_info()


def build_pickle(model: Any, dest: PathLike, *, requirements: Optional[Sequence[str]] = None,
                 check: bool = True, context: Optional[Mapping[str, Any]] = None,
                 **unsupported: Any) -> Dict[str, Any]:
    """Write only the pickled model, the requirements and the environment into the folder `dest`.

    Your own classes and functions are stored inside the pickle, so it loads on a machine that does not
    have your files. It needs the same Python and torch versions, and loading it runs code.
    """
    torch = _torch()
    if unsupported:
        raise ValidationError(
            f"flavor='{PICKLE_FLAVOR}' stores the whole object, so {sorted(unsupported)} are not used. "
            f"They belong to flavor='{FLAVOR}'.")
    model = _unwrap(model)
    if not isinstance(model, torch.nn.Module):
        raise ValidationError(f"flavor='{PICKLE_FLAVOR}' is for a torch.nn.Module, not {type(model).__name__}")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    _write_pickle(model, dest, by_value=True)
    (dest / ENVIRONMENT).write_text(requirements_text(), encoding="utf-8")
    pins = list(requirements) if requirements is not None else _auto_requirements(dest, _object_packages(model))
    (dest / REQUIREMENTS).write_text("\n".join(pins) + "\n", encoding="utf-8")
    manifest = {"format": MODEL_TYPE, "version": FORMAT_VERSION, "kind": "pickle", "class": type(model).__name__,
                "pickle": PICKLE, "environment": _environment(model), "git": git_info(),
                "container": _container(context)}
    (dest / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if check:
        _check_pickle(dest)
    manifest["files"] = _hashes(dest)
    (dest / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _check(root: Path, manifest: Mapping[str, Any], state: Mapping[str, Any]) -> set:
    """Prove the bundle can rebuild the class, without allocating the model (a 'meta' device).
    Returns the packages the class's module uses."""
    torch = _torch()
    packages: set = set()
    cls = _import_class(root, manifest["module"], manifest["class"], packages)
    args = _resolve(manifest["init_args"], root)
    try:
        with torch.device("meta"):
            net = cls(**args)
    except Exception:
        try:
            net = cls(**args)
        except Exception as e:
            raise ValidationError(
                f"The bundle cannot rebuild {manifest['class']} from its files: {type(e).__name__}: {e}. "
                "If the constructor needs arguments, pass them: init_args={...}") from e
    shape = lambda v: tuple(v.shape) if isinstance(v, torch.Tensor) else None  # noqa: E731
    expected = {k: shape(v) for k, v in net.state_dict().items()}
    saved = {k: shape(v) for k, v in state.items()}
    if expected != saved:
        missing = sorted(set(expected) - set(saved))[:3]
        extra = sorted(set(saved) - set(expected))[:3]
        wrong = [k for k in expected if k in saved and expected[k] != saved[k]][:3]
        raise ValidationError(
            f"The weights do not fit {manifest['class']} built with init_args={manifest['init_args']}: "
            f"missing {missing}, unexpected {extra}, different shapes {wrong}. "
            "Check init_args (the sizes your model was created with). If you change the model after "
            "creating it (pruning, LoRA, freezing), put those steps in a function and pass it as factory=.")
    if any(not isinstance(v, torch.Tensor) for v in state.values()):
        try:
            torch.load(root / WEIGHTS, map_location="cpu", weights_only=True)
        except Exception as e:
            raise ValidationError(
                "Some entries in the model's state_dict are not plain tensors (for example an object "
                "returned by get_extra_state()), and the bundle loads weights with weights_only=True so "
                f"that loading can never run code: {type(e).__name__}: {str(e).splitlines()[0][:150]}") from e
    return packages


# ---------------------------------------------------------------------------------------- load
def _under(root: Path, module: Any) -> bool:
    paths = [getattr(module, "__file__", None), *getattr(module, "__path__", [])]
    for p in paths:
        try:
            if p and Path(p).resolve().is_relative_to(root):
                return True
        except (OSError, ValueError):
            continue
    return False


@contextlib.contextmanager
def _bundle_modules(root: Path):
    """Inside this block, `import` finds the bundle's own files first, never a module of the same
    name that is already loaded (your architecture.py in the training process, or another bundle).
    Afterwards, the modules loaded from the bundle are forgotten and the old ones are put back."""
    root = root.resolve()
    tops = {p.stem for p in root.glob("*.py")} | {p.name for p in root.iterdir() if p.is_dir()}
    shadowed = {n: sys.modules.pop(n) for n in list(sys.modules) if n.split(".")[0] in tops}
    before = set(sys.modules)
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        sys.path.remove(str(root))
        for n in set(sys.modules) - before:
            if _under(root, sys.modules[n]):
                sys.modules.pop(n, None)
        sys.modules.update(shadowed)


def _import_class(root: Path, module: str, name: str, packages: Optional[set] = None):
    """Import `module` from `root` only and return `name` from it."""
    with _bundle_modules(root):
        try:
            mod = importlib.import_module(module)
            if packages is not None:
                packages.update(_namespace_packages(mod))
            return getattr(mod, name)
        except Exception as e:
            raise ValidationError(
                f"Could not import {name} from {module!r} in the bundle: {type(e).__name__}: {e}") from e


def read_manifest(root: PathLike) -> Dict[str, Any]:
    path = Path(root) / MANIFEST
    if not path.is_file():
        raise ValidationError(f"{str(root)!r} is not an authtracker bundle (no {MANIFEST})")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != MODEL_TYPE:
        raise ValidationError(f"{str(path)!r} is not an authtracker bundle manifest")
    if manifest.get("version", 0) > FORMAT_VERSION:
        raise ValidationError(
            f"This bundle uses format version {manifest['version']}; this authtracker reads up to "
            f"{FORMAT_VERSION}. Upgrade authtracker.")
    return manifest


def is_bundle(root: PathLike) -> bool:
    try:
        read_manifest(root)
        return True
    except (ValidationError, ValueError, OSError):
        return False


def _warn_if_environment_differs(manifest: Mapping[str, Any]) -> None:
    saved = manifest.get("environment", {})
    now_python, now_torch = platform.python_version(), _torch().__version__
    short = lambda v: ".".join(str(v).split("+")[0].split(".")[:2])  # noqa: E731
    notes = []
    if saved.get("python") and short(saved["python"]) != short(now_python):
        notes.append(f"Python {saved['python']} (this is {now_python})")
    if saved.get("torch") and short(saved["torch"]) != short(now_torch):
        notes.append(f"torch {saved['torch']} (this is {now_torch})")
    if notes:
        _log.warning("model.pkl was saved with %s. A pickle may fail to load, or behave differently, "
                     "on other versions; the folder bundle (architecture.py + weights.pth) does not have "
                     "this limit.", " and ".join(notes))


def _load_pickle(root: Path, manifest: Mapping[str, Any], device: str, eval: bool):
    torch = _torch()
    name = manifest.get("pickle") or PICKLE
    if not (root / name).is_file():
        raise ValidationError(f"This bundle has no {name}: it was saved without pickle=True")
    _warn_if_environment_differs(manifest)
    with _bundle_modules(root):                         # a pickle that names your classes finds this bundle's copy
        try:
            with open(root / name, "rb") as f:
                net = _pickle.load(f)
        except Exception as e:
            raise ValidationError(
                f"Could not load {name}: {type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}. "
                "A pickle needs the same Python and torch versions it was saved with.") from e
    if not isinstance(net, torch.nn.Module):
        raise ValidationError(f"{name} does not hold a torch.nn.Module")
    net.to(device)
    if eval:
        net.eval()
    return net


def load_dir(root: PathLike, *, device: str = "cpu", eval: bool = True, verify: bool = True,
             pickle: bool = False):
    """Rebuild the model from a bundle folder: the class from its source, then the weights.

    pickle=True loads model.pkl instead, which gives back the exact object (attributes and all).
    That runs code from the file, so it is opt-in; a folder saved with flavor='pickle' needs it.
    """
    torch = _torch()
    root = Path(root)
    manifest = read_manifest(root)
    if verify:                                          # before anything is imported or unpickled
        bad = [name for name, digest in manifest.get("files", {}).items()
               if not (root / name).is_file() or _sha256(root / name) != digest]
        if bad:
            raise ValidationError(f"The bundle was changed or is incomplete (hash mismatch): {bad[:5]}")
    if manifest.get("kind") == "pickle" and not pickle:
        raise ValidationError(
            "This model is stored as a pickle (flavor='pickle'). Loading a pickle runs code from the file, "
            "so it is opt-in: load it with pickle=True, and only if you trust where it came from.")
    if pickle:
        return _load_pickle(root, manifest, device, eval)
    cls = _import_class(root, manifest["module"], manifest["class"])
    args = _resolve(manifest["init_args"], root.resolve())
    try:
        net = cls(**args)
    except Exception as e:
        raise ValidationError(
            f"Could not build {manifest['class']}(**{args}): {type(e).__name__}: {e}. "
            "If the constructor reads files (a config, a vocabulary), ship them with assets=[...]") from e
    state = torch.load(root / manifest["weights"], map_location=device, weights_only=True)
    net.load_state_dict(state)
    for name, value in manifest.get("extras", {}).items():          # attributes that are not weights
        setattr(net, name, value)
    net.to(device)
    if eval:
        net.eval()
    _carry(net, root, manifest)
    return net


def _carry(net: Any, root: Path, manifest: Mapping[str, Any]) -> None:
    """Keep the bundle's code on the loaded model, so fine-tuning it and saving it again just works
    (the folder it was loaded from may be gone by then)."""
    skip = {WEIGHTS, MANIFEST, REQUIREMENTS, ENVIRONMENT}
    files = {name: (root / name).read_bytes() for name in manifest.get("files", {})
             if name not in skip and (root / name).is_file()}
    if sum(len(b) for b in files.values()) <= _CARRY_LIMIT:
        setattr(net, _CARRY, {"module": manifest["module"], "class": manifest["class"],
                              "factory": bool(manifest.get("factory")),
                              "init_args": dict(manifest["init_args"]),
                              "extras": dict(manifest.get("extras", {})), "files": files})
