"""Backend lookup: built-ins, runtime registration, and installed plugins.

Resolution order for AUTHTRACKER_BACKEND=<name>:
1. a backend registered in code with `register_backend`;
2. an installed package that declares it in the `authtracker.backends` entry-point group, e.g.
   in that package's pyproject.toml:

       [project.entry-points."authtracker.backends"]
       wandb = "authtracker_wandb:make_backend"      # callable(settings) -> Backend
"""
from __future__ import annotations

from importlib import metadata
from typing import Callable, Dict, List, Optional

from ..errors import BackendError
from ..settings import Settings
from .base import BACKEND_API_VERSION, Backend

ENTRY_POINT_GROUP = "authtracker.backends"

_REGISTRY: Dict[str, Callable[[Settings], Backend]] = {}


def register_backend(name: str, factory: Callable[[Settings], Backend]) -> None:
    _REGISTRY[name] = factory


def _make_local(settings: Settings) -> Backend:
    from .local import LocalBackend
    return LocalBackend(settings)


def _make_mlflow(settings: Settings) -> Backend:
    try:
        from .mlflow_backend import MlflowBackend
    except ImportError as e:  # mlflow not installed
        raise BackendError(
            "Backend 'mlflow' needs MLflow. Install one of: pip install 'authtracker[mlflow]' "
            "(full), or pip install 'authtracker[mlflow-skinny]' (light: only sends runs to a server)"
        ) from e
    return MlflowBackend(settings)


def _make_wandb(settings: Settings) -> Backend:
    try:
        from .wandb_backend import WandbBackend
    except ImportError as e:  # wandb not installed
        raise BackendError("Backend 'wandb' needs W&B: pip install 'authtracker[wandb]'") from e
    return WandbBackend(settings)


register_backend("mlflow", _make_mlflow)
register_backend("local", _make_local)
register_backend("wandb", _make_wandb)


def _entry_points() -> List[metadata.EntryPoint]:
    try:
        return list(metadata.entry_points(group=ENTRY_POINT_GROUP))
    except TypeError:  # Python < 3.10
        return list(metadata.entry_points().get(ENTRY_POINT_GROUP, []))


def available_backends() -> List[str]:
    return sorted(set(_REGISTRY) | {ep.name for ep in _entry_points()})


def _find_factory(name: str) -> Optional[Callable[[Settings], Backend]]:
    if name in _REGISTRY:
        return _REGISTRY[name]
    for ep in _entry_points():
        if ep.name == name:
            return ep.load()
    return None


def get_backend(settings: Settings) -> Backend:
    factory = _find_factory(settings.backend)
    if factory is None:
        raise BackendError(
            f"Unknown backend {settings.backend!r}. Available: {available_backends()}")
    backend = factory(settings)
    version = getattr(backend, "api_version", None)
    if version != BACKEND_API_VERSION:
        raise BackendError(
            f"Backend {settings.backend!r} implements backend API version {version}, "
            f"but this authtracker needs version {BACKEND_API_VERSION}")
    if settings.offline_queue and not backend.buffers_offline:
        from .resilient import ResilientBackend
        backend = ResilientBackend(backend, settings)
    return backend


__all__ = ["Backend", "get_backend", "register_backend", "available_backends"]
