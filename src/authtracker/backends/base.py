"""The only interface the rest of authtracker talks to.

To support another tracking platform, subclass `Backend`:

* implement the REQUIRED methods (runs, params, metrics, tags, artifacts, queries);
* optionally implement the model methods (`log_model`) and the registry methods
  (`register`, `set_alias`, `get_alias`, `version_run_id`, `load`). Anything you leave out
  raises a clear BackendError when a user calls it;
* register it with `register_backend("name", factory)`, or ship it in a separate package via the
  `authtracker.backends` entry-point group (no change to authtracker needed);
* check it with `authtracker.testing.check_backend(your_backend)`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..errors import BackendError, UnsupportedError

# Bump when a change to this interface would break existing backends.
BACKEND_API_VERSION = 1

# Tags the package itself writes.
TAG_LEVEL = "authtracker.level"          # "experiment" (parent) or "run" (child)
TAG_MODEL_URI = "authtracker.model_uri"  # where log_model put the model
TAG_RUN_ID = "authtracker.run_id"        # on model versions: source run

_TRANSIENT_NAMES = {"ConnectionError", "Timeout", "ConnectTimeout", "ReadTimeout",
                    "TimeoutError", "OperationalError"}


class Backend(ABC):
    api_version = BACKEND_API_VERSION
    # True if the platform already buffers writes while offline (for example W&B's offline mode).
    # authtracker then skips its own retry/queue layer for this backend.
    buffers_offline = False
    # True if the platform already records everything the program prints (W&B does). authtracker
    # then does not copy the console output into an `output.log` artifact itself.
    captures_console = False

    def is_transient(self, exc: Exception) -> bool:
        """True if `exc` looks like a temporary outage (worth retrying / queuing)."""
        return any(c.__name__ in _TRANSIENT_NAMES for c in type(exc).__mro__)

    def _unsupported(self, what: str) -> BackendError:
        return UnsupportedError(f"The {type(self).__name__} backend does not support {what}")

    # ------------------------------------------------------------------ required
    @abstractmethod
    def ping(self) -> None:
        """Raise if the backend is unreachable."""

    @abstractmethod
    def start_run(self, experiment: str, run_name: str, tags: Dict[str, str],
                  parent_id: Optional[str]) -> str:
        """Create a run in `experiment` (the model name) and return its id."""

    @abstractmethod
    def end_run(self, run_id: str, status: str) -> None:
        """Status is FINISHED, FAILED or KILLED."""

    @abstractmethod
    def set_tags(self, run_id: str, tags: Dict[str, str]) -> None: ...

    @abstractmethod
    def log_params(self, run_id: str, params: Dict[str, Any]) -> None: ...

    @abstractmethod
    def log_metrics(self, run_id: str, metrics: Dict[str, float], step: Optional[int]) -> None: ...

    @abstractmethod
    def log_artifact(self, run_id: str, path: str) -> None:
        """Store the file at `path` under its own file name."""

    @abstractmethod
    def get_run(self, run_id: str) -> Dict[str, Any]:
        """Return {'run_id','name','metrics','params','tags'} (metrics = latest values)."""

    @abstractmethod
    def search_runs(self, experiment: str, tags: Dict[str, str], metric: str,
                    mode: str, limit: int) -> List[Dict[str, Any]]:
        """Runs in `experiment` having all `tags` and a value for `metric`, best first
        ('max' = highest first, 'min' = lowest first), at most `limit`, same shape as get_run.
        An unknown experiment gives []."""

    # ------------------------------------------------------------------ optional: resuming runs
    def find_run(self, experiment: str, name: str, tags: Dict[str, str],
                 parent_id: Optional[str]) -> Optional[str]:
        """Id of the newest run in `experiment` that is called `name`, has all `tags` and is a child of
        `parent_id` (None for a top-level run); None if there is no such run."""
        raise self._unsupported("resuming runs")

    def resume_run(self, run_id: str, parent_id: Optional[str]) -> str:
        """Reopen a finished run so that more can be logged to it, and return its id. If `parent_id` is
        given, that run is the open run it is nested in."""
        raise self._unsupported("resuming runs")

    def last_step(self, run_id: str) -> Optional[int]:
        """The highest metric step logged to the run so far (None: no metrics, or not known)."""
        return None

    def download_artifact(self, run_id: str, name: str, folder: str) -> Optional[str]:
        """Copy the file `name` that was saved on the run into `folder` and return its path; None if the run has no such file."""
        raise self._unsupported("downloading files")

    # ------------------------------------------------------------------ optional: models
    def log_model(self, run_id: str, model: Any, flavor: Optional[str], **kwargs) -> str:
        """Log a model for the run, set the TAG_MODEL_URI tag on it and return the URI."""
        raise self._unsupported("logging models")

    def log_model_dir(self, run_id: str, local_dir: str) -> str:
        """Store a folder as the run's model (a PyTorch bundle), set TAG_MODEL_URI and return its URI.
        `register` must then be able to make a version from that URI."""
        raise self._unsupported("saving model bundles")

    def download_bundle(self, name: str, alias: str, dst: Optional[str] = None) -> Optional[str]:
        """If version `alias` of `name` is a model bundle, copy it into a NEW folder and return its
        path (the caller removes it); otherwise return None."""
        return None

    # ------------------------------------------------------------------ optional: registry
    def register(self, run_id: str, name: str, model_uri: str) -> str:
        """Create a new model version and return its version string."""
        raise self._unsupported("the model registry")

    def set_alias(self, name: str, version: str, alias: str) -> None:
        raise self._unsupported("the model registry")

    def get_alias(self, name: str, alias: str) -> Optional[str]:
        raise self._unsupported("the model registry")

    def version_run_id(self, name: str, version: str) -> str:
        raise self._unsupported("the model registry")

    def load(self, name: str, alias: str) -> Any:
        raise self._unsupported("the model registry")
