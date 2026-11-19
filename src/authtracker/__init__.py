"""authtracker: variable-configured experiment tracking and model versioning."""
from .errors import AuthTrackerError, BackendError, PromotionError, UnsupportedError, ValidationError
from .bundle import load_dir as load_bundle
from .registry import (best_run, export_all, export_model, import_all, import_model, load, promote, register,
                       rollback)
from .run import Run, run
from .simple import alert, current_run, finish, init, log, log_model, restore, save, set_tags, start, watch
from .settings import ConfigError, Settings, configure, get_settings, get_tracking_uri, set_tracking_uri

__version__ = "0.0.6"

__all__ = [
    "init", "start", "log", "save", "log_model", "restore", "watch", "finish", "set_tags", "current_run", "alert",
    "run", "Run", "register", "promote", "rollback", "load", "load_bundle", "best_run",
    "export_model", "import_model", "export_all", "import_all",
    "Settings", "configure", "get_settings", "set_tracking_uri", "get_tracking_uri", "ConfigError",
    "AuthTrackerError", "ValidationError", "PromotionError", "BackendError", "UnsupportedError",
]
