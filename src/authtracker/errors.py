class AuthTrackerError(Exception):
    """Base class for authtracker errors."""


class ValidationError(AuthTrackerError, ValueError):
    """A run violates the configured convention (tags, model name, metric names)."""


class PromotionError(AuthTrackerError):
    """A model version failed a promotion gate."""


class BackendError(AuthTrackerError):
    """The backend is unknown, missing a dependency, or does not support an operation."""


class UnsupportedError(BackendError):
    """The backend does not implement this optional feature (for example resuming runs)."""
