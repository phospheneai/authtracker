"""Configuration, read entirely from AUTHTRACKER_* environment variables.

No config file is required. Everything has a default, and code can override
settings explicitly with ``authtracker.configure(Settings(...))``.
"""
from __future__ import annotations

import getpass
import os
import re
import warnings
from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Tuple

PREFIX = "AUTHTRACKER_"

# What can be sent to the alert webhook: a run failing or being stopped, a run finishing, and your own at.alert() calls.
ALERT_EVENTS = ("failed", "stopped", "finished", "custom")
ALERT_DEFAULT = ("failed", "stopped", "custom")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


class ConfigError(ValueError):
    """An AUTHTRACKER_* variable has an invalid value."""


def _csv(value: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in value.split(",") if x.strip())


def _bool(name: str, value: str) -> bool:
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ConfigError(f"{PREFIX}{name}={value!r} is not a boolean (use true/false)")


def _float(name: str, value: str) -> float:
    try:
        return float(value)
    except ValueError:
        raise ConfigError(f"{PREFIX}{name}={value!r} is not a number") from None


def _int(name: str, value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise ConfigError(f"{PREFIX}{name}={value!r} is not an integer") from None
    if n < 0:
        raise ConfigError(f"{PREFIX}{name}={value!r} must be >= 0")
    return n


def _default_owner() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class Settings:
    backend: str = "mlflow"
    tracking_uri: Optional[str] = None
    registry_uri: Optional[str] = None
    owner: str = ""
    required_tags: Tuple[str, ...] = ()
    allowed_models: Tuple[str, ...] = ()
    metric_names: Tuple[str, ...] = ()
    capture_env: bool = True
    capture_logs: bool = True
    save_code: bool = True
    image: Optional[str] = None
    log_upload_interval: float = 60.0
    system_metrics: bool = True
    promote_metric: Optional[str] = None
    promote_min: Optional[float] = None
    promote_must_beat_prod: bool = False
    gated_aliases: Tuple[str, ...] = ("production",)
    offline_queue: bool = True
    queue_dir: Optional[str] = None
    retries: int = 3
    retry_wait: float = 1.0
    alert_webhook: Optional[str] = None      # a SECRET: only from AUTHTRACKER_ALERT_WEBHOOK, masked when shown
    alert_on: Tuple[str, ...] = ALERT_DEFAULT  # which events are sent to it (none are sent without a webhook)

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "Settings":
        env = os.environ if environ is None else environ

        def get(name: str) -> Optional[str]:
            v = env.get(PREFIX + name)
            return v if v not in (None, "") else None

        known = {
            "BACKEND", "TRACKING_URI", "REGISTRY_URI", "OWNER", "REQUIRED_TAGS",
            "ALLOWED_MODELS", "METRIC_NAMES", "CAPTURE_ENV", "CAPTURE_LOGS", "SAVE_CODE", "IMAGE", "LOG_UPLOAD_INTERVAL", "SYSTEM_METRICS", "PROMOTE_METRIC",
            "PROMOTE_MIN", "PROMOTE_MUST_BEAT_PROD", "GATED_ALIASES",
            "OFFLINE_QUEUE", "QUEUE_DIR", "RETRIES", "RETRY_WAIT", "ALERT_WEBHOOK", "ALERT_ON",
        }
        for key in env:
            if key.startswith(PREFIX) and key[len(PREFIX):] not in known:
                warnings.warn(f"Unknown variable {key} (typo?)", stacklevel=2)

        kw = {}
        if (v := get("BACKEND")) is not None:
            kw["backend"] = v.strip().lower()
        # Fall back to MLflow's own variable so existing setups keep working.
        uri = get("TRACKING_URI") or env.get("MLFLOW_TRACKING_URI") or None
        if uri:
            kw["tracking_uri"] = uri
        if (v := get("REGISTRY_URI")) is not None:
            kw["registry_uri"] = v
        kw["owner"] = get("OWNER") or _default_owner()
        if (v := get("REQUIRED_TAGS")) is not None:
            kw["required_tags"] = _csv(v)
        if (v := get("ALLOWED_MODELS")) is not None:
            kw["allowed_models"] = _csv(v)
        if (v := get("METRIC_NAMES")) is not None:
            kw["metric_names"] = _csv(v)
        if (v := get("CAPTURE_ENV")) is not None:
            kw["capture_env"] = _bool("CAPTURE_ENV", v)
        if (v := get("IMAGE")) is not None:
            kw["image"] = v
        if (v := get("SAVE_CODE")) is not None:
            kw["save_code"] = _bool("SAVE_CODE", v)
        if (v := get("LOG_UPLOAD_INTERVAL")) is not None:
            kw["log_upload_interval"] = _float("LOG_UPLOAD_INTERVAL", v)
            if kw["log_upload_interval"] < 0:
                raise ConfigError(f"{PREFIX}LOG_UPLOAD_INTERVAL={v!r} must be 0 or more")
        if (v := get("SYSTEM_METRICS")) is not None:
            kw["system_metrics"] = _bool("SYSTEM_METRICS", v)
        if (v := get("CAPTURE_LOGS")) is not None:
            kw["capture_logs"] = _bool("CAPTURE_LOGS", v)
        if (v := get("PROMOTE_METRIC")) is not None:
            kw["promote_metric"] = v
        if (v := get("PROMOTE_MIN")) is not None:
            kw["promote_min"] = _float("PROMOTE_MIN", v)
        if (v := get("PROMOTE_MUST_BEAT_PROD")) is not None:
            kw["promote_must_beat_prod"] = _bool("PROMOTE_MUST_BEAT_PROD", v)
        if (v := get("GATED_ALIASES")) is not None:
            kw["gated_aliases"] = _csv(v)
        if (v := get("OFFLINE_QUEUE")) is not None:
            kw["offline_queue"] = _bool("OFFLINE_QUEUE", v)
        if (v := get("QUEUE_DIR")) is not None:
            kw["queue_dir"] = v
        if (v := get("RETRIES")) is not None:
            kw["retries"] = _int("RETRIES", v)
        if (v := get("RETRY_WAIT")) is not None:
            kw["retry_wait"] = _float("RETRY_WAIT", v)
        if (v := get("ALERT_WEBHOOK")) is not None:
            if not v.lower().startswith(("https://", "http://")):
                raise ConfigError(f"{PREFIX}ALERT_WEBHOOK must be an http(s) URL (the value is not shown: it is a secret)")
            kw["alert_webhook"] = v.strip()
        if (v := get("ALERT_ON")) is not None:
            chosen = tuple(x.lower() for x in _csv(v))
            bad = [x for x in chosen if x not in ALERT_EVENTS]
            if bad:
                raise ConfigError(f"{PREFIX}ALERT_ON has unknown events {bad}; choose from {', '.join(ALERT_EVENTS)}")
            kw["alert_on"] = chosen
        return cls(**kw)

    def masked(self) -> dict:
        """Settings as a dict with credentials in URIs hidden."""
        d = asdict(self)
        for k in ("tracking_uri", "registry_uri"):
            if d[k]:
                d[k] = re.sub(r"://[^/@]+@", "://***@", d[k])
        if d["alert_webhook"]:                       # the path of a webhook URL is the secret
            d["alert_webhook"] = re.sub(r"^(https?://[^/]+/).*$", r"\1***", d["alert_webhook"])
        return d


_configured: Optional[Settings] = None


def configure(settings: Optional[Settings] = None) -> None:
    """Set settings explicitly (or pass None to go back to reading the environment)."""
    global _configured
    _configured = settings


def get_settings() -> Settings:
    return _configured if _configured is not None else Settings.from_env()
