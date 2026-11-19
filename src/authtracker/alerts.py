"""Alerts: a JSON message POSTed to a webhook URL when something you chose to hear about happens.

Everything is optional. With no AUTHTRACKER_ALERT_WEBHOOK nothing is sent. With one, only the events listed in
AUTHTRACKER_ALERT_ON are sent (a run failing, being stopped, finishing, or your own `at.alert(...)` calls).

The URL is a secret (anyone who has it can post to your channel), so it is only read from the environment, it is
masked wherever settings are shown, and it is never put in a log line or an error message. Sending never raises:
a broken webhook must not stop a training run.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional

from .settings import ALERT_EVENTS, Settings

log = logging.getLogger("authtracker")

LEVELS = ("info", "warn", "error")
TIMEOUT = 5.0                  # seconds: an alert must never hold a job up for long


def _post(url: str, payload: Mapping[str, Any], timeout: float = TIMEOUT) -> None:
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:      # noqa: S310 (scheme checked in Settings)
        response.read()


def payload(title: str, text: str, level: str, fields: Mapping[str, Any]) -> Dict[str, Any]:
    """The JSON that is sent. `text` is the whole message ready to read (chat tools whose incoming webhooks take a
    `text` field show it as is); the other keys are the same information as separate fields for anything that parses it."""
    known = {k: v for k, v in fields.items() if v not in (None, "")}
    lines = [f"[{level.upper()}] {title}"] + ([text] if text else []) + [f"{k}: {v}" for k, v in known.items()]
    return {"text": "\n".join(lines), "level": level, "title": title, "message": text, **known,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def wanted(settings: Settings, event: str) -> bool:
    """True if a webhook is set and `event` is one of the events chosen in AUTHTRACKER_ALERT_ON."""
    return bool(settings.alert_webhook) and event in settings.alert_on


def send(settings: Settings, event: str, title: str, text: str = "", level: str = "warn",
         fields: Optional[Mapping[str, Any]] = None, *, force: bool = False,
         post: Optional[Callable[..., None]] = None) -> bool:
    """Send one alert for `event` ("failed", "stopped", "finished" or "custom"). Returns True if the webhook accepted
    it. `force` sends even when the event was not chosen (for `authtracker test-alert`)."""
    if event not in ALERT_EVENTS:
        raise ValueError(f"unknown alert event {event!r}; use one of {ALERT_EVENTS}")
    url = settings.alert_webhook
    if not url or not (force or event in settings.alert_on):
        return False
    level = level if level in LEVELS else "warn"
    try:
        (post or _post)(url, payload(title, text, level, fields or {}))
        return True
    except Exception as e:                                 # never raise, and never print the URL
        log.warning("authtracker: could not send the alert (%s). Check AUTHTRACKER_ALERT_WEBHOOK.", type(e).__name__)
        return False


def run_fields(tags: Mapping[str, str], run_id: Optional[str] = None) -> Dict[str, Any]:
    return {"model": tags.get("model"), "experiment": tags.get("experiment"), "owner": tags.get("owner"),
            "run_id": run_id}


def one_line(error: BaseException, limit: int = 300) -> str:
    """The exception type and the first line of its message, shortened: enough to see what broke, and it keeps
    a long message (which may hold data) out of the channel."""
    first = (str(error).strip().splitlines() or [""])[0]
    return f"{type(error).__name__}: {first}"[:limit]
