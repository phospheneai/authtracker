"""Alerts: a JSON message POSTed to a webhook URL when something you chose to hear about happens.

How it fits together (everything about alerts is in this file):

* `Alert`            one message: which event it is for, a title, optional text, a level and some fields.
* `send`             delivers an Alert to the webhook, if one is set and the event was chosen. Never raises.
* `notify_run_end`   the one hook the tracking code calls when an experiment ends (failed, stopped or finished).
* `at.alert(...)`    (in simple.py) builds a "custom" Alert from your own code.

Everything is optional. With no AUTHTRACKER_ALERT_WEBHOOK nothing is sent; with one, only the events listed in
AUTHTRACKER_ALERT_ON are. The URL is a secret (anyone who has it can post to your channel): it is read only from the
environment, masked wherever settings are shown, and never put in a log line or an error message.

To add an event: add its name to ALERT_EVENTS in settings.py, then build an `Alert` for it where it happens.
To add another way of delivering (email, say): write a function like `_post` and pass it as `send(post=...)`.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional

from .settings import ALERT_EVENTS, Settings

log = logging.getLogger("authtracker")

FAILED, STOPPED, FINISHED, CUSTOM = ALERT_EVENTS
LEVELS = ("info", "warn", "error")
TIMEOUT = 5.0                   # seconds: an alert must never hold a job up for long


@dataclass(frozen=True)
class Alert:
    event: str                                  # one of ALERT_EVENTS
    title: str
    text: str = ""
    level: str = "warn"                         # one of LEVELS
    fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event not in ALERT_EVENTS:      # a typo in the code, so fail at once instead of never alerting
            raise ValueError(f"unknown alert event {self.event!r}; use one of {ALERT_EVENTS}")
        if self.level not in LEVELS:
            raise ValueError(f"unknown alert level {self.level!r}; use one of {LEVELS}")

    def payload(self) -> Dict[str, Any]:
        """The JSON that is sent. `text` is the whole message ready to read (chat tools whose incoming webhooks take a
        `text` field show it as it is); the other keys carry the same information separately, for anything that parses it."""
        known = {k: v for k, v in self.fields.items() if v not in (None, "")}
        lines = [f"[{self.level.upper()}] {self.title}"] + ([self.text] if self.text else [])
        lines += [f"{k}: {v}" for k, v in known.items()]
        return {"text": "\n".join(lines), "level": self.level, "title": self.title, "message": self.text, **known,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def _post(url: str, payload: Mapping[str, Any], timeout: float = TIMEOUT) -> None:
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:      # the scheme is checked in Settings
        response.read()


def send(settings: Settings, alert: Alert, *, force: bool = False,
         post: Optional[Callable[..., None]] = None) -> bool:
    """Deliver `alert`. Returns True if the webhook accepted it. False means: no webhook is set, the event was not
    chosen in AUTHTRACKER_ALERT_ON, or the webhook could not be reached. `force` sends even if the event was not
    chosen (for `authtracker test-alert`)."""
    url = settings.alert_webhook
    if not url or not (force or alert.event in settings.alert_on):
        return False
    try:
        (post or _post)(url, alert.payload())
        return True
    except Exception as e:                      # never raise, and never print the URL
        log.warning("authtracker: could not send the alert (%s). Check AUTHTRACKER_ALERT_WEBHOOK.", type(e).__name__)
        return False


def run_fields(tags: Mapping[str, str], run_id: Optional[str] = None) -> Dict[str, Any]:
    """Who and what an alert is about, taken from a run's tags."""
    return {"model": tags.get("model"), "experiment": tags.get("experiment"), "owner": tags.get("owner"),
            "run_id": run_id}


def one_line(error: Optional[BaseException], limit: int = 300) -> str:
    """The exception type and the first line of its message, shortened: enough to see what broke, and it keeps a
    long message (which may hold data) out of the channel."""
    if error is None:
        return ""
    first = (str(error).strip().splitlines() or [""])[0]
    return f"{type(error).__name__}: {first}"[:limit]


def notify_run_end(settings: Settings, tags: Mapping[str, str], run_id: str, status: str,
                   failure: Optional[BaseException], seconds: float) -> None:
    """Called when an experiment run ends. `status` is FINISHED, FAILED or KILLED. Never raises."""
    if not settings.alert_webhook:
        return
    try:
        fields = run_fields(tags, run_id)
        if status == "FAILED":
            alert = Alert(FAILED, "Run failed", level="error", fields={**fields, "error": one_line(failure)})
        elif status == "KILLED":
            alert = Alert(STOPPED, "Run stopped", "It was interrupted (Ctrl+C).", "warn", fields)
        else:
            alert = Alert(FINISHED, "Run finished", level="info", fields={**fields, "took": f"{seconds:.0f}s"})
        send(settings, alert)
    except Exception:                           # an alert is a convenience: never fail a run because of it
        log.debug("authtracker: could not build the run-end alert", exc_info=True)
