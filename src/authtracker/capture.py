"""Copy everything a run prints or logs into a file, so it can be saved as `output.log`.

While at least one capture is active, `sys.stdout` and `sys.stderr` are replaced by wrappers that
still write to the real streams, and also to every active capture file. Logging handlers that were
created before the capture started (for example by `logging.basicConfig`) hold the old stream, so
they are pointed at the wrapper for the duration. Everything is restored when the last capture ends.

Not captured: output of subprocesses or native code that writes straight to the terminal, and
loggers that write only to files.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Callable, List, Optional, Tuple

MAX_BYTES = 20 * 1024 * 1024     # stop copying after this much, so a chatty job cannot fill the disk

_lock = threading.RLock()
_active: List["Capture"] = []
_installed: dict = {}


class Capture:
    def __init__(self, path: str, periodic: Optional[Callable[[str], None]] = None,
                 interval: float = 0.0):
        self.path = path
        self._file = open(path, "w", encoding="utf-8", errors="replace")
        self._size = 0
        self._full = False
        self.periodic = periodic          # called with the file path every `interval` seconds
        self.interval = interval
        self.lock = threading.Lock()      # held while uploading, so closing waits for an upload
        self.closed = False
        self._uploaded_size = -1
        self._next_due = time.monotonic() + interval

    def write(self, text: str) -> None:
        if self._full:
            return
        size = len(text.encode("utf-8", "replace"))
        if self._size + size > MAX_BYTES:
            self._file.write(f"\n[authtracker: output.log truncated after {MAX_BYTES // 2**20} MB]\n")
            self._full = True
            return
        self._file.write(text)
        self._size += size

    def tick(self) -> None:
        """Upload the log so far, if it grew since the last upload. Never raises."""
        with self.lock:
            if self.closed or self._size == self._uploaded_size:
                return
            try:
                self._file.flush()
                self.periodic(self.path)
                self._uploaded_size = self._size
            except Exception:
                pass                      # best effort: the final upload still happens at the end

    def close(self) -> None:
        with self.lock:
            self.closed = True
            try:
                self._file.close()
            except Exception:
                pass


class _Tee:
    """Looks like the stream it wraps, and also copies what is written to the active captures."""

    def __init__(self, original):
        self._original = original

    def write(self, text):
        written = self._original.write(text)
        with _lock:
            for capture in _active:
                try:
                    capture.write(text)
                except Exception:
                    pass          # capturing must never break the program
        return written

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        self._original.flush()

    def __getattr__(self, name):      # isatty, encoding, fileno, ...
        return getattr(self._original, name)


def _loggers() -> List[logging.Logger]:
    return [logging.root] + [lg for lg in logging.root.manager.loggerDict.values()
                             if isinstance(lg, logging.Logger)]


def _install() -> None:
    out, err = sys.stdout, sys.stderr
    tee_out = _Tee(out) if out is not None else None
    tee_err = _Tee(err) if err is not None else None
    patched: List[Tuple[logging.StreamHandler, object]] = []
    for lg in _loggers():
        for h in lg.handlers:
            if isinstance(h, logging.StreamHandler):
                if tee_err is not None and h.stream is err:
                    patched.append((h, h.stream))
                    h.stream = tee_err
                elif tee_out is not None and h.stream is out:
                    patched.append((h, h.stream))
                    h.stream = tee_out
    if tee_out is not None:
        sys.stdout = tee_out
    if tee_err is not None:
        sys.stderr = tee_err
    _installed.update(out=out, err=err, tee_out=tee_out, tee_err=tee_err, patched=patched)


def _uninstall() -> None:
    if _installed.get("tee_out") is not None and sys.stdout is _installed["tee_out"]:
        sys.stdout = _installed["out"]
    if _installed.get("tee_err") is not None and sys.stderr is _installed["tee_err"]:
        sys.stderr = _installed["err"]
    for handler, stream in _installed.get("patched", []):
        if handler.stream in (_installed["tee_out"], _installed["tee_err"]):
            handler.stream = stream
    _installed.clear()


_ticker: Optional[threading.Thread] = None
_ticker_stop = threading.Event()


def _tick_loop() -> None:
    while not _ticker_stop.wait(0.5):
        with _lock:
            due = [c for c in _active if c.periodic and c.interval > 0
                   and time.monotonic() >= c._next_due]
        for capture in due:
            capture._next_due = time.monotonic() + capture.interval
            capture.tick()                # outside _lock: an upload can take a while


def start(path: str, periodic: Optional[Callable[[str], None]] = None,
          interval: float = 0.0) -> Capture:
    """Start copying output into `path`. With `periodic` and `interval`, the file is also handed to
    `periodic` every `interval` seconds while the run is going (a killed job keeps its log)."""
    global _ticker
    capture = Capture(path, periodic, interval)
    with _lock:
        if not _active:
            _install()
        _active.append(capture)
        if periodic and interval > 0 and (_ticker is None or not _ticker.is_alive()):
            _ticker_stop.clear()
            _ticker = threading.Thread(target=_tick_loop, name="authtracker-log-upload", daemon=True)
            _ticker.start()
    return capture


def stop(capture: Capture) -> None:
    with _lock:
        if capture in _active:
            _active.remove(capture)
        if not _active and _installed:
            _uninstall()
        if not _active:
            _ticker_stop.set()
    capture.close()                       # waits for an upload that is in progress
