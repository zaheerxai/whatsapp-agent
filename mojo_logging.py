"""
mojo_logging.py — Local application log with daily rotation + on-demand OneDrive upload.

Design (bandwidth-safe, free-tier aware):
- All logging is local only. No background upload thread.
- Local file lives under /tmp (or MOJO_LOG_DIR). On Render Free this is
  ephemeral — wiped on every restart / spin-down / deploy. That is expected.
- Local file starts clean every calendar day (UTC) while the process is up.
- Size-based safety rotation still applies.
- Owner runs /uploadlog to push the *current process* local log to OneDrive.
- Remote name is fixed (mojo_agent_live.log) → overwrite. If the user deletes
  the remote file, the next /uploadlog creates a fresh one with current local
  content only.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Optional

LOG_DIR = os.environ.get("MOJO_LOG_DIR", "/tmp/mojo_logs")
LOG_NAME = "mojo_agent.log"
REMOTE_NAME = "mojo_agent_live.log"  # fixed name → overwrite on demand
MAX_LOCAL_BYTES = int(os.environ.get("MOJO_LOG_MAX_BYTES", str(5 * 1024 * 1024)))  # 5 MB safety

_lock = threading.Lock()
_started = False
_log_path = os.path.join(LOG_DIR, LOG_NAME)
_current_day: Optional[str] = None  # YYYY-MM-DD (UTC)
_file_handler: Optional[logging.FileHandler] = None


def _ensure_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _touch_log_file():
    """Ensure the active log file exists on disk (empty is fine)."""
    _ensure_dir()
    if not os.path.isfile(_log_path):
        with open(_log_path, "a", encoding="utf-8"):
            pass


def _reopen_file_handler():
    """After rotation, point the FileHandler at a fresh stream for _log_path."""
    global _file_handler
    if _file_handler is None:
        return
    try:
        _file_handler.close()
    except Exception:
        pass
    try:
        _touch_log_file()
        _file_handler.baseFilename = os.path.abspath(_log_path)
        _file_handler.stream = _file_handler._open()  # type: ignore[attr-defined]
    except Exception as e:
        print(f"[mojo_logging] reopen handler failed: {e}")


def _rotate_daily_if_needed():
    """
    If the calendar day (UTC) has changed since the current log was opened,
    rename mojo_agent.log → mojo_agent.YYYY-MM-DD.log and start a fresh file.
    """
    global _current_day
    today = _today_utc()
    if _current_day is None:
        _current_day = today
        return
    if today == _current_day:
        return

    try:
        if os.path.isfile(_log_path) and os.path.getsize(_log_path) > 0:
            dated = os.path.join(LOG_DIR, f"mojo_agent.{_current_day}.log")
            if os.path.isfile(dated):
                with open(_log_path, "rb") as src, open(dated, "ab") as dst:
                    dst.write(src.read())
                os.remove(_log_path)
            else:
                os.replace(_log_path, dated)
            print(f"[mojo_logging] Daily rotate → {os.path.basename(dated)}")
            _reopen_file_handler()
        _current_day = today
    except Exception as e:
        print(f"[mojo_logging] daily rotate failed: {e}")


def _rotate_size_if_needed():
    """Safety: if active log exceeds MAX_LOCAL_BYTES, move it aside."""
    try:
        if os.path.isfile(_log_path) and os.path.getsize(_log_path) > MAX_LOCAL_BYTES:
            bak = _log_path + ".prev"
            if os.path.isfile(bak):
                os.remove(bak)
            os.replace(_log_path, bak)
            print(f"[mojo_logging] Size rotate → {os.path.basename(bak)}")
            _reopen_file_handler()
    except Exception as e:
        print(f"[mojo_logging] size rotate failed: {e}")


def _maybe_rotate():
    with _lock:
        _rotate_daily_if_needed()
        _rotate_size_if_needed()


class _FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        try:
            _maybe_rotate()
        except Exception:
            pass
        # If stream was closed by rotation, reopen before write
        if self.stream is None:
            try:
                self.stream = self._open()
            except Exception:
                pass
        super().emit(record)
        try:
            self.flush()
        except Exception:
            pass


def setup_logging():
    """Call once at startup. Local file only — no network. Idempotent."""
    global _started, _current_day, _file_handler
    if _started:
        _touch_log_file()
        return _log_path

    _ensure_dir()
    _current_day = _today_utc()
    _rotate_size_if_needed()
    _touch_log_file()

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = _FlushFileHandler(_log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    _file_handler = fh

    # Console mirror
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    ):
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    # Quiet noisy HTTP clients
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _started = True
    logging.getLogger("mojo").info(
        "Live log ready (local only, daily rotate, ephemeral on free tier): %s",
        _log_path,
    )
    return _log_path


def log_line(msg: str, level: int = logging.INFO):
    if not _started:
        setup_logging()
    logging.getLogger("mojo").log(level, msg)


def get_log_path() -> str:
    if not _started:
        setup_logging()
    return _log_path


def get_log_status() -> dict:
    """Return size / day info for the active local log (and any dated siblings)."""
    if not _started:
        setup_logging()
    _maybe_rotate()
    size = 0
    if os.path.isfile(_log_path):
        size = os.path.getsize(_log_path)
    dated = []
    try:
        for name in sorted(os.listdir(LOG_DIR)):
            if name.startswith("mojo_agent.") and name.endswith(".log") and name != LOG_NAME:
                p = os.path.join(LOG_DIR, name)
                if os.path.isfile(p):
                    dated.append({"name": name, "bytes": os.path.getsize(p)})
    except Exception:
        pass
    return {
        "path": _log_path,
        "bytes": size,
        "day": _current_day or _today_utc(),
        "remote_name": REMOTE_NAME,
        "dated": dated,
        "started": _started,
        "ephemeral_note": (
            "On Render Free, /tmp is wiped on every restart/spin-down/deploy. "
            "This file only holds logs from the current process lifetime."
        ),
    }


def upload_current_log(
    file_ops_module,
    remote_folder: Optional[str] = None,
    include_yesterday: bool = False,
) -> dict:
    """
    On-demand upload of the current local log to OneDrive.

    - Always overwrites the fixed remote name (mojo_agent_live.log).
    - If the remote file was deleted/moved by the user, Graph creates a new
      file with only the bytes we send now (current local content).
    - Does NOT pull any remote history back into the local file.
    - On free-tier Render, local content is only from the current process uptime.
    """
    if not _started:
        setup_logging()

    if not file_ops_module or not getattr(file_ops_module, "onedrive_configured", lambda: False)():
        raise RuntimeError("OneDrive not configured")

    _maybe_rotate()
    _touch_log_file()

    if not os.path.isfile(_log_path):
        raise FileNotFoundError(
            f"No local log at {_log_path}. "
            "On Render Free the disk is ephemeral — logs only exist for the "
            "current process. Send a few messages first, then /uploadlog again."
        )

    size = os.path.getsize(_log_path)
    # Allow empty file upload (still useful as a heartbeat) but flag it
    folder = (remote_folder or os.getenv("ONEDRIVE_FOLDER") or "MojoAgent").strip("/")

    with _lock:
        meta = file_ops_module.upload_to_onedrive(
            _log_path,
            remote_folder=folder,
            remote_name=REMOTE_NAME,
        )

    result = {
        "remote_folder": folder,
        "remote_name": REMOTE_NAME,
        "bytes": meta.get("size") or size,
        "webUrl": meta.get("webUrl"),
        "local_path": _log_path,
        "day": _current_day or _today_utc(),
        "empty": size == 0,
    }

    if include_yesterday:
        try:
            from datetime import timedelta

            yday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
            ypath = os.path.join(LOG_DIR, f"mojo_agent.{yday}.log")
            if os.path.isfile(ypath):
                ymeta = file_ops_module.upload_to_onedrive(
                    ypath,
                    remote_folder=folder,
                    remote_name=f"mojo_agent.{yday}.log",
                )
                result["yesterday"] = {
                    "remote_name": f"mojo_agent.{yday}.log",
                    "bytes": ymeta.get("size"),
                    "webUrl": ymeta.get("webUrl"),
                }
            else:
                result["yesterday_error"] = (
                    f"No local rotated file for {yday} "
                    "(expected on free tier after restart — only current process logs exist)"
                )
        except Exception as e:
            result["yesterday_error"] = str(e)

    logging.getLogger("mojo").info(
        "On-demand log upload → OneDrive/%s/%s (%s bytes)",
        folder,
        REMOTE_NAME,
        result["bytes"],
    )
    return result


def start_onedrive_log_sync(file_ops_module, interval_sec: int = 90):
    """
    DEPRECATED. Continuous OneDrive sync removed to stop bandwidth burn.
    Use /uploadlog (admin command) for on-demand upload instead.
    Kept as a no-op so older call sites do not crash on import/startup.
    """
    print(
        "[mojo_logging] start_onedrive_log_sync is a no-op. "
        "Auto-upload disabled — use /uploadlog for on-demand sync."
    )
    setup_logging()
