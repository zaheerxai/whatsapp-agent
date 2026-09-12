"""
mojo_logging.py — Local application log with daily rotation + on-demand OneDrive upload.

Design (bandwidth-safe):
- All logging is local only. No background upload thread.
- Local file starts clean every calendar day (UTC date for determinism on cloud).
- Size-based safety rotation still applies.
- Owner runs /uploadlog (or /synclog) to push the current local log to OneDrive.
- Remote file is always the same name (mojo_agent_live.log) → overwrite.
  If the user deletes/moves that remote file, the next /uploadlog creates a
  brand-new file containing only the current local log (no old history).
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


def _ensure_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


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
            # Avoid clobbering an existing dated file from an earlier process
            if os.path.isfile(dated):
                # Append current content into the dated file, then clear active
                with open(_log_path, "rb") as src, open(dated, "ab") as dst:
                    dst.write(src.read())
                os.remove(_log_path)
            else:
                os.replace(_log_path, dated)
            print(f"[mojo_logging] Daily rotate → {os.path.basename(dated)}")
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
    except Exception as e:
        print(f"[mojo_logging] size rotate failed: {e}")


def _maybe_rotate():
    with _lock:
        _rotate_daily_if_needed()
        _rotate_size_if_needed()


class _FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        # Rotate before writing so a new day always starts clean
        try:
            _maybe_rotate()
        except Exception:
            pass
        super().emit(record)
        try:
            self.flush()
        except Exception:
            pass


def setup_logging():
    """Call once at startup. Local file only — no network."""
    global _started, _current_day
    if _started:
        return _log_path
    _ensure_dir()
    _current_day = _today_utc()
    _rotate_size_if_needed()

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
        "Live log ready (local only, daily rotate): %s", _log_path
    )
    return _log_path


def log_line(msg: str, level: int = logging.INFO):
    logging.getLogger("mojo").log(level, msg)


def get_log_path() -> str:
    return _log_path


def get_log_status() -> dict:
    """Return size / day info for the active local log (and any dated siblings)."""
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
    """
    if not file_ops_module or not getattr(file_ops_module, "onedrive_configured", lambda: False)():
        raise RuntimeError("OneDrive not configured")

    _maybe_rotate()

    if not os.path.isfile(_log_path):
        raise FileNotFoundError(f"No local log yet at {_log_path}")

    folder = (remote_folder or os.getenv("ONEDRIVE_FOLDER") or "MojoAgent").strip("/")
    size = os.path.getsize(_log_path)

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
    }

    # Optional: also push yesterday's rotated file under a dated name
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
        except Exception as e:
            result["yesterday_error"] = str(e)

    logging.getLogger("mojo").info(
        "On-demand log upload → OneDrive/%s/%s (%s bytes)",
        folder,
        REMOTE_NAME,
        result["bytes"],
    )
    return result


# ---------------------------------------------------------------------------
# Backward-compat stubs (old continuous sync is intentionally gone)
# ---------------------------------------------------------------------------

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
    # Still ensure local logging is ready
    setup_logging()
