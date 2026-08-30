"""
mojo_logging.py — Full application log → local file + periodic OneDrive sync.

Design for API limits & continuity:
- Everything is written locally first (fast, unlimited).
- On boot/redeploy, pulls the existing live log from OneDrive so history continues.
- OneDrive upload overwrites the SAME remote file on a timer (default 90s),
  so Graph API calls stay low (~40/hour) and file size continuously grows.
- Optional size rotation keeps the local file bounded.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

LOG_DIR = os.environ.get("MOJO_LOG_DIR", "/tmp/mojo_logs")
LOG_NAME = "mojo_agent.log"
REMOTE_NAME = "mojo_agent_live.log"  # fixed name → overwrite, not spam new files
UPLOAD_INTERVAL_SEC = int(os.environ.get("MOJO_LOG_UPLOAD_SEC", "90"))
MAX_LOCAL_BYTES = int(os.environ.get("MOJO_LOG_MAX_BYTES", str(5 * 1024 * 1024)))  # 5 MB

_lock = threading.Lock()
_started = False
_log_path = os.path.join(LOG_DIR, LOG_NAME)


def _ensure_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def _rotate_if_needed():
    try:
        if os.path.isfile(_log_path) and os.path.getsize(_log_path) > MAX_LOCAL_BYTES:
            bak = _log_path + ".prev"
            if os.path.isfile(bak):
                os.remove(bak)
            os.replace(_log_path, bak)
    except Exception as e:
        print(f"[mojo_logging] rotate failed: {e}")


def _hydrate_from_onedrive(file_ops_module, remote_folder: Optional[str] = None):
    """
    Download existing live log from OneDrive on startup if ephemeral disk was wiped.
    Ensures continuous log history across container deployments/restarts.
    """
    if not file_ops_module or not getattr(file_ops_module, "onedrive_configured", lambda: False)():
        return

    folder = (remote_folder or os.getenv("ONEDRIVE_FOLDER") or "MojoAgent").strip("/")
    remote_path = f"{folder}/{REMOTE_NAME}"
    temp_download = _log_path + ".remote"

    try:
        _ensure_dir()
        # Fetch existing log from OneDrive
        file_ops_module.download_from_onedrive(remote_path, local_path=temp_download)

        if os.path.isfile(temp_download) and os.path.getsize(temp_download) > 0:
            with _lock:
                local_bytes = b""
                if os.path.isfile(_log_path):
                    with open(_log_path, "rb") as f:
                        local_bytes = f.read()

                with open(temp_download, "rb") as rf:
                    remote_bytes = rf.read()

                # Merge downloaded remote log + any startup logs written locally before hydration
                with open(_log_path, "wb") as lf:
                    lf.write(remote_bytes)
                    if local_bytes and not remote_bytes.endswith(local_bytes):
                        lf.write(local_bytes)

            print(f"[mojo_logging] Hydrated live log from OneDrive ({os.path.getsize(_log_path)} bytes)")
    except Exception as e:
        # Normal on first-ever run if remote file doesn't exist yet
        print(f"[mojo_logging] Hydration skipped (file new or unavailable): {e}")
    finally:
        if os.path.isfile(temp_download):
            try:
                os.remove(temp_download)
            except Exception:
                pass


class _FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        try:
            self.flush()
        except Exception:
            pass


def setup_logging():
    """Call once at startup. Mirrors root + key loggers to the live file."""
    global _started
    if _started:
        return _log_path
    _ensure_dir()
    _rotate_if_needed()

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

    # Also keep console
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    # Quiet noisy httpx polling but keep real errors
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _started = True
    logging.getLogger("mojo").info("Live log file ready: %s", _log_path)
    return _log_path


def log_line(msg: str, level: int = logging.INFO):
    logging.getLogger("mojo").log(level, msg)


def _upload_once(file_ops_module, remote_folder: Optional[str] = None):
    if not file_ops_module or not getattr(file_ops_module, "onedrive_configured", lambda: False)():
        return
    if not os.path.isfile(_log_path):
        return
    folder = remote_folder or os.getenv("ONEDRIVE_FOLDER") or "MojoAgent"
    try:
        with _lock:
            meta = file_ops_module.upload_to_onedrive(
                _log_path,
                remote_folder=folder,
                remote_name=REMOTE_NAME,
            )
        logging.getLogger("mojo").info(
            "Log synced → OneDrive/%s/%s (%s bytes)",
            folder,
            REMOTE_NAME,
            meta.get("size"),
        )
    except Exception as e:
        # Don't crash the bot for log upload failures
        print(f"[mojo_logging] OneDrive upload failed: {e}")


def start_onedrive_log_sync(file_ops_module, interval_sec: int = UPLOAD_INTERVAL_SEC):
    """Background thread: hydrate log on startup then sync continuously every interval_sec."""
    # Step 1: Hydrate prior logs from OneDrive before initializing handlers
    _hydrate_from_onedrive(file_ops_module)
    
    # Step 2: Bind local logger to the hydrated file
    setup_logging()

    def loop():
        # Initial small delay so startup noise is included
        time.sleep(15)
        while True:
            try:
                _rotate_if_needed()
                _upload_once(file_ops_module)
            except Exception as e:
                print(f"[mojo_logging] sync loop error: {e}")
            time.sleep(max(30, interval_sec))

    t = threading.Thread(target=loop, name="mojo-log-sync", daemon=True)
    t.start()
    logging.getLogger("mojo").info(
        "OneDrive log sync started (every %ss → %s/%s)",
        interval_sec,
        os.getenv("ONEDRIVE_FOLDER") or "MojoAgent",
        REMOTE_NAME,
    )