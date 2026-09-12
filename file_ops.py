"""
file_ops.py — shared file create / upload / download for all agents.

Env:
  ONEDRIVE_CLIENT_ID
  ONEDRIVE_CLIENT_SECRET   (optional)
  ONEDRIVE_REFRESH_TOKEN
  ONEDRIVE_FOLDER          (default: AgentLogs)
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _folder() -> str:
    return (os.getenv("ONEDRIVE_FOLDER") or "AgentLogs").strip("/")


def onedrive_configured() -> bool:
    return bool(os.getenv("ONEDRIVE_CLIENT_ID") and os.getenv("ONEDRIVE_REFRESH_TOKEN"))


# ---------------------------------------------------------------------------
# Local file helpers
# ---------------------------------------------------------------------------

def write_text_file(
    content: str,
    filename: str,
    directory: Optional[str] = None,
) -> str:
    """
    Write UTF-8 text to a local file. Returns absolute path.
    If directory is None, uses system temp dir.
    """
    directory = directory or tempfile.gettempdir()
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content if content.endswith("\n") else content + "\n")
    return os.path.abspath(path)


def make_timestamped_name(prefix: str, ext: str = "txt") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ext = ext.lstrip(".")
    return f"{prefix}_{ts}.{ext}"


def safe_remove(path: str) -> None:
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except OSError as e:
        print(f"[file_ops] remove failed {path}: {e}")


# ---------------------------------------------------------------------------
# OneDrive auth (delegated refresh token — required for personal OneDrive)
# ---------------------------------------------------------------------------

def _onedrive_access_token() -> str:
    client_id = os.getenv("ONEDRIVE_CLIENT_ID")
    client_secret = os.getenv("ONEDRIVE_CLIENT_SECRET")
    refresh = os.getenv("ONEDRIVE_REFRESH_TOKEN")
    if not client_id or not refresh:
        raise RuntimeError(
            "OneDrive not configured. Set ONEDRIVE_CLIENT_ID and ONEDRIVE_REFRESH_TOKEN."
        )

    data = {
        "client_id": client_id,
        "grant_type": "refresh_token",
        "refresh_token": refresh,
        "scope": "https://graph.microsoft.com/Files.ReadWrite offline_access",
    }
    if client_secret:
        data["client_secret"] = client_secret

    r = requests.post(
        "https://login.microsoftonline.com/common/oauth2/v2.0/token",
        data=data,
        timeout=30,
    )
    r.raise_for_status()
    body = r.json()
    # Optional: if body has a new refresh_token, log a reminder to update env
    if body.get("refresh_token") and body["refresh_token"] != refresh:
        print("[file_ops] NOTE: Microsoft rotated refresh_token — update ONEDRIVE_REFRESH_TOKEN")
    return body["access_token"]


# ---------------------------------------------------------------------------
# Upload / download
# ---------------------------------------------------------------------------

def upload_to_onedrive(
    local_path: str,
    remote_folder: Optional[str] = None,
    remote_name: Optional[str] = None,
) -> dict:
    """
    Upload a local file to OneDrive folder.
    Returns Graph driveItem-ish dict: {name, webUrl, id, size, ...}
    """
    if not os.path.isfile(local_path):
        raise FileNotFoundError(local_path)

    folder = (remote_folder or _folder()).strip("/")
    name = remote_name or os.path.basename(local_path)
    access = _onedrive_access_token()

    url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{folder}/{name}:/content"
    headers = {
        "Authorization": f"Bearer {access}",
        "Content-Type": "application/octet-stream",
    }
    with open(local_path, "rb") as f:
        data = f.read()

    up = requests.put(url, headers=headers, data=data, timeout=180)
    up.raise_for_status()
    meta = up.json()
    print(f"[file_ops] uploaded → {folder}/{name} ({meta.get('size')} bytes)")
    return meta


def download_from_onedrive(
    remote_path: str,
    local_path: Optional[str] = None,
) -> str:
    """
    Download file from OneDrive path relative to drive root, e.g. 'AgentLogs/foo.txt'.
    Returns local absolute path.
    """
    remote_path = remote_path.lstrip("/")
    access = _onedrive_access_token()
    url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{remote_path}:/content"
    headers = {"Authorization": f"Bearer {access}"}

    r = requests.get(url, headers=headers, timeout=180)
    r.raise_for_status()

    if not local_path:
        local_path = os.path.join(tempfile.gettempdir(), os.path.basename(remote_path))
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(r.content)
    print(f"[file_ops] downloaded {remote_path} → {local_path}")
    return os.path.abspath(local_path)


def list_onedrive_folder(remote_folder: Optional[str] = None) -> list[dict]:
    """List children of a folder under drive root."""
    folder = (remote_folder or _folder()).strip("/")
    access = _onedrive_access_token()
    url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{folder}:/children"
    items: list[dict] = []
    while url:
        r = requests.get(
            url,
            headers={"Authorization": f"Bearer {access}"},
            timeout=60,
            params={"$top": 200} if ":" in url and "children" in url else None,
        )
        r.raise_for_status()
        body = r.json()
        items.extend(body.get("value") or [])
        url = body.get("@odata.nextLink")
    return items


def ensure_onedrive_folder(remote_folder: str) -> dict:
    """
    Ensure a folder path exists under drive root (creates intermediate folders).
    Returns the Graph folder item for the leaf.
    """
    remote_folder = (remote_folder or "").strip().strip("/")
    if not remote_folder:
        raise ValueError("remote_folder required")
    access = _onedrive_access_token()
    headers = {
        "Authorization": f"Bearer {access}",
        "Content-Type": "application/json",
    }
    parts = [p for p in remote_folder.split("/") if p]
    parent_path = ""
    leaf_meta: dict = {}
    for part in parts:
        current = f"{parent_path}/{part}" if parent_path else part
        # Does it exist?
        probe = requests.get(
            f"https://graph.microsoft.com/v1.0/me/drive/root:/{current}",
            headers={"Authorization": f"Bearer {access}"},
            timeout=30,
        )
        if probe.status_code == 200:
            leaf_meta = probe.json()
            parent_path = current
            continue
        # Create under parent (or root)
        if parent_path:
            create_url = (
                f"https://graph.microsoft.com/v1.0/me/drive/root:/{parent_path}:/children"
            )
        else:
            create_url = "https://graph.microsoft.com/v1.0/me/drive/root/children"
        body = {
            "name": part,
            "folder": {},
            "@microsoft.graph.conflictBehavior": "fail",
        }
        cr = requests.post(create_url, headers=headers, json=body, timeout=30)
        if cr.status_code in (200, 201):
            leaf_meta = cr.json()
        elif cr.status_code == 409:
            # Race: already exists
            again = requests.get(
                f"https://graph.microsoft.com/v1.0/me/drive/root:/{current}",
                headers={"Authorization": f"Bearer {access}"},
                timeout=30,
            )
            again.raise_for_status()
            leaf_meta = again.json()
        else:
            cr.raise_for_status()
        parent_path = current
        print(f"[file_ops] ensured folder → {current}")
    return leaf_meta


def list_onedrive_recursive(
    remote_folder: str,
    extensions: Optional[set] = None,
    max_items: int = 500,
) -> list[dict]:
    """
    Depth-first list of files under remote_folder.
    Each item: {name, path, size, id, webUrl, lastModifiedDateTime}.
    """
    remote_folder = (remote_folder or "").strip().strip("/")
    access = _onedrive_access_token()
    out: list[dict] = []
    stack = [remote_folder]

    while stack and len(out) < max_items:
        folder = stack.pop()
        try:
            children = list_onedrive_folder(folder)
        except Exception as e:
            print(f"[file_ops] list failed for {folder}: {e}")
            continue
        for item in children:
            name = item.get("name") or ""
            is_folder = "folder" in item
            # Graph path: prefer parentReference.path + name
            parent_ref = item.get("parentReference") or {}
            parent_path = parent_ref.get("path") or ""
            # parent path looks like /drive/root:/Documents/aimojo
            if ":/" in parent_path:
                rel_parent = parent_path.split(":/", 1)[-1]
            else:
                rel_parent = folder
            full_path = f"{rel_parent}/{name}".strip("/") if rel_parent else name

            if is_folder:
                stack.append(full_path)
                continue
            if extensions:
                ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
                if ext not in extensions:
                    continue
            out.append(
                {
                    "name": name,
                    "path": full_path,
                    "remote_path": full_path,
                    "size": item.get("size"),
                    "id": item.get("id"),
                    "webUrl": item.get("webUrl"),
                    "lastModifiedDateTime": item.get("lastModifiedDateTime"),
                }
            )
            if len(out) >= max_items:
                break
    return out


# ---------------------------------------------------------------------------
# High-level: write text + upload (used by exportlog)
# ---------------------------------------------------------------------------

def write_and_upload_text(
    content: str,
    filename_prefix: str,
    remote_folder: Optional[str] = None,
    delete_local: bool = True,
) -> dict:
    """
    Create a timestamped .txt, upload to OneDrive, optionally delete local copy.
    Returns {local_path, remote_name, webUrl, size, ...}
    """
    name = make_timestamped_name(filename_prefix, "txt")
    path = write_text_file(content, name)
    try:
        meta = upload_to_onedrive(path, remote_folder=remote_folder, remote_name=name)
        return {
            "local_path": path,
            "remote_name": name,
            "webUrl": meta.get("webUrl"),
            "id": meta.get("id"),
            "size": meta.get("size"),
            "folder": remote_folder or _folder(),
        }
    finally:
        if delete_local:
            safe_remove(path)


def write_and_upload_file(
    local_path: str,
    filename_prefix: str,
    ext: str = "jpg",
    remote_folder: Optional[str] = None,
    delete_local: bool = True,
) -> dict:
    """
    Upload an existing local file with a timestamped name.
    Returns {local_path, remote_name, webUrl, size, folder, ...}
    """
    name = make_timestamped_name(filename_prefix, ext)
    try:
        meta = upload_to_onedrive(local_path, remote_folder=remote_folder, remote_name=name)
        return {
            "local_path": local_path,
            "remote_name": name,
            "webUrl": meta.get("webUrl"),
            "id": meta.get("id"),
            "size": meta.get("size"),
            "folder": remote_folder or _folder(),
        }
    finally:
        if delete_local:
            safe_remove(local_path)