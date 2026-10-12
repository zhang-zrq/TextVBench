"""Redact credentials in artifacts and local paths in review responses."""

from __future__ import annotations

import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any


_CREDENTIAL_KEYS = {"api_key", "apikey", "authorization", "password", "passwd", "token", "secret", "credentials"}
_PRIVATE_PATH = re.compile(r"(?<![A-Za-z0-9:])/(?:mnt|home|root|tmp|var|opt|private|workspace|data)/[^\s\"'<>]+")
_WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|\\\\)[^\s\"'<>]+")
_URL = re.compile(r"https?://[^\s\"'<>]+")


def _credential_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in _CREDENTIAL_KEYS or normalized.endswith(("_api_key", "_token", "_secret", "_password"))


def redact_credentials(value: Any) -> Any:
    """Return a copy with named credentials removed, preserving runtime config."""
    if isinstance(value, dict):
        return {
            key: "[redacted]" if _credential_key(str(key)) else redact_credentials(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_credentials(item) for item in value]
    return value


def _absolute_basename(value: str) -> str | None:
    for path_type in (PureWindowsPath, PurePosixPath):
        path = path_type(value)
        if path.is_absolute():
            return path.name
    return None


def redact_debug_value(value: Any, *, key: str = "") -> Any:
    """Sanitize review JSON even when bundles came from another OS."""
    if isinstance(value, dict):
        return {
            name: "[redacted]" if _credential_key(str(name)) or str(name).lower() in {"base_url", "endpoint"}
            else redact_debug_value(item, key=str(name).lower())
            for name, item in value.items()
        }
    if isinstance(value, list):
        return [redact_debug_value(item, key=key) for item in value]
    if isinstance(value, str):
        basename = _absolute_basename(value)
        path_key = key in {"input", "output", "result", "marker_image", "project_root"} or key.endswith(("_path", "_dir", "_root"))
        if basename is not None and (path_key or not any(char.isspace() for char in value)):
            return basename
        value = _PRIVATE_PATH.sub(lambda match: f"[path]/{PurePosixPath(match.group()).name}", value)
        value = _WINDOWS_PATH.sub(lambda match: f"[path]/{PureWindowsPath(match.group()).name}", value)
        if key in {"error", "exception", "message"}:
            value = _URL.sub("[endpoint]", value)
    return value
