"""Configuration loading and runtime overrides for public evaluators."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    value = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"config must contain a mapping: {config_path}")
    return apply_environment(value)


def apply_environment(config: dict[str, Any]) -> dict[str, Any]:
    """Apply optional credentials and endpoint settings without editing YAML."""
    llm = config.setdefault("llm", {})
    profile_name = os.environ.get("TEXTVBENCH_VLM_PROFILE", llm.get("active_profile"))
    if profile_name:
        llm["active_profile"] = profile_name
    profile = llm.setdefault("profiles", {}).setdefault(profile_name, {}) if profile_name else llm
    overrides = {
        "base_url": os.environ.get("TEXTVBENCH_VLM_BASE_URL"),
        "model": os.environ.get("TEXTVBENCH_VLM_MODEL"),
        "api_key": os.environ.get("TEXTVBENCH_VLM_API_KEY"),
    }
    for key, value in overrides.items():
        if value:
            profile[key] = value
    return config
