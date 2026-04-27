"""Shared JSON-backed user config at ``~/.tune/config.json``.

Read at startup, written on changes. Robust against missing/corrupt files —
errors during load/save are swallowed so a bad config never crashes the
app, and ``update_config`` always re-loads first to avoid clobbering keys
written by another module.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONFIG_PATH = Path.home() / ".tune" / "config.json"


def load_config() -> dict[str, Any]:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(cfg: dict[str, Any]) -> None:
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass


def update_config(**kwargs: Any) -> dict[str, Any]:
    """Load, merge in ``kwargs``, save, return the updated dict."""
    cfg = load_config()
    cfg.update(kwargs)
    save_config(cfg)
    return cfg
