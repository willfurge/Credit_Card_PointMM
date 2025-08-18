# src/utils/config_loader.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../Credit_Card_PointMM
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.json"

# Load .env once at import
load_dotenv(_PROJECT_ROOT / ".env", override=False)


class Config:
    """
    Minimal, dependency-free config loader with:
      - JSON + environment variable overlay
      - dot-path get() access
      - reload() support
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else _DEFAULT_CONFIG_PATH
        self._cfg: Dict[str, Any] = {}
        self.reload()

    def _read_json(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _overlay_env(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Overlay environment variables onto matching config keys.

        Convention:
          - Double underscore __ => dot path separator (e.g., MODEL__CONFIDENCE_LEVEL -> model.confidence_level)
          - Single underscores are preserved as part of the key name
        Only ENV keys that match an existing config path are applied (case-insensitive).
        """
        flat: Dict[str, Any] = {}

        def walk(prefix: str, obj: Any):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    walk(f"{prefix}.{k}" if prefix else k, v)
            else:
                flat[prefix] = obj

        walk("", data)

        # Build a lowercase lookup of existing keys for case-insensitive matching
        flat_keys_lower = {k.lower(): k for k in flat.keys()}

        for env_key, env_val in os.environ.items():
            # Only convert double underscores to dots; keep single underscores
            # Example: MODEL__CONFIDENCE_LEVEL -> "model.confidence_level"
            dot_key_lower = env_key.lower().replace("__", ".")
            if dot_key_lower in flat_keys_lower:
                real_key = flat_keys_lower[dot_key_lower]
                current_val = flat[real_key]

                # Type-aware casting
                casted: Any = env_val
                if isinstance(current_val, bool):
                    casted = env_val.lower() in {"1", "true", "yes", "on"}
                elif isinstance(current_val, int):
                    try:
                        casted = int(env_val)
                    except ValueError:
                        pass
                elif isinstance(current_val, float):
                    try:
                        casted = float(env_val)
                    except ValueError:
                        pass

                self._assign(data, real_key, casted)

        return data

    def _assign(self, root: Dict[str, Any], dot_key: str, value: Any) -> None:
        keys = dot_key.split(".")
        d = root
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    def reload(self) -> None:
        base = self._read_json()
        # Provide defaults if keys are missing
        defaults = {
            "paths": {
                "data_raw": "data/raw",
                "data_processed": "data/processed",
                "logs": "logs",
            },
            "model": {
                "reserve_horizon_days": 365,
                "confidence_level": 0.995,
            },
            "dashboard": {"port": 8501},
        }
        # Merge defaults (defaults -> base)
        merged = defaults.copy()

        def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(a)
            for k, v in b.items():
                if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                    out[k] = deep_merge(out[k], v)
                else:
                    out[k] = v
            return out

        merged = deep_merge(merged, base)
        merged = self._overlay_env(merged)
        self._cfg = merged

    def get(self, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Dot-path access. If key is None, return whole config dict.
        """
        if key is None:
            return self._cfg
        node: Any = self._cfg
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


# Singleton-style accessor
_config_singleton: Optional[Config] = None


def get_config(path: Optional[Path] = None) -> Config:
    global _config_singleton
    if _config_singleton is None or path is not None:
        _config_singleton = Config(path)
    return _config_singleton
