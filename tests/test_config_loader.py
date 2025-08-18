import json
import os
from pathlib import Path

from src.utils.config_loader import Config, get_config


def test_defaults_without_file(tmp_path):
    # No config.json -> defaults only
    cfg = Config(path=tmp_path / "config.json")
    assert cfg.get("paths.data_raw") == "data/raw"
    assert cfg.get("model.confidence_level") == 0.995


def test_with_config_file_and_env_overlay(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "paths": {"data_raw": "data/raw_custom"},
                "model": {"reserve_horizon_days": 180, "confidence_level": 0.9},
            }
        ),
        encoding="utf-8",
    )

    # Overlay env variable onto an existing key (note: underscores -> dots)
    monkeypatch.setenv("MODEL__CONFIDENCE_LEVEL", "0.97")

    cfg = Config(path=cfg_path)
    assert cfg.get("paths.data_raw") == "data/raw_custom"
    assert cfg.get("model.reserve_horizon_days") == 180
    assert cfg.get("model.confidence_level") == 0.97


def test_get_whole_config_dict(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text("{}", encoding="utf-8")
    cfg = Config(path=cfg_path)
    whole = cfg.get()
    assert isinstance(whole, dict)
    assert "paths" in whole and "model" in whole
