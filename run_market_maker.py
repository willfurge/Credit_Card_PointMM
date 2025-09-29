from __future__ import annotations

from src.market_maker.app import run_main

if __name__ == "__main__":
    run_main("config.json", env_prefix="APP_")
