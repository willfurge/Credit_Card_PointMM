from __future__ import annotations

import argparse

from src.market_maker.app import run_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit Card Points Market Maker")
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to config.json"
    )
    parser.add_argument(
        "--env-prefix",
        type=str,
        default="APP_",
        help="Environment variable prefix for overrides",
    )
    args = parser.parse_args()

    out = run_main(config_path=args.config, env_prefix=args.env_prefix)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
