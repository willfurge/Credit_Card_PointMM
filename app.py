# app.py
from pathlib import Path

from src.utils.config_loader import get_config
from src.utils.logger import get_logger


def main():
    cfg = get_config()
    logger = get_logger("startup")

    logger.info("Booting Credit Card Points MM project...")
    logger.debug(f"Full config loaded: {cfg.get()}")

    data_raw = Path(cfg.get("paths.data_raw"))
    data_raw.mkdir(parents=True, exist_ok=True)
    logger.info(f"Verified data directory: {data_raw.resolve()}")

    logger.info("Ready to code 🚀")


if __name__ == "__main__":
    main()
