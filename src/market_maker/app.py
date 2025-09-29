from __future__ import annotations

import csv
import json
import logging
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

from src.market_maker.core import MarketMakerParams, State
from src.market_maker.sim import SimConfig, run_sim

# ---- Robust imports with graceful fallbacks --------------------------------


# Config loader resolver
def _resolve_load_config():
    # Try canonical location
    try:
        import src.utils.config_loader as cfg_mod  # type: ignore

        for name in ("load_config", "get_config", "load", "read_config"):
            if hasattr(cfg_mod, name):
                return getattr(cfg_mod, name)
    except Exception:
        pass

    # Fallback: simple JSON + env override loader compatible with our docs
    def _bool_cast(s: str) -> bool | None:
        ss = s.strip().lower()
        if ss in ("true", "1", "yes", "y", "on"):
            return True
        if ss in ("false", "0", "no", "n", "off"):
            return False
        return None

    def _set_dotted(d: dict, dotted: str, value: Any) -> None:
        keys = dotted.split(".")
        cur = d
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = value

    def _merge_env(cfg: dict, prefix: str | None) -> dict:
        if not prefix:
            return cfg
        plen = len(prefix)
        for k, v in os.environ.items():
            if not k.startswith(prefix):
                continue
            path = k[plen:].replace("__", ".")
            # Try casts: bool, int, float, fallback to str
            b = _bool_cast(v)
            if b is not None:
                val = b
            else:
                try:
                    val = int(v)
                except ValueError:
                    try:
                        val = float(v)
                    except ValueError:
                        val = v
            _set_dotted(cfg, path, val)
        return cfg

    def _fallback_load_config(
        path: str, env_prefix: str | None = None, **_: Any
    ) -> dict:
        with open(path, "r") as f:
            cfg = json.load(f)
        return _merge_env(cfg, env_prefix)

    return _fallback_load_config


load_config = _resolve_load_config()


# Logger resolver
def _resolve_get_logger():
    try:
        import src.utils.logger as log_mod  # type: ignore

        for name in ("get_logger", "make_logger", "init_logger", "logger"):
            if hasattr(log_mod, name):
                return getattr(log_mod, name)
    except Exception:
        pass

    # Fallback basic logger
    def _get_logger(
        name: str = "app",
        *,
        level: int | str = "INFO",
        log_file: str | None = None,
        **_: Any,
    ) -> logging.Logger:
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if not logger.handlers:
            fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            logger.addHandler(sh)
            if log_file:
                Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                fh = logging.FileHandler(log_file)
                fh.setFormatter(fmt)
                logger.addHandler(fh)
        logger.propagate = False
        return logger

    return _get_logger


get_logger = _resolve_get_logger()


# ---- Helpers ---------------------------------------------------------------


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _lognormal_params(mean: float, std: float) -> Tuple[float, float]:
    """Convert desired lognormal mean/std to mu/sigma on log scale.

    For X ~ LogNormal(mu, sigma^2):
        E[X] = exp(mu + sigma^2 / 2)
        Var[X] = (exp(sigma^2) - 1) * exp(2mu + sigma^2)

    Given mean m and std s, we solve:
        sigma^2 = ln(1 + (s^2 / m^2))
        mu = ln(m) - 0.5 * sigma^2
    """
    if mean <= 0:
        raise ValueError("transaction_volume_mean must be positive")
    if std < 0:
        raise ValueError("transaction_volume_std must be non-negative")
    if std == 0:
        return (math.log(mean), 1e-9)
    sigma2 = math.log(1.0 + (std * std) / (mean * mean))
    sigma = math.sqrt(sigma2)
    mu = math.log(mean) - 0.5 * sigma2
    return mu, sigma


def build_from_config(cfg: Dict[str, Any]):
    """Build objects (State, Params, SimConfig) and filesystem paths from config dict."""
    # Paths
    data_dir = Path(cfg.get("paths", {}).get("data_dir", "data"))
    logs_dir = Path(cfg.get("paths", {}).get("logs_dir", "logs"))
    _ensure_dir(data_dir)
    _ensure_dir(logs_dir)

    # Logging
    log_level = cfg.get("logging", {}).get("level", "INFO")
    log_file = cfg.get("logging", {}).get("file", str(logs_dir / "app.log"))
    logger = get_logger(name="market_maker", level=log_level)

    # Initial state (allow overrides)
    state_cfg = cfg.get("state", {})
    init = State(
        reserve_cash=float(state_cfg.get("reserve_cash", 5_000_000.0)),
        liability_points=float(state_cfg.get("liability_points", 450_000_000.0)),
        earn_rate_ppd=float(state_cfg.get("earn_rate_ppd", 1.5)),
        redeem_cpp=float(state_cfg.get("redeem_cpp", 0.011)),
        ema_cost_cpp=float(state_cfg.get("ema_cost_cpp", 0.012)),
    )

    # Controller params
    p = cfg.get("controller", {})
    params = MarketMakerParams(
        target_reserve_ratio=float(p.get("target_reserve_ratio", 0.60)),
        band_width=float(p.get("band_width", 0.12)),
        interchange_rate=float(p.get("interchange_rate", 0.018)),
        elasticity_redeem=float(p.get("elasticity_redeem", 2.0)),
        ema_alpha=float(p.get("ema_alpha", 0.15)),
        k_redeem=float(p.get("k_redeem", 0.40)),
        k_issue=float(p.get("k_issue", 0.25)),
        redeem_cpp_bounds=tuple(p.get("redeem_cpp_bounds", (0.005, 0.025))),  # type: ignore
        earn_rate_bounds=tuple(p.get("earn_rate_bounds", (0.5, 5.0))),  # type: ignore
        max_redeem_step=float(p.get("max_redeem_step", 0.0015)),
        max_issue_step=float(p.get("max_issue_step", 0.25)),
    )

    # Simulation config
    sim_cfg = cfg.get("simulation", {})
    model_cfg = cfg.get("model", {})
    mean = float(sim_cfg.get("transaction_volume_mean", 10000))
    std = float(sim_cfg.get("transaction_volume_std", 2500))
    mu, sigma = _lognormal_params(mean, std)

    baseline_rr = float(model_cfg.get("base_redemption_rate", 0.003))
    cost_cpp_mean = float(model_cfg.get("cost_per_point_mean", 0.012))
    cost_cpp_sigma = float(model_cfg.get("cost_per_point_sigma", 0.0015))

    sim = SimConfig(
        days=int(sim_cfg.get("days", 120)),
        seed=int(sim_cfg.get("seed", 42)),
        spend_mu=mu,
        spend_sigma=sigma,
        baseline_redeem_rate=baseline_rr,
        cost_cpp_mean=cost_cpp_mean,
        cost_cpp_sigma=cost_cpp_sigma,
    )

    return init, params, sim, logger, data_dir


def run_main(config_path: str = "config.json", env_prefix: str = "APP_") -> Path:
    """Load config, run sim, persist artifacts, and log a summary.

    Returns path to the written JSON artifact.
    """
    cfg = load_config(config_path)
    init, params, sim, logger, data_dir = build_from_config(cfg)

    logger.info(
        "Starting Market Maker simulation", extra={"days": sim.days, "seed": sim.seed}
    )

    result = run_sim(sim, init, params)

    # Write history CSV and JSON
    hist = result["history"]
    out_json = data_dir / "sim_output.json"
    out_csv = data_dir / "sim_history.csv"

    out_json.write_text(
        json.dumps(result, default=lambda o: getattr(o, "__dict__", str(o)), indent=2)
    )

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "day",
                "spend",
                "cost_cpp",
                "issued_points",
                "redeem_points",
                "reserve_cash",
                "liability_points",
                "ema_cost_cpp",
                "reserve_ratio",
                "new_redeem_cpp",
                "new_earn_rate",
                "in_band",
            ],
        )
        writer.writeheader()
        for r in hist:
            writer.writerow(
                {
                    "day": r.get("day"),
                    "spend": r.get("spend"),
                    "cost_cpp": r.get("cost_cpp"),
                    "issued_points": r.get("issued_points"),
                    "redeem_points": r.get("redeem_points"),
                    "reserve_cash": r.get("reserve_cash"),
                    "liability_points": r.get("liability_points"),
                    "ema_cost_cpp": r.get("ema_cost_cpp"),
                    "reserve_ratio": r.get("reserve_ratio"),
                    "new_redeem_cpp": r.get("new_redeem_cpp"),
                    "new_earn_rate": r.get("new_earn_rate"),
                    "in_band": r.get("in_band"),
                }
            )

    # Summary logs
    in_band_days = sum(1 for r in hist if r.get("in_band"))
    logger.info(
        "Simulation complete",
        extra={
            "in_band_days": in_band_days,
            "total_days": len(hist),
            "final_reserve_ratio": hist[-1]["reserve_ratio"],
            "final_reserve_cash": hist[-1]["reserve_cash"],
            "final_liability_points": hist[-1]["liability_points"],
            "final_redeem_cpp": hist[-1]["new_redeem_cpp"],
            "final_earn_rate_ppd": hist[-1]["new_earn_rate"],
            "json_path": str(out_json),
            "csv_path": str(out_csv),
        },
    )

    return out_json
