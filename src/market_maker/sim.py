from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List

from .core import DayInputs, MarketMaker, MarketMakerParams, State


@dataclass
class SimConfig:
    days: int = 90
    seed: int = 42
    # Base daily spend distribution (lognormal-ish)
    spend_mu: float = 9.5  # log-mean -> exp(mu) approx 13359
    spend_sigma: float = 0.5  # log-std
    # Baseline redemption propensity per day
    baseline_redeem_rate: float = 0.003  # 0.3% of outstanding per day
    # Cost per point dynamics (in $/pt)
    cost_cpp_mean: float = 0.012  # 1.2 cents
    cost_cpp_sigma: float = 0.0015


def _draw_spend(mu: float, sigma: float, rng: random.Random) -> float:
    # Lognormal sample via underlying normal on ln scale
    z = rng.normalvariate(mu, sigma)
    return math.exp(z)


def _draw_cost(mean: float, sigma: float, rng: random.Random) -> float:
    # Truncated normal at a small positive floor
    c = rng.normalvariate(mean, sigma)
    return max(0.004, c)  # floor at 0.4¢


def run_sim(cfg: SimConfig, init: State, params: MarketMakerParams) -> Dict[str, Any]:
    rng = random.Random(cfg.seed)
    mm = MarketMaker(params)

    hist: List[Dict[str, Any]] = []

    s = init
    for day in range(cfg.days):
        spend = _draw_spend(cfg.spend_mu, cfg.spend_sigma, rng)
        cost = _draw_cost(cfg.cost_cpp_mean, cfg.cost_cpp_sigma, rng)
        x = DayInputs(
            spend=spend, baseline_redeem_rate=cfg.baseline_redeem_rate, cost_cpp=cost
        )
        s, info = mm.step(s, x)
        info["day"] = day + 1
        info["spend"] = spend
        info["cost_cpp"] = cost
        hist.append(info)

    return {"history": hist, "final_state": s, "params": params, "sim_config": cfg}
