from __future__ import annotations

import math

from src.market_maker.core import (DayInputs, MarketMaker, MarketMakerParams,
                                   State)


def test_redemption_monotonicity():
    params = MarketMakerParams(elasticity_redeem=2.0)
    mm = MarketMaker(params)
    s = State(
        reserve_cash=1_000_000,
        liability_points=100_000_000,
        earn_rate_ppd=1.5,
        redeem_cpp=0.010,
        ema_cost_cpp=0.012,
    )
    x = DayInputs(
        spend=100_000, baseline_redeem_rate=0.003, cost_cpp=0.012, ref_cpp=0.01
    )

    # Lower price
    s_low, info_low = mm.step(s, x)
    # Higher price -> expect more redemptions
    s_high = State(**{**s.__dict__})
    s_high.redeem_cpp = 0.012  # +20% price
    s_high, info_high = mm.step(s_high, x)

    assert info_high["redeem_points"] > info_low["redeem_points"]


def test_controller_moves_toward_band():
    params = MarketMakerParams(target_reserve_ratio=0.6, band_width=0.1)
    mm = MarketMaker(params)
    # Start with very low reserve ratio -> controller should NOT increase redeem quote
    s = State(
        reserve_cash=100_000,
        liability_points=200_000_000,
        earn_rate_ppd=2.0,
        redeem_cpp=0.012,
        ema_cost_cpp=0.012,
    )
    earn, redeem, dbg = mm.quote(s)
    assert (
        redeem <= s.redeem_cpp
    )  # should not raise redemption price when under-reserved
    assert earn <= s.earn_rate_ppd  # should reduce issuance when under-reserved
