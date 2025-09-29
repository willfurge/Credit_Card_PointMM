from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# --- Data Models ------------------------------------------------------------


@dataclass
class DayInputs:
    """Exogenous daily inputs for the market.

    Attributes:
        spend: Total card spend in dollars for the day.
        baseline_redeem_rate: Baseline fraction of outstanding points that would be redeemed
            if redemption value equals ref_cpp (units: fraction per day).
        cost_cpp: Bank's realized dollar cost per point redeemed (e.g., 0.012 == 1.2¢).
        ref_cpp: Reference "neutral" redemption value in dollars per point.
    """

    spend: float
    baseline_redeem_rate: float
    cost_cpp: float
    ref_cpp: float = 0.01


@dataclass
class State:
    """System state carried across days.

    Attributes:
        reserve_cash: Available cash reserve to fund redemptions (dollars).
        liability_points: Outstanding points (liability units).
        earn_rate_ppd: Points per dollar issued on spend (≥0).
        redeem_cpp: Redemption quote offered to customers ($/point).
        ema_cost_cpp: Internal estimate of average cost per redeemed point ($/pt).
    """

    reserve_cash: float
    liability_points: float
    earn_rate_ppd: float
    redeem_cpp: float
    ema_cost_cpp: float


@dataclass
class MarketMakerParams:
    """Control & economic parameters of the bot."""

    target_reserve_ratio: float = 0.60  # target cash / (points * cost) ratio
    band_width: float = 0.10  # ± band around target for "ok" zone
    interchange_rate: float = 0.018  # revenue on spend (1.8% default)
    elasticity_redeem: float = 2.0  # sensitivity of redemption to price (unitless)
    ema_alpha: float = 0.15  # smoothing for cost estimate
    # Controller gains (simple proportional control)
    k_redeem: float = 0.40  # controls redeem_cpp (↑ when ratio above target)
    k_issue: float = 0.25  # controls earn_rate_ppd (↓ when ratio below target)
    # Bounds (safety rails)
    redeem_cpp_bounds: Tuple[float, float] = (0.005, 0.025)  # $/pt, 0.5¢ to 2.5¢
    earn_rate_bounds: Tuple[float, float] = (0, 5.0)  # pts per $
    # Damping to avoid oscillation
    max_redeem_step: float = 0.0015  # max $/pt change per day
    max_issue_step: float = 0.25  # max pts/$ change per day


class MarketMaker:
    """Market-making controller for credit card points.

    The bot adjusts:
      - redeem_cpp: the redemption value customers get per point ($/point)
      - earn_rate_ppd: points issued per dollar of spend

    Objective: keep reserve_ratio near a target while managing liability.
    reserve_ratio := reserve_cash / (liability_points * ema_cost_cpp + 1e-9)
    """

    def __init__(self, params: MarketMakerParams) -> None:
        self.p = params

    def _clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _reserve_ratio(self, s: State) -> float:
        denom = s.liability_points * max(s.ema_cost_cpp, 1e-9) + 1e-9
        return s.reserve_cash / denom

    def quote(self, s: State) -> Tuple[float, float, Dict[str, Any]]:
        rr = self._reserve_ratio(s)
        p = self.p
        lo, hi = p.redeem_cpp_bounds
        e_lo, e_hi = p.earn_rate_bounds

        # error defined so err>0 means "under target" (need to tighten)
        err = max(0.0, p.target_reserve_ratio - rr)
        err_relax = max(0.0, rr - p.target_reserve_ratio)

        # deadband: no movement inside [target - band, target + band]
        in_band = abs(rr - p.target_reserve_ratio) <= p.band_width

        redeem_cpp = s.redeem_cpp
        earn_rate = s.earn_rate_ppd

        if not in_band:
            # tighten when under-reserved: cut redeem quote, cut issuance
            if err > 0.0:
                redeem_adj = (
                    -p.k_redeem * err
                )  # lower redeem value to slow cash outflow
                issue_adj = -2.0 * p.k_issue * err  # stronger push on issuance down
            else:
                # relax when over-reserved: raise redeem value slightly, allow more issuance
                redeem_adj = p.k_redeem * err_relax
                issue_adj = 0.5 * p.k_issue * err_relax
            # damp daily moves
            redeem_adj = self._clamp(redeem_adj, -p.max_redeem_step, p.max_redeem_step)
            issue_adj = self._clamp(issue_adj, -p.max_issue_step, p.max_issue_step)
            redeem_cpp = self._clamp(redeem_cpp + redeem_adj, lo, hi)
            earn_rate = self._clamp(earn_rate + issue_adj, e_lo, e_hi)

        dbg = {
            "reserve_ratio": rr,
            "err_under": err,
            "err_over": err_relax,
            "new_redeem_cpp": redeem_cpp,
            "new_earn_rate": earn_rate,
            "in_band": in_band,
        }

        return earn_rate, redeem_cpp, dbg

    def step(self, s: State, x: DayInputs) -> Tuple[State, Dict[str, Any]]:
        """Advance one day, applying quotes and accounting flows.

        Demand model:
            redeem_points = (baseline_rate * outstanding_points) * exp(elasticity * (redeem_cpp / ref_cpp - 1))
            clipped to outstanding_points.
        Issuance:
            issued_points = spend * earn_rate_ppd

        Cash:
            + interchange = spend * interchange_rate
            - redemption_cost = redeem_points * cost_cpp
        """
        # Get control decisions
        earn_rate_ppd, redeem_cpp, dbg = self.quote(s)
        rr = self._reserve_ratio(s)
        if rr < (
            self.p.target_reserve_ratio - 0.10
        ):  # e.g., target 0.60 → threshold 0.50
            earn_rate_ppd = 0.01  # stop issuing new points
        elif rr < (self.p.target_reserve_ratio - 0.05):
            earn_rate_ppd *= 0.25
        # Issuance and outstanding
        issued_points = max(0.0, x.spend) * earn_rate_ppd
        outstanding_points = s.liability_points + issued_points

        # Redemption demand (log-response to relative price)
        redeemable_points = s.liability_points
        price_rel = (redeem_cpp / max(x.ref_cpp, 1e-9)) - 1.0
        redeem_multiplier = math.exp(self.p.elasticity_redeem * price_rel)
        base_redeem = max(0.0, x.baseline_redeem_rate) * redeemable_points
        redeem_points = min(redeemable_points, base_redeem * redeem_multiplier)

        # Cash accounting
        interchange = max(0.0, x.spend) * self.p.interchange_rate
        redemption_cost = redeem_points * max(x.cost_cpp, 0.0)
        new_reserve = s.reserve_cash + interchange - redemption_cost

        # Liability update
        new_liability = s.liability_points - redeem_points + issued_points

        # Update EMA of cost
        ema_cost = (
            self.p.ema_alpha * x.cost_cpp + (1 - self.p.ema_alpha) * s.ema_cost_cpp
        )

        new_state = State(
            reserve_cash=new_reserve,
            liability_points=new_liability,
            earn_rate_ppd=earn_rate_ppd,
            redeem_cpp=redeem_cpp,
            ema_cost_cpp=ema_cost,
        )

        # Diagnostics
        rr = self._reserve_ratio(new_state)
        band_lo = self.p.target_reserve_ratio - self.p.band_width
        band_hi = self.p.target_reserve_ratio + self.p.band_width
        in_band = band_lo <= rr <= band_hi

        info = {
            **dbg,
            "issued_points": issued_points,
            "redeem_points": redeem_points,
            "interchange": interchange,
            "redemption_cost": redemption_cost,
            "reserve_cash": new_reserve,
            "liability_points": new_liability,
            "ema_cost_cpp": ema_cost,
            "reserve_ratio": rr,
            "in_band": in_band,
        }
        return new_state, info
