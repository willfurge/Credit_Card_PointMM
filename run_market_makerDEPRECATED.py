from __future__ import annotations

import json
from pathlib import Path

from src.market_maker.core import MarketMakerParams, State
from src.market_maker.sim import SimConfig, run_sim


def main() -> None:
    # Initial state (tune as needed)
    init = State(
        reserve_cash=5_000_000.0,  # $5m reserve
        liability_points=450_000_000,  # 450m points outstanding
        earn_rate_ppd=1.5,  # 1.5 pts per $
        redeem_cpp=0.011,  # 1.1¢ per point
        ema_cost_cpp=0.012,  # initial cost estimate 1.2¢
    )

    params = MarketMakerParams(
        target_reserve_ratio=0.60,
        band_width=0.12,
        interchange_rate=0.018,
        elasticity_redeem=2.0,
        ema_alpha=0.15,
        k_redeem=0.40,
        k_issue=0.25,
        redeem_cpp_bounds=(0.005, 0.025),
        earn_rate_bounds=(0.5, 5.0),
        max_redeem_step=0.0015,
        max_issue_step=0.25,
    )

    simcfg = SimConfig(days=120, seed=42)

    result = run_sim(simcfg, init, params)

    out_path = Path("sim_output.json")
    out_path.write_text(json.dumps(result, default=lambda o: o.__dict__, indent=2))
    print(f"Wrote {out_path.resolve()}")
    # Quick summary
    hist = result["history"]
    in_band_days = sum(1 for r in hist if r["in_band"])
    print(f"In-band days: {in_band_days}/{len(hist)} ({in_band_days/len(hist):.1%})")
    print(f"Final reserve_ratio: {hist[-1]['reserve_ratio']:.3f}")
    print(f"Final reserve_cash: ${hist[-1]['reserve_cash']:,.0f}")
    print(f"Final liability_points: {hist[-1]['liability_points']:,.0f}")
    print(
        f"Final redeem_cpp: {hist[-1]['new_redeem_rate'] if 'new_redeem_rate' in hist[-1] else 'n/a'}"
    )


if __name__ == "__main__":
    main()
