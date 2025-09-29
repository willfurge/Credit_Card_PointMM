
Market Making Bot — Credit Card Points (MVP)
===========================================

Files created:
- src/market_maker/core.py     -> MarketMaker controller, data models, step function
- src/market_maker/sim.py      -> Simple stochastic simulator
- run_market_maker.py          -> Runs a 120-day sim and writes sim_output.json
- tests/test_market_maker.py   -> Basic invariants

Quickstart
----------
1) Ensure numpy is installed (if not already in requirements):
   pip install numpy

2) Run tests:
   python -m pytest -q

3) Run the simulation:
   python run_market_maker.py

Concept
-------
We model the "market" for points with two knobs:
- earn_rate_ppd (points per dollar issued)
- redeem_cpp ($ per point redeemed)

The MarketMaker adjusts these daily to keep the reserve ratio:
    reserve_cash / (liability_points * ema_cost_cpp)
near a target band, while the simulator generates daily spend and redemption costs.
